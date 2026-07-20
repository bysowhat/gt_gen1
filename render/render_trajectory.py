"""回放并渲染【一整条关键帧下采样后的轨迹】：机械臂沿 (L,8) 逐关键帧运动，工件+障碍固定，
每个关键帧渲染左右目 RGB + 深度。与 render/render_seam.py（钉死机械臂@retract、动工件）相反：
本脚本钉死工件+障碍、动机械臂，并行环境沿轨迹关键帧铺开。

输入是 scripts/traj_downsample.py 的产物（Scene pkl，含 sampled_trajectories/工件/障碍/init pose/焊缝）。
渲染阶段关节角已存在 (L,8) 里，Isaac 自己驱动关节 + 相机挂 Link6 算位姿——【不需要 curobo/warp】，
故单进程即可（先 Scene.load 抽成纯 numpy，再启动 AppLauncher，之后只用 numpy+isaac）。

相机内外参、robot USD/cfg 全部取自 configs/default.yaml（单一真源，与下采样 FK 同源）。

用法（无显示器，渲染 pkl 内第一条轨迹）：
    conda run -n env_isaaclab python render/render_trajectory.py \
        --pkl /media/a/新加卷/tempt/5_ds/BEAM_..._type2_seam0.pkl \
        --out /tmp/render_traj --headless
    渲染指定轨迹：... --seam-id 0 --hand forehand --traj-index -1
    渲染 pkl 内全部有采样结果的成功轨迹：... --all

输出（轨迹目录下分 left/right 两个平铺文件夹，各帧按帧号前缀命名，每侧一个含全帧信息的 render_info.npy）：
    <out>/<part_stem>/seam{sid}_{hand}{index}_traj{j}/
        left/  {k}_rgb.jpg  {k}_depth.exr ...  render_info.npy   # render_info 含【所有帧】信息(按帧堆叠)
        right/ {k}_rgb.jpg  {k}_depth.exr ...  render_info.npy
        _traj_meta.npy       # 轨迹级：(L,8) 原数组 + observe/goal + workpiece_pose7 + seam/hand/index …
        _DONE_...            # 完成哨兵（断点续跑）
    （k = 关键帧行号 0..L-1；--observe-only 时只有 observe 帧，帧号可能不连续，render_info.frame_indices 记录映射）
"""
import argparse
import os
import sys
from pathlib import Path

# OPENCV_IO_ENABLE_OPENEXR 必须在 import cv2 前设置
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RENDER_DIR = os.path.dirname(os.path.abspath(__file__))
_TS = os.path.join(_PROJECT_ROOT, "traj_sampler")
for _p in (_PROJECT_ROOT, _TS, _RENDER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_WAREHOUSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "default.yaml")

# ===== 左右目相机内外参：单一真源 = configs/default.yaml（同 render_seam）=====
CAM_LEFT_POS = CAM_LEFT_ROT = CAM_RIGHT_POS = CAM_RIGHT_ROT = None
CAM_WIDTH = CAM_HEIGHT = CAM_FOCAL_LENGTH = CAM_FOCUS_DISTANCE = None
CAM_H_APERTURE = CAM_V_APERTURE = CAM_CLIP = None
DEPTH_KEY = "distance_to_image_plane"

# ===== 整组离地高度约束（同 render_seam）=====
FLOOR_CLEARANCE = 0.0
MAX_LIFT = 5.0
ROBOT_Z_RANGE_EST = (0.0, 3.0)


def parse_args():
    p = argparse.ArgumentParser(description="回放并渲染整条采样后轨迹（左右目 RGB+深度）")
    p.add_argument("--pkl", required=True, help="下采样 Scene pkl（traj_downsample.py 产物）")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="default.yaml 路径")
    p.add_argument("--out", required=True, help="输出根目录")
    # 轨迹选择（不给 --all 时按下面单选一条，语义同 scene_viz.show_trajectory_isaacsim）
    p.add_argument("--seam-id", type=int, default=None, help="焊缝号（默认取第一条有采样结果的）")
    p.add_argument("--hand", default=None, choices=["forehand", "backhand"],
                   help="只渲该手别（默认不限）")
    p.add_argument("--traj-index", type=int, default=-1, help="过滤后第几条（默认 -1=最新）")
    p.add_argument("--all", action="store_true",
                   help="渲染 pkl 内【全部】有采样结果的成功轨迹（忽略上面单选）")
    p.add_argument("--observe-only", action="store_true",
                   help="只渲染 observe==1 的关键帧（默认关：渲染全部 L 个关键帧=整条轨迹）")
    # 通用（同 render_seam）
    p.add_argument("--warehouse-usd", default=DEFAULT_WAREHOUSE_USD, help="环境 USD")
    p.add_argument("--max-envs", type=int, default=8, help="并行环境数上限")
    p.add_argument("--spacing", type=float, default=40.0, help="相邻环境间距(米)")
    p.add_argument("--settle-steps", type=int, default=12, help="读图前 step 帧数")
    p.add_argument("--force-convert", action="store_true", help="强制重转工件 USD")
    return p


# ======================= 阶段一：启动 Isaac 前，先把渲染所需数据从 Scene 抽成纯 numpy =======================
# （Scene.load 会经 __init__ 惰性 import plan_init_pose；把它放在 AppLauncher 之前、且渲染阶段只用
#  抽出的 numpy，避免 warp/curobo 与运行中的 isaac 交织——与 demo_scene --task viz 同一安全顺序。）

def _load_cameras_from_config(config_path):
    """从 default.yaml 的 sensor.camera(左)+sensor.camera_right(右) 读左右目内外参，覆盖模块级常量。"""
    import yaml
    global CAM_LEFT_POS, CAM_LEFT_ROT, CAM_RIGHT_POS, CAM_RIGHT_ROT
    global CAM_WIDTH, CAM_HEIGHT, CAM_FOCAL_LENGTH, CAM_FOCUS_DISTANCE
    global CAM_H_APERTURE, CAM_V_APERTURE, CAM_CLIP
    with open(config_path, "r") as f:
        sensor = yaml.safe_load(f)["sensor"]
    left = sensor["camera"]
    right = sensor.get("camera_right", left)
    CAM_LEFT_POS = tuple(float(v) for v in left["extrinsic_pos"])
    CAM_LEFT_ROT = tuple(float(v) for v in left["extrinsic_quat_wxyz"])
    CAM_RIGHT_POS = tuple(float(v) for v in right["extrinsic_pos"])
    CAM_RIGHT_ROT = tuple(float(v) for v in right["extrinsic_quat_wxyz"])
    CAM_WIDTH = int(left["width"]); CAM_HEIGHT = int(left["height"])
    CAM_FOCAL_LENGTH = float(left["focal_length"]); CAM_FOCUS_DISTANCE = float(left["focus_distance"])
    CAM_H_APERTURE = float(left["horizontal_aperture"]); CAM_V_APERTURE = float(left["vertical_aperture"])
    CAM_CLIP = tuple(float(v) for v in left["clipping_range"])
    print(f"[main] 相机内外参 ← {config_path}（{CAM_WIDTH}x{CAM_HEIGHT}，"
          f"左目 pos={CAM_LEFT_POS}，右目 pos={CAM_RIGHT_POS}）")


def load_robot_cfg(robot_cfg_path):
    """读 cuRobo 机器人 yml，返回 (urdf_path, joint_names, retract_config)。"""
    import yaml
    with open(robot_cfg_path, "r") as f:
        d = yaml.safe_load(f)
    kin = d["robot_cfg"]["kinematics"]
    cspace = kin["cspace"]
    return kin["urdf_path"], list(cspace["joint_names"]), list(cspace["retract_config"])


def collect_render_jobs(pkl_path, seam_id, hand, traj_index, render_all):
    """Scene.load → 抽出待渲染轨迹的纯 numpy 数据（每条一个 job dict）。

    job = dict(key=(sid,hand,index,j), positions=(L,6), observe=(L,), goal=(L,), sampled=(L,8),
               workpiece_pose7=(7,), obstacles=[(verts,faces,color)...], seam_line_base=(N,3)|None)
    工件几何路径 workpiece_obj 单个 pkl 内恒定，单独返回。返回 (workpiece_obj, jobs)。
    """
    import numpy as np
    from gt_gen.scene import Scene

    scene = Scene.load(pkl_path)
    n_seam = len(scene.seams)

    # 铺平「有采样结果」的成功轨迹（与 show_trajectory_isaacsim 的 ds=True 口径一致）
    def _iter_sampled(sid):
        smap = scene.sampled_trajectories.get(sid, {})
        tmap = scene.trajectories.get(sid, {})
        for (h, idx), slst in smap.items():
            if hand is not None and h != hand:
                continue
            tlst = tmap.get((h, idx), [])
            for j, samp in enumerate(slst):
                if samp is None:
                    continue                       # 非成功轨迹：占位 None，跳过
                entry = tlst[j] if j < len(tlst) else {}
                yield (sid, h, idx, j, samp, entry)

    if render_all:
        candidates = [t for sid in sorted(scene.sampled_trajectories) for t in _iter_sampled(sid)]
    else:
        sid = (scene.seam_id if seam_id is None
               else int(seam_id) % n_seam)
        if seam_id is None:
            # 默认取第一条有采样结果的焊缝
            for s in sorted(scene.sampled_trajectories):
                if any(True for _ in _iter_sampled(s)):
                    sid = s; break
        flat = list(_iter_sampled(sid))
        if not flat:
            raise RuntimeError(f"无可渲染的采样轨迹：seam_id={sid}"
                               f"{'' if hand is None else f' hand={hand}'}"
                               f"（请先 scripts/traj_downsample.py 生成 sampled_trajectories）")
        n = len(flat)
        if not (-n <= int(traj_index) < n):
            raise IndexError(f"traj_index 越界：{traj_index}，共 {n} 条")
        candidates = [flat[int(traj_index)]]

    if not candidates:
        raise RuntimeError("pkl 内没有任何有采样结果的成功轨迹（sampled_trajectories 为空）")

    jobs = []
    for (sid, h, idx, j, samp, entry) in candidates:
        # 切到该轨迹所属焊缝 + init pose，取固定工件位姿 + 该手别障碍（mesh 系 → base 系）
        scene._set_cur_seam(sid)
        cand = scene.set_init_pose(h, idx)
        T = np.asarray(cand.T_workpiece_in_base, float)            # mesh 系 → base 系
        R_T, t_T = T[:3, :3], T[:3, 3]
        wp_pose7 = np.asarray(cand.workpiece_pose7, float)         # 工件在 base 系 pose7(wxyz)

        arr = np.asarray(samp, float)                              # (L,8)=[q1..q6, observe, goal]
        positions = arr[:, :6]
        observe = arr[:, 6].astype(np.int64)
        goal = arr[:, 7].astype(np.int64)

        # 障碍：实体 trimesh（mesh 系）→ base 系顶点，存 (verts,faces,color)。统一灰蓝色（纯视觉）。
        obstacles = []
        for tm in scene._obstacle_solid_trimeshes(h):
            verts = np.asarray(tm.vertices, float) @ R_T.T + t_T
            faces = np.asarray(tm.faces, np.int64).reshape(-1, 3)
            obstacles.append((verts, faces, (0.62, 0.64, 0.67)))

        # 当前焊缝折线（mesh 系 → base 系），供 render_info（失败则 None）
        seam_line_base = None
        try:
            seam_pts = np.asarray(scene._seam_frame()[6], float)   # (N,3) mesh 系
            seam_line_base = seam_pts @ R_T.T + t_T
        except Exception as e:
            print(f"[main] seam#{sid} 取焊缝线失败（忽略）：{e}")

        jobs.append(dict(
            key=(sid, h, idx, j),
            positions=positions, observe=observe, goal=goal, sampled=arr,
            workpiece_pose7=wp_pose7, obstacles=obstacles, seam_line_base=seam_line_base,
            status=entry.get("status"), goal_index=entry.get("goal_index"),
            variant=entry.get("variant"),
        ))
    return scene.workpiece_obj, jobs


# ---- 解析 CLI + 抽数据（都在启动 Isaac 之前）----
_parser = parse_args()
from isaaclab.app import AppLauncher  # noqa: E402  仅 import（不启动），加 --headless 等参数
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.enable_cameras = True

_load_cameras_from_config(args_cli.config)
import yaml as _yaml  # noqa: E402
with open(args_cli.config, "r") as _f:
    _robot_yaml = _yaml.safe_load(_f)["robot"]
_robot_cfg_path = _robot_yaml["cfg_path"]
ROBOT_USD = _robot_yaml.get("usd_path")
if not ROBOT_USD or not os.path.isfile(ROBOT_USD):
    raise FileNotFoundError(f"default.yaml 的 robot.usd_path 无效: {ROBOT_USD}")
_URDF, JOINT_NAMES, RETRACT = load_robot_cfg(_robot_cfg_path)
print(f"[main] robot usd : {ROBOT_USD}")
print(f"[main] joints    : {JOINT_NAMES}")

WORKPIECE_OBJ, JOBS = collect_render_jobs(
    args_cli.pkl, args_cli.seam_id, args_cli.hand, args_cli.traj_index, args_cli.all)
print(f"[main] 工件 obj  : {WORKPIECE_OBJ}")
print(f"[main] 待渲染轨迹: {len(JOBS)} 条（{'全部' if args_cli.all else '单选'}）")

# ======================= 阶段二：启动 Isaac，之后只用 numpy + isaac =======================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Gf, UsdPhysics  # noqa: E402

import depth_io  # noqa: E402
import asset_convert  # noqa: E402


# ======================= 几何/位姿辅助（同 render_seam）=======================
def grid_offsets(num_envs, spacing):
    cols = math.ceil(math.sqrt(num_envs))
    out = []
    for i in range(num_envs):
        r, c = divmod(i, cols)
        out.append(np.array([c * spacing, r * spacing, 0.0], float))
    return out


def set_prim_pose(prim_path, pos, quat_wxyz):
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    t = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    t.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    o = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    w, x, y, z = [float(v) for v in quat_wxyz]
    o.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def prim_world_z_range(prim_path):
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return float(rng.GetMin()[2]), float(rng.GetMax()[2])


def compute_group_lift(workpiece_path):
    wp = prim_world_z_range(workpiece_path)
    wp_min, wp_max = wp if wp is not None else (0.0, 0.0)
    rb_min, rb_max = ROBOT_Z_RANGE_EST
    group_min, group_max = min(wp_min, rb_min), max(wp_max, rb_max)
    lift = max(0.0, FLOOR_CLEARANCE - group_min)
    span = group_max - group_min
    if span + FLOOR_CLEARANCE > MAX_LIFT:
        print(f"  [warn] 整组 z 跨度 {span:.2f}m > 上限 {MAX_LIFT}m，按最低点贴地处理")
    elif group_max + lift > MAX_LIFT:
        lift = max(0.0, MAX_LIFT - group_max)
    return lift


def make_camera_cfg(prim_path, pos, rot_wxyz):
    return TiledCameraCfg(
        prim_path=prim_path, update_period=0, width=CAM_WIDTH, height=CAM_HEIGHT,
        data_types=["rgb", DEPTH_KEY],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM_FOCAL_LENGTH, focus_distance=CAM_FOCUS_DISTANCE,
            horizontal_aperture=CAM_H_APERTURE, vertical_aperture=CAM_V_APERTURE,
            clipping_range=CAM_CLIP),
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot_wxyz, convention="ros"))


# ======================= render_info 辅助（对齐 render_seam）=======================
_POSE_CAM_ROS_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def _quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)


def _R_to_quat_wxyz(R):
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z], float)


def _pose7_to_T(p7):
    p7 = np.asarray(p7, float)
    T = np.eye(4); T[:3, :3] = _quat_wxyz_to_R(p7[3:]); T[:3, 3] = p7[:3]
    return T


def _T_to_pose7(T):
    T = np.asarray(T)
    return np.concatenate([T[:3, 3], _R_to_quat_wxyz(T[:3, :3])])


def _cam_pose_arm(cam_pos_w, cam_quat_w, base_pos_w, base_quat_w):
    """相机世界位姿(ROS) → arm(base) 系并翻到 USD 光学约定，返回 pose7 (1,7)（同 render_seam）。"""
    T_cam = _pose7_to_T(np.concatenate([np.asarray(cam_pos_w, float), np.asarray(cam_quat_w, float)]))
    T_base = _pose7_to_T(np.concatenate([np.asarray(base_pos_w, float), np.asarray(base_quat_w, float)]))
    T_arm = np.linalg.inv(T_base) @ T_cam @ _POSE_CAM_ROS_TO_USD
    return _T_to_pose7(T_arm)[None, :]


# ======================= 场景构建 / 渲染 =======================
def ordered_joint_row(sim_names, q_row):
    """把一行关节角 q_row（按 cfg JOINT_NAMES 顺序）重排成 robot 内部关节顺序。"""
    name2val = {n: float(v) for n, v in zip(JOINT_NAMES, q_row)}
    return [name2val.get(n, 0.0) for n in sim_names]


def build_scene(robot_usd, workpiece_usd, link6_sub, num_envs, spacing, warehouse_usd):
    """一次性建 num_envs 个并行环境：warehouse + robot + workpiece(占位) + 左右相机。
    机械臂初始摆 retract（每帧渲染前再改成关键帧关节角）。障碍逐 job 单独 spawn（见 prepare_job）。"""
    offsets = grid_offsets(num_envs, spacing)
    joint_pos_dict = {n: float(v) for n, v in zip(JOINT_NAMES, RETRACT)}
    robots, workpiece_paths, warehouse_paths = [], [], []
    for i, off in enumerate(offsets):
        env_root = f"/World/envs/env_{i:02d}"
        bg = sim_utils.UsdFileCfg(
            usd_path=warehouse_usd,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
        wh = f"{env_root}/warehouse"
        bg.func(wh, bg, translation=(float(off[0]), float(off[1]), 0.0))
        warehouse_paths.append(wh)

        rcfg = ArticulationCfg(
            prim_path=f"{env_root}/robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=joint_pos_dict,
                pos=(float(off[0]), float(off[1]), 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                  stiffness=None, damping=None)})
        robots.append(Articulation(cfg=rcfg))

        wp = f"{env_root}/workpiece"
        wcfg = sim_utils.UsdFileCfg(usd_path=workpiece_usd)
        wcfg.func(wp, wcfg)
        workpiece_paths.append(wp)

    left = TiledCamera(make_camera_cfg(
        f"/World/envs/env_.*/robot/{link6_sub}/camera_left", CAM_LEFT_POS, CAM_LEFT_ROT))
    right = TiledCamera(make_camera_cfg(
        f"/World/envs/env_.*/robot/{link6_sub}/camera_right", CAM_RIGHT_POS, CAM_RIGHT_ROT))
    return dict(offsets=offsets, robots=robots, workpiece_paths=workpiece_paths,
                warehouse_paths=warehouse_paths, left_cam=left, right_cam=right)


def _disable_physics(prim_path):
    stg = omni.usd.get_context().get_stage()
    for pr in Usd.PrimRange(stg.GetPrimAtPath(prim_path)):
        if pr.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)


def spawn_obstacle_mesh(path, verts, faces, color):
    """把 base 系实体三角网 spawn 成一个静态 UsdGeom.Mesh。"""
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    m.CreateFaceVertexCountsAttr([3] * len(faces))
    m.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    m.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    m.CreateDoubleSidedAttr(True)


def prepare_job(scene, job):
    """把本条轨迹的固定工件位姿 + 障碍摆到各 env（每 env 叠加各自偏移）。返回各 env 的 lift。
    工件几何 build_scene 已 spawn（单 pkl 恒定），此处只改位姿；障碍先清旧再 spawn。"""
    stage = omni.usd.get_context().get_stage()
    wp7 = job["workpiece_pose7"]
    lifts = {}
    for env_idx, off in enumerate(scene["offsets"]):
        # 工件：base 系 pose7 + env 偏移 → 世界系
        wpath = scene["workpiece_paths"][env_idx]
        set_prim_pose(wpath, np.asarray(wp7[:3], float) + off, np.asarray(wp7[3:7], float))
        _disable_physics(wpath)

        # 障碍：清旧组、按 base 系顶点 + env 偏移 spawn 新的
        grp = f"/World/envs/env_{env_idx:02d}/obstacles"
        if stage.GetPrimAtPath(grp).IsValid():
            stage.RemovePrim(grp)
        UsdGeom.Xform.Define(stage, grp)
        for oi, (verts, faces, color) in enumerate(job["obstacles"]):
            spawn_obstacle_mesh(f"{grp}/obs_{oi}", verts + off, faces, color)

        # 离地抬升（按固定工件 bbox 估；机械臂运动范围由 ROBOT_Z_RANGE_EST 兜底）
        lift = compute_group_lift(wpath)
        lifts[env_idx] = lift
        set_prim_pose(scene["warehouse_paths"][env_idx],
                      (off[0], off[1], -lift), (1.0, 0.0, 0.0, 0.0))
    return lifts


def render_batch(sim, scene, batch_rows, lifts, settle_steps):
    """batch_rows: list[(env_idx, q_row(6,))]。设各 env 机械臂关节角 → step → 读左右目数据。"""
    sim_dt = sim.get_physics_dt()
    device = scene["robots"][0].data.default_root_state.device
    dtype = scene["robots"][0].data.default_root_state.dtype

    for env_idx, q_row in batch_rows:
        robot = scene["robots"][env_idx]
        q = torch.tensor([ordered_joint_row(robot.data.joint_names, q_row)],
                         device=device, dtype=dtype)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()

    for _ in range(settle_steps):
        sim.step()
        for env_idx, _q in batch_rows:
            scene["robots"][env_idx].update(dt=sim_dt)
        scene["left_cam"].update(dt=sim_dt)
        scene["right_cam"].update(dt=sim_dt)

    left, right = scene["left_cam"].data, scene["right_cam"].data
    lrgb, ldep = left.output["rgb"], left.output[DEPTH_KEY]
    rrgb, rdep = right.output["rgb"], right.output[DEPTH_KEY]
    lK = left.intrinsic_matrices.detach().cpu().numpy()
    rK = right.intrinsic_matrices.detach().cpu().numpy()
    lpos, lquat = left.pos_w.detach().cpu().numpy(), left.quat_w_ros.detach().cpu().numpy()
    rpos, rquat = right.pos_w.detach().cpu().numpy(), right.quat_w_ros.detach().cpu().numpy()

    out = {}
    for env_idx, _q in batch_rows:
        rb = scene["robots"][env_idx].data
        out[env_idx] = dict(
            left_rgb=lrgb[env_idx].detach().cpu().numpy(),
            left_depth=ldep[env_idx, :, :, 0].detach().cpu().numpy(),
            right_rgb=rrgb[env_idx].detach().cpu().numpy(),
            right_depth=rdep[env_idx, :, :, 0].detach().cpu().numpy(),
            left_K=lK[env_idx], right_K=rK[env_idx],
            left_pos_w=lpos[env_idx], left_quat_w=lquat[env_idx],
            right_pos_w=rpos[env_idx], right_quat_w=rquat[env_idx],
            base_pos_w=rb.root_link_pos_w[0].detach().cpu().numpy(),
            base_quat_w=rb.root_link_quat_w[0].detach().cpu().numpy(),
            z_lift=lifts.get(env_idx, 0.0))
    torch.cuda.empty_cache()
    return out


def frame_record_for_side(side, r, job, row_idx, q_row):
    """单帧单侧的【逐帧变化量】记录（相机内外参世界位姿、本帧关节角、observe/goal 等）。
    这些字段稍后按帧堆叠进整侧的一个 render_info.npy；不含轨迹级常量（见 build_side_render_info）。"""
    K = r[f"{side}_K"]
    cam_pose7 = _cam_pose_arm(r[f"{side}_pos_w"], r[f"{side}_quat_w"],
                              r["base_pos_w"], r["base_quat_w"])[0]   # (7,)
    return {
        "row_index": int(row_idx),
        "cam_pos": cam_pose7[:3].astype(np.float64),          # arm(base)系相机位姿(USD光学)
        "cam_quat": cam_pose7[3:].astype(np.float64),
        "cam_intrinsic": np.asarray(K, np.float32),
        "jointstates": np.asarray(q_row, np.float32),         # 本关键帧关节角(6,)
        "observe": int(job["observe"][row_idx]),
        "goal": int(job["goal"][row_idx]),
        "cam_pose_w_pos": np.asarray(r[f"{side}_pos_w"], np.float64),
        "cam_pose_w_quat_wxyz": np.asarray(r[f"{side}_quat_w"], np.float64),
        "base_pose_w_pos": np.asarray(r["base_pos_w"], np.float64),
        "base_pose_w_quat_wxyz": np.asarray(r["base_quat_w"], np.float64),
        "z_lift": float(r.get("z_lift", 0.0)),
    }


def build_side_render_info(side, job, records):
    """把某侧全部帧的 frame_record 按帧堆叠，加上轨迹级常量，组成一个 render_info（存该侧 render_info.npy）。

    逐帧字段（沿 axis0=帧堆叠，F=帧数）：frame_indices(F,)、cam_pos_list(F,3)、cam_quat_list(F,4)、
        cam_intrinsic(F,3,3)、jointstates(F,6)、observe(F,)、goal(F,)、
        cam_pose_w_pos(F,3)/cam_pose_w_quat_wxyz(F,4)、base_pose_w_pos(F,3)/base_pose_w_quat_wxyz(F,4)、z_lift(F,)
    轨迹级常量：内外参、joint_names、workpiece_pose7、seam_line_base、约定串等（与 render_seam 同义）。
    """
    sid, hand, index, jj = job["key"]
    stk = lambda k: np.stack([rec[k] for rec in records], axis=0)
    return {
        # —— 逐帧堆叠 ——
        "frame_indices": np.asarray([rec["row_index"] for rec in records], np.int64),
        "cam_pos_list": stk("cam_pos"),
        "cam_quat_list": stk("cam_quat"),
        "cam_intrinsic": stk("cam_intrinsic"),
        "jointstates": stk("jointstates"),
        "observe": np.asarray([rec["observe"] for rec in records], np.int64),
        "goal": np.asarray([rec["goal"] for rec in records], np.int64),
        "cam_pose_w_pos": stk("cam_pose_w_pos"),
        "cam_pose_w_quat_wxyz": stk("cam_pose_w_quat_wxyz"),
        "base_pose_w_pos": stk("base_pose_w_pos"),
        "base_pose_w_quat_wxyz": stk("base_pose_w_quat_wxyz"),
        "z_lift": np.asarray([rec["z_lift"] for rec in records], np.float64),
        "n_frames": int(len(records)),
        # —— 轨迹级常量 ——
        "sampled_len": int(len(job["positions"])),
        "traj_key": {"seam_id": int(sid), "hand": hand, "index": int(index), "traj_j": int(jj)},
        "seam_line_base": (None if job["seam_line_base"] is None
                           else np.asarray(job["seam_line_base"], np.float64)),
        "side": side,
        "cam_pose_arm_convention": "usd",
        "cam_pose_w_convention": "ros",
        "joint_names": JOINT_NAMES,
        "workpiece_pose7": np.asarray(job["workpiece_pose7"], np.float64),
        "left_extrinsic_pos": np.asarray(CAM_LEFT_POS),
        "left_extrinsic_quat_wxyz": np.asarray(CAM_LEFT_ROT),
        "right_extrinsic_pos": np.asarray(CAM_RIGHT_POS),
        "right_extrinsic_quat_wxyz": np.asarray(CAM_RIGHT_ROT),
        "extrinsic_convention": "ros", "extrinsic_ref_link": "Link6",
        "depth_type": DEPTH_KEY,
    }



def render_job(sim, scene, job, part_stem, out_base, observe_only, settle_steps, num_envs):
    """渲染一条轨迹的全部（或 observe==1）关键帧。左右目各存平铺 {k}_rgb.jpg/{k}_depth.exr +
    一个含全帧信息的 render_info.npy。返回已保存帧数。"""
    sid, hand, index, jj = job["key"]
    traj_dir = Path(out_base) / part_stem / f"seam{sid}_{hand}{index}_traj{jj}"
    done = traj_dir / f"_DONE_seam{sid}_{hand}{index}_traj{jj}"
    if done.exists():
        print(f"[traj {traj_dir.name}] 已完成，跳过")
        return "skipped"

    positions = job["positions"]
    rows = (np.nonzero(job["observe"] == 1)[0].tolist() if observe_only
            else list(range(len(positions))))
    print(f"[traj {traj_dir.name}] status={job['status']} 关键帧 {len(positions)}，"
          f"渲染 {len(rows)} 帧{'（仅 observe）' if observe_only else ''}")

    left_dir, right_dir = traj_dir / "left", traj_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    lifts = prepare_job(scene, job)
    records = {"left": [], "right": []}   # 逐帧记录（按渲染顺序，即帧号升序）
    saved = 0
    for start in range(0, len(rows), num_envs):
        chunk = rows[start:start + num_envs]
        batch_rows = [(env_idx, positions[k]) for env_idx, k in enumerate(chunk)]
        rendered = render_batch(sim, scene, batch_rows, lifts, settle_steps)
        for env_idx, k in enumerate(chunk):
            r = rendered[env_idx]
            depth_io.store_rgb(left_dir / f"{k}_rgb.jpg", r["left_rgb"])
            depth_io.store_depth(left_dir / f"{k}_depth.exr", r["left_depth"])
            depth_io.store_rgb(right_dir / f"{k}_rgb.jpg", r["right_rgb"])
            depth_io.store_depth(right_dir / f"{k}_depth.exr", r["right_depth"])
            records["left"].append(frame_record_for_side("left", r, job, k, positions[k]))
            records["right"].append(frame_record_for_side("right", r, job, k, positions[k]))
            saved += 1
        print(f"  帧 {chunk[0]}~{chunk[-1]} 已存（{len(chunk)}）")

    # 每侧一个含全帧信息的 render_info.npy
    np.save(left_dir / "render_info.npy",
            build_side_render_info("left", job, records["left"]), allow_pickle=True)
    np.save(right_dir / "render_info.npy",
            build_side_render_info("right", job, records["right"]), allow_pickle=True)

    # 轨迹级 meta
    np.save(traj_dir / "_traj_meta.npy", {
        "sampled": job["sampled"], "observe": job["observe"], "goal": job["goal"],
        "positions": positions, "workpiece_pose7": job["workpiece_pose7"],
        "seam_id": int(sid), "hand": hand, "index": int(index), "traj_j": int(jj),
        "joint_names": JOINT_NAMES, "n_obstacles": len(job["obstacles"]),
        "status": job["status"], "goal_index": job["goal_index"], "variant": job["variant"],
        "observe_only": bool(observe_only), "n_rendered": saved,
        "frame_indices": np.asarray(rows, np.int64),
    }, allow_pickle=True)
    done.write_text(f"saved={saved}\ntraj={traj_dir}\n")
    print(f"[traj {traj_dir.name}] 完成，共 {saved} 帧 → {traj_dir}")
    return saved


def main():
    t0 = time.time()
    workpiece_usd = asset_convert.convert_workpiece_obj(WORKPIECE_OBJ, force=args_cli.force_convert)
    link6_sub = asset_convert.find_link_subpath(ROBOT_USD, "Link6")
    print(f"[main] Link6 子路径: {link6_sub}  工件 USD: {workpiece_usd}")

    # 并行环境数不超过最长轨迹要渲的帧数（够用即可，省显存）
    max_rows = max((int((j["observe"] == 1).sum()) if args_cli.observe_only else len(j["positions"]))
                   for j in JOBS)
    num_envs = max(1, min(args_cli.max_envs, max_rows))
    print(f"[main] num_envs={num_envs}（max_envs={args_cli.max_envs}，最长轨迹 {max_rows} 帧）")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 0.5])
    scene = build_scene(ROBOT_USD, workpiece_usd, link6_sub, num_envs,
                        args_cli.spacing, args_cli.warehouse_usd)
    sim.reset(); sim.play()

    obj_stem = asset_convert._ascii_safe(WORKPIECE_OBJ)
    part_stem = (obj_stem[:-len("_watertight")] if obj_stem.endswith("_watertight") else obj_stem)

    total = 0
    for job in JOBS:
        r = render_job(sim, scene, job, part_stem, args_cli.out,
                       args_cli.observe_only, args_cli.settle_steps, num_envs)
        if r != "skipped":
            total += int(r)
    print(f"[main] 全部完成：{len(JOBS)} 条轨迹，共渲染 {total} 帧，用时 {time.time() - t0:.1f}s")
    print("RENDER_TRAJECTORY_DONE")


if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    main()
    simulation_app.close()
