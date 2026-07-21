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

分割类别：{k}_seg.png 含 robot / workpiece / obstacle_* / seam 四类。焊缝(seam)是沿焊缝线建的一根实体管，
【只出现在分割里】——它平时 invisible，仅在渲染时的第二个 seg-only pass 临时置 visible，故 RGB / depth
与「无焊缝」逐像素一致（详见 render_batch 双 pass 说明与 docs/渲染整条采样轨迹-方案.md）。
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
# 2D 实例分割：每个打了语义标签的 prim 子树一个整数 id（工件/机械臂/各障碍/焊缝各一 id，
# 机械臂整体一个 id 不被拆成各 link）。相机侧须 colorize=False 才拿到整数 id 图。
SEG_KEY = "instance_segmentation_fast"

# ===== 焊缝分割：沿焊缝线建一根实体管(tube)，打 "seam" 语义标签 =====
# 关键约束：焊缝【只能出现在实例分割里】，RGB / depth 必须与「无焊缝」逐像素一致。
# Isaac 的 rgb/depth/seg 是同一次光追出的三张 AOV，无「只进 seg、不进 rgb/depth」的单-prim 开关，
# 故 tube 平时 invisible（rgb/depth 不含它），仅在 seg-only 的第二 pass 里临时置 visible（见 render_batch）。
SEAM_TUBE_RADIUS = 0.004   # 焊缝管半径(米)，默认 4mm（可由 --seam-radius 覆盖）
SEAM_TUBE_SIDES = 8        # 焊缝管截面多边形边数

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
    p.add_argument("--seam-radius", type=float, default=SEAM_TUBE_RADIUS,
                   help="焊缝分割管半径(米，默认 4mm)")
    p.add_argument("--seam-settle-steps", type=int, default=4,
                   help="seg-only 第二 pass 的 step 帧数（焊缝置 visible 后等 seg 更新）")
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

try:  # 给裸 UsdGeom.Mesh（障碍）打语义标签用；新旧 Isaac 命名空间兜底
    from isaacsim.core.utils.semantics import add_update_semantics  # noqa: E402
except ImportError:  # pragma: no cover
    from omni.isaac.core.utils.semantics import add_update_semantics  # noqa: E402

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


def compute_group_lift(workpiece_path, robot_path=None):
    wp = prim_world_z_range(workpiece_path)
    wp_min, wp_max = wp if wp is not None else (0.0, 0.0)
    # 机械臂 z 范围优先实测（含底座/法兰低于 base 原点的几何），实测不到才退回估计值
    rb = prim_world_z_range(robot_path) if robot_path else None
    rb_min, rb_max = rb if rb is not None else ROBOT_Z_RANGE_EST
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
        data_types=["rgb", DEPTH_KEY, SEG_KEY],
        colorize_instance_segmentation=False,  # 拿 (H,W) 整数 id 图（否则输出上色 RGBA）
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


def _link6_index(scene):
    """在机械臂 articulation 的 body 列表里定位 Link6 的下标（相机挂它上面）。"""
    names = list(scene["robots"][0].data.body_names)
    for i, n in enumerate(names):
        if n == "Link6" or str(n).split("/")[-1] == "Link6":
            return i
    raise KeyError(f"body_names 里找不到 Link6：{names}")


def _cam_pose_w_from_link6(l6_pos, l6_quat_wxyz, ext_pos, ext_quat_wxyz):
    """由 Link6 的【活体】世界位姿 ∘ 相机相对 Link6 的固定外参(ROS) → 相机世界位姿(ROS 光学)。

    背景：TiledCamera.pos_w/quat_w_ros 对「挂在关节体、由物理驱动」的相机读回的是陈旧
    （冻结在初始）值——RTX 渲染读 Fabric 活变换故图像正确，但传感器位姿读 USD 未回写故不变。
    这里改从物理 tensor 的 Link6 link 位姿(body_link_pos_w/quat_w，逐帧更新)重建：
    相机世界位姿 = T_link6_w ∘ T_extrinsic(config, ROS)。extrinsic 与相机 spawn 用的
    OffsetCfg(convention="ros") 同一份，故组合出的正是 ROS 光学相机帧（等价原 quat_w_ros）。
    """
    T_l6 = _pose7_to_T(np.concatenate([np.asarray(l6_pos, float), np.asarray(l6_quat_wxyz, float)]))
    T_ext = _pose7_to_T(np.concatenate([np.asarray(ext_pos, float), np.asarray(ext_quat_wxyz, float)]))
    T_cam = T_l6 @ T_ext
    return T_cam[:3, 3], _R_to_quat_wxyz(T_cam[:3, :3])


def sanity_check_cam_pose(sim, scene, settle=6, tol=2e-3):
    """retract 初始态自检：此刻 TiledCamera.pos_w/quat_w_ros 尚未变陈旧(=正确初值)，
    用它校验「Link6活位姿 ∘ config extrinsic」重建法是否与传感器一致（即 extrinsic 约定没搞反）。
    偏差应≈0；偏大则说明外参/约定组合有误。只在开跑前跑一次，不影响渲染。"""
    dt = sim.get_physics_dt()
    for _ in range(settle):
        sim.step()
        scene["robots"][0].update(dt=dt)
        scene["left_cam"].update(dt=dt)
        scene["right_cam"].update(dt=dt)
    l6_idx = _link6_index(scene)
    rb = scene["robots"][0].data
    l6_pos = rb.body_link_pos_w[0, l6_idx].detach().cpu().numpy()
    l6_quat = rb.body_link_quat_w[0, l6_idx].detach().cpu().numpy()
    for side, cam, epos, equat in (
            ("left", scene["left_cam"], CAM_LEFT_POS, CAM_LEFT_ROT),
            ("right", scene["right_cam"], CAM_RIGHT_POS, CAM_RIGHT_ROT)):
        cpos, cquat = _cam_pose_w_from_link6(l6_pos, l6_quat, epos, equat)
        spos = cam.data.pos_w[0].detach().cpu().numpy()
        squat = cam.data.quat_w_ros[0].detach().cpu().numpy()
        dp = float(np.linalg.norm(cpos - spos))
        dq = float(min(np.linalg.norm(cquat - squat), np.linalg.norm(cquat + squat)))  # 四元数符号无关
        flag = "OK" if (dp < tol and dq < tol) else "⚠ 偏大！检查 extrinsic 约定/组合顺序"
        print(f"[sanity:{side}] retract 相机位姿 计算 vs 传感器: dpos={dp:.5f}m dquat={dq:.5f}  {flag}")


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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                semantic_tags=[("class", "robot")]),  # 机械臂整体一个实例 id
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=joint_pos_dict,
                pos=(float(off[0]), float(off[1]), 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                  stiffness=None, damping=None)})
        robots.append(Articulation(cfg=rcfg))

        wp = f"{env_root}/workpiece"
        wcfg = sim_utils.UsdFileCfg(usd_path=workpiece_usd,
                                    semantic_tags=[("class", "workpiece")])
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


def spawn_obstacle_mesh(path, verts, faces, color, label=None):
    """把 base 系实体三角网 spawn 成一个静态 UsdGeom.Mesh，返回该 Mesh。
    color 为 None 时不写 DisplayColor（焊缝管走此路：反正 RGB 不渲染它）。
    label 非空则给该 prim 打 class 语义标签（实例分割用，逐障碍不同 → obstacle_0/1/...；焊缝 → seam）。"""
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    m.CreateFaceVertexCountsAttr([3] * len(faces))
    m.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    if color is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    m.CreateDoubleSidedAttr(True)   # 双面：无需管心面朝向/绕序
    if label is not None:
        add_update_semantics(m.GetPrim(), str(label))
    return m


def _seam_tube_mesh(seam_line, radius, n_sides=SEAM_TUBE_SIDES):
    """沿 base 系焊缝折线 seam_line(N,3) 生成一根半径 radius 的实体管，返回 (verts(M,3), faces(K,3))。

    每个折线点放一圈 n_sides 顶点(在与切线垂直的平面上)，相邻两圈用四边形(两三角)连成侧面，
    并封两端。用平行传输(parallel transport)沿折线滚动截面基，避免折线拐弯处圈发生扭转。
    点数 <2 → 返回 None（无法成管）。双面渲染，故不在意三角绕序/朝向。
    """
    P = np.asarray(seam_line, float).reshape(-1, 3)
    if len(P) < 2:
        return None
    n = len(P)
    tang = np.empty_like(P)                 # 逐点切线（中点用中心差分，端点用单边差分）
    tang[1:-1] = P[2:] - P[:-2]
    tang[0] = P[1] - P[0]
    tang[-1] = P[-1] - P[-2]
    tang /= np.clip(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12, None)

    def _perp(t):                           # 取一个与 t 垂直的单位向量
        a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(t, a)
        return u / (np.linalg.norm(u) + 1e-12)

    ang = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    u = _perp(tang[0]); v = np.cross(tang[0], u)
    rings = []
    for i in range(n):
        if i > 0:                           # 平行传输：把上一 u 投影回当前法平面再正交化
            u = u - float(np.dot(u, tang[i])) * tang[i]
            if np.linalg.norm(u) < 1e-9:
                u = _perp(tang[i])
            u /= np.linalg.norm(u) + 1e-12
            v = np.cross(tang[i], u)
        rings.append(P[i] + radius * (cos * u[None, :] + sin * v[None, :]))
    verts = np.vstack(rings)                # (n*n_sides, 3)

    faces = []
    for i in range(n - 1):
        b0, b1 = i * n_sides, (i + 1) * n_sides
        for k in range(n_sides):
            kn = (k + 1) % n_sides
            faces.append([b0 + k, b0 + kn, b1 + kn])
            faces.append([b0 + k, b1 + kn, b1 + k])
    c0 = len(verts)                         # 两端盖中心点
    verts = np.vstack([verts, P[0][None, :], P[-1][None, :]])
    last = (n - 1) * n_sides
    for k in range(n_sides):
        kn = (k + 1) % n_sides
        faces.append([c0, k, kn])           # 首端盖
        faces.append([c0 + 1, last + k, last + kn])   # 末端盖
    return verts, np.asarray(faces, np.int64)


def _set_seam_visible(scene, visible):
    """把各 env 的焊缝管 prim 统一置 visible/invisible（双 pass 渲染切换用）。缺失则忽略。"""
    stage = omni.usd.get_context().get_stage()
    for env_idx in range(len(scene["offsets"])):
        pr = stage.GetPrimAtPath(f"/World/envs/env_{env_idx:02d}/seam/tube")
        if pr.IsValid():
            img = UsdGeom.Imageable(pr)
            img.MakeVisible() if visible else img.MakeInvisible()



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
            spawn_obstacle_mesh(f"{grp}/obs_{oi}", verts + off, faces, color,
                                label=f"obstacle_{oi}")

        # 焊缝：沿 base 系焊缝线 + env 偏移建一根 tube，打 "seam" 标签，【初始 invisible】
        # （只在 render_batch 的 seg-only 第二 pass 临时 visible；RGB/depth 恒不含它）。
        sgrp = f"/World/envs/env_{env_idx:02d}/seam"
        if stage.GetPrimAtPath(sgrp).IsValid():
            stage.RemovePrim(sgrp)
        if job["seam_line_base"] is not None:
            tube = _seam_tube_mesh(np.asarray(job["seam_line_base"], float) + off,
                                   float(args_cli.seam_radius))
            if tube is not None:
                UsdGeom.Xform.Define(stage, sgrp)
                m = spawn_obstacle_mesh(f"{sgrp}/tube", tube[0], tube[1], None, label="seam")
                UsdGeom.Imageable(m.GetPrim()).MakeInvisible()

        # 离地抬升（工件 + 机械臂真实 bbox 实测 z 范围，保证二者都在地板上）
        rpath = f"/World/envs/env_{env_idx:02d}/robot"
        lift = compute_group_lift(wpath, rpath)
        lifts[env_idx] = lift
        set_prim_pose(scene["warehouse_paths"][env_idx],
                      (off[0], off[1], -lift), (1.0, 0.0, 0.0, 0.0))
    return lifts


def _normalize_seg_labels(raw):
    """把 Isaac 的 idToLabels 规整成 {int_id: label_str}。

    instance_segmentation_fast 的值通常形如 {"2": {"class": "workpiece"}}；也兜底纯串/其它键。
    背景/未标注（BACKGROUND/UNLABELLED）一并保留，便于下游按需过滤。
    """
    out = {}
    for k, v in dict(raw).items():
        try:
            kid = int(k)
        except (ValueError, TypeError):
            continue
        if isinstance(v, dict):
            label = v.get("class") or v.get("semantic") or v.get("instance") or str(v)
        else:
            label = str(v)
        out[kid] = label
    return out


def _seg_labels_from_info(info, env_idx=None):
    """从 camera.data.info 取实例分割 idToLabels 并规整成 {int_id: label}；缺失则空 dict。

    坑：TiledCamera 的 camera.data.info 是 {data_type: info} 的【单 dict】（tiled_camera.py:236/395），
    并非按 env 的 list——整块 tiled 渲染共用一份 idToLabels，覆盖所有 env 的 prim。故此处按 data_type
    直接取，不能再用 info[env_idx] 索引（int 索引 dict 会 KeyError，早先被 except 吞掉导致映射恒为空）。
    env_idx 仅为兼容旧签名保留，不再使用。
    """
    try:
        raw = info[SEG_KEY]["idToLabels"]
    except (TypeError, KeyError, IndexError):
        return {}
    return _normalize_seg_labels(raw)


def render_batch(sim, scene, batch_rows, lifts, settle_steps, seam_settle_steps):
    """batch_rows: list[(env_idx, q_row(6,))]。设各 env 机械臂关节角后【双 pass 渲染】：
      pass1：焊缝管 invisible → 读 RGB/depth（此帧=无焊缝场景，逐像素与不加焊缝一致）；
      pass2：焊缝管 visible → 只读 seg（RGB/depth 丢弃）→ 复位 invisible。
    相机内参与世界位姿是几何量、两 pass 相同，在 pass1 后算一次即可。"""
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

    def _step(n):
        for _ in range(n):
            sim.step()
            for env_idx, _q in batch_rows:
                scene["robots"][env_idx].update(dt=sim_dt)
            scene["left_cam"].update(dt=sim_dt)
            scene["right_cam"].update(dt=sim_dt)

    # ---- pass1：焊缝隐藏 → RGB/depth ----
    _set_seam_visible(scene, False)
    _step(settle_steps)
    left, right = scene["left_cam"].data, scene["right_cam"].data
    lrgb, ldep = left.output["rgb"], left.output[DEPTH_KEY]
    rrgb, rdep = right.output["rgb"], right.output[DEPTH_KEY]
    lK = left.intrinsic_matrices.detach().cpu().numpy()
    rK = right.intrinsic_matrices.detach().cpu().numpy()
    # 注意：不再用 left.pos_w/quat_w_ros——它们对关节驱动的相机是陈旧(冻结)值。
    # 改从 Link6 活位姿(body_link) ∘ config extrinsic 逐帧重建相机世界位姿(见 _cam_pose_w_from_link6)。
    l6_idx = _link6_index(scene)
    # 立即把 RGB/depth 拷到 CPU（pass2 会覆写 camera.output），同时算好相机世界位姿(几何量)。
    rgbd, campose = {}, {}
    for env_idx, _q in batch_rows:
        rgbd[env_idx] = dict(
            left_rgb=lrgb[env_idx].detach().cpu().numpy(),
            left_depth=ldep[env_idx, :, :, 0].detach().cpu().numpy(),
            right_rgb=rrgb[env_idx].detach().cpu().numpy(),
            right_depth=rdep[env_idx, :, :, 0].detach().cpu().numpy())
        rb = scene["robots"][env_idx].data
        l6_pos = rb.body_link_pos_w[0, l6_idx].detach().cpu().numpy()      # Link6 活位姿(世界系)
        l6_quat = rb.body_link_quat_w[0, l6_idx].detach().cpu().numpy()    # wxyz
        l_pos_w, l_quat_w = _cam_pose_w_from_link6(l6_pos, l6_quat, CAM_LEFT_POS, CAM_LEFT_ROT)
        r_pos_w, r_quat_w = _cam_pose_w_from_link6(l6_pos, l6_quat, CAM_RIGHT_POS, CAM_RIGHT_ROT)
        campose[env_idx] = dict(
            left_pos_w=l_pos_w, left_quat_w=l_quat_w,
            right_pos_w=r_pos_w, right_quat_w=r_quat_w,
            base_pos_w=rb.root_link_pos_w[0].detach().cpu().numpy(),
            base_quat_w=rb.root_link_quat_w[0].detach().cpu().numpy())

    # ---- pass2：焊缝显示 → 只取 seg（RGB/depth 丢弃）----
    _set_seam_visible(scene, True)
    _step(seam_settle_steps)
    lseg = scene["left_cam"].data.output[SEG_KEY]     # (N,H,W,1) 整数 id 图
    rseg = scene["right_cam"].data.output[SEG_KEY]
    linfo, rinfo = scene["left_cam"].data.info, scene["right_cam"].data.info   # {data_type:info} 单 dict（含 seam）
    lseg_lab = _seg_labels_from_info(linfo)
    rseg_lab = _seg_labels_from_info(rinfo)
    seg = {env_idx: dict(
        left_seg=lseg[env_idx, :, :, 0].detach().cpu().numpy(),
        right_seg=rseg[env_idx, :, :, 0].detach().cpu().numpy())
        for env_idx, _q in batch_rows}
    _set_seam_visible(scene, False)   # 复位，供下一个 batch 的 pass1

    out = {}
    for env_idx, _q in batch_rows:
        out[env_idx] = dict(
            **rgbd[env_idx], **seg[env_idx], **campose[env_idx],
            left_seg_labels=lseg_lab, right_seg_labels=rseg_lab,
            left_K=lK[env_idx], right_K=rK[env_idx],
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
        "seg_id_to_label": dict(r.get(f"{side}_seg_labels", {})),  # 本帧 id→标签映射(逐帧存)
        "cam_pose_w_pos": np.asarray(r[f"{side}_pos_w"], np.float64),
        "cam_pose_w_quat_wxyz": np.asarray(r[f"{side}_quat_w"], np.float64),
        "base_pose_w_pos": np.asarray(r["base_pos_w"], np.float64),
        "base_pose_w_quat_wxyz": np.asarray(r["base_quat_w"], np.float64),
        "z_lift": float(r.get("z_lift", 0.0)),
    }


def build_side_render_info(side, job, records):
    """把某侧全部帧的 frame_record 按帧堆叠，加上轨迹级常量，组成一个 render_info（存该侧 render_info.npy）。

    逐帧字段（沿 axis0=帧堆叠，F=帧数）：frame_indices(F,)、cam_pos_list(F,3)、cam_quat_list(F,4)、
        cam_intrinsic(F,3,3)、jointstates(F,6)、cam_2d(F,)、camera_3d(F,)、
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
        "cam_2d": np.asarray([rec["observe"] for rec in records], np.int64),
        "camera_3d": np.asarray([rec["goal"] for rec in records], np.int64),
        "cam_pose_w_pos": stk("cam_pose_w_pos"),
        "cam_pose_w_quat_wxyz": stk("cam_pose_w_quat_wxyz"),
        "base_pose_w_pos": stk("base_pose_w_pos"),
        "base_pose_w_quat_wxyz": stk("base_pose_w_quat_wxyz"),
        "z_lift": np.asarray([rec["z_lift"] for rec in records], np.float64),
        "n_frames": int(len(records)),
        # —— 2D 实例分割：id→标签映射【逐帧各一份】（list 长度 F，与各帧 {k}_seg.png 对应）——
        "seg_type": SEG_KEY,
        "seg_id_to_label_list": [rec["seg_id_to_label"] for rec in records],
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



def render_job(sim, scene, job, part_stem, out_base, observe_only, settle_steps, seam_settle_steps, num_envs):
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
        rendered = render_batch(sim, scene, batch_rows, lifts, settle_steps, seam_settle_steps)
        for env_idx, k in enumerate(chunk):
            r = rendered[env_idx]
            depth_io.store_rgb(left_dir / f"{k}_rgb.jpg", r["left_rgb"])
            depth_io.store_depth(left_dir / f"{k}_depth.exr", r["left_depth"])
            depth_io.store_seg(left_dir / f"{k}_seg.png", r["left_seg"])
            depth_io.store_rgb(right_dir / f"{k}_rgb.jpg", r["right_rgb"])
            depth_io.store_depth(right_dir / f"{k}_depth.exr", r["right_depth"])
            depth_io.store_seg(right_dir / f"{k}_seg.png", r["right_seg"])
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

    # 开跑前一次性自检：验证「Link6活位姿 ∘ config extrinsic」重建相机位姿与传感器一致
    sanity_check_cam_pose(sim, scene)

    obj_stem = asset_convert._ascii_safe(WORKPIECE_OBJ)
    part_stem = (obj_stem[:-len("_watertight")] if obj_stem.endswith("_watertight") else obj_stem)

    total = 0
    for job in JOBS:
        r = render_job(sim, scene, job, part_stem, args_cli.out,
                       args_cli.observe_only, args_cli.settle_steps,
                       args_cli.seam_settle_steps, num_envs)
        if r != "skipped":
            total += int(r)
    print(f"[main] 全部完成：{len(JOBS)} 条轨迹，共渲染 {total} 帧，用时 {time.time() - t0:.1f}s")
    print("RENDER_TRAJECTORY_DONE")


if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    main()
    simulation_app.close()
