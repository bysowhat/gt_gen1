"""并行环境快速渲染：对工件相对机械臂的每个位姿，在固定初始关节角(retract_config)下
渲染左右目 RGB + 深度。参考 va_simulation/va_sim23_multi.py 的并行环境做法。

用法（无显示器自检）：
    conda run -n env_isaaclab python render/render_seam.py \
        --obj '/media/a/upan/tempt/2/柱_1JdzFk001Mz34qC38vE3On_part_watertight.obj' \
        --seam-npy '/media/a/upan/tempt/2/柱_1JdzFk001Mz34qC38vE3On/seam_40.npy' \
        --out /tmp/render_out --headless

输出：
    <out>/<obj_stem>/<seam_stem>/pose_{p}/
        left_rgb.png  left_depth.exr  right_rgb.png  right_depth.exr  meta.npy
"""
import argparse
import os
import time
from functools import wraps
from pathlib import Path

# OPENCV_IO_ENABLE_OPENEXR 必须在 import cv2 前设置；depth_io 顶部已设，但这里也兜底。
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# ======================= 默认路径 =======================
DEFAULT_WAREHOUSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "default.yaml")

# ======================= 左右目相机外参（相对 Link6，用户标定，ros 约定）=======================
CAM_LEFT_POS = (0.05018328, 0.09963676, 0.14782703)
CAM_LEFT_ROT = (0.02269110, 0.03172106, 0.00070220, -0.99923861)   # wxyz
CAM_RIGHT_POS = (-0.06923272, 0.09421033, 0.14022987)
CAM_RIGHT_ROT = (0.02269110, 0.03172106, 0.00070220, -0.99923861)  # wxyz

# ======================= 相机内参（用户给的 PinholeCameraCfg）=======================
CAM_WIDTH = 2208
CAM_HEIGHT = 1242
CAM_FOCAL_LENGTH = 4.01
CAM_FOCUS_DISTANCE = 480.0
CAM_H_APERTURE = 8.305
CAM_V_APERTURE = 4.672
CAM_CLIP = (1e-5, 1e3)
DEPTH_KEY = "distance_to_image_plane"   # 用户写的 "depth" 的真实键（针孔 z 深度，单位 m）


def timer(stage_name):
    def deco(func):
        @wraps(func)
        def wrapper(*a, **k):
            t0 = time.time()
            r = func(*a, **k)
            print(f"[TIMER] {stage_name}: {time.time() - t0:.2f}s")
            return r
        return wrapper
    return deco


def parse_args():
    parser = argparse.ArgumentParser(description="并行环境渲染左右目 RGB+深度")
    parser.add_argument("--obj", required=True, help="工件 .obj（watertight）路径")
    parser.add_argument("--seam-npy", required=True, help="seam_*.npy（含 workpiece_pose7）")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="default.yaml 路径")
    parser.add_argument("--warehouse-usd", default=DEFAULT_WAREHOUSE_USD, help="环境 USD")
    parser.add_argument("--out", required=True, help="输出根目录")
    parser.add_argument("--max-envs", type=int, default=2, help="并行环境数上限")
    parser.add_argument("--spacing", type=float, default=40.0, help="相邻环境间距(米)")
    parser.add_argument("--settle-steps", type=int, default=12, help="读图前 step 帧数")
    parser.add_argument("--force-convert", action="store_true", help="强制重转 USD")
    return parser


# ---- AppLauncher 必须在导入其余 isaaclab/pxr 之前启动 ----
from isaaclab.app import AppLauncher  # noqa: E402

_parser = parse_args()
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.enable_cameras = True   # 渲染相机必需
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ======================= 启动后再导入 =======================
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema  # noqa: E402

# 项目内模块
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_io  # noqa: E402
import asset_convert  # noqa: E402


def load_robot_cfg(robot_cfg_path):
    """读 cuRobo 机器人 yml，返回 (urdf_path, joint_names, retract_config)。"""
    with open(robot_cfg_path, "r") as f:
        d = yaml.safe_load(f)
    kin = d["robot_cfg"]["kinematics"]
    urdf_path = kin["urdf_path"]
    cspace = kin["cspace"]
    return urdf_path, list(cspace["joint_names"]), list(cspace["retract_config"])


def load_poses(seam_npy):
    """读 workpiece_pose7 → (N,7) [x,y,z, qw,qx,qy,qz]（T_base←workpiece，米，wxyz）。"""
    data = np.load(seam_npy, allow_pickle=True).item()
    poses = np.asarray(data["workpiece_pose7"], dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


def grid_offsets(num_envs, spacing):
    cols = math.ceil(math.sqrt(num_envs))
    offsets = []
    for i in range(num_envs):
        r, c = divmod(i, cols)
        offsets.append(np.array([c * spacing, r * spacing, 0.0], dtype=np.float64))
    return offsets


def bake_joint_state_to_usd(num_envs, joint_names, retract_config):
    """把 retract 关节角回写进 USD 的 RevoluteJoint（drive target + joint state）。

    运行时关节角只存在于 PhysX/Fabric，不写进 USD；直接 stage.Export() 导出的臂型
    停在转换时的默认姿态。导出前调用本函数，在每个关节 prim 上 author：
      - UsdPhysics DriveAPI 目标角（targetPosition，度）
      - PhysxSchema JointStateAPI 初始状态（state:angular:physics:position，度）
    这样把 debug USD 重新载入 Isaac Sim 时，机械臂据此摆到 retract。返回写入的关节数。
    """
    stage = omni.usd.get_context().get_stage()
    name2deg = {n: math.degrees(float(v)) for n, v in zip(joint_names, retract_config)}
    n = 0
    for i in range(num_envs):
        root_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/robot")
        if not root_prim or not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            deg = name2deg.get(prim.GetName())
            if deg is None:
                continue
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTargetPositionAttr().Set(deg)
            js = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            js.CreatePositionAttr().Set(deg)
            js.CreateVelocityAttr().Set(0.0)
            n += 1
    return n


def set_prim_pose(prim_path, pos, quat_wxyz):
    """把静态 prim 的世界位姿设为 (pos, quat_wxyz)。"""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    # 用双精度（spawn 出的 prim 默认 xformOp:orient 为 quatd，须匹配）
    t = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    t.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    o = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    w, x, y, z = [float(v) for v in quat_wxyz]
    o.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def hide_prims_by_name(substrings):
    """把名字/路径包含任一 substring 的可渲染 prim 设为不可见。

    用途：左右目相机挂在 Link6 上，相机外参恰好落在机械臂自带的「相机外壳/护罩」
    网格内部（如 camera_cover），导致该目整幅贴着外壳 ~1cm、全灰无效。真实镜头位于
    外壳表面，故渲染时把这些外壳网格隐藏。返回隐藏的 prim 数。
    """
    if not substrings:
        return 0
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if any(s in path for s in substrings):
            img = UsdGeom.Imageable(prim)
            if img:
                img.MakeInvisible()
                n += 1
    return n


def make_camera_cfg(prim_path, pos, rot_wxyz):
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        data_types=["rgb", DEPTH_KEY],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM_FOCAL_LENGTH,
            focus_distance=CAM_FOCUS_DISTANCE,
            horizontal_aperture=CAM_H_APERTURE,
            vertical_aperture=CAM_V_APERTURE,
            clipping_range=CAM_CLIP,
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot_wxyz, convention="ros"),
    )


@timer("构建并行场景")
def build_scene(robot_usd, workpiece_usd, link6_subpath, num_envs, spacing,
                warehouse_usd, joint_names, retract_config):
    """一次性建 num_envs 个并行环境：warehouse + robot + workpiece(占位) + 左右相机。"""
    offsets = grid_offsets(num_envs, spacing)
    joint_pos_dict = {n: float(v) for n, v in zip(joint_names, retract_config)}

    robots = []
    workpiece_paths = []
    for i, off in enumerate(offsets):
        env_root = f"/World/envs/env_{i:02d}"

        # 背景 warehouse（关碰撞）
        bg_cfg = sim_utils.UsdFileCfg(
            usd_path=warehouse_usd,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
        bg_cfg.func(f"{env_root}/warehouse", bg_cfg,
                    translation=(float(off[0]), float(off[1]), 0.0))

        # 机械臂（固定底座、关重力、初始关节角=retract）
        robot_cfg = ArticulationCfg(
            prim_path=f"{env_root}/robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=joint_pos_dict,
                pos=(float(off[0]), float(off[1]), 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                  stiffness=None, damping=None)},
        )
        robots.append(Articulation(cfg=robot_cfg))

        # 工件：静态 prim，先 spawn 占位（位姿在每批里设）
        wp_path = f"{env_root}/workpiece"
        wp_cfg = sim_utils.UsdFileCfg(usd_path=workpiece_usd)
        wp_cfg.func(wp_path, wp_cfg)
        workpiece_paths.append(wp_path)

    # 左右目 TiledCamera（正则匹配所有 env 的 Link6）
    sub = link6_subpath
    left_cam = TiledCamera(make_camera_cfg(
        f"/World/envs/env_.*/robot/{sub}/camera_left", CAM_LEFT_POS, CAM_LEFT_ROT))
    right_cam = TiledCamera(make_camera_cfg(
        f"/World/envs/env_.*/robot/{sub}/camera_right", CAM_RIGHT_POS, CAM_RIGHT_ROT))

    return {
        "offsets": offsets,
        "robots": robots,
        "workpiece_paths": workpiece_paths,
        "left_cam": left_cam,
        "right_cam": right_cam,
    }


def ordered_joint_tensor(robot, joint_names, retract_config, device, dtype):
    """按 robot 内部关节顺序排好 retract 关节角，返回 (1, njoints) tensor。"""
    name_to_val = {n: v for n, v in zip(joint_names, retract_config)}
    sim_names = robot.data.joint_names
    vals = [name_to_val.get(n, 0.0) for n in sim_names]
    return torch.tensor([vals], device=device, dtype=dtype)


@timer("渲染一批")
def render_batch(sim, scene, batch_poses, offsets, robots, joint_names,
                 retract_config, settle_steps):
    """设置本批每个 env 的工件位姿 + 机械臂初始关节角，step 若干帧后读左右目数据。

    Args:
        batch_poses: list[(env_idx, pose7)]，长度 ≤ num_envs
    Returns:
        dict env_idx -> {left_rgb,left_depth,right_rgb,right_depth, left_K, right_K}
    """
    sim_dt = sim.get_physics_dt()
    device = robots[0].data.default_root_state.device
    dtype = robots[0].data.default_root_state.dtype

    # 设工件位姿（base 系 pose + env 偏移 → 世界系）
    for env_idx, pose7 in batch_poses:
        off = offsets[env_idx]
        world_pos = (pose7[0] + off[0], pose7[1] + off[1], pose7[2] + off[2])
        set_prim_pose(scene["workpiece_paths"][env_idx], world_pos, pose7[3:7])

    # 设机械臂初始关节角
    for env_idx, _ in batch_poses:
        q = ordered_joint_tensor(robots[env_idx], joint_names, retract_config, device, dtype)
        robots[env_idx].write_joint_state_to_sim(q, torch.zeros_like(q))
        robots[env_idx].set_joint_position_target(q)
        robots[env_idx].write_data_to_sim()

    # step 让相机缓冲填充
    for _ in range(settle_steps):
        sim.step()
        for env_idx, _ in batch_poses:
            robots[env_idx].update(dt=sim_dt)
        scene["left_cam"].update(dt=sim_dt)
        scene["right_cam"].update(dt=sim_dt)

    left = scene["left_cam"].data
    right = scene["right_cam"].data
    left_rgb = left.output["rgb"]
    left_depth = left.output[DEPTH_KEY]
    right_rgb = right.output["rgb"]
    right_depth = right.output[DEPTH_KEY]
    left_K = left.intrinsic_matrices.detach().cpu().numpy()
    right_K = right.intrinsic_matrices.detach().cpu().numpy()

    out = {}
    for env_idx, _ in batch_poses:
        out[env_idx] = {
            "left_rgb": left_rgb[env_idx].detach().cpu().numpy(),
            "left_depth": left_depth[env_idx, :, :, 0].detach().cpu().numpy(),
            "right_rgb": right_rgb[env_idx].detach().cpu().numpy(),
            "right_depth": right_depth[env_idx, :, :, 0].detach().cpu().numpy(),
            "left_K": left_K[env_idx],
            "right_K": right_K[env_idx],
        }
    torch.cuda.empty_cache()
    return out


def save_pose(out_dir, rendered, pose7, joint_names, retract_config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_io.store_rgb(out_dir / "left_rgb.png", rendered["left_rgb"])
    depth_io.store_rgb(out_dir / "right_rgb.png", rendered["right_rgb"])
    depth_io.store_depth(out_dir / "left_depth.exr", rendered["left_depth"])
    depth_io.store_depth(out_dir / "right_depth.exr", rendered["right_depth"])
    np.save(out_dir / "meta.npy", {
        "joint_names": joint_names,
        "retract_config": np.asarray(retract_config, dtype=np.float64),
        "workpiece_pose7": np.asarray(pose7, dtype=np.float64),
        "left_intrinsic": rendered["left_K"],
        "right_intrinsic": rendered["right_K"],
        "left_extrinsic_pos": np.asarray(CAM_LEFT_POS),
        "left_extrinsic_quat_wxyz": np.asarray(CAM_LEFT_ROT),
        "right_extrinsic_pos": np.asarray(CAM_RIGHT_POS),
        "right_extrinsic_quat_wxyz": np.asarray(CAM_RIGHT_ROT),
        "extrinsic_convention": "ros",
        "extrinsic_ref_link": "Link6",
        "depth_type": DEPTH_KEY,
    }, allow_pickle=True)


@timer("完整渲染")
def main():
    # 解析机器人 cfg
    with open(args_cli.config, "r") as f:
        robot_cfg_path = yaml.safe_load(f)["robot"]["cfg_path"]
    urdf_path, joint_names, retract_config = load_robot_cfg(robot_cfg_path)
    print(f"[main] robot cfg : {robot_cfg_path}")
    print(f"[main] urdf      : {urdf_path}")
    print(f"[main] joints    : {joint_names}")
    print(f"[main] retract   : {retract_config}")

    poses = load_poses(args_cli.seam_npy)
    n_poses = len(poses)
    print(f"[main] poses     : {n_poses} 个")

    # 转换资产
    robot_usd = asset_convert.convert_robot_urdf(urdf_path, force=args_cli.force_convert)
    workpiece_usd = asset_convert.convert_workpiece_obj(args_cli.obj, force=args_cli.force_convert)
    link6_sub = asset_convert.find_link_subpath(robot_usd, "Link6")
    print(f"[main] Link6 子路径: {link6_sub}")

    num_envs = min(n_poses, args_cli.max_envs)
    print(f"[main] num_envs  : {num_envs}")

    # 仿真上下文
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 0.5])

    scene = build_scene(robot_usd, workpiece_usd, link6_sub, num_envs, args_cli.spacing,
                        args_cli.warehouse_usd, joint_names, retract_config)

    sim.reset()
    sim.play()

    obj_stem = asset_convert._ascii_safe(args_cli.obj)
    seam_stem = Path(args_cli.seam_npy).stem
    out_root = Path(args_cli.out) / obj_stem / seam_stem

    # 按 num_envs 分批
    saved = 0
    for start in range(0, n_poses, num_envs):
        chunk = list(range(start, min(start + num_envs, n_poses)))
        batch_poses = [(env_idx, poses[p]) for env_idx, p in enumerate(chunk)]
        print(f"[main] 批 {start}~{chunk[-1]}（{len(chunk)} 个 pose）")
        rendered = render_batch(sim, scene, batch_poses, scene["offsets"],
                                scene["robots"], joint_names, retract_config,
                                args_cli.settle_steps)
        # [debug] 保存整个场景 USD，便于离线检查相机/工件/机械臂相对位姿
        dbg_usd = out_root / f"scene_batch_{start}.usd"
        dbg_usd.parent.mkdir(parents=True, exist_ok=True)
        n_baked = bake_joint_state_to_usd(num_envs, joint_names, retract_config)
        omni.usd.get_context().get_stage().Export(str(dbg_usd))
        print(f"  [debug] 场景 USD -> {dbg_usd}（回写 {n_baked} 个关节角）")
        for env_idx, p in enumerate(chunk):
            save_pose(out_root / f"pose_{p}", rendered[env_idx], poses[p],
                      joint_names, retract_config)
            saved += 1
            print(f"  saved pose_{p} -> {out_root / f'pose_{p}'}")

    print(f"[main] 完成，共保存 {saved} 个 pose 到 {out_root}")


if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    main()
    simulation_app.close()
