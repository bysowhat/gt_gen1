"""
在 Isaac Sim (Isaac Lab) 中可视化 plan_path_stomp.py / plan_path_stomp_obstacle.py
规划出的机械臂关节轨迹，并可同时渲染障碍物（方块/圆柱）与工件 mesh。

读取 traj.npy (形状 (T, 6)，列为 xiaoyu_arm_joint1..6 的关节角)，
加载本项目机器人 USD (robot/a1_ur12e.usd，经 A1_CFG)，
逐时间步把关节角直接写入仿真并渲染，循环往复播放，便于肉眼检查轨迹。

障碍物来源与 plan_path_stomp_obstacle.py 完全一致：
    - 基础障碍物：直接复用该文件的 DEFAULT_OBSTACLES（gt_gen_hanfeng 原语 -> cuRobo
      Cuboid/Cylinder），保证可视化的障碍与规划时用到的障碍是同一套；
    - 工件 mesh：--workpiece_mesh（.usd 直接加载；.obj/.stl 仅提示，无法直接渲染）。
    障碍位姿在「机器人 base 系」下给出，渲染时会按机器人 base 的世界位姿做变换对齐。

注意：
    - 必须用 Isaac Lab 的 Python 环境运行（env_isaaclab）。
    - 默认带 GUI（不要加 --headless）。
    - A1_CFG 已 disable_gravity，stiffness=0，所以直接 write_joint_state_to_sim
      置位后机械臂会停在该姿态，与本项目 Traj 阶段写关节角的方式一致。

运行：
    python visualize_traj_isaacsim.py --traj traj_obs.npy            # 含障碍物
    python visualize_traj_isaacsim.py --traj traj.npy --no_obstacles # 不渲染障碍物
    python visualize_traj_isaacsim.py --traj traj_obs.npy --hold 3   # 每个路点停留更久(更慢)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Sim 轨迹可视化")
parser.add_argument("--traj", type=str, default="traj.npy", help="轨迹 .npy 文件 (T,6)")
parser.add_argument("--hold", type=int, default=2, help="每个路点渲染帧数(越大越慢)")
parser.add_argument("--pause_ends", type=int, default=30, help="到达起点/终点时额外停留帧数")
parser.add_argument("--once", action="store_true", help="只播放一次后停在终点(默认来回循环)")
parser.add_argument("--no_obstacles", action="store_true", help="不渲染障碍物(只看机械臂)")
parser.add_argument("--workpiece_mesh", type=str, default=None,
                    help="工件 mesh 路径(.usd 可直接渲染; .obj/.stl 仅提示)")
parser.add_argument("--workpiece_pose", type=float, nargs=7, default=None,
                    help="工件位姿 x y z qw qx qy qz (base系)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制带 GUI 可视化
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 以下需在 app 启动后再 import ----
import os
import sys
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from isaaclab.utils.math import quat_mul, quat_apply

from robot.a1_cfg import A1_CFG

JOINT_NAMES = [f"xiaoyu_arm_joint{i}" for i in range(1, 7)]


def get_obstacle_geoms():
    """复用 plan_path_stomp_obstacle.DEFAULT_OBSTACLES，返回 cuRobo 几何描述。

    返回 (cuboids, cylinders)：
      cuboids:   [dict(pose=[x,y,z,qw,qx,qy,qz], dims=[dx,dy,dz]), ...]
      cylinders: [dict(pose=[...], radius, height), ...]
    位姿均在「机器人 base 系」下。失败(找不到 gt_gen)时返回空列表并提示。
    """
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    if proj_dir not in sys.path:
        sys.path.insert(0, proj_dir)
    gt_gen_root = os.environ.get("GT_GEN_ROOT", "/home/a/Projects/Github/gt_gen_hanfeng")
    if gt_gen_root not in sys.path:
        sys.path.insert(0, gt_gen_root)
    try:
        from plan_path_stomp_obstacle import DEFAULT_OBSTACLES
        import gt_gen.compat  # noqa: F401  warp/trimesh shim
        gt_gen.compat.apply_trimesh_shim()
        from gt_gen import obstacles as obs_lib
    except Exception as e:  # noqa: BLE001
        print(f"[viz][warn] 无法载入障碍物定义({e})，将只显示机械臂。")
        return [], []

    prims = []
    for name, pos, rpy, shape in DEFAULT_OBSTACLES:
        prims.extend(obs_lib.build(name, pos, rpy, **shape))

    cuboids, cylinders = [], []
    for o in obs_lib.to_curobo(prims):
        cls = o.__class__.__name__
        if cls == "Cuboid":
            cuboids.append(dict(pose=list(o.pose), dims=list(o.dims)))
        elif cls == "Cylinder":
            cylinders.append(dict(pose=list(o.pose),
                                  radius=float(o.radius), height=float(o.height)))
    print(f"[viz] 障碍物: {len(cuboids)} cuboid + {len(cylinders)} cylinder "
          f"(来自 DEFAULT_OBSTACLES)")
    return cuboids, cylinders


def _to_world(base_pos, base_quat, local_pos, local_quat):
    """把 base 系下的位姿变换到世界系。pos/quat 均为 (3,)/(4,) torch tensor, quat=wxyz。"""
    lp = torch.tensor(local_pos, dtype=torch.float, device=base_pos.device)
    lq = torch.tensor(local_quat, dtype=torch.float, device=base_pos.device)
    world_pos = base_pos + quat_apply(base_quat.unsqueeze(0), lp.unsqueeze(0)).squeeze(0)
    world_quat = quat_mul(base_quat.unsqueeze(0), lq.unsqueeze(0)).squeeze(0)
    return world_pos.cpu().tolist(), world_quat.cpu().tolist()


def spawn_world(base_pos, base_quat):
    """在机器人 base 世界位姿下渲染障碍物方块/圆柱与(可选)工件 mesh。"""
    cuboids, cylinders = get_obstacle_geoms()

    obs_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.45, 0.10),
                                          roughness=0.6)
    for i, c in enumerate(cuboids):
        cfg = sim_utils.CuboidCfg(size=tuple(c["dims"]), visual_material=obs_mat)
        wp, wq = _to_world(base_pos, base_quat, c["pose"][:3], c["pose"][3:])
        cfg.func(f"/World/obstacles/cuboid_{i}", cfg, translation=tuple(wp),
                 orientation=tuple(wq))
    for i, c in enumerate(cylinders):
        cfg = sim_utils.CylinderCfg(radius=c["radius"], height=c["height"],
                                    visual_material=obs_mat)
        wp, wq = _to_world(base_pos, base_quat, c["pose"][:3], c["pose"][3:])
        cfg.func(f"/World/obstacles/cylinder_{i}", cfg, translation=tuple(wp),
                 orientation=tuple(wq))

    # 工件 mesh（仅 .usd 可直接渲染）
    if args_cli.workpiece_mesh:
        path = args_cli.workpiece_mesh
        pose = args_cli.workpiece_pose or [0, 0, 0, 1, 0, 0, 0]
        wp, wq = _to_world(base_pos, base_quat, pose[:3], pose[3:])
        if path.lower().endswith((".usd", ".usda", ".usdc")):
            cfg = sim_utils.UsdFileCfg(usd_path=path)
            cfg.func("/World/obstacles/workpiece", cfg, translation=tuple(wp),
                     orientation=tuple(wq))
            print(f"[viz] 工件 mesh 已加载: {path}")
        else:
            print(f"[viz][warn] 工件 mesh 为非 USD 格式({path})，Isaac 无法直接渲染，"
                  f"已跳过(规划仍使用了它)。")


@configclass
class VizSceneCfg(InteractiveSceneCfg):
    """最小可视化场景：地面 + 灯光 + 机械臂。"""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )
    # A1_CFG 的 prim_path 已是 /World/envs/env_.*/robot
    robot = A1_CFG.replace(prim_path="/World/envs/env_.*/robot")


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.6], target=[0.0, 0.0, 0.4])

    scene = InteractiveScene(VizSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]

    # 渲染障碍物（按机器人 base 的世界位姿对齐；障碍位姿是 base 系）
    if not args_cli.no_obstacles:
        base_w_pos = robot.data.root_pos_w[0].clone()
        base_w_quat = robot.data.root_quat_w[0].clone()
        print(f"[viz] 机器人 base 世界位姿 pos={base_w_pos.cpu().tolist()} "
              f"quat(wxyz)={base_w_quat.cpu().tolist()}")
        spawn_world(base_w_pos, base_w_quat)

    # 读取轨迹
    traj_path = args_cli.traj
    if not os.path.isabs(traj_path):
        traj_path = os.path.join(os.getcwd(), traj_path)
    traj_np = np.load(traj_path)
    assert traj_np.ndim == 2 and traj_np.shape[1] == 6, f"轨迹形状应为 (T,6)，实际 {traj_np.shape}"
    traj = torch.tensor(traj_np, dtype=torch.float, device=sim.device)
    T = traj.shape[0]
    print(f"[viz] 载入轨迹 {traj_path}，形状 (T,D) = {traj_np.shape}")

    # 轨迹列 -> 仿真关节索引 映射（保证顺序与 JOINT_NAMES 一致）
    joint_ids, found_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    joint_ids = torch.tensor(joint_ids, device=sim.device, dtype=torch.long)
    print(f"[viz] 关节映射: {found_names} -> idx {joint_ids.tolist()}")
    print(f"[viz] 机械臂全部关节: {robot.joint_names}")

    sim_dt = sim.get_physics_dt()
    base_pos = robot.data.default_joint_pos.clone()  # (1, num_joints)

    def write_q(q6: torch.Tensor):
        """把 6 维关节角写入仿真并保持。"""
        pos = base_pos.clone()
        pos[:, joint_ids] = q6.unsqueeze(0)
        vel = torch.zeros_like(pos)
        robot.write_joint_state_to_sim(pos, vel)
        scene.write_data_to_sim()

    def step_frames(q6: torch.Tensor, n: int):
        for _ in range(n):
            if not simulation_app.is_running():
                return False
            write_q(q6)            # 每帧重写，防止漂移
            sim.step()
            scene.update(sim_dt)
        return True

    print("[viz] 开始播放（关闭窗口或 Ctrl-C 退出）...")
    idx = 0
    direction = 1
    while simulation_app.is_running():
        q6 = traj[idx]
        if not step_frames(q6, args_cli.hold):
            break

        # 端点停留
        if idx == 0 or idx == T - 1:
            if not step_frames(q6, args_cli.pause_ends):
                break
            if args_cli.once and idx == T - 1:
                print("[viz] 已播放到终点（--once），保持终点姿态。")
                while simulation_app.is_running():
                    if not step_frames(traj[T - 1], 10):
                        break
                break

        idx += direction
        if idx >= T - 1:
            idx = T - 1
            direction = -1
        elif idx <= 0:
            idx = 0
            direction = 1

    simulation_app.close()


if __name__ == "__main__":
    main()
