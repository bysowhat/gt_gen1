"""
带障碍物的 STOMP 单段路径规划（基于本项目 STOMP 方法 + cuRobo 碰撞世界）。

与 plan_path_stomp.py（free 空间）的区别：
    把恒为 0 的「状态代价」换成**真实碰撞代价**，碰撞世界 = 工件 mesh + 基础障碍物，
    构建方式与 /home/a/Projects/Github/gt_gen_hanfeng 项目一致：
      - 基础障碍物：用 gt_gen_hanfeng 的 gt_gen/obstacles.py 原语（plate/pipe/open_box/...）
        生成 → to_curobo → cuRobo Cuboid/Cylinder；
      - 工件：cuRobo Mesh(file_path=*.obj/.stl, pose=...)；
      - 合成 cuRobo WorldConfig。
    碰撞代价通过 cuRobo `RobotWorld` 批量查询：把每条 rollout 各时间步的关节角喂进
    get_world_self_collision_distance_from_joints，得到「世界碰撞 + 自碰撞」距离场
    (>0 表示进入安全缓冲/穿透)，作为 STOMP 的状态代价 —— 机械臂碰撞球 vs 障碍，
    与 gt_gen_hanfeng 的避障判据同源。

STOMP 主体（控制代价矩阵、N(0,R⁻¹) 噪声、概率加权、平滑滤波、固定首末点）
完全复用 plan_path_stomp.py 的 FreeSpaceStomp。

依赖：cuRobo（在 env_isaaclab 中已装）、gt_gen_hanfeng（提供 obstacles.py）。
必须用 GPU（cuRobo RobotWorld 仅 cuda）。

运行（仅基础障碍物）：
    python plan_path_stomp_obstacle.py --output traj_obs.npy
带工件 mesh：
    python plan_path_stomp_obstacle.py --workpiece_mesh /path/piece.obj \
        --workpiece_pose 0 0 0 1 0 0 0 --output traj_obs.npy
"""

import os
import sys
import argparse

import numpy as np
import torch

# 本项目 STOMP 核心
_PROJ_DIR = os.path.dirname(os.path.realpath(__file__))
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)
from plan_path_stomp import FreeSpaceStomp, load_robot_cfg, SEED  # noqa: E402

# gt_gen_hanfeng（提供基础障碍物原语 obstacles.py 与 warp/trimesh shim）
# 默认 = 本文件上两级目录（stomp_planner/ 的父目录即项目根），与机器无关；env GT_GEN_ROOT 可覆盖。
GT_GEN_ROOT = os.environ.get("GT_GEN_ROOT", os.path.dirname(_PROJ_DIR))
if GT_GEN_ROOT not in sys.path:
    sys.path.insert(0, GT_GEN_ROOT)

torch.set_default_dtype(torch.float)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------------- #
# 基础障碍物场景定义（编辑此处即可改障碍布局；用 obstacles.py 原语，base 系: +X前 +Y左 +Z上）
# 每项: (障碍类型名, anchor位置[x,y,z], anchor姿态rpy(度), 形状参数dict)
# 可选类型见 gt_gen/obstacles.py: plate/l_bracket/u_channel/open_box/pipe/parallel_pipes/
#   crossed_pipes/box_beam/rect_frame/gantry/braced_frame/tripod/steps/box_with_pipe/...
# ----------------------------------------------------------------------------- #
DEFAULT_OBSTACLES = [
    # 一块挡在直线路径中段的方块（两端构型都在其外、直线路径穿过它 → 演示避障）
    ("box_beam", [0.15, -0.05, 0.80], (0, 0, 0), dict(length=0.16, side=0.16, axis="y")),
]


def build_world(workpiece_mesh=None, workpiece_pose=None, use_obstacles=True,
                obstacle_spec=None):
    """构建 cuRobo WorldConfig：工件 mesh + 基础障碍物（与 gt_gen_hanfeng 一致）。

    返回 (world_config, checker_type)。
    """
    import gt_gen.compat  # noqa: F401  warp/trimesh shim，须在 import curobo 前
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    from gt_gen import obstacles as obs_lib

    cuboids, cylinders, meshes = [], [], []

    # 基础障碍物原语 → cuRobo Cuboid/Cylinder
    if use_obstacles:
        spec = obstacle_spec if obstacle_spec is not None else DEFAULT_OBSTACLES
        prims = []
        for name, pos, rpy, shape in spec:
            prims.extend(obs_lib.build(name, pos, rpy, **shape))
        for o in obs_lib.to_curobo(prims):
            if o.__class__.__name__ == "Cuboid":
                cuboids.append(o)
            elif o.__class__.__name__ == "Cylinder":
                cylinders.append(o)
        print(f"[world] 基础障碍物原语 {len(prims)} 个 -> "
              f"{len(cuboids)} cuboid + {len(cylinders)} cylinder")

    # 工件 mesh
    if workpiece_mesh:
        pose = list(workpiece_pose) if workpiece_pose is not None else [0, 0, 0, 1, 0, 0, 0]
        meshes.append(Mesh(name="workpiece", file_path=workpiece_mesh, pose=pose))
        print(f"[world] 工件 mesh: {workpiece_mesh}  pose={pose}")

    world = WorldConfig(mesh=meshes, cuboid=cuboids, cylinder=cylinders)

    # 有 mesh 时统一转 mesh 世界用 MESH 检查器（与 gt_gen_hanfeng build_world 一致）；
    # 纯原语时用 PRIMITIVE 检查器（精确且快）。
    if meshes:
        world = world.get_mesh_world(merge_meshes=False)
        checker = CollisionCheckerType.MESH
    else:
        checker = CollisionCheckerType.PRIMITIVE
    return world, checker


class CollisionOracle:
    """cuRobo RobotWorld 封装：批量关节角 -> (世界碰撞 + 自碰撞) 距离 (>0 = 碰撞/进缓冲)。"""

    def __init__(self, robot_yml, world_config, checker_type,
                 activation_distance=0.1, chunk=8192):
        import gt_gen.compat  # noqa: F401
        gt_gen.compat.apply_trimesh_shim()
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import load_yaml
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

        self.ta = TensorDeviceType()
        self.device = self.ta.device
        rd = load_yaml(robot_yml)
        n_cub = max(50, len(getattr(world_config, "cuboid", []) or []) + 5)
        n_mesh = max(50, len(getattr(world_config, "mesh", []) or []) + 5)
        cfg = RobotWorldConfig.load_from_config(
            rd, world_config, self.ta,
            collision_checker_type=checker_type,
            collision_activation_distance=activation_distance,
            n_cuboids=n_cub, n_meshes=n_mesh,
        )
        self.rw = RobotWorld(cfg)
        self.chunk = chunk
        self.joint_names = list(self.rw.kinematics.joint_names)

    @torch.no_grad()
    def distance(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, dof) -> (N,) 碰撞距离 (clamp>=0)。"""
        q = q.to(self.device, dtype=torch.float32)
        out = []
        for i in range(0, q.shape[0], self.chunk):
            qc = q[i:i + self.chunk].contiguous()
            dw, ds = self.rw.get_world_self_collision_distance_from_joints(qc)
            out.append((dw.clamp(min=0) + ds.clamp(min=0)))
        return torch.cat(out, dim=0)


class ObstacleStomp(FreeSpaceStomp):
    """在 FreeSpaceStomp 基础上，用 CollisionOracle 计算碰撞状态代价。"""

    def __init__(self, oracle: CollisionOracle, collision_cost_weight=80.0, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        self.collision_cost_weight = collision_cost_weight
        # free 空间分支阈值在父类用到 collision_cost_weight；这里 noise/filter 固定低档即可

    def computeStateCost(self, rollouts: torch.Tensor):
        B, R, D, T = rollouts.shape
        q = rollouts.permute(0, 1, 3, 2).reshape(-1, D)        # (B*R*T, D)
        d = self.oracle.distance(q).reshape(B, R, T)           # 每个构型碰撞距离
        collision_cost = (d.unsqueeze(2).expand(B, R, D, T).contiguous()
                          * self.collision_cost_weight)
        vision_cost = torch.zeros_like(collision_cost)
        return collision_cost, vision_cost


# ----------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="带障碍物的 STOMP 路径规划")
    p.add_argument("--robot_yml", type=str,
                   default="/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml")
    p.add_argument("--cur", type=float, nargs=6, default=None)
    p.add_argument("--target", type=float, nargs=6, default=None)
    p.add_argument("--workpiece_mesh", type=str, default=None, help="工件 mesh 路径(.obj/.stl)")
    p.add_argument("--workpiece_pose", type=float, nargs=7, default=None,
                   help="工件位姿 x y z qw qx qy qz (base系)")
    p.add_argument("--no_obstacles", action="store_true", help="不加基础障碍物(只用工件)")
    p.add_argument("--num_timesteps", type=int, default=51)
    p.add_argument("--num_iterations", type=int, default=80)
    p.add_argument("--num_batch", type=int, default=4, help="并行搜索条数(取最优1条输出)")
    p.add_argument("--delta_t", type=float, default=0.1)
    p.add_argument("--collision_weight", type=float, default=80.0)
    p.add_argument("--buffer", type=float, default=0.1, help="碰撞安全缓冲/梯度带(m)")
    p.add_argument("--noise_scale", type=float, default=1.5)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo RobotWorld 需要 CUDA，但当前不可用。")
    device = "cuda"

    cur_cfg = args.cur if args.cur is not None else [
        1.5707824230194092, -2.0071660480894984, 1.3613484541522425,
        -0.9599629205516357, -1.570770565663473, 0.0]
    target_cfg = args.target if args.target is not None else [
        3.442432165145874, -2.472576379776001, 1.4129290580749512,
        -0.4071854054927826, -2.634472131729126, -3.2598252296447754]

    robot = load_robot_cfg(args.robot_yml)
    lower, upper = robot["lower_limit"], robot["upper_limit"]
    print("关节下限:", [round(x, 4) for x in lower])
    print("关节上限:", [round(x, 4) for x in upper])

    # 构建碰撞世界（工件 mesh + 基础障碍物）
    world, checker = build_world(
        workpiece_mesh=args.workpiece_mesh,
        workpiece_pose=args.workpiece_pose,
        use_obstacles=not args.no_obstacles,
        obstacle_spec=DEFAULT_OBSTACLES,
    )
    oracle = CollisionOracle(args.robot_yml, world, checker,
                             activation_distance=args.buffer)
    print("[oracle] 关节顺序:", oracle.joint_names)

    # 端点碰撞自检
    qends = torch.tensor([cur_cfg, target_cfg], device=device)
    dends = oracle.distance(qends).cpu().numpy()
    print(f"[check] cur_cfg 碰撞距离={dends[0]:.4f}, target_cfg 碰撞距离={dends[1]:.4f} "
          f"(>0 表示在缓冲/碰撞内)")

    # STOMP
    stomp = ObstacleStomp(
        oracle=oracle, collision_cost_weight=args.collision_weight,
        lower_limit=lower, upper_limit=upper, device=device,
        num_batch=args.num_batch, num_timesteps=args.num_timesteps,
        delta_t=args.delta_t, num_iterations=args.num_iterations,
        noise_scale=args.noise_scale, filter_scale=2.0)

    cur = torch.tensor(cur_cfg, device=device)
    target = torch.tensor(target_cfg, device=device)
    fixed_pts = torch.stack([cur, target], dim=0).unsqueeze(0).repeat(args.num_batch, 1, 1)

    traj_all, total_cost = stomp.solve(fixed_pts)        # (B, D, T), (B,)

    # 选碰撞(状态)代价最小的一条（vision=0，故 state_cost 即碰撞代价）
    best = int(torch.argmin(stomp.parameters_state_cost).item())
    traj = traj_all[best]                                # (D, T)
    traj_np = traj.transpose(0, 1).cpu().numpy()         # (T, 6)

    # 评估：整条轨迹的最大碰撞距离（0 = 全程无碰）
    dtraj = oracle.distance(traj.transpose(0, 1)).cpu().numpy()  # (T,)
    n_coll = int((dtraj > 1e-4).sum())
    print(f"\n规划完成: 选第 {best} 条 batch，轨迹形状 (T,D)={traj_np.shape}")
    print(f"  碰撞代价(该条)={float(stomp.parameters_state_cost[best]):.4f}, "
          f"总代价={float(total_cost[best]):.4f}")
    print(f"  全程碰撞距离 max={dtraj.max():.4f} mean={dtraj.mean():.4f}, "
          f"进入缓冲/碰撞的时间步数={n_coll}/{len(dtraj)}")
    print("  起点(应=cur):   ", np.round(traj_np[0], 4))
    print("  终点(应=target):", np.round(traj_np[-1], 4))
    in_limit = (np.all(traj_np >= np.array(lower) - 1e-5)
                and np.all(traj_np <= np.array(upper) + 1e-5))
    print("  全程在关节限位内:", bool(in_limit))
    if n_coll == 0:
        print("  ✅ 轨迹无碰撞")
    else:
        print("  ⚠ 仍有时间步在缓冲/碰撞内：可增大 --collision_weight / --num_iterations / "
              "--noise_scale / --num_batch，或确认两端点本身是否可行")

    if args.output:
        np.save(args.output, traj_np)
        print("轨迹已保存:", args.output)

    if args.plot:
        import matplotlib.pyplot as plt
        t = np.arange(traj_np.shape[0]) * args.delta_t
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for d in range(6):
            ax1.plot(t, traj_np[:, d], label=f"joint{d+1}")
        ax1.set_ylabel("joint angle (rad)"); ax1.legend(); ax1.grid(True)
        ax1.set_title("STOMP trajectory with obstacles")
        ax2.plot(t, dtraj, "r-")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("collision dist (m)")
        ax2.grid(True)
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
