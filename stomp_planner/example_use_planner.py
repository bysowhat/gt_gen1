"""示例：在【别的项目】里调用 gt_overall 的 STOMP 避障规划能力。

演示两个对外函数（来自 stomp_planning_api.py）：
    函数1 plan_to_joint_config(cur_cfg, target_cfg, world)  —— 规划到目标关节角（返回所有候选）
          plan_to_joint_single/-_multi(...)                —— 同上，但直接返回最优/全部合格 traj
    函数2 plan_to_pose(cur_cfg, target_pose, world)         —— 规划到目标末端位姿

=== 在别的项目里怎么用（3 步）===
1) 让 Python 找得到本项目目录（gt_overall），把它加进 sys.path：
       import sys
       sys.path.insert(0, "/home/a/Projects/Github/kejian_guihua/gt_overall")
   （或把 gt_overall 装成包 / 设 PYTHONPATH。若要用 gt_gen 原语建障碍，
     还需保证 gt_gen_hanfeng 可导入，可设 GT_GEN_ROOT 环境变量。）
2) 准备一个 cuRobo 碰撞世界 `world`（curobo.geom.types.WorldConfig）：
       - 自己构造：WorldConfig(cuboid=[Cuboid(...)], cylinder=[...], mesh=[Mesh(...)])
       - 或用本模块辅助：build_world_from_obstacles(...) / empty_world()
       - 若你已有 gt_gen 的 h_expl 这类封装，把它底层的 WorldConfig 传进来即可。
   注意：world 内位姿与 target_pose 都在【机器人 base 系】，四元数 wxyz。
3) 调用函数，拿到 (T,6) 的关节轨迹 ndarray，存盘或直接驱动你的机械臂。

运行本示例（须在 env_isaaclab 环境、有 CUDA）：
    python example_use_planner.py
    python example_use_planner.py --empty_world      # 不放障碍（仅自碰撞）
"""

import argparse
import sys

import numpy as np

# --- 第 1 步：把 gt_overall 加入 sys.path（在你自己的项目里改成绝对路径）---
GT_OVERALL_DIR = "/home/a/Projects/Github/kejian_guihua/gt_overall"
if GT_OVERALL_DIR not in sys.path:
    sys.path.insert(0, GT_OVERALL_DIR)

from stomp_planning_api import (
    plan_to_joint_config,        # 函数1：返回所有 batch 候选 (trajs, infos)
    plan_to_joint_single,        # 函数1-single：仅返回最优合格 traj（无则 None）
    plan_to_joint_multi,         # 函数1-multi：返回所有合格 traj 列表（无则 None）
    plan_to_pose_single,         # 函数2-single：仅返回最优合格 traj（无则 None）
    plan_to_pose_multi,          # 函数2-multi：返回所有合格 traj 列表（无则 None）
    build_world_from_obstacles,  # 可选：用 gt_gen 原语建障碍世界
    empty_world,                 # 可选：空世界
    StompPlanner,                # 可选：复用同一世界多次规划
)


def build_world_manually():
    """方式A：完全用 cuRobo 原生 API 构造碰撞世界（不依赖 gt_gen）。

    在 base 系放一个 0.16m 立方体障碍挡在路径中段（与项目 DEFAULT_OBSTACLES 等价）。
    """
    from curobo.geom.types import WorldConfig, Cuboid  # noqa: 须在有 curobo 的环境
    box = Cuboid(
        name="box_beam",
        pose=[0.15, -0.05, 0.80, 1.0, 0.0, 0.0, 0.0],  # x,y,z, qw,qx,qy,qz (wxyz)
        dims=[0.16, 0.16, 0.16],
    )
    from curobo.geom.sdf.world import CollisionCheckerType
    return WorldConfig(cuboid=[box], cylinder=[], mesh=[]), CollisionCheckerType.PRIMITIVE


def main():
    ap = argparse.ArgumentParser(description="STOMP 规划 API 使用示例")
    ap.add_argument("--empty_world", action="store_true", help="用空世界(仅自碰撞)")
    ap.add_argument("--manual_world", action="store_true",
                    help="用纯 cuRobo API 手搓障碍世界(不经 gt_gen)")
    ap.add_argument("--buffer", type=float, default=0.1,
                    help="碰撞安全缓冲/激活距离(m)，>0 即视为进入缓冲(代价>0)")
    ap.add_argument("--out_joint", default="example_traj_joint.npy")
    ap.add_argument("--out_pose", default="example_traj_pose.npy")
    args = ap.parse_args()

    # 与项目脚本一致的两端构型（base 系关节角，弧度）
    cur_cfg = [1.5707824230194092, -2.0071660480894984, 1.3613484541522425,
               -0.9599629205516357, -1.570770565663473, 0.0]
    target_cfg = [3.442432165145874, -2.472576379776001, 1.4129290580749512,
                  -0.4071854054927826, -2.634472131729126, -3.2598252296447754]

    # === 第 2 步：准备 cuRobo 碰撞世界 ===
    if args.empty_world:
        world, checker = empty_world()
        print("[example] 碰撞世界：空（仅自碰撞）")
    elif args.manual_world:
        world, checker = build_world_manually()
        print("[example] 碰撞世界：手搓 cuRobo Cuboid 障碍")
    else:
        '''
        world, checker = build_world_from_obstacles(
        workpiece_mesh="/path/piece.obj", workpiece_pose=[0,0,0,1,0,0,0],
        use_obstacles=True)          # 基础障碍 + 工件 mesh，内部已转 mesh + MESH checker
        '''
        # 用 gt_gen_hanfeng 原语建障碍（与 plan_path_stomp_obstacle 同源）
        world, checker = build_world_from_obstacles()
        print("[example] 碰撞世界：gt_gen 原语 DEFAULT_OBSTACLES")

    '''
    num_iterations, num_batch参数：
        num_iterations=120 —— 迭代次数（搜索深度）

            STOMP 是迭代式优化：每轮在当前轨迹上撒噪声、按代价加权更新、平滑滤波，反复逼近低代价轨迹。num_iterations 就是这个循环跑多少轮。

            - 越大 → 收敛越充分，更可能把轨迹彻底推出障碍/缓冲区，碰撞代价压得更低；代价是耗时线性增加。
            - 越小 → 快，但可能没收敛完，留下 n_collision_steps > 0（仍有时间步在缓冲/碰撞内）。
            - 默认 80；示例里调到 120 是因为有障碍要绕，给它更多轮次确保绕开。

        num_batch=8 —— 并行搜索条数（搜索宽度）

            同一对首末点，开 num_batch 条相互独立的 STOMP 同时跑（用不同随机噪声探索不同绕行方向），最后取状态代价最小的那条输出（stomp_planning_api.py:153-156）：

            fixed_pts = torch.stack([cur, target]).unsqueeze(0).repeat(num_batch, 1, 1)  # B 条都固定同一对端点
            traj_all, _ = stomp.solve(fixed_pts)
            best = argmin(stomp.parameters_state_cost)   # 取最优 1 条

            - 越大 → 更不容易陷在局部最优（比如该从障碍左边绕还是右边绕，多条并行各试一种，挑成功的），鲁棒性更好；GPU 上是并行的，显存够时时间增加不明显。
            - 越小 → 省显存，但运气不好那一条卡在局部最优时没有备选。
            - 默认 4；示例里调到 8 是为避障多撒几条候选。

    '''

    # === 第 3 步之一：函数1 —— 规划到目标关节角（返回所有 batch 候选）===
    print("\n==== 函数1: plan_to_joint_config (cur -> target_cfg) ====")
    trajs_joint, infos_joint = plan_to_joint_config(
        cur_cfg, target_cfg, world, checker_type=checker,
        buffer=args.buffer,                      # 碰撞安全缓冲(m)
        num_iterations=120, num_batch=8)         # 透传 STOMP 旋钮（可选）

    # trajs_joint: (B,T,6) 所有候选；infos_joint: 各 batch 诊断 dict 列表
    print(f"  候选轨迹形状 (B,T,6) = {trajs_joint.shape}")
    for info in infos_joint:
        print(f"  batch{info['batch']}: state_cost={info['state_cost']:.3f} "
              f"碰撞max={info['collision_max']:.4f} "
              f"进缓冲/碰撞步数={info['n_collision_steps']}/{info['n_steps']} "
              f"在限位内={info['in_limit']}")
    # 自己挑一条用（例：满足无碰撞+在限位内、state_cost 最小的）
    ok = [i for i in infos_joint if i["n_collision_steps"] == 0 and i["in_limit"]]
    if ok:
        best = min(ok, key=lambda i: i["state_cost"])["batch"]
        np.save(args.out_joint, trajs_joint[best])
        print(f"  选用 batch{best} 保存: {args.out_joint}")
    else:
        print("  ⚠ 没有完全合格(无碰撞+在限位)的候选；可加大 num_iterations/num_batch")

    # 上面的「挑一条」也可直接用便捷封装（与 plan_to_pose_single/multi 对称）：
    print("\n==== 函数1-single: plan_to_joint_single (cur -> target_cfg) ====")
    traj_joint_best = plan_to_joint_single(
        cur_cfg, target_cfg, world, checker_type=checker,
        buffer=args.buffer, num_iterations=120, num_batch=8)
    if traj_joint_best is None:
        print("  ⚠ 无满足(无碰撞+在限位)的轨迹，返回 None")
    else:
        print(f"  最优合格轨迹形状 (T,6) = {traj_joint_best.shape}")

    print("\n==== 函数1-multi: plan_to_joint_multi (cur -> target_cfg) ====")
    trajs_joint_ok = plan_to_joint_multi(
        cur_cfg, target_cfg, world, checker_type=checker,
        buffer=args.buffer, num_iterations=120, num_batch=8)
    if trajs_joint_ok is None:
        print("  ⚠ 无满足(无碰撞+在限位)的轨迹，返回 None")
    else:
        print(f"  合格轨迹条数 = {len(trajs_joint_ok)}，每条形状 = {trajs_joint_ok[0].shape}")

    # === 第 3 步之二：函数2 —— 规划到目标末端位姿 ===
    # 用一个示例位姿（这里取 target_cfg 的 FK 结果作目标，确保可达；
    # 你的项目里换成自己的目标位姿即可，格式 [x,y,z,qw,qx,qy,qz]，base 系）。
    target_pose = make_demo_pose(target_cfg)
    print("\n==== 函数2-single: plan_to_pose_single (cur -> target_pose) ====")
    print(f"  目标位姿(base系) = {np.round(target_pose, 4).tolist()}")
    traj_pose = plan_to_pose_single(
        cur_cfg, target_pose, world, checker_type=checker,
        buffer=args.buffer, num_iterations=120, num_batch=8)
    if traj_pose is None:
        print("  ⚠ 无满足(无碰撞+在限位)的轨迹，返回 None")
    else:
        print(f"  最优合格轨迹形状 (T,6) = {traj_pose.shape}")
        np.save(args.out_pose, traj_pose)
        print(f"  已保存: {args.out_pose}")

    print("\n==== 函数2-multi: plan_to_pose_multi (cur -> target_pose) ====")
    trajs_pose = plan_to_pose_multi(
        cur_cfg, target_pose, world, checker_type=checker,
        buffer=args.buffer, num_iterations=120, num_batch=8)
    if trajs_pose is None:
        print("  ⚠ 无满足(无碰撞+在限位)的轨迹，返回 None")
    else:
        print(f"  合格轨迹条数 = {len(trajs_pose)}，每条形状 = {trajs_pose[0].shape}")

    # === （可选）复用同一世界多次规划：直接用 StompPlanner，避免重建碰撞世界 ===
    print("\n==== 复用示例: StompPlanner ====")
    planner = StompPlanner(world, checker_type=checker, buffer=args.buffer)
    t1, i1 = planner.plan_joint(cur_cfg, target_cfg, num_iterations=80)
    t2, i2 = planner.plan_pose(cur_cfg, target_pose, num_iterations=80)
    print(f"  复用规划器各得到一批候选: plan_joint {t1.shape}, plan_pose {t2.shape}")

    print("\n[example] 完成。可用 visualize_traj_isaacsim.py 回放查看避障效果：")
    print(f"    python visualize_traj_isaacsim.py --traj {args.out_joint}")


def make_demo_pose(cfg):
    """对给定关节角做一次 cuRobo FK，得到末端位姿 [x,y,z,qw,qx,qy,qz]（仅为示例造一个可达目标）。"""
    from stomp_planning_api import _apply_shim, DEFAULT_ROBOT_YML
    _apply_shim()
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.util_file import load_yaml

    ta = TensorDeviceType()
    robot_cfg = load_yaml(DEFAULT_ROBOT_YML)["robot_cfg"]
    kin_cfg = CudaRobotModelConfig.from_data_dict(robot_cfg["kinematics"], ta)
    model = CudaRobotModel(kin_cfg)
    q = ta.to_device(cfg).view(1, -1)
    state = model.get_state(q)
    pos = state.ee_position[0].detach().cpu().numpy()
    quat = state.ee_quaternion[0].detach().cpu().numpy()   # wxyz
    return [float(x) for x in (*pos, *quat)]


if __name__ == "__main__":
    main()
