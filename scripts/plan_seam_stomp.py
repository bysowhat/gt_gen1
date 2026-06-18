"""plan_seam.py 的 STOMP 版：用基于 STOMP 的避障规划器规划 retract → 焊缝目标位姿。

与 scripts/plan_seam.py 的区别只在【规划器】：
  - plan_seam.py     用 cuRobo MotionGen（graph + trajopt）规划；
  - plan_seam_stomp  用参考项目 gt_overall 的 STOMP 规划器（stomp_planning_api.py）。
其余完全一致：焊缝几何 → 末端目标位姿(base 系)、工件 obj 当障碍(MESH 世界)、retract 当起点；
产出 npz 字段与 plan_seam.py 对齐，故可直接用 scripts/viz_seam_isaacsim.py 回放。

碰撞世界（即 plan_two_cfgs.py 里 h_expl 那种「cuRobo 碰撞世界」）这里取【只含工件 mesh 的
MESH 世界】，与 plan_seam.py 同源；--empty_world 可切成空世界(仅自碰撞)做对照。

STOMP 规划能力来自参考项目（默认路径见 GT_OVERALL_DIR，可用环境变量 GT_OVERALL_DIR 覆盖）：
    函数2 plan_to_pose(cur_cfg, target_pose, world, ...) —— 内部先 IK 解目标关节角，再 STOMP 到关节角。

运行：conda run -n env_isaaclab python scripts/plan_seam_stomp.py --seam <seam_x.pkl> --out /tmp/seam_traj_stomp.npz
回放：conda run -n env_isaaclab python scripts/viz_seam_isaacsim.py --traj /tmp/seam_traj_stomp.npz
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# 参考项目 gt_overall（含 stomp_planning_api）。逐机器不同：默认走下面，env 可覆盖。
GT_OVERALL_DIR = os.environ.get("GT_OVERALL_DIR",
                                "/home/a/Projects/Github/kejian_guihua/gt_overall")
if GT_OVERALL_DIR not in sys.path:
    sys.path.insert(0, GT_OVERALL_DIR)

# 复用 plan_seam.py 已验证的焊缝几何 → 末端位姿、找 obj
from plan_seam import seam_ee_pose, find_obj, DEFAULT_SEAM  # noqa: E402


def _mesh_pose_to_robot(robot_pose7, piece_pose7):
    """工件相对机械臂基座位姿 = inv(T_world_robot) ∘ T_world_piece（cuRobo Pose, wxyz）。

    与 plan_seam.py main() 内的算法逐字一致（torch + cuRobo Pose，需 CUDA）。
    """
    import torch
    from curobo.types.math import Pose

    def _pose(p7):
        return Pose(
            position=torch.tensor([p7[:3]], dtype=torch.float32, device="cuda"),
            quaternion=torch.tensor([p7[3:7]], dtype=torch.float32, device="cuda"),
        )

    robot_inv = _pose(robot_pose7).inverse()
    T_robot_piece = robot_inv.multiply(_pose(piece_pose7))
    return T_robot_piece.get_pose_vector()[0].detach().cpu().numpy().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--out", default="/tmp/seam_traj_stomp.npz")
    ap.add_argument("--offset_cm", type=float, default=10.0,
                    help="目标沿两面角平分方向(bis)平移的距离(cm)，正值远离工件(standoff)")
    ap.add_argument("--empty_world", action="store_true",
                    help="用空世界(仅自碰撞，不放工件)做对照")
    # STOMP 旋钮（透传 stomp_planning_api）
    ap.add_argument("--num_iterations", type=int, default=120, help="STOMP 迭代轮数(搜索深度)")
    ap.add_argument("--num_batch", type=int, default=8, help="STOMP 并行候选条数(搜索宽度)")
    ap.add_argument("--buffer", type=float, default=0.1, help="碰撞安全缓冲/激活距离(m)")
    args = ap.parse_args()

    from gt_gen.config import load_config
    from curobo.util_file import load_yaml
    from stomp_planning_api import (
        StompPlanner, build_world_from_obstacles, empty_world,
    )

    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))

    # 1) 焊缝几何 → 末端目标位姿(base 系) + 调试方向（复用 plan_seam）
    (pos, quat), seam_dbg = seam_ee_pose(d, offset_m=args.offset_cm / 100.0)
    target_pose = list(pos) + list(quat)              # [x,y,z, qw,qx,qy,qz]

    target = np.asarray(d["joint_angles"], dtype=float).tolist()
    robot_pos = np.asarray(d["robot_pose"][0], dtype=float)
    piece_pose = np.asarray(d["piece_pose"][0], dtype=float)
    obj_path = find_obj(args.seam)
    print("seam   :", args.seam)
    print("obj    :", obj_path)
    print("目标位姿(base) [x,y,z,qw,qx,qy,qz]:", np.round(target_pose, 4).tolist())

    # 2) 构造碰撞世界（与 plan_seam 同源：只含工件 mesh 的 MESH 世界）
    if args.empty_world:
        world, checker = empty_world()
        mesh_pose = [0, 0, 0, 1, 0, 0, 0]
        print("碰撞世界：空（仅自碰撞）")
    else:
        mesh_pose = _mesh_pose_to_robot(robot_pos.tolist(), piece_pose.tolist())
        print("工件@机械臂基座 pose [x,y,z,qw,qx,qy,qz]:", np.round(mesh_pose, 4).tolist())
        world, checker = build_world_from_obstacles(
            workpiece_mesh=obj_path, workpiece_pose=mesh_pose, use_obstacles=False)
        print("碰撞世界：工件 mesh（MESH checker）")

    # 3) STOMP 规划 retract → 目标位姿（内部先 IK 解目标关节角，再 STOMP 到关节角）
    retract = list(map(float, cfg.retract_config))
    print("\n== STOMP plan_to_pose: retract -> 焊缝目标位姿 ==")
    # plan_pose 返回所有 batch 候选 (trajs, infos)；这里挑「无碰撞+在限位」中 state_cost 最小的，
    # 若无完全合格者则退回 state_cost 最小的一条（仍写盘，便于排查）。
    planner = StompPlanner(world, checker_type=checker, buffer=args.buffer)
    trajs, infos = planner.plan_pose(
        retract, target_pose,
        num_iterations=args.num_iterations, num_batch=args.num_batch)
    ok = [i for i in range(len(infos))
          if infos[i]["n_collision_steps"] == 0 and infos[i]["in_limit"]]
    best = (min(ok, key=lambda i: infos[i]["state_cost"]) if ok
            else int(np.argmin([x["state_cost"] for x in infos])))
    traj, info = trajs[best], infos[best]
    print(f"候选 {len(infos)} 条，合格(无碰撞+在限位) {len(ok)} 条，选用 batch{best}")

    print(f"IK 选中目标关节角 = {np.round(info['target_cfg'], 4).tolist()}")
    print(f"IK 位置误差={info['ik_position_error']:.5f}m  无碰撞={info['ik_collision_free']}")
    print(f"轨迹形状 (T,6) = {traj.shape}  进缓冲/碰撞步数={info['n_collision_steps']}/{info['n_steps']}  "
          f"在限位内={info['in_limit']}")

    # 4) 落盘（字段与 plan_seam.py 对齐，供 viz_seam_isaacsim.py 回放）
    robot_usd = load_yaml(cfg.robot_cfg_path)["robot_cfg"]["kinematics"]["usd_path"]
    np.savez(
        args.out,
        positions=traj,                                # (T,6)
        joint_names=np.array(cfg.joint_names),
        retract=np.array(retract),
        target=np.array(target),
        target_cfg=np.array(info["target_cfg"]),       # STOMP 实际终点(IK 解)
        piece_pose_to_robot=mesh_pose,
        seam_mid=np.array(seam_dbg["seam_mid"]),
        seam_bisector=np.array(seam_dbg["seam_bisector"]),
        obj_path=obj_path,
        robot_usd=robot_usd,
        dt=0.1,                                        # STOMP delta_t 默认 0.1
    )
    print("已保存:", args.out)
    print("起点:", np.round(traj[0], 3))
    print("终点:", np.round(traj[-1], 3), " IK目标:", np.round(info["target_cfg"], 3))
    if info["n_collision_steps"] == 0 and info["in_limit"]:
        print("PLAN_OK")
    else:
        print(f"PLAN_WARN: 仍有 {info['n_collision_steps']} 步进入缓冲/碰撞或越限"
              f"（可加大 --num_iterations / --num_batch，或调小 --buffer）")


if __name__ == "__main__":
    main()
