"""用工件 obj 当障碍、seam pkl 的 joint_angles 当目标、retract 当起点，规划轨迹并保存。

（这是端到端连通性验证：obj 当【已知】障碍，不同于未知探索主线。）

运行：conda run -n env_isaaclab python scripts/plan_seam.py --seam <seam_x.pkl> --out /tmp/seam_traj.npz
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl")


def find_obj(seam_path):
    d = os.path.dirname(seam_path)
    for pat in ("*_watertight.obj", "*.obj"):
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no obj in {d}")


def seam_ee_pose(d, index=None):
    """焊缝几何 -> 末端目标位姿（机械臂基座系）。返回 (pos[3], quat_wxyz[4])。

    - 位置 = 焊缝中点（seam_line[middle]）。
    - 接近轴（末端 x 轴）= -normalize(dir1+dir2)，即相交两面的角平分方向
      （与 FK(joint_angles) 实测一致：末端 x 轴·角平分线 = -1）。
    - 绕接近轴的自转(roll)是自由 DOF，这里用 seam_tangent 对齐作规范选择。
    seam_line/seam_tangent/seam_limits 都在世界系，按 robot_pose 换算到基座系。
    """
    from scipy.spatial.transform import Rotation as Rsp

    line = np.asarray(d["seam_line"], float)         # (N,3) world
    tang = np.asarray(d["seam_tangent"], float)      # (N,3) world
    limits = np.asarray(d["seam_limits"], float)     # (N,2,3) world
    i = int(d.get("middle", len(line) // 2)) if index is None else index

    rp = np.asarray(d["robot_pose"][0], float)       # [x,y,z,qw,qx,qy,qz]
    Rwr = Rsp.from_quat(np.r_[rp[4:7], rp[3]]).as_matrix()   # wxyz -> xyzw
    to_base_p = lambda p: Rwr.T @ (p - rp[:3])
    to_base_v = lambda v: Rwr.T @ v

    pos = to_base_p(line[i])
    t = to_base_v(tang[i]); t /= np.linalg.norm(t)
    d1 = to_base_v(limits[i, 0]); d2 = to_base_v(limits[i, 1])
    bis = d1 + d2; bis /= np.linalg.norm(bis)
    x = -bis                                          # 接近轴
    y = t - np.dot(t, x) * x; y /= np.linalg.norm(y)  # 去掉沿 x 分量，对齐切向
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    q_xyzw = Rsp.from_matrix(R).as_quat()
    return pos.tolist(), np.r_[q_xyzw[3], q_xyzw[:3]].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--out", default="/tmp/seam_traj.npz")
    ap.add_argument("--max_attempts", type=int, default=20)
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType

    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))
    goal_pose = seam_ee_pose(d)

    target = np.asarray(d["joint_angles"], dtype=float).tolist()
    robot_pos = np.asarray(d["robot_pose"][0], dtype=float)
    piece_pose = np.asarray(d['piece_pose'][0], dtype=float)
    obj_path = find_obj(args.seam)
    print("seam   :", args.seam)
    print("obj    :", obj_path)
    print("robot_pos(world):", np.round(robot_pos, 4))
    print("target joint_angles:", np.round(target, 4))

    # 工件相对机械臂基座位姿 = inv(T_world_robot) ∘ T_world_piece（cuRobo Pose, wxyz）
    import torch
    from curobo.types.math import Pose

    def _pose(p7):
        return Pose(
            position=torch.tensor([p7[:3]], dtype=torch.float32, device="cuda"),
            quaternion=torch.tensor([p7[3:7]], dtype=torch.float32, device="cuda"),
        )

    robot_pos_inv = _pose(robot_pos).inverse()
    T_robot_piece = robot_pos_inv.multiply(_pose(piece_pose))
    mesh_pose = T_robot_piece.get_pose_vector()[0].detach().cpu().numpy().tolist()
    print("工件@机械臂基座 pose [x,y,z,qw,qx,qy,qz]:", np.round(mesh_pose, 4))

    mesh = Mesh(name="workpiece", file_path=obj_path, pose=mesh_pose)
    world = WorldConfig(mesh=[mesh])

    print("\n== init_curobo（MESH 世界含工件；焊枪 link 排除碰撞）==")
    # h = ci.init_curobo(cfg, world_model=world,
    #                    collision_checker_type=CollisionCheckerType.MESH,
    #                    drop_collision_links=["xiaoyu_accessory_link"])
    '''
        ┌────────────────────┬────────────┐
        │ rotation_threshold │ 对应角度 θ │
        ├────────────────────┼────────────┤
        │ 0.05（默认）        │ ≈ 5.7°     │
        ├────────────────────┼────────────┤
        │ 0.1                │ ≈ 11.5°    │
        ├────────────────────┼────────────┤
        │ 0.2                │ ≈ 23.1°    │
        ├────────────────────┼────────────┤
        │ 0.3                │ ≈ 34.9°    │
        ├────────────────────┼────────────┤
        │ 0.5                │ ≈ 60°      │
        └────────────────────┴────────────┘
        │ 1.0                │ ≈ 180°      │
        └────────────────────┴────────────┘
    '''
    h = ci.init_curobo(cfg, world_model=world,
                       collision_checker_type=CollisionCheckerType.MESH,
                    #    drop_collision_links=["xiaoyu_accessory_link"],
                       position_threshold=0.05, rotation_threshold=0.5)

    retract = cfg.retract_config
    metric = ci.free_pose_metric(h, free_rot=(0,))      # 放开焊枪绕接近轴自转

    # goal_pose 对应的关节角（与规划用同一 metric，便于失败时诊断终点）
    goal_cfg = ci.ik_best_config(h, ci.solve_ik(h, goal_pose, pose_cost_metric=metric))

    print("\n== plan_to_pose: retract -> 焊缝几何位姿 ==")
    # res = ci.plan_to_config(h, retract, target, max_attempts=args.max_attempts)
    res = ci.plan_to_pose(h, retract, goal_pose,
                          max_attempts=args.max_attempts,
                          pose_cost_metric=metric)
    # res = ci.plan_to_pose2(h, retract, goal_pose, max_attempts=args.max_attempts)

    ok = res is not None and bool(res.success.item())
    print("success:", ok, " status:", getattr(res, "status", None))
    if not ok:
        if goal_cfg is None:
            print("PLAN_FAIL: IK 解不出 goal_pose（无终点可检查）")
        else:
            print("PLAN_FAIL:", ci.explain_endpoints(h, retract, goal_cfg))
        return

    traj = res.get_interpolated_plan()
    pos = traj.position.detach().cpu().numpy()
    np.savez(
        args.out,
        positions=pos,
        joint_names=np.array(cfg.joint_names),
        retract=np.array(retract),
        target=np.array(target),
        piece_pose_to_robot=mesh_pose,
        obj_path=obj_path,
        robot_usd=h.config.robot_cfg["robot_cfg"]["kinematics"]["usd_path"],
        dt=0.02,
    )
    print("轨迹点数:", pos.shape, " 已保存:", args.out)
    print("起点(deg近似):", np.round(pos[0], 3))
    print("终点:", np.round(pos[-1], 3), " 目标:", np.round(target, 3))
    print("PLAN_OK")


if __name__ == "__main__":
    main()
