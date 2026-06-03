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
    target = np.asarray(d["joint_angles"], dtype=float).tolist()
    robot_pos = np.asarray(d["robot_pose"][0], dtype=float)
    piece_pose = np.asarray(d['piece_pose'][0], dtype=float)
    obj_path = find_obj(args.seam)
    print("seam   :", args.seam)
    print("obj    :", obj_path)
    print("robot_pos(world):", np.round(robot_pos, 4))
    print("target joint_angles:", np.round(target, 4))

    mesh = Mesh(
        name="workpiece",
        file_path=obj_path,
        pose=[piece_pose[0], piece_pose[1], piece_pose[2], piece_pose[3], piece_pose[4], piece_pose[5], piece_pose[6]],
    )
    world = WorldConfig(mesh=[mesh])

    print("\n== init_curobo（MESH 世界含工件；焊枪 link 排除碰撞）==")
    # h = ci.init_curobo(cfg, world_model=world,
    #                    collision_checker_type=CollisionCheckerType.MESH,
    #                    drop_collision_links=["xiaoyu_accessory_link"])
    h = ci.init_curobo(cfg, world_model=world,
                       collision_checker_type=CollisionCheckerType.MESH)

    retract = cfg.retract_config
    print("\n== plan_to_config: retract -> joint_angles ==")
    res = ci.plan_to_config(h, retract, target, max_attempts=args.max_attempts)
    ok = res is not None and bool(res.success.item())
    print("success:", ok, " status:", getattr(res, "status", None))
    if not ok:
        print("PLAN_FAIL")
        return

    traj = res.get_interpolated_plan()
    pos = traj.position.detach().cpu().numpy()
    np.savez(
        args.out,
        positions=pos,
        joint_names=np.array(cfg.joint_names),
        retract=np.array(retract),
        target=np.array(target),
        robot_pose=np.asarray(d["robot_pose"][0], dtype=float),
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
