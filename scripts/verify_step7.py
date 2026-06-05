"""Step 7 验证：reach_pt + 阻塞段 B。

真值(工件 mesh)上规划 P* = retract → seam 关节角；在不同 voxmap 状态下验证：
- 全 UNKNOWN：reach_idx=0（鼻子尖前就是未知），B 非空且全 UNKNOWN；
- 把 P* 前半段扫掠体积标 FREE：reach_pt 前进到前半段末、停在未知前沿；
- 把整条 P* 扫掠标 FREE：reach_idx 到终点，B 空。

运行：conda run -n env_isaaclab python scripts/verify_step7.py
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
        "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl")


def main():
    import glob
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap, FREE, UNKNOWN
    from gt_gen.swept import swept_volume
    from gt_gen.reach_b import compute_reach_pt, compute_blocking_B
    from plan_seam import seam_ee_pose                       # 由 pkl 几何算 goal_pose
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    import torch
    from curobo.types.math import Pose

    cfg = load_config()
    d = pickle.load(open(SEAM, "rb"))
    robot_pose = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)
    goal_pose = seam_ee_pose(d)                              # (pos[3], quat_wxyz[4])，非 pkl 关节角
    obj = sorted(glob.glob(os.path.dirname(SEAM) + "/*_watertight.obj"))[0]

    def _p(p7):
        return Pose(position=torch.tensor([p7[:3]], dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor([p7[3:7]], dtype=torch.float32, device="cuda"))
    mp = _p(robot_pose).inverse().multiply(_p(piece_pose)).get_pose_vector()[0].cpu().numpy().tolist()

    # 真值 handle（工件当障碍，焊枪排除）
    world = WorldConfig(mesh=[Mesh(name="workpiece", file_path=obj, pose=mp)])
    print("初始化真值 cuRobo（含工件 MESH）...")
    h = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH,
                       drop_collision_links=["xiaoyu_accessory_link"])

    retract = cfg.retract_config
    metric = ci.free_pose_metric(h, free_rot=(0,))          # 放开焊枪绕接近轴自转
    print("goal_pose:", np.round(goal_pose[0], 3), np.round(goal_pose[1], 3))
    P = ci.plan_on_truth(h, retract, goal_pose, max_attempts=20, pose_cost_metric=metric)
    assert P is not None, "真值上 P*(到 goal_pose) 规划失败"
    print("P* 路点数:", P.shape)
    k = cfg.params["nbv"]["k_lookahead"]

    # 1) 全 UNKNOWN
    vm = build_roi_voxmap(cfg)
    ri = compute_reach_pt(h, vm, P)
    B = compute_blocking_B(h, vm, P, ri, k)
    print(f"\n[全UNKNOWN] reach_idx={ri}  B={B.shape[0]} 格")
    assert ri == 0, "全未知时前方紧贴未知 → reach_pt 应停在起点"
    assert B.shape[0] > 0 and np.all(np.asarray(vm.get(B)) == UNKNOWN)

    # 2) 前半段扫掠标 FREE
    half = len(P) // 2
    for i in range(half):
        vm.set_many(swept_volume(h, vm, P[i], P[i + 1]), FREE)
    ri2 = compute_reach_pt(h, vm, P)
    B2 = compute_blocking_B(h, vm, P, ri2, k)
    print(f"[前半FREE] reach_idx={ri2}/{len(P)-1}  B={B2.shape[0]} 格")
    assert ri2 == half, f"应前进到前半段末 {half}，实际 {ri2}"
    assert B2.shape[0] > 0 and np.all(np.asarray(vm.get(B2)) == UNKNOWN), "B 须全 UNKNOWN"

    # 3) 整条 P* 扫掠标 FREE
    for i in range(len(P) - 1):
        vm.set_many(swept_volume(h, vm, P[i], P[i + 1]), FREE)
    ri3 = compute_reach_pt(h, vm, P)
    B3 = compute_blocking_B(h, vm, P, ri3, k)
    print(f"[整条FREE] reach_idx={ri3}/{len(P)-1}  B={B3.shape[0]} 格")
    assert ri3 == len(P) - 1, "整条已自由 → 应能走到终点"
    assert B3.shape[0] == 0, "无未知 → B 空"

    print("\nSTEP7_OK")


if __name__ == "__main__":
    main()
