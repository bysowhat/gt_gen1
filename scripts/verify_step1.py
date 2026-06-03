"""Step 1 验证：cuRobo 封装 IK + 规划（VOXEL 全自由世界）。

运行：conda run -n env_isaaclab python scripts/verify_step1.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci

    cfg = load_config()
    print("== init_curobo（VOXEL 全自由世界）+ warmup ==")
    h = ci.init_curobo(cfg)
    print("OK；voxel 世界:", h.voxel["dims"], "@", h.voxel["voxel_size"], "m")

    retract = cfg.retract_config

    # 1) FK：retract -> ee 位姿
    ee_pos, ee_quat, link_pose = ci.fk(h, retract)
    print("\n== FK ==")
    print("ee_pos :", np.round(ee_pos, 4), " ee_quat(wxyz):", np.round(ee_quat, 4))
    print("Link6 in link_pose:", "Link6" in link_pose)

    # 2) IK：把上面的 ee 位姿解回关节角，应成功
    print("\n== IK ==")
    ik = ci.solve_ik(h, (ee_pos.tolist(), ee_quat.tolist()))
    succ = bool(ik.success.any().item()) if hasattr(ik.success, "any") else bool(ik.success)
    print("IK success:", succ)
    if succ:
        sol = ik.solution.detach().cpu().numpy().reshape(-1)[: len(retract)]
        print("IK solution:", np.round(sol, 4))

    # 3) plan_to_pose：retract -> 另一个构型的 ee 位姿（内部 IK->关节规划）
    print("\n== plan_to_pose ==")
    goal_cfg = list(retract); goal_cfg[0] += 0.5; goal_cfg[1] += 0.3
    g_pos, g_quat, _ = ci.fk(h, goal_cfg)
    ikg = ci.solve_ik(h, (g_pos.tolist(), g_quat.tolist()))
    print("goal 位姿 IK: success=", bool(ikg.success.any().item()),
          " pos_err=", float(ikg.position_error.min().item()))
    res = ci.plan_to_pose(h, retract, (g_pos.tolist(), g_quat.tolist()), max_attempts=10)
    ok = res is not None and bool(res.success.item())
    print("plan success:", ok)
    if ok:
        traj = res.get_interpolated_plan()
        print("轨迹点数:", traj.position.shape[0], " 维度:", traj.position.shape[1])

    # 4) plan_to_config：retract -> goal_cfg
    print("\n== plan_to_config ==")
    res2 = ci.plan_to_config(h, retract, goal_cfg)
    ok2 = bool(res2.success.item()) if hasattr(res2.success, "item") else bool(res2.success)
    print("success:", ok2)
    if ok2:
        traj2 = res2.get_interpolated_plan()
        print("轨迹点数:", traj2.position.shape[0])

    print("\nSTEP1_OK")


if __name__ == "__main__":
    main()
