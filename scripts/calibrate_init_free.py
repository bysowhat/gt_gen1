"""标定"尽量小的初始引导 FREE 空间"——从 tmp/plan_seam 已规划轨迹定 init_free.dq_rad。

这是【离线一次性 / 复核】工具，不在运行时调用。它回答一个问题：
gt_gen/init_free.set_initial_free_space 的 dq 该取多少，才能让机械臂在探索起步时动得了，
同时空间尽量小？做法：扫一组 dq，对每条轨迹用该 FREE 空间跑 compute_reach_pt，统计起步
成功率 / 能走多远 / blob 体素数。机器人、retract 或轨迹集变化后，重跑本脚本复核即可。

运行：
  conda run -n env_isaaclab python scripts/calibrate_init_free.py
  conda run -n env_isaaclab python scripts/calibrate_init_free.py --check-workpiece --n-workpiece 20

字段读取约定同 scripts/plan_seam.py / scripts/viz_seam_isaacsim.py：
  npz: positions(T,6), retract(6), piece_pose_to_robot(=mesh_pose,[x,y,z,qw,qx,qy,qz]), obj_path
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_TRAJ_DIR = os.path.join(ROOT, "tmp", "plan_seam")
DQ_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)


def link6_pos(handle, q):
    import torch
    st = handle.mg.kinematics.get_state(
        torch.tensor([list(q)], dtype=torch.float32, device="cuda"))
    return st.link_pose["Link6"].position[0].detach().cpu().numpy()


def check_workpiece_clear(cells_world, traj_files, n_sample, voxel_size):
    """安全性核查：blob 体素中心是否贴近某工件表面（< 1 个体素即视为"触碰"）。

    退回 retract 是安全 home，理论上 blob 不应与任何工件相交；这里抽样核查。
    """
    from gt_gen.sensor import load_truth_scene
    import trimesh  # noqa: F401  (load_truth_scene 内部用)

    files = traj_files[:: max(1, len(traj_files) // n_sample)][:n_sample]
    worst = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        obj = str(d["obj_path"])
        if not os.path.exists(obj):
            print(f"  [skip] obj 不存在: {obj}")
            continue
        mp = np.asarray(d["piece_pose_to_robot"], float)
        scene = load_truth_scene(obj, mesh_pose=mp)
        # 体素中心到工件表面的最近距离
        _, dist, _ = scene.nearest.on_surface(cells_world)
        near = int((dist < voxel_size).sum())
        worst.append((near, float(dist.min()), os.path.basename(f)))
    worst.sort(reverse=True)
    print("\n== 工件安全性核查（blob 体素 vs 工件表面）==")
    print(f"  抽样 {len(worst)} 个工件；'触碰'= 体素中心到工件面 < {voxel_size} m")
    for near, dmin, name in worst[:5]:
        print(f"    {name}: 触碰体素={near}  最近距离={dmin:.3f} m")
    total_touch = sum(w[0] for w in worst)
    print(f"  合计触碰体素={total_touch}（应为 0：blob 落在工件外才安全标 FREE）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--dq", type=float, nargs="*", default=list(DQ_GRID))
    ap.add_argument("--check-workpiece", action="store_true",
                    help="抽样核查 blob 是否贴近工件表面（默认关，较慢）")
    ap.add_argument("--n-workpiece", type=int, default=20)
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap, FREE
    from gt_gen.reach_b import compute_reach_pt
    from gt_gen.init_free import set_initial_free_space

    cfg = load_config()
    h = ci.init_curobo(cfg)                       # 无 world → 全自由 VOXEL 世界 + 运动学
    retract = np.asarray(cfg.retract_config, float)

    fs = sorted(glob.glob(os.path.join(args.traj_dir, "*.npz")))
    if not fs:
        print(f"未找到轨迹: {args.traj_dir}/*.npz")
        return
    Ps = [np.asarray(np.load(f, allow_pickle=True)["positions"], float) for f in fs]
    print(f"retract: {np.round(retract, 4)}")
    print(f"n traj : {len(Ps)}  (来自 {args.traj_dir})")

    # 基线：全 UNKNOWN（应全 0 → 起步不了）
    vm0 = build_roi_voxmap(cfg)
    ri_base = np.array([compute_reach_pt(h, vm0, P) for P in Ps])
    print(f"\n全 UNKNOWN 基线 reach_idx: min/med/max = "
          f"{ri_base.min()}/{int(np.median(ri_base))}/{ri_base.max()}  (期望全 0)")

    print("\n dq(rad) | blobFREE体素 | reach_idx min/med/max | EE行进(m) med/max | reach>=1")
    last_cells_world = None
    for dq in args.dq:
        vm = build_roi_voxmap(cfg)
        n_free, cells = set_initial_free_space(h, vm, config=cfg, dq=dq, return_cells=True)
        ris, travels = [], []
        for P in Ps:
            ri = compute_reach_pt(h, vm, P)
            ris.append(ri)
            travels.append(float(np.linalg.norm(link6_pos(h, P[ri]) - link6_pos(h, P[0]))))
        ris = np.array(ris); travels = np.array(travels)
        print(f"  {dq:4.2f}  | {n_free:7d}     | "
              f"{ris.min():3d}/{int(np.median(ris)):3d}/{ris.max():3d}        | "
              f"{np.median(travels):.3f}/{travels.max():.3f}   | {(ris >= 1).mean()*100:.0f}%")
        if abs(dq - cfg.init_free_dq) < 1e-9:
            last_cells_world = vm.voxel_to_world(cells)

    # 默认 dq 下做工件安全性核查
    if args.check_workpiece:
        if last_cells_world is None:
            vm = build_roi_voxmap(cfg)
            _, cells = set_initial_free_space(h, vm, config=cfg, return_cells=True)
            last_cells_world = vm.voxel_to_world(cells)
        check_workpiece_clear(last_cells_world, fs, args.n_workpiece, cfg.voxel_size_m)

    print(f"\n默认 init_free.dq_rad = {cfg.init_free_dq}（configs/default.yaml）")
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
