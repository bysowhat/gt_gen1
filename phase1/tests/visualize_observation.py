"""M1.3 视觉验证: 围着焊缝中点撒一堆候选 viewpoint, 按 4 条约束着色画出来.

颜色:
    绿色   ✓ 4 条都过 (valid)
    红色   ✗ distance 失败
    橙色   ✗ frustum 失败
    黄色   ✗ line_of_sight 失败
    紫色   ✗ (未使用)

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_observation.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"

输出: 同时打印两种模式的统计:
    strict      — M1.7 真实行为, 视线穿 unknown 算失败
    optimistic  — 假设地图完整, 仅检查算法本身; 用于可视化
窗口里画的是 optimistic 结果 (能看到 4 条约束都生效).

合格标准:
    - 离 P_weld 太近 / 太远的 viewpoint 应是红色 (distance fail)
    - 在 P_weld 背面 (被工件挡住) 的 viewpoint 应是黄色 (line_of_sight fail)
    - 中间一圈合规位置应是绿色
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d
import torch

from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.mapping import VoxelMap
from phase1.observation_check import Viewpoint, is_valid_observation
from phase1.tests._viz_helpers import (
    load_seam_data,
    look_at,
    make_seam_geoms,
)


COLORS = {
    None:           [0.20, 0.85, 0.20],
    "distance":     [0.95, 0.20, 0.20],
    "frustum":      [1.00, 0.55, 0.10],
    "line_of_sight":[0.95, 0.85, 0.20],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--n-candidates", type=int, default=200)
    parser.add_argument("--radius-min", type=float, default=0.15)
    parser.add_argument("--radius-max", type=float, default=0.85)
    parser.add_argument("--d-min", type=float, default=0.30)
    parser.add_argument("--d-max", type=float, default=0.60)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    K = CameraIntrinsics.from_fov(86.0, 57.0, 360, 640)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K, device=device)
    print(f"[load] {sim.num_triangles} triangles")

    seam = load_seam_data(args.pkl)
    print(f"[seam] {seam.label}")
    print(f"       p_weld = {seam.p_weld.round(4)}, tangent = {seam.tangent.round(3)}")

    bounds = np.stack([sim.mesh.bounds[0] - 0.6, sim.mesh.bounds[1] + 0.6])
    vm = VoxelMap(bounds=bounds, resolution=0.02, device=device, max_range=2.0)
    n_pre = 8
    for i in range(n_pre):
        ang = 2 * np.pi * i / n_pre
        cam_pos = seam.p_weld + np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang), 0.3])
        depth = sim.render(look_at(cam_pos, seam.p_weld))
        vm.integrate(depth, K, look_at(cam_pos, seam.p_weld))
    counts = vm.num_voxels_by_state()
    print(f"[map] occupied={counts['occupied']}, free={counts['free']}, "
          f"unknown={counts['unknown']}")

    # 候选采样: 上半球壳
    radii = rng.uniform(args.radius_min, args.radius_max, args.n_candidates)
    u = rng.uniform(0, 1, args.n_candidates)
    v = rng.uniform(0, 1, args.n_candidates)
    theta = np.arccos(1 - u)
    phi = 2 * np.pi * v
    offsets = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1) * radii[:, None]
    positions = seam.p_weld + offsets

    results_strict, results_opt = [], []
    t0 = time.time()
    for pos in positions:
        cam_dir = (seam.p_weld - pos)
        cam_dir /= np.linalg.norm(cam_dir)
        vp = Viewpoint(pos=pos, dir=cam_dir, up=np.array([0, 0, 1.0]))
        results_strict.append(is_valid_observation(
            vp, seam.p_weld, seam.tangent, vm, K,
            d_min=args.d_min, d_max=args.d_max, los_unknown_as_block=True,
        ))
        results_opt.append(is_valid_observation(
            vp, seam.p_weld, seam.tangent, vm, K,
            d_min=args.d_min, d_max=args.d_max, los_unknown_as_block=False,
        ))
    print(f"[eval] {args.n_candidates} candidates × 2 modes "
          f"in {(time.time()-t0)*1000:.0f} ms")

    def _stats(label, rs):
        n_valid = sum(1 for r in rs if r.valid)
        reasons = Counter([r.fail_reason for r in rs])
        print(f"\n[{label}]  valid: {n_valid}/{args.n_candidates}")
        for reason, n in reasons.most_common():
            tag = "✓ valid" if reason is None else f"✗ {reason}"
            print(f"  {tag:<18s} {n}")

    _stats("strict (M1.7 真实行为)", results_strict)
    _stats("optimistic (验证 4 条约束本身)", results_opt)

    results = results_opt

    # ---- Open3D ----
    geoms = []
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)

    by_reason: dict = {}
    for pos, r in zip(positions, results):
        by_reason.setdefault(r.fail_reason, []).append(pos)
    for reason, pts in by_reason.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack(pts))
        pcd.paint_uniform_color(COLORS[reason])
        geoms.append(pcd)

    geoms.extend(make_seam_geoms(seam))

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=seam.p_weld))

    print()
    print("Legend:")
    print("  灰 mesh        — 工件")
    print("  品红粗线        — 当前 evaluating 的焊缝 (整条)")
    print("  亮绿大球        — 焊缝起点")
    print("  亮红大球        — 焊缝终点")
    print("  橙球           — 焊缝中点 P_weld")
    print("  蓝色短线段     — 焊缝切线方向")
    print("  绿色点         — ✓ 4 条都过 (valid)")
    print("  红色点         — ✗ distance 失败")
    print("  橙色点         — ✗ frustum 失败")
    print("  黄色点         — ✗ line_of_sight 失败")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"M1.3  ObservationCheck  |  {seam.label}",
            width=1280, height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
