"""M1.2 视觉验证: 用 M1.1 RaycastSim 从多视角渲染 BEAM, 喂进 VoxelMap, 在 Open3D 里看体素状态.

核心检查:
    "occupied 体素 (红) 应聚集在 BEAM mesh 的表面附近"
    "free 体素 (绿, 稀疏) 应在相机和工件之间"

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_voxel.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d
import torch

from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.mapping import STATE_FREE, STATE_OCCUPIED, VoxelMap
from phase1.tests._viz_helpers import (
    load_seam_data,
    look_at,
    make_camera_frustum,
    make_seam_geoms,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--cam-radius", type=float, default=0.5)
    parser.add_argument("--cam-z-offset", type=float, default=0.3)
    parser.add_argument("--voxel-res", type=float, default=0.02)
    parser.add_argument("--bounds-pad", type=float, default=0.6)
    parser.add_argument("--show-free", action="store_true")
    parser.add_argument("--free-subsample", type=int, default=20)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K, device=device)
    print(f"[load] mesh: {sim.num_triangles} triangles, image {K.width}x{K.height}")

    seam = load_seam_data(args.pkl)
    print(f"[seam] {seam.label}")

    mesh_bb = sim.mesh.bounds
    bounds = np.stack([mesh_bb[0] - args.bounds_pad, mesh_bb[1] + args.bounds_pad])
    print(f"[bounds] {bounds[0].round(2)} → {bounds[1].round(2)}")
    vm = VoxelMap(bounds=bounds, resolution=args.voxel_res, device=device, max_range=2.0)
    print(f"[map] {vm.shape} = {np.prod(vm.shape)} voxels")

    poses = []
    for i in range(args.n_views):
        ang = 2 * np.pi * i / args.n_views
        cam_pos = seam.p_weld + np.array([
            args.cam_radius * np.cos(ang),
            args.cam_radius * np.sin(ang),
            args.cam_z_offset,
        ])
        poses.append(look_at(cam_pos, seam.p_weld))

    t0 = time.time()
    for i, pose in enumerate(poses):
        depth = sim.render(pose)
        vm.integrate(depth, K, pose)
        if i % max(1, args.n_views // 4) == 0:
            print(f"  view {i+1}/{args.n_views}  done")
    elapsed = time.time() - t0
    print(f"[total] {args.n_views} frames in {elapsed:.2f} s "
          f"({elapsed / args.n_views * 1000:.0f} ms/frame avg)")

    counts = vm.num_voxels_by_state()
    total = sum(counts.values())
    print(f"[state] unknown={counts['unknown']:>8d} ({counts['unknown']/total:5.1%})  "
          f"free={counts['free']:>8d} ({counts['free']/total:5.1%})  "
          f"occupied={counts['occupied']:>6d} ({counts['occupied']/total:5.1%})")

    # ---- Open3D ----
    state_cpu = vm.state.cpu().numpy()
    occ_idx = np.argwhere(state_cpu == STATE_OCCUPIED)
    free_idx = np.argwhere(state_cpu == STATE_FREE)
    occ_pts = bounds[0] + (occ_idx + 0.5) * args.voxel_res
    free_pts = bounds[0] + (free_idx + 0.5) * args.voxel_res

    geoms = []
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)

    if len(occ_pts) > 0:
        pcd_occ = o3d.geometry.PointCloud()
        pcd_occ.points = o3d.utility.Vector3dVector(occ_pts)
        pcd_occ.paint_uniform_color([1.0, 0.1, 0.1])
        geoms.append(pcd_occ)

    if args.show_free and len(free_pts) > 0:
        free_sub = free_pts[::args.free_subsample]
        pcd_free = o3d.geometry.PointCloud()
        pcd_free.points = o3d.utility.Vector3dVector(free_sub)
        pcd_free.paint_uniform_color([0.2, 0.9, 0.2])
        geoms.append(pcd_free)
        print(f"[viz] free subsampled: {len(free_sub)}/{len(free_pts)}")

    geoms.extend(make_seam_geoms(seam))
    for pose in poses:
        geoms.append(make_camera_frustum(pose, K, far=0.2))

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=seam.p_weld))
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounds[0], max_bound=bounds[1])
    bbox.color = (0.5, 0.5, 0.5)
    geoms.append(bbox)

    print()
    print("Legend:")
    print("  灰 mesh        — BEAM 工件")
    print("  品红粗线        — 当前 evaluating 的焊缝")
    print("  亮绿大球        — 焊缝起点")
    print("  亮红大球        — 焊缝终点")
    print("  橙球           — 焊缝中点 P_weld")
    print("  红色点云       — occupied voxel (应贴在 mesh 表面)")
    if args.show_free:
        print("  绿色点云稀疏  — free voxel")
    print("  蓝视锥 ×N      — 相机机位")
    print("  灰色 bbox      — VoxelMap 边界")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"M1.2  VoxelMap  |  {seam.label}",
            width=1280, height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
