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
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d
import torch

from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.mapping import STATE_FREE, STATE_OCCUPIED, VoxelMap


def look_at(cam_pos, target, up_hint=np.array([0.0, 0.0, 1.0])):
    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    if abs(np.dot(z_axis, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up_hint, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cam_pos
    return T


def make_camera_frustum(camera_pose, intrin, far=0.3, color=(0.1, 0.4, 1.0)):
    corners_uv = np.array([
        [0, 0], [intrin.width - 1, 0],
        [intrin.width - 1, intrin.height - 1], [0, intrin.height - 1],
    ])
    dirs_cam = np.stack([
        (corners_uv[:, 0] - intrin.cx) / intrin.fx,
        (corners_uv[:, 1] - intrin.cy) / intrin.fy,
        np.ones(4),
    ], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    far_corners = t + far * (dirs_cam @ R.T)
    points = np.vstack([t.reshape(1, 3), far_corners])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--height", type=int, default=180, help="depth 分辨率 H")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--n-views", type=int, default=8, help="围着 seam 渲染多少张")
    parser.add_argument("--cam-radius", type=float, default=0.5)
    parser.add_argument("--cam-z-offset", type=float, default=0.3)
    parser.add_argument("--voxel-res", type=float, default=0.02)
    parser.add_argument("--bounds-pad", type=float, default=0.6,
                        help="VoxelMap bounds 在工件 bbox 外向四周扩多少米")
    parser.add_argument("--show-free", action="store_true",
                        help="同时画 free 体素 (默认只画 occupied, 不然太密)")
    parser.add_argument("--free-subsample", type=int, default=20,
                        help="--show-free 时每 N 个 free 取一个")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)

    # ---- 加载 ----
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K, device=device)
    print(f"[load] mesh: {sim.num_triangles} triangles, image {K.width}x{K.height}")

    with open(args.pkl, "rb") as f:
        pkl = pickle.load(f)
    seam_line = np.asarray(pkl["seam_line"])
    seam_mid = seam_line[len(seam_line) // 2]
    print(f"[scene] seam_mid = {seam_mid.round(4)}")

    # ---- 算 VoxelMap bounds: 工件 bbox + 一点 padding ----
    mesh_bb = sim.mesh.bounds  # (2, 3)
    bounds = np.stack([
        mesh_bb[0] - args.bounds_pad,
        mesh_bb[1] + args.bounds_pad,
    ])
    print(f"[bounds] {bounds[0].round(2)} -> {bounds[1].round(2)} "
          f"({(bounds[1] - bounds[0]).round(2)} m)")

    vm = VoxelMap(bounds=bounds, resolution=args.voxel_res, device=device, max_range=2.0)
    print(f"[map] {vm.shape} = {np.prod(vm.shape)} voxels")

    # ---- 在 seam_mid 周围撒 N 个相机位姿, 每个朝 seam_mid ----
    poses = []
    for i in range(args.n_views):
        ang = 2 * np.pi * i / args.n_views
        cam_pos = seam_mid + np.array([
            args.cam_radius * np.cos(ang),
            args.cam_radius * np.sin(ang),
            args.cam_z_offset,
        ])
        poses.append(look_at(cam_pos, seam_mid))

    # ---- 跑 N 帧 + integrate ----
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

    if counts["occupied"] == 0:
        print("[WARN] 0 个 occupied 体素! 检查 bounds 是否包含工件, depth 是否正常.")

    # ---- 提取 occupied / free 体素中心 ----
    state_cpu = vm.state.cpu().numpy()
    occ_idx = np.argwhere(state_cpu == STATE_OCCUPIED)  # (M, 3)
    free_idx = np.argwhere(state_cpu == STATE_FREE)
    occ_pts = bounds[0] + (occ_idx + 0.5) * args.voxel_res
    free_pts = bounds[0] + (free_idx + 0.5) * args.voxel_res

    # ---- Open3D 几何 ----
    geoms = []

    # mesh 半透明灰
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)

    # occupied 红色点云
    if len(occ_pts) > 0:
        pcd_occ = o3d.geometry.PointCloud()
        pcd_occ.points = o3d.utility.Vector3dVector(occ_pts)
        pcd_occ.paint_uniform_color([1.0, 0.1, 0.1])
        geoms.append(pcd_occ)

    # free 绿色点云 (subsampled)
    if args.show_free and len(free_pts) > 0:
        free_sub = free_pts[::args.free_subsample]
        pcd_free = o3d.geometry.PointCloud()
        pcd_free.points = o3d.utility.Vector3dVector(free_sub)
        pcd_free.paint_uniform_color([0.2, 0.9, 0.2])
        geoms.append(pcd_free)
        print(f"[viz] free subsampled: {len(free_sub)}/{len(free_pts)}")

    # 焊缝点 + 中点
    for p in seam_line:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=8)
        s.translate(p); s.paint_uniform_color([1.0, 0.6, 0.6])
        s.compute_vertex_normals()
        geoms.append(s)
    seam_mid_sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.012, resolution=12)
    seam_mid_sph.translate(seam_mid); seam_mid_sph.paint_uniform_color([1.0, 0.5, 0.0])
    seam_mid_sph.compute_vertex_normals()
    geoms.append(seam_mid_sph)

    # 所有相机视锥
    for pose in poses:
        geoms.append(make_camera_frustum(pose, K, far=0.2))

    # 世界坐标轴 + bounds 外框
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=seam_mid)
    geoms.append(axes)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounds[0], max_bound=bounds[1])
    bbox.color = (0.5, 0.5, 0.5)
    geoms.append(bbox)

    print()
    print("Legend:")
    print("  灰 mesh      — BEAM 工件原始几何")
    print("  红色点云     — occupied voxel (应贴在 mesh 表面)")
    if args.show_free:
        print("  绿色点云(稀疏) — free voxel")
    print("  红小球       — 焊缝采样点")
    print("  橙球         — 焊缝中点 P_weld")
    print("  蓝视锥 ×N    — 相机机位")
    print("  灰色 bbox    — VoxelMap 边界")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="M1.2 VoxelMap 3D verification",
            width=1280, height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
