"""M1.1 3D 反投验证: 在 Open3D 里同时画 mesh + depth 反投点云 + 相机视锥 + 焊缝.

如果 RaycastSim 的几何/坐标全对, depth 反投出来的点云 (绿色) 会完美贴在 mesh (灰) 表面.
任何 R 矩阵转置反 / FOV 算反 / 像素 (u,v) 顺序倒 / OpenCV vs OpenGL 系混淆 都会让点云明显错位.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_depth_3d.py \
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

from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.tests._viz_helpers import (
    load_seam_data,
    look_at,
    make_camera_frustum,
    make_seam_geoms,
    make_line,
    make_sphere,
)


def reproject_depth_to_world(
    depth: np.ndarray, intrin: CameraIntrinsics, camera_pose: np.ndarray
) -> np.ndarray:
    H, W = depth.shape
    u = np.arange(W); v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x_c = (uu - intrin.cx) / intrin.fx
    y_c = (vv - intrin.cy) / intrin.fy
    z_c = np.ones_like(x_c)
    dirs_cam = np.stack([x_c, y_c, z_c], -1).astype(np.float64)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    R = camera_pose[:3, :3]; t = camera_pose[:3, 3]
    dirs_world = dirs_cam @ R.T
    points = t + depth[..., None] * dirs_world
    return points[depth > 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--cam-offset", type=float, nargs=3, default=[0.3, 0.3, 0.5])
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K)
    print(f"[load] {sim.num_triangles} triangles, image {K.width}x{K.height}")

    seam = load_seam_data(args.pkl)
    print(f"[seam] {seam.label}")
    print(f"       p_weld = {seam.p_weld.round(4)}")

    cam_pos = seam.p_weld + np.asarray(args.cam_offset)
    cam_pose = look_at(cam_pos, seam.p_weld)
    print(f"[camera] pos = {cam_pos.round(4)} → p_weld")

    sim.render(cam_pose)  # warmup
    t0 = time.time()
    depth = sim.render(cam_pose).cpu().numpy()
    print(f"[render] {(time.time() - t0) * 1000:.1f} ms, hits = {(depth > 0).sum()}/{depth.size}")

    cloud_pts = reproject_depth_to_world(depth, K, cam_pose)
    print(f"[reproject] {len(cloud_pts)} 3D points")

    # ---- Open3D ----
    geoms = []

    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_pts)
    pcd.paint_uniform_color([0.2, 0.9, 0.2])
    geoms.append(pcd)

    geoms.extend(make_seam_geoms(seam))

    geoms.append(make_camera_frustum(cam_pose, K, far=0.4))
    geoms.append(make_sphere(cam_pos, 0.015, color=(0.1, 0.4, 1.0)))
    geoms.append(make_line(cam_pos, seam.p_weld, color=(1.0, 0.9, 0.0)))

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=seam.p_weld))

    print()
    print("Legend:")
    print("  灰 mesh        — 工件")
    print("  品红粗线        — 当前 evaluating 的焊缝 (整条)")
    print("  亮绿大球        — 焊缝起点 (idx=0)")
    print("  亮红大球        — 焊缝终点 (idx=N-1)")
    print("  橙球           — 焊缝中点 P_weld")
    print("  粉色小球        — 焊缝其他采样点")
    print("  蓝色短线段     — 焊缝切线方向 (在 P_weld 处)")
    print("  绿色点云       — depth 反投回 world (应贴在 mesh 表面)")
    print("  蓝球+视锥      — 相机位置 + 视场角")
    print("  黄线           — 相机到 P_weld 视线")
    print("  RGB 轴         — XYZ (R=X G=Y B=Z)")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"M1.1  RaycastSim  |  {seam.label}",
            width=1280, height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
