"""M1.1 3D 反投验证: 在 Open3D 里同时画 mesh + depth 反投点云 + 相机视锥 + 焊缝.

如果 RaycastSim 的几何/坐标全对, depth 反投出来的点云 (绿色) 会完美贴在 mesh (灰) 表面.
任何 R 矩阵转置反 / FOV 算反 / 像素 (u,v) 顺序倒 / OpenCV vs OpenGL 系混淆 都会让点云明显错位.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_depth_3d.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"

操作:
    左键拖拽: 旋转视角
    Ctrl+左键拖拽: 平移
    滚轮: 缩放
    R: 重置视角
    Q / ESC: 退出
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

# 让 import phase1.xxx 能跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d

from phase1.depth_source import CameraIntrinsics, RaycastSim


def look_at(cam_pos: np.ndarray, target: np.ndarray, up_hint=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """OpenCV 系: x 右 y 下 z 前."""
    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    if abs(np.dot(z_axis, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(z_axis[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return T


def reproject_depth_to_world(
    depth: np.ndarray, intrin: CameraIntrinsics, camera_pose: np.ndarray
) -> np.ndarray:
    """把 depth (H, W) 反投回 world 系 3D 点云.

    我们的 depth 是沿光线的 *Euclidean 距离* (因为射线方向 normalize 过).
    所以 P_world = origin + dist * dir_world.
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x_c = (uu - intrin.cx) / intrin.fx
    y_c = (vv - intrin.cy) / intrin.fy
    z_c = np.ones_like(x_c)
    dirs_cam = np.stack([x_c, y_c, z_c], axis=-1).astype(np.float64)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    dirs_world = dirs_cam @ R.T   # (H, W, 3)

    points = t + depth[..., None] * dirs_world  # (H, W, 3)
    valid = depth > 0
    return points[valid]


def make_camera_frustum(
    camera_pose: np.ndarray, intrin: CameraIntrinsics, far: float = 0.5
) -> o3d.geometry.LineSet:
    """画相机视锥 (4 棱锥 + 4 边)."""
    # 4 个图像角的方向 (camera frame)
    corners_uv = np.array([
        [0, 0],
        [intrin.width - 1, 0],
        [intrin.width - 1, intrin.height - 1],
        [0, intrin.height - 1],
    ])
    dirs_cam = np.stack([
        (corners_uv[:, 0] - intrin.cx) / intrin.fx,
        (corners_uv[:, 1] - intrin.cy) / intrin.fy,
        np.ones(4),
    ], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    far_corners_world = t + far * (dirs_cam @ R.T)

    points = np.vstack([t.reshape(1, 3), far_corners_world])  # 5 个点
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],            # 顶到 4 角
        [1, 2], [2, 3], [3, 4], [4, 1],            # 远端 4 边
    ]
    colors = [[0.1, 0.4, 1.0]] * len(lines)        # 蓝色

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_sphere(center: np.ndarray, radius: float, color=(1.0, 0.2, 0.2)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def make_line(p_a: np.ndarray, p_b: np.ndarray, color=(1.0, 0.9, 0.0)) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([p_a, p_b]))
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--height", type=int, default=120, help="降低分辨率, 点云数 < 15K 才不会卡")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--cam-offset", type=float, nargs=3, default=[0.3, 0.3, 0.5],
                        help="相机相对 seam_mid 的偏移 (米)")
    parser.add_argument("--no-window", action="store_true", help="无窗口 (CI 测试用)")
    args = parser.parse_args()

    # ---- 加载 ----
    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K)
    print(f"[load] {sim.num_triangles} triangles, image {K.width}x{K.height}")

    with open(args.pkl, "rb") as f:
        pkl = pickle.load(f)
    seam_line = np.asarray(pkl["seam_line"])               # (20, 3)
    seam_mid = seam_line[len(seam_line) // 2]
    print(f"[scene] seam mid (world) = {seam_mid.round(4)}")

    cam_pos = seam_mid + np.asarray(args.cam_offset)
    cam_pose = look_at(cam_pos, seam_mid)
    print(f"[camera] pos = {cam_pos.round(4)}, target = {seam_mid.round(4)}")

    # ---- 渲染 ----
    sim.render(cam_pose)  # warmup
    t0 = time.time()
    depth = sim.render(cam_pose).cpu().numpy()
    print(f"[render] {(time.time() - t0) * 1000:.1f} ms, "
          f"hits = {(depth > 0).sum()}/{depth.size}")

    # ---- 反投点云 ----
    cloud_pts = reproject_depth_to_world(depth, K, cam_pose)
    print(f"[reproject] {len(cloud_pts)} 3D points")

    # ---- 构造 Open3D 几何 ----
    # 1. mesh (灰色, 半透明)
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # 2. 反投点云 (绿色)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_pts)
    pcd.paint_uniform_color([0.2, 0.9, 0.2])

    # 3. 焊缝点 (红色)
    seam_spheres = [make_sphere(p, 0.005, color=(1.0, 0.2, 0.2)) for p in seam_line]

    # 4. 焊缝中点 (橙色, 更大)
    p_weld_sphere = make_sphere(seam_mid, 0.012, color=(1.0, 0.5, 0.0))

    # 5. 相机视锥 (蓝)
    frustum = make_camera_frustum(cam_pose, K, far=0.4)
    cam_sphere = make_sphere(cam_pos, 0.015, color=(0.1, 0.4, 1.0))

    # 6. 视线 (黄)
    los = make_line(cam_pos, seam_mid, color=(1.0, 0.9, 0.0))

    # 7. 世界坐标轴 (放在工件附近)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=seam_mid - np.array([0, 0, 0]),
    )

    geoms = [mesh, pcd, p_weld_sphere, cam_sphere, frustum, los, axes] + seam_spheres

    print()
    print("Legend:")
    print("  灰色      — 工件 mesh (BEAM)")
    print("  绿色点云  — depth 反投回 world (应贴在 mesh 表面)")
    print("  红球      — 焊缝采样点 (20 个)")
    print("  橙球      — 焊缝中点 P_weld")
    print("  蓝球+视锥 — 相机位置 + 视场角")
    print("  黄线      — 相机到 P_weld 视线")
    print("  RGB 轴   — XYZ (R=X, G=Y, B=Z)")
    print()
    print("操作: 左键拖拽旋转, 滚轮缩放, Q/ESC 退出")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="M1.1 RaycastSim 3D verification",
            width=1280,
            height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
