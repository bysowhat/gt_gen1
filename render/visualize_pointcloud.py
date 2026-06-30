"""读取 render_seam.py 保存的某个 pose 目录，用 open3d 反投影成彩色 3D 点云并
可视化（左右目融合到世界系）。

依赖：numpy、opencv-python、open3d（不需要 isaaclab）。

用法：
    python render/visualize_pointcloud.py <pose_dir> [--eye both|left|right]
        [--max-depth 50] [--voxel 0.0] [--frame world|camera] [--save out.ply] [--no-vis]

例：
    python render/visualize_pointcloud.py \
        /home/a/Downloads/render_out2/<part>/<part>_seam_40_pose0

目录格式（render_seam.py save_pose 写出，每侧一个子目录）：
    <pose_dir>/left/   0_rgb.jpg  0_depth.exr  render_info.npy
    <pose_dir>/right/  0_rgb.jpg  0_depth.exr  render_info.npy
  每侧的 render_info.npy 都含两目完整信息（left/right 内参 + 各自世界位姿），
  故本脚本任取一侧的 render_info.npy 当 meta 用。

坐标与约定（与 render_info.npy 一致）：
  - depth 是 distance_to_image_plane（针孔 z 深度，单位 m），标准针孔反投影即可。
  - 内参 left_intrinsic/right_intrinsic 为 3x3。
  - 相机世界位姿 *_pose_w_pos / *_pose_w_quat_wxyz 为 ROS 约定（+Z 朝前、-Y 朝上、
    +X 朝右），恰为 OpenCV 光学系，故反投影点直接由该位姿变换到世界系。
  - meta 里的世界位姿是「底座立于 z≈0」的真实渲染坐标；若要换算到相对地板，
    把世界 z 再 + z_lift（本脚本默认按真实渲染坐标显示，不加）。
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import open3d as o3d


def quat_wxyz_to_R(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_rgb(path):
    """读 store_rgb 写的 BGR jpg -> (H,W,3) RGB uint8。"""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取 RGB: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path):
    """读 store_depth 写的 EXR -> (H,W) float32 z 深度（米）。"""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"无法读取深度: {path}")
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def backproject(rgb, depth, K, max_depth):
    """针孔反投影 -> (相机 ROS 光学系点 (M,3), 颜色 (M,3) float[0,1])。"""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = np.isfinite(z) & (z > 1e-6)
    if max_depth is not None and max_depth > 0:
        valid &= z < max_depth

    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    zc = z[valid].astype(np.float64)
    xc = (u - cx) / fx * zc
    yc = (v - cy) / fy * zc
    pts_cam = np.stack([xc, yc, zc], axis=1)          # ROS 光学系：x右 y下 z前

    cols = rgb[valid].astype(np.float64) / 255.0      # (M,3) RGB[0,1]
    return pts_cam, cols


def cam_to_world(pts_cam, pos_w, quat_w_wxyz):
    R = quat_wxyz_to_R(quat_w_wxyz)
    return pts_cam @ R.T + np.asarray(pos_w, dtype=np.float64)[None, :]


def build_eye_pcd(pose_dir, meta, eye, max_depth, frame):
    rgb = load_rgb(pose_dir / eye / "0_rgb.jpg")
    depth = load_depth(pose_dir / eye / "0_depth.exr")
    K = np.asarray(meta[f"{eye}_intrinsic"], dtype=np.float64)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"{eye}: rgb {rgb.shape[:2]} 与 depth {depth.shape} 尺寸不一致")

    pts_cam, cols = backproject(rgb, depth, K, max_depth)
    if frame == "world":
        pts = cam_to_world(pts_cam, meta[f"{eye}_pose_w_pos"], meta[f"{eye}_pose_w_quat_wxyz"])
    else:
        pts = pts_cam

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    print(f"  [{eye}] 有效点 {len(pts):,} (depth {np.nanmin(depth):.3f}~"
          f"{np.nanmax(depth[np.isfinite(depth)]):.3f}m)")
    return pcd


def main():
    ap = argparse.ArgumentParser(description="open3d 可视化左右目彩色点云")
    ap.add_argument("pose_dir", help="pose 目录（含 left/ right/ 子目录，每侧 0_rgb.jpg+0_depth.exr+render_info.npy）")
    ap.add_argument("--eye", choices=["both", "left", "right"], default="both")
    ap.add_argument("--max-depth", type=float, default=50.0,
                    help="丢弃 z 大于该值的点（米，0=不限）")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="体素下采样尺寸（米，0=不降采样）")
    ap.add_argument("--frame", choices=["world", "camera"], default="world",
                    help="world=融合到世界系（默认）；camera=各目自身光学系")
    ap.add_argument("--save", default=None, help="另存合并点云为 .ply")
    ap.add_argument("--no-vis", action="store_true", help="只保存不弹窗")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir)
    # meta 用任一侧的 render_info.npy（每侧都含两目完整内参与世界位姿）
    meta = None
    for side in ("left", "right"):
        info_p = pose_dir / side / "render_info.npy"
        if info_p.exists():
            meta = np.load(info_p, allow_pickle=True).item()
            break
    if meta is None:
        raise FileNotFoundError(f"{pose_dir} 下未找到 left/render_info.npy 或 right/render_info.npy")
    print(f"[meta] z_lift={meta.get('z_lift', 0.0):.3f}m  depth_type={meta.get('depth_type')}")

    eyes = ["left", "right"] if args.eye == "both" else [args.eye]
    if args.frame == "camera" and len(eyes) > 1:
        print("  [warn] frame=camera 下左右目各在自身光学系，不会对齐；建议 --frame world")

    pcds = [build_eye_pcd(pose_dir, meta, e, args.max_depth, args.frame) for e in eyes]
    merged = pcds[0]
    for p in pcds[1:]:
        merged += p

    if args.voxel and args.voxel > 0:
        before = len(merged.points)
        merged = merged.voxel_down_sample(args.voxel)
        print(f"  体素下采样 {args.voxel}m: {before:,} -> {len(merged.points):,}")

    if args.save:
        o3d.io.write_point_cloud(args.save, merged)
        print(f"  已保存点云 -> {args.save}")

    if not args.no_vis:
        geoms = [merged]
        # 世界系：加坐标轴 + 标出左右相机原点
        if args.frame == "world":
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
            for e, c in [("left", (1, 0, 0)), ("right", (0, 0, 1))]:
                if e in eyes:
                    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    s.translate(np.asarray(meta[f"{e}_pose_w_pos"], dtype=np.float64))
                    s.paint_uniform_color(c)
                    geoms.append(s)
        o3d.visualization.draw_geometries(geoms, window_name=f"pointcloud: {pose_dir.name}")


if __name__ == "__main__":
    main()
