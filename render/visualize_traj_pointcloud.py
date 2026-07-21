"""读取 render_trajectory.py 保存的【一条轨迹目录】，把该轨迹【多帧】的左右目
RGB+深度反投影、并融合到【同一个坐标系】下的彩色 3D 点云，用 open3d 可视化。

依赖：numpy、opencv-python、open3d（不需要 isaaclab）。

与 visualize_pointcloud.py 的区别：那个是 render_seam 的【单帧单 pose】目录（render_info
里含左右目两套内参+世界位姿）；本脚本针对 render_trajectory 的【多帧轨迹】目录——每侧
render_info.npy 是「按帧堆叠」的（cam_intrinsic (F,3,3)、cam_pos_list/cam_quat_list (F,·)、frame_indices (F,)），
RGB/深度按帧号平铺成 {k}_rgb.jpg / {k}_depth.exr。

用法：
    python render/visualize_traj_pointcloud.py <traj_dir> [--eye both|left|right]
        [--max-depth 5] [--voxel 0.0] [--stride 1]
        [--frames 0 3 7] [--show-cams] [--save out.ply] [--no-vis] [--seg]

例：
    python render/visualize_traj_pointcloud.py \
        /tmp/render_traj/<part_stem>/seam0_forehand0_traj0

--seg：不用真实 RGB，而是把每帧的【实例分割 id 图 {k}_seg.png】按物体上色成彩色点云，
    用来肉眼核对分割 label 是否正确。上色【按 label 串】（workpiece/obstacle_0/robot/背景…）
    而非原始整数 id——因多帧分摊在不同 env、整数 id 逐 env 可能不同，只有 label 跨帧稳定；
    障碍本就是逐个不同标签(obstacle_0/1/…)，故按 label 上色既跨帧一致又能区分实例。
    命令行会打印「颜色 → 类别名」对照表。

轨迹目录格式（render_trajectory.py render_job 写出）：
    <traj_dir>/left/   {k}_rgb.jpg  {k}_depth.exr ...  render_info.npy   # render_info 含全帧信息
    <traj_dir>/right/  {k}_rgb.jpg  {k}_depth.exr ...  render_info.npy
  （k = 关键帧行号；--observe-only 渲染时帧号可能不连续，以 render_info.frame_indices 为准）

坐标与约定（与 render_info.npy 一致）：
  - depth 是 distance_to_image_plane（针孔 z 深度，单位 m），标准针孔反投影即可。
  - 点云坐标系【固定】为 arm(base) 系、USD 光学约定，不提供坐标系参数。相机位姿直接取
    render_info 的 cam_pos_list/cam_quat_list（本就是 base 系 + USD 光学，即已消 env 偏移）。
  - 反投影得到的是 ROS 光学系点（x右 y下 z前），乘 diag([1,-1,-1]) 翻到 USD 光学系
    （x右 y上 z后）后，再用 cam_pos_list/cam_quat_list 变到 base 系。
  * 一条轨迹的多帧分摊在【多个并行 env（网格间距数十米）】里渲，世界系相机位姿带各自 env
    偏移；改用 base 系的 cam_pos_list/cam_quat_list 后偏移已被消掉，多帧天然对齐
    （工件在 base 系固定）。故无需再区分 base/world。
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import open3d as o3d

# ROS 光学系(x右 y下 z前) -> USD 光学系(x右 y上 z后) 的轴翻转（与 render_info
# 的 cam_pos_list/cam_quat_list 约定对齐：diag([1,-1,-1])）。
_ROS_TO_USD_OPTICAL = np.array([1.0, -1.0, -1.0], dtype=np.float64)

# --seg 上色用的高区分度调色板（RGB 0-255）。前景物体(工件/各障碍/机械臂)从这里按
# label 排序依次取色；背景/未标注单独给一个暗灰，避免抢占鲜艳色。
_SEG_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40),
    (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128),
    (255, 225, 25), (0, 0, 0), (255, 215, 180), (128, 128, 128),
]
_BG_COLOR = (30, 30, 30)          # 背景/未标注固定暗灰
_SEG_DEFAULT = (10, 10, 10)       # id 在映射里找不到 label 时的兜底色


def _is_bg_label(label):
    """判断是否背景/未标注类。"""
    return str(label).upper() in ("BACKGROUND", "UNLABELLED", "UNLABELED")


def build_label_color_map(labels):
    """把一组 label 串映射到固定 RGB 颜色（跨帧、跨左右目一致）。

    背景/未标注给暗灰；其余前景 label 按字典序稳定地从调色板取色，
    保证同一 label 无论哪一帧都得到同一颜色。
    """
    fg = sorted(l for l in labels if not _is_bg_label(l))
    cmap = {l: _BG_COLOR for l in labels if _is_bg_label(l)}
    for i, lab in enumerate(fg):
        cmap[lab] = _SEG_PALETTE[i % len(_SEG_PALETTE)]
    return cmap


def _normalize_id2label(raw):
    """把 render_info 里某帧的 seg_id_to_label（可能 int/str 键）规整成 {int: label}。"""
    out = {}
    for k, v in dict(raw).items():
        try:
            out[int(k)] = str(v)
        except (ValueError, TypeError):
            continue
    return out


def load_seg(path):
    """读 store_seg 写的 16-bit 单通道 PNG -> (H,W) int64 实例 id 图。"""
    s = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if s is None:
        raise FileNotFoundError(f"无法读取分割图: {path}")
    if s.ndim == 3:
        s = s[..., 0]
    return s.astype(np.int64)


def seg_to_color(seg, id2label, label2color):
    """把整数 id 图 (H,W) 按 id->label->颜色 上色成 (H,W,3) uint8。

    每个 id 先查本帧 id2label 得 label，再查全局 label2color 得颜色；
    这样同一 label 跨帧同色，不同实例(不同 label)不同色。
    """
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for sid in np.unique(seg):
        lab = id2label.get(int(sid))
        col = label2color.get(lab, _SEG_DEFAULT) if lab is not None else _SEG_DEFAULT
        out[seg == sid] = col
    return out


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


def pose_to_T(pos, quat_wxyz):
    """pos(3,)+quat(w,x,y,z) -> 4x4 齐次变换。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


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


def build_side_pcd(side_dir, meta, max_depth, sel_indices, seg=False, label2color=None):
    """把某侧 render_info 里选中的多帧全部反投影 + 变换到 arm(base) 系，合并成一个点云。

    相机位姿【固定】取 render_info 的 cam_pos_list/cam_quat_list（已是 base 系 + USD 光学约定），
    反投影点由 ROS 光学翻到 USD 光学（乘 diag([1,-1,-1])）后再用该位姿变到 base 系；
    多帧因此自动对齐（消掉各 env 偏移，工件在 base 系固定）。
    返回 (合并点云, [各选中帧相机在 base 系的原点 (3,)])。

    seg=True 时点颜色来自 {k}_seg.png 按 label2color 上色（核对分割用）；否则用真实 RGB。
    """
    frame_ids = np.asarray(meta["frame_indices"]).reshape(-1)      # (F,) 文件名用的帧号 k
    Ks = np.asarray(meta["cam_intrinsic"], dtype=np.float64)       # (F,3,3)
    cam_pos = np.asarray(meta["cam_pos_list"], dtype=np.float64)   # (F,3) base 系 USD 光学
    cam_quat = np.asarray(meta["cam_quat_list"], dtype=np.float64)  # (F,4)
    F = len(frame_ids)
    seg_list = meta.get("seg_id_to_label_list") if seg else None
    if seg and seg_list is None:
        raise KeyError(f"{side_dir.name}/render_info.npy 无 seg_id_to_label_list 字段，"
                       f"该轨迹渲染时未产出分割标签，无法用 --seg")

    pcd = o3d.geometry.PointCloud()
    cam_origins = []
    n_pts = 0
    for i in sel_indices:
        if i < 0 or i >= F:
            continue
        k = int(frame_ids[i])
        depth = load_depth(side_dir / f"{k}_depth.exr")
        if seg:
            seg_img = load_seg(side_dir / f"{k}_seg.png")
            id2label = _normalize_id2label(seg_list[i])
            color_img = seg_to_color(seg_img, id2label, label2color)   # (H,W,3) uint8
        else:
            color_img = load_rgb(side_dir / f"{k}_rgb.jpg")            # (H,W,3) uint8 RGB
        if color_img.shape[:2] != depth.shape:
            raise ValueError(f"{side_dir.name} 帧{k}: 图像 {color_img.shape[:2]} 与 depth {depth.shape} 尺寸不一致")

        pts_ros, cols = backproject(color_img, depth, Ks[i], max_depth)  # ROS 光学系点
        pts_usd = pts_ros * _ROS_TO_USD_OPTICAL                    # -> USD 光学系
        T = pose_to_T(cam_pos[i], cam_quat[i])                     # cam(USD 光学) -> base
        pts = pts_usd @ T[:3, :3].T + T[:3, 3][None, :]

        fp = o3d.geometry.PointCloud()
        fp.points = o3d.utility.Vector3dVector(pts)
        fp.colors = o3d.utility.Vector3dVector(cols)
        pcd += fp
        cam_origins.append(T[:3, 3].copy())
        n_pts += len(pts)
        print(f"    帧 k={k:<3d} 有效点 {len(pts):>8,}")

    print(f"  [{meta.get('side', side_dir.name)}] {len(cam_origins)} 帧，合计 {n_pts:,} 点")
    return pcd, cam_origins


def main():
    ap = argparse.ArgumentParser(description="open3d 融合一条轨迹的多帧左右目彩色点云到同一坐标系")
    ap.add_argument("traj_dir", help="轨迹目录（含 left/ right/ 子目录，每侧多帧 {k}_rgb.jpg/{k}_depth.exr+render_info.npy）")
    ap.add_argument("--eye", choices=["both", "left", "right"], default="both")
    ap.add_argument("--max-depth", type=float, default=5.0,
                    help="丢弃 z 大于该值的点（米，0=不限；轨迹里含仓库背景，默认 5m 只留工件附近）")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="体素下采样尺寸（米，0=不降采样；多帧点多时建议 0.003~0.01）")
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔几帧取一帧（默认 1=全取；帧太多时调大）")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="只用这些【堆叠下标 i】（0..F-1，非帧号 k）；给了则忽略 --stride")
    ap.add_argument("--show-cams", action="store_true",
                    help="在目标系里用小球标出各帧相机原点（左红右蓝）")
    ap.add_argument("--seg", action="store_true",
                    help="用实例分割 id 图 {k}_seg.png 按物体上色（核对分割 label），而非真实 RGB")
    ap.add_argument("--save", default=None, help="另存合并点云为 .ply")
    ap.add_argument("--no-vis", action="store_true", help="只保存不弹窗")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    eyes = ["left", "right"] if args.eye == "both" else [args.eye]

    # 第一遍：加载各侧 meta 并确定选帧（--seg 时同时汇总所有帧的 label，构建跨帧一致的颜色表）
    sides = []                # [(eye, side_dir, meta, sel)]
    all_labels = set()
    for eye in eyes:
        side_dir = traj_dir / eye
        info_p = side_dir / "render_info.npy"
        if not info_p.exists():
            raise FileNotFoundError(f"未找到 {info_p}")
        meta = np.load(info_p, allow_pickle=True).item()
        F = int(meta.get("n_frames", len(np.asarray(meta["frame_indices"]).reshape(-1))))
        if args.frames is not None:
            sel = list(args.frames)
        else:
            sel = list(range(0, F, max(1, args.stride)))
        print(f"[{eye}] 共 {F} 帧，选取 {len(sel)} 帧（融合到 base 系 / USD 光学）")
        sides.append((eye, side_dir, meta, sel))

        if args.seg:
            seg_list = meta.get("seg_id_to_label_list")
            if seg_list is None:
                raise KeyError(f"{info_p} 无 seg_id_to_label_list 字段，该轨迹未产出分割标签，无法用 --seg")
            for i in sel:
                if 0 <= i < len(seg_list):
                    all_labels.update(_normalize_id2label(seg_list[i]).values())

    label2color = None
    if args.seg:
        if not all_labels:
            raise RuntimeError(
                "开了 --seg 但选中帧的 seg_id_to_label 全为空 —— 该轨迹的分割映射没存进去"
                "（旧数据命中 _seg_labels_from_info 的 env_idx 索引 bug，映射恒空，上色会全黑）。"
                "请用修复后的 render_trajectory.py 重新渲染该轨迹再来 --seg。")
        label2color = build_label_color_map(all_labels)
        print("分割上色对照（颜色 RGB → 类别，跨帧一致）：")
        for lab, col in sorted(label2color.items(), key=lambda kv: (_is_bg_label(kv[0]), kv[0])):
            tag = "（背景）" if _is_bg_label(lab) else ""
            rgb_str = str(tuple(int(c) for c in col))
            print(f"  RGB{rgb_str:<18}  <-  {lab}{tag}")

    merged = o3d.geometry.PointCloud()
    cam_markers = []   # (origin(3,), color)
    for eye, side_dir, meta, sel in sides:
        pcd, origins = build_side_pcd(side_dir, meta, args.max_depth, sel,
                                      seg=args.seg, label2color=label2color)
        merged += pcd
        if args.show_cams:
            c = (1, 0, 0) if eye == "left" else (0, 0, 1)
            cam_markers.extend((o, c) for o in origins)

    print(f"[merged] 合计 {len(merged.points):,} 点")
    if args.voxel and args.voxel > 0:
        before = len(merged.points)
        merged = merged.voxel_down_sample(args.voxel)
        print(f"  体素下采样 {args.voxel}m: {before:,} -> {len(merged.points):,}")

    if args.save:
        o3d.io.write_point_cloud(args.save, merged)
        print(f"  已保存点云 -> {args.save}")

    if not args.no_vis:
        geoms = [merged, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
        for origin, color in cam_markers:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            s.translate(np.asarray(origin, dtype=np.float64))
            s.paint_uniform_color(color)
            geoms.append(s)
        o3d.visualization.draw_geometries(geoms, window_name=f"traj pointcloud: {traj_dir.name}")


if __name__ == "__main__":
    main()
