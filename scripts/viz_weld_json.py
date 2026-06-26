#!/usr/bin/env python3
"""可视化工件网格 + 焊缝（不含机械臂，纯几何）。

参考 scripts/viz_kejian2_npy.py 的 open3d 可视化方式，但本脚本不读 .npy、不画机械臂，
只读：
  --obj       工件 mesh（_part.obj / _watertight.obj）——顶点即 world/mesh 系；
  --weld-json 焊缝 json（corrected_p0/p1 + bisector + boundary_dirs，见 load_welds）；
  --seam-id   可选：只画这一条焊缝（缺省=画全部焊缝）。

工件停在自身 mesh 坐标（不施加任何 base 变换），焊缝端点/中点/bisector 与工件同框直接叠加：
  · 浅灰工件网格；
  · 每条焊缝 p0→p1 红色圆柱；
  · 原点坐标架（size 自适应工件尺度）。

相机视角沿 bisector 方向望向焊缝（相机在 +bisector 一侧、距焊缝 0.5 米，看的方向 = -bisector）：
单焊缝用该缝的 bisector/中点；全焊缝用各中点质心 + 平均 bisector。

用法：
  python scripts/viz_weld_json.py \
    --obj /…/BEAM_…_part_watertight.obj \
    --weld-json /…/BEAM_…_weld_angle3.json \
    [--seam-id 1]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 复用 scripts/plan_init_pose_kejian2.py 的 load_welds（不改它）
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import plan_init_pose_kejian2 as k2  # noqa: E402


def _seam_geoms(weld: dict, scale: float):
    """单条焊缝的 open3d 几何（mesh/world 系直接画）：红色圆柱 p0→p1。"""
    import open3d as o3d

    p0 = np.asarray(weld["p0_world"], dtype=np.float64)
    p1 = np.asarray(weld["p1_world"], dtype=np.float64)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    print(f"[viz] seam {int(weld['idx']):>4}  焊缝长度 {L * 100:.2f} cm（{L:.4f} m）")
    if L < 1e-6:
        return []

    radius = 0.01                                 # 固定 2cm 直径（半径 1cm），不随工件尺度变化
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L)
    cyl.rotate(k2._align_rotmat([0.0, 0.0, 1.0], d / L), center=(0.0, 0.0, 0.0))  # +Z → 焊缝方向
    cyl.translate((0.5 * (p0 + p1)).tolist())     # 圆柱中心 = 焊缝中点
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color([0.9, 0.1, 0.1])
    return [cyl]


def _camera_extrinsic(lookat, bis, dist: float) -> np.ndarray:
    """相机外参（world→camera 的 4×4）：相机置于 lookat + dist·bisector，朝 -bisector 望向 lookat。
    open3d 相机系约定 x 右 / y 下 / z 前（视线）。"""
    lookat = np.asarray(lookat, dtype=np.float64)
    bis = np.asarray(bis, dtype=np.float64)
    bis = bis / (np.linalg.norm(bis) + 1e-12)
    C = lookat + dist * bis                       # 相机在 +bisector 一侧、距焊缝 dist 米
    z = lookat - C
    z = z / (np.linalg.norm(z) + 1e-12)           # 视线（前）= -bisector
    w = np.array([0.0, 0.0, 1.0])                 # 世界「上」提示
    if abs(float(np.dot(w, z))) > 0.95:           # 视线近竖直 → 换提示避免退化
        w = np.array([0.0, 1.0, 0.0])
    y = -(w - np.dot(w, z) * z)                   # 相机「下」轴 = 世界 -up 在 ⟂z 平面的投影
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)                            # 右 = down × forward（右手系）
    R_cw = np.column_stack([x, y, z])             # camera→world 旋转（列=相机轴在世界系）
    T_cw = np.eye(4); T_cw[:3, :3] = R_cw; T_cw[:3, 3] = C
    return np.linalg.inv(T_cw)                    # world→camera


def visualize(obj_fp: str, weld_json: str, seam_id):
    import open3d as o3d

    welds = k2.load_welds(weld_json)
    if seam_id is not None:
        welds = [w for w in welds if int(w["idx"]) == int(seam_id)]
        if not welds:
            print(f"[viz] weld_json 中找不到 seam-id={seam_id}")
            return

    mesh = o3d.io.read_triangle_mesh(obj_fp)
    if not mesh.has_vertices():
        print(f"[viz] open3d 读不到工件网格: {obj_fp}")
        return
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.72])

    # 坐标架/圆柱半径随工件包围盒尺度自适应（焊缝坐标与工件同框，绝对值可能很大）
    ext = mesh.get_axis_aligned_bounding_box().get_extent()
    diag = float(np.linalg.norm(ext))
    scale = max(diag, 1.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2 * scale)

    geoms = [mesh, frame]
    mids = []
    biss = []
    for w in welds:
        L = k2._seam_len_m(w)
        print(f"[viz] seam {int(w['idx']):>4}  长度 {L * 100:.2f} cm  "
              f"mid={np.asarray(w['mid_world']).round(3).tolist()}")
        geoms.extend(_seam_geoms(w, scale))
        mids.append(np.asarray(w["mid_world"], dtype=np.float64))
        b = np.asarray(w["bisector_world"], dtype=np.float64)
        biss.append(b / (np.linalg.norm(b) + 1e-12))

    # 视角：相机沿 +bisector 一侧、距焊缝 0.5 米，看向焊缝中点（看的方向 = -bisector）。
    #   单焊缝用该缝；全焊缝用各中点质心 + 平均 bisector（退化则取第一条）。
    lookat = np.mean(mids, axis=0)
    bis = np.sum(biss, axis=0)
    nb = float(np.linalg.norm(bis))
    bis = biss[0] if nb < 1e-6 else bis / nb
    extrinsic = _camera_extrinsic(lookat, bis, 0.5)

    title = (f"工件 + 焊缝 {('seam ' + str(seam_id)) if seam_id is not None else f'全部 {len(welds)} 条'}"
             "  （红=焊缝圆柱；视角沿 bisector 望向焊缝，距 0.5m）")
    print(f"[viz] {os.path.basename(obj_fp)}：画 {len(welds)} 条焊缝；关窗退出")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    for g in geoms:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description="可视化工件网格 + 焊缝（不含机械臂）")
    ap.add_argument("--obj", required=True, help="工件 mesh（_part.obj / _watertight.obj）")
    ap.add_argument("--weld-json", required=True, help="焊缝 json（_weld_angle3.json）")
    ap.add_argument("--seam-id", type=int, default=None,
                    help="只画这一条焊缝（缺省=画全部）")
    args = ap.parse_args()
    if not os.path.isfile(args.obj):
        ap.error(f"--obj 不存在：{args.obj}")
    if not os.path.isfile(args.weld_json):
        ap.error(f"--weld-json 不存在：{args.weld_json}")
    visualize(args.obj, args.weld_json, args.seam_id)


if __name__ == "__main__":
    main()
