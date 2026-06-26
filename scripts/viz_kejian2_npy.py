#!/usr/bin/env python3
"""可视化 plan_init_pose_kejian2.py --out 保存的 .npy 结果（挨个看）。

输入：
  --obj  工件 mesh（_part.obj / _watertight.obj）
  --npy  plan_init_pose_kejian2 存的结构化数组 .npy
         （字段：hand / joint_angles / workpiece_pose7，见 _save_kejian2_npy）

每条记录单开一个窗口（标题写「正手/反手 第 i/N 个」），画：
  · 该 joint_angles 下整臂碰撞球 + init_free 盒 + base 坐标架（init_space_geometries）；
  · 按 workpiece_pose7（工件在 base_link 下的 pose）摆放的工件网格（浅灰）；
  · 焊缝红色圆柱（npy 里的原始焊缝 dict 在 mesh 系，套同一 workpiece_pose7 变到 base 系）。
按【C 键】切到下一条；相机默认沿该记录焊缝 bisector、距 0.5m 望向焊缝中点（参考 viz_weld_json）。
直接关窗（不按 C）则退出。

用法：
  python scripts/viz_kejian2_npy.py \
    --obj /…/BEAM_…_watertight.obj \
    --npy /tmp/kejian2_seam1.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 复用 scripts/plan_init_pose_kejian2.py 的几何 helper（不改它）
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import plan_init_pose_kejian2 as k2  # noqa: E402

_HAND_CN = {"forehand": "正手", "backhand": "反手"}


def _weld_in_base(weld_raw: dict, T: np.ndarray):
    """npy 存的原始焊缝 dict（mesh/world 系）经工件位姿 T(base←mesh) 变到 base 系。
    返回 (p0_base, p1_base, mid_base, bisector_base)；字段缺失返回 None。"""
    if not isinstance(weld_raw, dict) or "corrected_p0" not in weld_raw:
        return None
    p0 = np.asarray(weld_raw["corrected_p0"], dtype=np.float64)
    p1 = np.asarray(weld_raw["corrected_p1"], dtype=np.float64)
    bis = np.asarray(weld_raw.get("bisector", [0.0, 0.0, 1.0]), dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    p0b = R @ p0 + t
    p1b = R @ p1 + t
    bisb = R @ bis
    bisb = bisb / (np.linalg.norm(bisb) + 1e-12)
    return p0b, p1b, 0.5 * (p0b + p1b), bisb


def _weld_cylinder(p0b, p1b):
    """焊缝红色圆柱（base 系 p0→p1，固定 2cm 直径）；参考 viz_weld_json._seam_geoms。返回 (cyl, L)。"""
    import open3d as o3d

    d = p1b - p0b
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None, L
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=L)  # 固定 2cm 直径
    cyl.rotate(k2._align_rotmat([0.0, 0.0, 1.0], d / L), center=(0.0, 0.0, 0.0))  # +Z → 焊缝方向
    cyl.translate((0.5 * (p0b + p1b)).tolist())
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color([0.9, 0.1, 0.1])
    return cyl, L


def _camera_extrinsic(lookat, bis, dist: float) -> np.ndarray:
    """相机外参（world→camera 4×4）：相机置于 lookat + dist·bisector，朝 -bisector 望向 lookat。
    open3d 相机系约定 x 右 / y 下 / z 前（视线）。整段照搬 viz_weld_json._camera_extrinsic。"""
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
    R_cw = np.column_stack([x, y, z])
    T_cw = np.eye(4); T_cw[:3, :3] = R_cw; T_cw[:3, 3] = C
    return np.linalg.inv(T_cw)                    # world→camera


def _entry_geoms(cfg, obj_fp: str, rec, weld_raw):
    """单条记录的 open3d 几何：整臂碰撞球 + init_free 盒 + base 架 + 工件网格（按 pose7 摆放）
    + 焊缝红色圆柱（同一 pose7 变换到 base 系）。返回 (geoms, cam_target)，cam_target=(mid,bis) 或 None。"""
    import open3d as o3d

    q = np.asarray(rec["joint_angles"], dtype=np.float64).reshape(-1)
    geoms, _ = k2.init_space_geometries(cfg, q=q)

    T = k2.pose7_to_mat44(np.asarray(rec["workpiece_pose7"], dtype=np.float64).reshape(-1))
    mesh = o3d.io.read_triangle_mesh(obj_fp)
    if not mesh.has_vertices():
        print(f"[warn] open3d 读不到工件网格: {obj_fp}（仅画机械臂）")
    else:
        mesh.transform(T)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.72])
        geoms.append(mesh)

    # 焊缝：原始 dict 在 mesh 系，套同一工件位姿 T 变到 base 系，与工件网格同框叠加。
    cam_target = None
    wb = _weld_in_base(weld_raw, T)
    if wb is not None:
        p0b, p1b, mid_b, bis_b = wb
        cyl, L = _weld_cylinder(p0b, p1b)
        if cyl is not None:
            geoms.append(cyl)
            cam_target = (mid_b, bis_b)
            print(f"[viz]   焊缝长度 {L * 100:.2f} cm（{L:.4f} m）")
    return geoms, cam_target


def visualize_npy(obj_fp: str, npy_fp: str):
    import open3d as o3d
    from gt_gen.config import load_config

    cfg = load_config()
    # _save_kejian2_npy 存的是「0 维 object 数组包一个并列列 dict」（hand/joint_angles/workpiece_pose7），
    # 读回须 .item() 取出 dict，再按下标切各列；不是逐记录的 record array。
    loaded = np.load(npy_fp, allow_pickle=True)
    data = loaded.item() if (loaded.ndim == 0 and loaded.dtype == object) else loaded
    hands = np.asarray(data["hand"]).reshape(-1)
    joints = np.asarray(data["joint_angles"], dtype=np.float64).reshape(len(hands), -1)
    poses = np.asarray(data["workpiece_pose7"], dtype=np.float64).reshape(len(hands), -1)
    weld_raw = data.get("weld", {})            # 该焊缝原始 json dict（mesh 系），所有记录共用
    n = int(len(hands))
    if n == 0:
        print(f"[viz] {npy_fp} 为空，无可视化")
        return
    n_fore = int(np.sum(hands == "forehand"))
    n_back = int(np.sum(hands == "backhand"))
    print(f"[viz] {npy_fp}：共 {n} 条（正手 {n_fore} / 反手 {n_back}）；"
          f"按 C 切下一条，直接关窗退出")

    cam = {"params": None}   # 跨窗口沿用相机视角
    idx = 0
    while idx < n:
        rec = {"hand": hands[idx], "joint_angles": joints[idx], "workpiece_pose7": poses[idx]}
        hand = str(rec["hand"])
        hand_cn = _HAND_CN.get(hand, hand)
        title = f"kejian2 npy: {hand_cn} 第 {idx + 1}/{n} 条 — 按 C 下一个 / 关窗退出"
        q = np.asarray(rec["joint_angles"], dtype=np.float64).reshape(-1)
        print(f"[viz] 第 {idx + 1}/{n} 条 {hand_cn}  q(deg)="
              f"[{', '.join(f'{np.degrees(v):.1f}' for v in q)}]")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=title)
        geoms, cam_target = _entry_geoms(cfg, obj_fp, rec, weld_raw)
        for g in geoms:
            vis.add_geometry(g)
        # 视角：优先沿该记录焊缝的 bisector、距 0.5m 望向焊缝中点（参考 viz_weld_json）；
        #   无焊缝信息时回退到跨窗口沿用的手动视角。
        if cam_target is not None:
            try:
                ctr = vis.get_view_control()
                params = ctr.convert_to_pinhole_camera_parameters()
                params.extrinsic = _camera_extrinsic(cam_target[0], cam_target[1], 0.5)
                ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            except Exception:
                pass
        elif cam["params"] is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    cam["params"], allow_arbitrary=True)
            except Exception:
                pass

        advance = {"go": False}

        def _next(v):
            advance["go"] = True
            v.close()
            return False

        vis.register_key_callback(ord("C"), _next)
        vis.run()
        try:
            cam["params"] = vis.get_view_control().convert_to_pinhole_camera_parameters()
        except Exception:
            pass
        vis.destroy_window()
        if not advance["go"]:
            print("[viz] 直接关窗，结束可视化")
            break
        idx += 1


def main():
    ap = argparse.ArgumentParser(description="可视化 plan_init_pose_kejian2 保存的 .npy 结果")
    ap.add_argument("--obj", required=True, help="工件 mesh（_part.obj / _watertight.obj）")
    ap.add_argument("--npy", required=True, help="plan_init_pose_kejian2 --out 保存的 .npy")
    args = ap.parse_args()
    if not os.path.isfile(args.obj):
        ap.error(f"--obj 不存在：{args.obj}")
    if not os.path.isfile(args.npy):
        ap.error(f"--npy 不存在：{args.npy}")
    visualize_npy(args.obj, args.npy)


if __name__ == "__main__":
    main()
