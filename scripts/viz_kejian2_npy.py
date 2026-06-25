#!/usr/bin/env python3
"""可视化 plan_init_pose_kejian2.py --out 保存的 .npy 结果（挨个看）。

输入：
  --obj  工件 mesh（_part.obj / _watertight.obj）
  --npy  plan_init_pose_kejian2 存的结构化数组 .npy
         （字段：hand / joint_angles / workpiece_pose7，见 _save_kejian2_npy）

每条记录单开一个窗口（标题写「正手/反手 第 i/N 个」），画：
  · 该 joint_angles 下整臂碰撞球 + init_free 盒 + base 坐标架（init_space_geometries）；
  · 按 workpiece_pose7（工件在 base_link 下的 pose）摆放的工件网格（浅灰）。
按【C 键】切到下一条、相机视角自动沿用；直接关窗（不按 C）则退出。

注：npy 里没有焊缝/ bisector 信息，故不画焊缝线与判据箭头（那需 weld_json，见 kejian2 自带可视化）。

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


def _entry_geoms(cfg, obj_fp: str, rec):
    """单条记录的 open3d 几何：整臂碰撞球 + init_free 盒 + base 架 + 工件网格（按 pose7 摆放）。"""
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
    return geoms


def visualize_npy(obj_fp: str, npy_fp: str):
    import open3d as o3d
    from gt_gen.config import load_config

    cfg = load_config()
    arr = np.load(npy_fp, allow_pickle=False)
    n = int(arr.shape[0])
    if n == 0:
        print(f"[viz] {npy_fp} 为空，无可视化")
        return
    n_fore = int(np.sum(arr["hand"] == "forehand"))
    n_back = int(np.sum(arr["hand"] == "backhand"))
    print(f"[viz] {npy_fp}：共 {n} 条（正手 {n_fore} / 反手 {n_back}）；"
          f"按 C 切下一条，直接关窗退出")

    cam = {"params": None}   # 跨窗口沿用相机视角
    idx = 0
    while idx < n:
        rec = arr[idx]
        hand = str(rec["hand"])
        hand_cn = _HAND_CN.get(hand, hand)
        title = f"kejian2 npy: {hand_cn} 第 {idx + 1}/{n} 条 — 按 C 下一个 / 关窗退出"
        q = np.asarray(rec["joint_angles"], dtype=np.float64).reshape(-1)
        print(f"[viz] 第 {idx + 1}/{n} 条 {hand_cn}  q(deg)="
              f"[{', '.join(f'{np.degrees(v):.1f}' for v in q)}]")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=title)
        for g in _entry_geoms(cfg, obj_fp, rec):
            vis.add_geometry(g)
        if cam["params"] is not None:
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
