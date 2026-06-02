#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D 可视化 OBJ 网格 (基于 open3d)。

功能:
  - 交互窗口查看 (鼠标左键旋转 / 右键或滚轮缩放 / 中键平移)
  - 半透明上色 + 黑色线框, 方便看清结构与是否闭合
  - --compare: 把 *_part.obj 与对应的 *_part_watertight.obj 并排对比
  - --wireframe: 纯线框模式
  - --screenshot out.png: 无头离屏渲染存图 (无需弹窗)

交互快捷键 (open3d 自带):
  W 切换线框   N 显示法线   +/- 调点/线大小   R 重置视角   Q/Esc 退出

用法:
  python view_obj.py <file.obj>                     # 看单个文件
  python view_obj.py <dir>                           # 看目录下所有 _part.obj
  python view_obj.py <file.obj> --compare            # 原始 vs 闭合 并排
  python view_obj.py <file.obj> --screenshot a.png   # 存图不弹窗
"""

import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d


# 一组区分度高的颜色, 多个网格时循环使用
PALETTE = [
    (0.40, 0.65, 0.95), (0.95, 0.55, 0.35), (0.55, 0.80, 0.45),
    (0.85, 0.45, 0.75), (0.95, 0.80, 0.30), (0.50, 0.75, 0.85),
]


def load_o3d(path):
    """读取为 open3d 三角网格, 计算法线; 失败抛异常。"""
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"未读到三角面 (文件可能只含线/点): {path}")
    mesh.compute_vertex_normals()
    return mesh


def colorize(mesh, rgb):
    mesh.paint_uniform_color(rgb)
    return mesh


def wireframe_of(mesh, color=(0.0, 0.0, 0.0)):
    """生成网格的线框 (LineSet), 叠加显示边界更清楚。"""
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(color)
    return ls


def build_geometries(paths, compare=False, wireframe=False):
    """根据输入路径列表构造待渲染的 open3d 几何体。"""
    geoms = []

    if compare:
        # 每个输入文件, 找它的 _part / _watertight 配对, 左右并排
        for path in paths:
            base = path[:-4]
            if base.endswith("_watertight"):
                src = base[:-len("_watertight")] + ".obj"
                wt = path
            else:
                src = path
                wt = base + "_watertight.obj"
            pair = [(src, "原始"), (wt, "闭合")]
            existing = [(p, tag) for p, tag in pair if os.path.exists(p)]
            if not existing:
                print(f"  [警告] 找不到配对文件: {path}")
                continue
            # 沿 X 方向把两个模型拉开一个身位
            span = None
            for i, (p, tag) in enumerate(existing):
                m = load_o3d(p)
                ext = m.get_axis_aligned_bounding_box().get_extent()
                span = ext[0] if span is None else span
                m.translate((i * span * 1.3, 0, 0))
                colorize(m, PALETTE[i % len(PALETTE)])
                geoms.append(m)
                geoms.append(wireframe_of(m))
                print(f"  [{tag}] {os.path.basename(p)}  面={len(m.triangles)}")
        return geoms

    # 普通模式: 所有文件叠加显示
    for i, path in enumerate(paths):
        m = load_o3d(path)
        if wireframe:
            geoms.append(wireframe_of(m, PALETTE[i % len(PALETTE)]))
        else:
            colorize(m, PALETTE[i % len(PALETTE)])
            geoms.append(m)
            geoms.append(wireframe_of(m))
        print(f"  {os.path.basename(path)}  顶点={len(m.vertices)} 面={len(m.triangles)}")
    return geoms


def show_window(geoms, title="OBJ viewer"):
    o3d.visualization.draw_geometries(
        geoms, window_name=title, width=1280, height=860,
        mesh_show_back_face=True,   # 双面渲染, 看内壁/翻转面不漏
    )


def save_screenshot(geoms, out, size=(1600, 1100)):
    """离屏渲染存 PNG (无需弹窗)。"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size[0], height=size[1])
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([1, 1, 1])
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    vis.capture_screen_image(out, do_render=True)
    vis.destroy_window()
    print(f"已保存截图 -> {out}")


def resolve_inputs(targets):
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            files += sorted(glob.glob(os.path.join(t, "**", "*_part.obj"), recursive=True))
        else:
            print(f"  [警告] 跳过不存在的路径: {t}")
    # 默认不重复显示已生成的 watertight (除非用户显式传入)
    return [f for f in files]


def main():
    ap = argparse.ArgumentParser(description="3D 可视化 OBJ 网格")
    ap.add_argument("targets", nargs="+", help="OBJ 文件或目录")
    ap.add_argument("--compare", action="store_true",
                    help="把 _part 与 _part_watertight 并排对比")
    ap.add_argument("--wireframe", action="store_true", help="纯线框模式")
    ap.add_argument("--screenshot", metavar="PNG", default=None,
                    help="离屏渲染存图而不弹窗")
    args = ap.parse_args()

    paths = resolve_inputs(args.targets)
    if not paths:
        print("没有可显示的 OBJ。")
        return 1

    print(f"加载 {len(paths)} 个文件:")
    geoms = build_geometries(paths, compare=args.compare, wireframe=args.wireframe)
    if not geoms:
        print("没有可渲染的几何体。")
        return 1

    if args.screenshot:
        save_screenshot(geoms, args.screenshot)
    else:
        title = "compare" if args.compare else os.path.basename(paths[0])
        show_window(geoms, title=f"OBJ viewer - {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
