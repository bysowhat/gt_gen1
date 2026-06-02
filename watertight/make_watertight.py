#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 *_part.obj 工件网格转换为闭合(watertight)网格, 另存为 *_part_watertight.obj.

背景
----
这些 OBJ 是 IFC 结构构件导出的装配体: 文件里有几十个独立的 `o` 对象, 每个对象本身
是一个闭合的实体(立方体/板件), 但它们相互穿插、重叠, 只是被简单拼接在一个文件里。
因此整体不是"一个闭合实体"——内部存在大量被包裹的内壁面, 且不同部件接触处会出现
非流形边(一条边被 4 个面共享)。直接合并顶点反而会破坏流形性。

正确做法是对所有部件做**布尔并集(boolean union)**, 得到去除内部几何的单一闭合外表面。
布尔并集是精确运算(基于 manifold3d), 不像体素重建那样损失薄板等细节。

用法
----
    python make_watertight.py                 # 处理默认目录下所有 *_part.obj
    python make_watertight.py <dir_or_file>...  # 指定目录或文件
    python make_watertight.py --force          # 覆盖已存在的 *_watertight.obj
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import trimesh


DEFAULT_ROOT = "/media/a/新加卷/hanfeng/segment_sub_output"

# 体积小于该阈值(相对于整体包围盒体积)的连通块视为布尔运算产生的退化碎片, 丢弃。
REL_VOLUME_EPS = 1e-6


def find_part_files(targets):
    """收集所有待处理的 *_part.obj (排除已生成的 *_watertight.obj)。"""
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            files += glob.glob(os.path.join(t, "**", "*_part.obj"), recursive=True)
            files += glob.glob(os.path.join(t, "*_part.obj"))
        else:
            print(f"  [警告] 跳过不存在的路径: {t}")
    files = sorted({f for f in files if not f.endswith("_watertight.obj")})
    return files


def components_as_solids(mesh):
    """拆分为连通块, 每块各自合并重复顶点 -> 一组独立闭合实体。

    关键: 只在单个连通块内部合并顶点。跨部件全局合并会在接触面制造非流形边。
    """
    solids = []
    for comp in mesh.split(only_watertight=False):
        c = comp.copy()
        c.merge_vertices()  # 同一实体内焊接重复顶点, 使其成为有效流形实体
        solids.append(c)
    return solids


def drop_degenerate(mesh):
    """布尔并集后会残留零体积碎片(共面接触面产生), 仅保留体积显著的连通块。"""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    bbox_vol = float(np.prod(mesh.bounds[1] - mesh.bounds[0])) or 1.0
    keep = [p for p in parts if abs(p.volume) > REL_VOLUME_EPS * bbox_vol]
    if not keep:
        return mesh
    return trimesh.util.concatenate(keep)


def make_watertight(path):
    """读取一个 _part.obj, 返回闭合后的 Trimesh。"""
    mesh = trimesh.load(path, process=False, force="mesh")
    solids = components_as_solids(mesh)

    # 对所有部件求布尔并集 -> 去除内部几何的单一闭合外壳
    try:
        union = trimesh.boolean.union(solids, engine="manifold")
    except Exception as e:  # 退而求其次: 直接拼接(仍可能闭合, 因每块本就闭合)
        print(f"  [警告] 布尔并集失败({e}), 回退为直接拼接各闭合部件")
        union = trimesh.util.concatenate(solids)

    # manifold 引擎输出已是有效流形, 不做删面类清理(会戳破闭合面)。
    # 仅丢弃共面接触产生的零体积碎片, 并焊接重复顶点。
    union = drop_degenerate(union)
    union.merge_vertices()
    return mesh, union


def main():
    ap = argparse.ArgumentParser(description="将 _part.obj 转为闭合网格 _part_watertight.obj")
    ap.add_argument("targets", nargs="*", default=[DEFAULT_ROOT],
                    help="目录或 .obj 文件 (默认: %(default)s)")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 _watertight.obj")
    args = ap.parse_args()

    files = find_part_files(args.targets)
    if not files:
        print("未找到任何 *_part.obj 文件。")
        return 1

    print(f"共找到 {len(files)} 个文件\n")
    n_ok = n_skip = n_fail = 0
    for path in files:
        out = path[:-4] + "_watertight.obj"
        name = os.path.basename(path)
        if os.path.exists(out) and not args.force:
            print(f"[跳过] {name} (已存在, 用 --force 覆盖)")
            n_skip += 1
            continue
        try:
            t = time.time()
            src, wt = make_watertight(path)
            parts = wt.split(only_watertight=False)
            all_wt = all(p.is_watertight for p in parts)
            wt.export(out)
            tag = "闭合" if all_wt else "未完全闭合"
            print(f"[完成] {name}")
            print(f"        原始: {len(src.vertices)}v/{len(src.faces)}f  ->  "
                  f"结果: {len(wt.vertices)}v/{len(wt.faces)}f  "
                  f"({len(parts)}个闭合壳, {tag}, 体积={wt.volume:.4f}, {time.time()-t:.2f}s)")
            print(f"        -> {out}")
            n_ok += 1
            if not all_wt:
                n_fail += 1
        except Exception as e:
            print(f"[失败] {name}: {e}")
            n_fail += 1

    print(f"\n完成: {n_ok} 成功, {n_skip} 跳过, {n_fail} 有问题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
