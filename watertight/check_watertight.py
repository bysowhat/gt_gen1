#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测 OBJ/网格是否为闭合(watertight)。

闭合(watertight)的标准定义是拓扑性质: 表面是一个闭合的 2-流形。
判定核心: 焊接重复顶点后, 统计每条边被多少个面共享。
  - 全部恰好被 2 个面共享        -> 闭合(watertight)
  - 存在只被 1 个面共享的边      -> 有破洞(边界边)
  - 存在被 >=3 个面共享的边      -> 非流形边(部件接触/重叠处常见)

注意: "无自相交"(non-self-intersecting)是另一个独立性质, 不属于 watertight。
一个网格可以闭合但自相交。本脚本把自相交作为**附加诊断**单独报告(--strict),
不影响 watertight 的判定结论。

用法:
    python check_watertight.py                      # 检测默认目录下所有 .obj
    python check_watertight.py <dir_or_file> ...     # 指定目录或文件
    python check_watertight.py --pattern '*_watertight.obj'   # 自定义匹配
    python check_watertight.py --strict              # 额外报告自相交/退化面(独立诊断)
"""

import argparse
import collections
import glob
import os
import sys

import numpy as np
import trimesh
from trimesh.grouping import group_rows


DEFAULT_ROOT = "/media/a/新加卷/hanfeng/segment_sub_output"


def edge_multiplicity(mesh):
    """返回 {边被共享的面数: 这样的边有多少条}, 例如 {2: 100, 1: 4}。"""
    groups = group_rows(mesh.edges_sorted)
    return dict(collections.Counter(len(g) for g in groups))


def check_mesh(path, strict=False):
    """检测单个文件, 返回结果字典。"""
    # process=True 会自动焊接重复顶点 —— 这一步对正确判定 watertight 至关重要,
    # 否则 OBJ 里同位置的重复顶点会让相邻面"看起来"不共享边而误报有洞。
    mesh = trimesh.load(path, process=True, force="mesh")

    parts = mesh.split(only_watertight=False)
    mult = edge_multiplicity(mesh)
    n_boundary = mult.get(1, 0)        # 破洞边
    n_nonmanifold = sum(c for k, c in mult.items() if k >= 3)  # 非流形边

    res = {
        "path": path,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "shells": len(parts),
        "is_watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "volume": float(mesh.volume),
        "boundary_edges": n_boundary,
        "nonmanifold_edges": n_nonmanifold,
        "all_shells_watertight": all(p.is_watertight for p in parts),
        "edge_multiplicity": mult,
        "degenerate_faces": int((mesh.area_faces < 1e-12).sum()),
        "self_intersecting": None,
    }

    if strict:
        # 自相交是独立于 watertight 的附加诊断(open3d 的 is_watertight 把它捆绑进去了,
        # 所以这里只取 is_self_intersecting, 不用它的 is_watertight 结论)。
        try:
            import open3d as o3d
            mo = o3d.io.read_triangle_mesh(path)
            res["self_intersecting"] = bool(mo.is_self_intersecting())
        except Exception as e:
            res["self_intersecting"] = f"open3d检查失败: {e}"

    return res


def find_files(targets, pattern):
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            files += glob.glob(os.path.join(t, "**", pattern), recursive=True)
            files += glob.glob(os.path.join(t, pattern))
        else:
            print(f"  [警告] 跳过不存在的路径: {t}")
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description="检测网格是否为闭合(watertight)")
    ap.add_argument("targets", nargs="*", default=[DEFAULT_ROOT],
                    help="目录或网格文件 (默认: %(default)s)")
    ap.add_argument("--pattern", default="*.obj",
                    help="目录下文件匹配模式 (默认: %(default)s)")
    ap.add_argument("--strict", action="store_true",
                    help="额外用 open3d 检查自相交(更严格)")
    args = ap.parse_args()

    files = find_files(args.targets, args.pattern)
    if not files:
        print("未找到任何匹配文件。")
        return 1

    print(f"共检测 {len(files)} 个文件\n")
    n_pass = 0
    for path in files:
        try:
            r = check_mesh(path, strict=args.strict)
        except Exception as e:
            print(f"[错误] {os.path.basename(path)}: {e}\n")
            continue

        # 判定只看拓扑闭合性; 自相交/退化面是单独诊断, 不影响结论。
        ok = r["is_watertight"] and r["all_shells_watertight"]
        n_pass += ok

        mark = "✅ 闭合" if ok else "❌ 未闭合"
        print(f"{mark}  {os.path.basename(path)}")
        print(f"     顶点={r['vertices']} 面={r['faces']} 闭合壳数={r['shells']}")
        print(f"     is_watertight={r['is_watertight']}  绕向一致={r['winding_consistent']}  "
              f"欧拉数={r['euler_number']}  体积={r['volume']:.4f}")
        if r["boundary_edges"]:
            print(f"     ⚠ 破洞边(边界边)={r['boundary_edges']} 条")
        if r["nonmanifold_edges"]:
            print(f"     ⚠ 非流形边={r['nonmanifold_edges']} 条")
        if not r["all_shells_watertight"]:
            print(f"     ⚠ 存在未闭合的连通块")
        if args.strict:
            # 附加诊断(不影响闭合判定)
            print(f"     [附加诊断] 自相交={r['self_intersecting']}  "
                  f"退化面(零面积)={r['degenerate_faces']}")
        print(f"     边多重度分布={r['edge_multiplicity']}")
        print()

    print(f"结果: {n_pass}/{len(files)} 个文件闭合")
    # 全部闭合返回 0, 否则返回 2 (便于脚本/CI 判断)
    return 0 if n_pass == len(files) else 2


if __name__ == "__main__":
    sys.exit(main())
