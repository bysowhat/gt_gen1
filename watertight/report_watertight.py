#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事后扫描: 遍历已生成的 *_part_watertight.obj, 检查每个是否真正闭合(watertight),
汇总列出"未完全闭合"的文件清单。

用途
----
make_watertight.py 在布尔并集失败时会回退为直接拼接(警告 "Not all meshes are
volumes!"), 结果可能不闭合。本脚本不重跑转换, 只读已生成的 *_watertight.obj 做体检,
快速告诉你: 哪些 part 没闭合、占比多大, 便于判断影响面。

闭合判据
--------
拆连通块, 要求每一块 is_watertight (无开放边/无非流形边)。任一块不闭合即判该文件未闭合,
并附带原因(open_edges / 非流形 / 0体积 等)与体积、连通块数。

用法
----
    python report_watertight.py                      # 默认目录
    python report_watertight.py <dir_or_file>...     # 指定目录/文件
    python report_watertight.py <dir> --out bad.txt  # 把未闭合清单另存到文件
    python report_watertight.py <dir> --jobs 8        # 多进程并行体检(默认 1)

    # 服务器(无 conda):
    #   /isaac-sim/python.sh watertight/report_watertight.py <dir>
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh


DEFAULT_ROOT = "/media/a/新加卷/hanfeng/1/segment_output_sub"


def find_watertight_files(targets):
    """收集所有待体检的 *_part_watertight.obj。"""
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            files += glob.glob(os.path.join(t, "**", "*_part_watertight.obj"), recursive=True)
            files += glob.glob(os.path.join(t, "*_part_watertight.obj"))
        else:
            print(f"  [警告] 跳过不存在的路径: {t}", file=sys.stderr)
    return sorted(set(files))


def check_one(path):
    """体检单个文件, 返回 dict: path, ok(bool), reason, n_parts, n_bad, volume。

    ok=True 表示每个连通块都 is_watertight。失败时 reason 给出第一块不闭合的原因。
    """
    try:
        mesh = trimesh.load(path, process=False, force="mesh")
    except Exception as e:
        return dict(path=path, ok=False, reason=f"load_error:{e}",
                    n_parts=0, n_bad=0, volume=0.0)

    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        parts = [mesh]

    n_bad = 0
    first_reason = ""
    for p in parts:
        if p.is_watertight:
            continue
        n_bad += 1
        if first_reason:
            continue
        # 诊断第一块不闭合的原因: 开放边数 + winding 一致性 + 欧拉数
        try:
            n_open = int(len(p.edges_unique) - len(p.face_adjacency))
        except Exception:
            n_open = -1
        is_wnd = bool(getattr(p, "is_winding_consistent", True))
        first_reason = (f"非闭合块: faces={len(p.faces)} "
                        f"开放边≈{n_open} winding_consistent={is_wnd} "
                        f"euler={p.euler_number}")

    ok = (n_bad == 0)
    reason = "watertight" if ok else first_reason or f"{n_bad}/{len(parts)} 块未闭合"
    return dict(path=path, ok=ok, reason=reason,
                n_parts=len(parts), n_bad=n_bad, volume=float(mesh.volume))


def main():
    ap = argparse.ArgumentParser(description="体检 *_part_watertight.obj 的闭合性, 列出未闭合清单")
    ap.add_argument("targets", nargs="*", default=[DEFAULT_ROOT],
                    help="目录或 .obj 文件 (默认: %(default)s)")
    ap.add_argument("--out", default=None, help="把未闭合文件清单另存到此路径(每行一个)")
    ap.add_argument("--jobs", type=int, default=1, help="并行体检进程数(默认 1)")
    args = ap.parse_args()

    files = find_watertight_files(args.targets)
    if not files:
        print("未找到任何 *_part_watertight.obj 文件。", file=sys.stderr)
        return 1

    print(f"共体检 {len(files)} 个 *_part_watertight.obj\n")

    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            results = pool.map(check_one, files)
    else:
        results = [check_one(p) for p in files]

    bad = [r for r in results if not r["ok"]]
    n_ok = len(results) - len(bad)

    for r in bad:
        print(f"[未闭合] {r['path']}")
        print(f"          {r['n_bad']}/{r['n_parts']} 块不闭合  体积={r['volume']:.4f}  {r['reason']}")

    print(f"\n汇总: {n_ok} 闭合 / {len(bad)} 未闭合 / 共 {len(results)}"
          f"  (未闭合占比 {100.0*len(bad)/max(1,len(results)):.1f}%)")

    if args.out and bad:
        with open(args.out, "w") as f:
            for r in bad:
                f.write(r["path"] + "\n")
        print(f"未闭合清单已写出: {args.out}  ({len(bad)} 行)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
