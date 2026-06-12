#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 report_watertight.py 产出的「未闭合清单」删除对应的 part 文件夹。

输入清单(每行一个 *_part_watertight.obj 路径)由 report_watertight.py 的 --out 写出。
本脚本把每个 obj 的【父目录】(即 <part>_part/ 文件夹)收集去重, 整文件夹删除
(连同 _part.obj / _part_watertight.obj / seam_*.pkl 一并删)。

安全设计
--------
- 默认 dry-run: 只打印将删除的文件夹, 不动磁盘。确认无误后加 --apply 才真删。
- 只删【根目录 root 的直接子目录】: 父目录必须正好在 root 下一层, 否则跳过并告警
  (防止清单里出现异常路径导致误删 root 本身或上层目录)。
- 父目录名需以 _part 结尾(part 文件夹约定), 否则跳过并告警。

用法
----
    # 1) 先看会删哪些(dry-run, 不动磁盘):
    python drop_not_watertight.py /root/no_watertight.txt \
        --root /kpfs_dataset/dataset/baiyu/dataset_v2/debug/1/segment_output_sub

    # 2) 核对无误后真删:
    python drop_not_watertight.py /root/no_watertight.txt \
        --root /kpfs_dataset/dataset/baiyu/dataset_v2/debug/1/segment_output_sub --apply

    # 服务器(无 conda):
    #   /isaac-sim/python.sh watertight/drop_not_watertight.py /root/no_watertight.txt --root <root> [--apply]
"""

import argparse
import os
import shutil
import sys


def collect_dirs(list_path, root):
    """读清单 → 返回 (待删文件夹去重列表, 跳过列表[(行, 原因)])。"""
    root = os.path.abspath(root)
    keep_order = []
    seen = set()
    skipped = []

    with open(list_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = os.path.abspath(line)
            parent = os.path.dirname(obj)
            # 约束①: parent 必须正好是 root 的直接子目录
            if os.path.dirname(parent) != root:
                skipped.append((line, f"父目录不在 root 直接下一层: {parent}"))
                continue
            # 约束②: parent 必须是 *_part 文件夹
            if not os.path.basename(parent).endswith("_part"):
                skipped.append((line, f"父目录非 *_part 文件夹: {os.path.basename(parent)}"))
                continue
            if parent not in seen:
                seen.add(parent)
                keep_order.append(parent)
    return keep_order, skipped


def main():
    ap = argparse.ArgumentParser(description="按未闭合清单删除对应 part 文件夹(默认 dry-run)")
    ap.add_argument("list_file", help="report_watertight.py --out 写出的未闭合清单(每行一个 obj 路径)")
    ap.add_argument("--root", required=True, help="part 文件夹所在根目录(segment_output_sub)")
    ap.add_argument("--apply", action="store_true", help="真删除(默认仅 dry-run 打印)")
    args = ap.parse_args()

    if not os.path.isfile(args.list_file):
        print(f"清单文件不存在: {args.list_file}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.root):
        print(f"root 目录不存在: {args.root}", file=sys.stderr)
        return 1

    dirs, skipped = collect_dirs(args.list_file, args.root)

    if skipped:
        print(f"[告警] 跳过 {len(skipped)} 行(不符合安全约束, 不删):")
        for line, why in skipped:
            print(f"    {why}\n        ← {line}")
        print()

    if not dirs:
        print("没有符合条件的待删文件夹。")
        return 0

    mode = "真删除" if args.apply else "DRY-RUN(不动磁盘)"
    print(f"== {mode} ==  待删 {len(dirs)} 个 part 文件夹(root={args.root})\n")
    for d in dirs:
        exists = os.path.isdir(d)
        print(f"  {'[删]' if args.apply else '[待删]'} {d}" + ("" if exists else "  (已不存在, 跳过)"))

    if not args.apply:
        print(f"\n这是 dry-run, 未删除任何文件。确认无误后加 --apply 执行。")
        return 0

    n_del = 0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        shutil.rmtree(d)
        n_del += 1
    print(f"\n已删除 {n_del} 个文件夹(清单 {len(dirs)} 个, 其余已不存在)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
