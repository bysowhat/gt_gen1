#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 render_seam_scheduler.sh 的渲染结果生成「平放合格」白名单。

背景与判据
----------
render 输出布局：
    <root>/<子目录=一个工件>/<..._pose#>/{left,right}/render_info.npy
同一工件目录下的所有 pose 都来自【同一个平放(lay_flat)结果】，只是相对机械臂的 xyz 不同，
所以每个子目录只需检测【一次】即可判定其平放是否合格。

render_info.npy 里的 `workpiece_pose7 = [x,y,z, qw,qx,qy,qz]`（wxyz，T_base←workpiece）。
渲染时工件朝向 Rv = {I,长轴翻180°}×{I,竖轴翻180°}×Rz(90°)×R_lay，右边全是 90°/180°
（立方体旋转群元素），故 ⟺ 仅当 R_lay 是绕 x/y/z 轴 90° 整倍时 Rv 才是【有符号置换矩阵】。
旧代码的 45° 之类滚转会让 Rv 偏离。判据即：workpiece_pose7 的旋转 ≈ 有符号置换矩阵
（元素∈{0,±1}、det=+1，与最近 90°-整倍朝向残差角 < 阈值）。

输出
----
只产【一个】txt：合格的子目录名，一行一个。每验证出一个合格工件就【立即追加并 flush】，
中途中断也保住已写结果。

用法（在远程跑；仅依赖 numpy）
------------------------------
  ssh debug 'python3 /kpfs_dataset/dataset/baiyu/code/render/gt_gen_hanfeng/scripts/bash/kejian/gen_render_whitelist.py'
  # 覆盖：
  python3 gen_render_whitelist.py --root /kpfs_dataset/dataset/render_kejian/render_outs \
      --out /path/whitelist.txt --tol-deg 5
"""
import argparse
import math
import os
import sys

import numpy as np


def quat_wxyz_to_R(q):
    """四元数 (w,x,y,z) → 3×3 旋转矩阵。"""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def is_ninety_multiple(pose7, tol_deg):
    """workpiece_pose7 的旋转是否为绕 xyz 90° 整倍（有符号置换矩阵、det=+1、残差角<tol）。

    返回 (valid, residual_deg)。"""
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)
    R = quat_wxyz_to_R(pose7[3:7])
    # 贪心吸附到最近的有符号置换矩阵 P（按各列最大绝对值置信度分配，保证行不重复）
    P = np.zeros((3, 3))
    used = set()
    for j in sorted(range(3), key=lambda c: -np.max(np.abs(R[:, c]))):
        i = max((k for k in range(3) if k not in used), key=lambda k: abs(R[k, j]))
        P[i, j] = 1.0 if R[i, j] >= 0 else -1.0
        used.add(i)
    c = (np.trace(R @ P.T) - 1.0) / 2.0
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))
    valid = (ang < float(tol_deg)) and (float(np.linalg.det(P)) > 0.5)
    return valid, ang


def read_one_pose7(stem_dir):
    """从工件目录里【任取一个】pose 单元读 workpiece_pose7（先 left 后 right）。读不到返回 None。"""
    try:
        names = sorted(os.listdir(stem_dir))
    except OSError:
        return None
    for name in names:
        unit = os.path.join(stem_dir, name)
        if not (os.path.isdir(unit) and "_pose" in name):
            continue
        for side in ("left", "right"):
            fp = os.path.join(unit, side, "render_info.npy")
            if not os.path.isfile(fp):
                continue
            try:
                info = np.load(fp, allow_pickle=True).item()
            except Exception:
                continue
            p = info.get("workpiece_pose7")
            if p is None and isinstance(info.get("seam_npy"), dict):
                p = info["seam_npy"].get("workpiece_pose7")   # 兜底
            if p is not None:
                p = np.asarray(p, dtype=np.float64).reshape(-1)
                if p.size >= 7:
                    return p[:7]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="render_outs 平放合格白名单（每个子目录只检测一次；只出一个 txt）。")
    ap.add_argument("--root", default="/kpfs_dataset/dataset/render_kejian/render_outs",
                    help="渲染输出根目录（其下每个子目录=一个工件）")
    ap.add_argument("--out", default=None,
                    help="白名单 txt 路径；缺省 = <root>/whitelist.txt")
    ap.add_argument("--tol-deg", type=float, default=5.0,
                    help="残差角阈值(度)：与最近 90° 整倍朝向夹角 < 此值即合格；默认 5")
    ap.add_argument("--limit", type=int, default=0, help=">0：只处理前 N 个子目录（冒烟测试）")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[err] 渲染根目录不存在: {root}", file=sys.stderr)
        sys.exit(1)
    out_fp = args.out or os.path.join(root, "whitelist.txt")

    stem_dirs = [n for n in sorted(os.listdir(root))
                 if os.path.isdir(os.path.join(root, n)) and n not in ("logs", "flat", "orig")]
    if args.limit > 0:
        stem_dirs = stem_dirs[:args.limit]

    print(f"[wl] 根目录 : {root}")
    print(f"[wl] 白名单 : {out_fp}   阈值 tol={args.tol_deg}°")
    print(f"[wl] 子目录 : {len(stem_dirs)} 个，逐个检测一次 ...")

    n_ok = n_bad = n_nodata = 0
    # 每验证出一个合格工件就【立即写入并 flush】——中断也保住已写结果
    with open(out_fp, "w") as f:
        for name in stem_dirs:
            p7 = read_one_pose7(os.path.join(root, name))
            if p7 is None:
                n_nodata += 1
                continue
            valid, ang = is_ninety_multiple(p7, args.tol_deg)
            if valid:
                f.write(name + "\n")
                f.flush()
                n_ok += 1
            else:
                n_bad += 1
                print(f"[wl][剔除] {name}  残差角={ang:.3f}°（非 90° 整倍平放）")

    print("")
    print("===== 汇总 =====")
    print(f"合格(已写入白名单) : {n_ok}")
    print(f"不合格(剔除)       : {n_bad}")
    print(f"无数据(读不到)     : {n_nodata}")
    print(f"白名单文件         : {out_fp}")


if __name__ == "__main__":
    main()
