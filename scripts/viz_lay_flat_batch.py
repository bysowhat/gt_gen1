#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量「平放工件 + 拍照」脚本（独立，不依赖工程其他模块）。

用途
----
给定一个输入文件夹（例如 /media/a/新加卷/hanfeng/segment_sub_output），其下有多个子文件夹，
每个子文件夹里有一个 *_part_watertight.obj 工件网格。本脚本对每个工件：

  ① 用 lay_flat 把工件「摆平」放到 z=0 地面上；相对 origin 的旋转严格限定为
     「绕 x/y/z 轴 90° 整数倍」的组合（在 24 个立方体旋转里挑长轴水平+落地质心最低者）；
  ② 另做一版「不旋转」的对照（旋转=单位阵，仅平移使最低点落到 z=0、xy 居中）；

然后从【固定的斜上方视角 + 固定距离】离屏渲染，各存 1 张图片，共 2 张：
    <stem>_flat.png   摆平后（应见工件长轴水平、平贴地面）
    <stem>_orig.png   原始朝向（未旋转，仅落地，用于对比是否原本就歪/立）

两图都画 z=0 地面 + base 坐标架 + 相同相机参数，肉眼一眼可判「整体形状」与「是否平放」。

运行
----
    python3 scripts/viz_lay_flat_batch.py /media/a/新加卷/hanfeng/segment_sub_output
    # 常用可调项
    python3 scripts/viz_lay_flat_batch.py <dir> --out-dir /tmp/lay_flat_imgs \
        --width 1280 --height 960 --azim 45 --elev 30 --dist-factor 2.2

注意：用 open3d OffscreenRenderer（EGL/filament）离屏渲染，无需窗口/显示器，
无头服务器（如远程 debug 节点）可直接运行。
"""
import argparse
import glob
import os
import sys

import numpy as np


# ============================================================================
# ① lay_flat：从 scripts/plan_init_pose_kejian2.py 原样拷贝（去掉 open3d 可视化调用，只返回 T）
# ============================================================================
def _dominant_seam_points(obj_fp, long_len, weld_json=None, seam_ratio=0.9):
    """读 obj 同目录同前缀的 <stem>_weld_angle3.json，返回「最长那批焊缝」的所有端点 (M,3)。

    仅当【最长焊缝 > 工件最长轴长度的一半】时才启用焊缝逻辑（否则返回 None，交回质心兜底）；
    启用时取长度 ≥ seam_ratio·最长 的那批「差不多长」焊缝的全部端点。
    坐标为工件 mesh 系（corrected_p0/p1，与 obj 顶点同系）。
    返回 (端点(M,3) 或 None, 用到的 json 路径 或 None, 最长焊缝长度)。
    """
    import json
    fp = weld_json
    if fp is None:                                              # 自动定位：同目录同前缀
        d = os.path.dirname(obj_fp); base = os.path.basename(obj_fp)
        for suf in ("_part_watertight.obj", ".obj"):
            if base.endswith(suf):
                cand = os.path.join(d, base[:-len(suf)] + "_weld_angle3.json")
                if os.path.isfile(cand):
                    fp = cand
                    break
    if not fp or not os.path.isfile(fp):
        return None, None, 0.0
    try:
        with open(fp, "r") as f:
            data = json.load(f)
    except Exception:
        return None, fp, 0.0
    segs = []
    for w in data:
        try:
            p0 = np.asarray(w["corrected_p0"], dtype=np.float64)
            p1 = np.asarray(w["corrected_p1"], dtype=np.float64)
        except Exception:
            continue
        if p0.shape == (3,) and p1.shape == (3,):
            segs.append((p0, p1))
    if not segs:
        return None, fp, 0.0
    lens = np.array([np.linalg.norm(p1 - p0) for p0, p1 in segs])
    max_seam = float(lens.max())
    if max_seam <= 0.5 * float(long_len):                       # 最长焊缝没过最长轴一半 → 不用焊缝逻辑
        return None, fp, max_seam
    keep = lens >= seam_ratio * max_seam                        # 「差不多长」的那批
    pts = np.array([p for (p0, p1), k in zip(segs, keep) if k for p in (p0, p1)])
    return pts, fp, max_seam


def lay_flat(obj_fp, viz=False, weld_json=None, seam_ratio=0.9) -> np.ndarray:
    """把工件「摆平」放到 z=0 地面上，返回 4×4 变换 T（p_world = T · p_obj）。

    硬约束：相对 origin 的旋转 **必须是「绕 x/y/z 轴 90° 整数倍」的组合**——
    即 R 只能取立方体旋转群的有符号置换矩阵（det=+1）之一，
    绝不允许出现 45° 之类的任意滚转角（旧方案B 用凸包支撑边外法向定滚转，会破坏这一点）。

    做法：
      ① 在 mesh 自身坐标系里量各轴包围盒长度，最长者为「长轴」；
      ② 把长轴对到世界 +X（长轴水平），只剩「绕 +X 滚转 90° 整倍」4 种朝向；
      ③ 选滚转：若同目录 _weld_angle3.json 里存在「长度 > 最长轴一半」的长焊缝，
         则取最长那批焊缝、让它们尽量共处一个水平面（端点世界 z 跨度最小）；
         否则（无此长焊缝 / 无 json / 4 者难分）退回「落地质心最低」；
      ④ 平移使 min z = 0 贴地、xy 居中。
    因约束在 mesh 系内做，故长轴/滚转都恰好是 90° 整倍，L/工字/槽型梁也停在真实平面上。
    """
    import trimesh as _trimesh

    tm = _trimesh.load(obj_fp, force="mesh", process=False)
    V = np.asarray(tm.vertices, dtype=np.float64)                # (N,3)
    # 质心：watertight 用体质心，否则退化为顶点形心
    try:
        com = np.asarray(tm.center_mass, dtype=np.float64)
        if not np.all(np.isfinite(com)):
            raise ValueError
    except Exception:
        com = V.mean(axis=0)

    # ① mesh 系各轴包围盒长度 → 最长轴（origin 已轴对齐，故长轴必是某条 mesh 轴）
    ext_mesh = V.max(axis=0) - V.min(axis=0)
    i_long = int(np.argmax(ext_mesh))

    # ② 长轴（mesh 第 i_long 轴）→ 世界 +X，其余两轴放 Y/Z，构一个 det=+1 的基准旋转 R0
    a, b = [k for k in range(3) if k != i_long]
    R0 = np.zeros((3, 3))
    R0[:, i_long] = (1.0, 0.0, 0.0)
    R0[:, a] = (0.0, 1.0, 0.0)
    R0[:, b] = (0.0, 0.0, 1.0)
    if np.linalg.det(R0) < 0:
        R0[:, b] = (0.0, 0.0, -1.0)                             # 翻一轴符号 → det=+1（右手系）

    # ③ 长轴恒沿 +X，仅剩「绕 +X 滚转 90° 整倍」4 种朝向。
    #    有「超过最长轴一半」的长焊缝 → 让最长那批焊缝尽量共处一个水平面（端点 z 跨度最小）；
    #    否则（含无焊缝、4 者 z 跨度难分）退回「落地质心最低」。
    dom_pts, used_json, max_seam = _dominant_seam_points(
        obj_fp, ext_mesh[i_long], weld_json, seam_ratio)
    z_tol = max(1e-4, 0.01 * float(np.linalg.norm(ext_mesh)))   # z 跨度差 < 1% 对角线视为不可区分

    Rx90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    best = None  # (stable, -zkey(焊缝 z 跨度桶，越小越好), -com_h, R, zspread)
    Rk = R0
    for _ in range(4):
        Vw = (Rk @ V.T).T
        cw = Rk @ com
        mn = Vw.min(axis=0); mx = Vw.max(axis=0)
        com_h = float(cw[2] - mn[2])                            # 落地后质心高度（越低越平/越稳）
        stable = (mn[0] - 1e-9 <= cw[0] <= mx[0] + 1e-9 and
                  mn[1] - 1e-9 <= cw[1] <= mx[1] + 1e-9)        # 质心 xy 落在底面投影内
        if dom_pts is not None:
            zc = (Rk @ dom_pts.T).T[:, 2]
            zspread = float(zc.max() - zc.min())                # 最长那批焊缝端点的世界 z 跨度
        else:
            zspread = 0.0
        zkey = int(round(zspread / z_tol))                      # 量化：跨度相近者同桶 → 交给质心兜底
        cand = (1 if stable else 0, -zkey, -com_h, Rk, zspread)
        if best is None or cand[:3] > best[:3]:
            best = cand
        Rk = Rx90 @ Rk                                          # 绕 +X 再滚 90°
    R = best[3]

    # ④ 平移：min z = 0 贴地，xy 居中（用 bbox 中心）
    Vw = (R @ V.T).T
    mn = Vw.min(axis=0); mx = Vw.max(axis=0)
    t = np.array([-(mn[0] + mx[0]) / 2.0, -(mn[1] + mx[1]) / 2.0, -mn[2]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t

    mode = ("焊缝面水平" if dom_pts is not None
            else ("焊缝≤半长→质心" if used_json else "无焊缝json→质心"))
    print(f"[lay_flat] mesh 三轴长(米): {ext_mesh.round(4)}  最长轴#{i_long}  依据={mode}")
    print(f"[lay_flat] 选中 90°-整倍旋转 R(行)= {R[0].astype(int)} {R[1].astype(int)} {R[2].astype(int)}")
    if dom_pts is not None:
        print(f"[lay_flat] 最长焊缝={max_seam:.4f}(>半长{0.5*float(ext_mesh[i_long]):.4f}) "
              f"参与端点={len(dom_pts)} 选中焊缝 z 跨度={best[4]:.4f}")
    print(f"[lay_flat] 落地后质心高={-best[2]:.4f} 稳定={bool(best[0])}")
    print(f"[lay_flat] 摆平后包围盒(米): 长×宽×高 = "
          f"{(mx-mn)[0]:.4f} × {(mx-mn)[1]:.4f} × {(mx-mn)[2]:.4f}")
    return T


def ground_align_no_rotate(obj_fp) -> np.ndarray:
    """「不旋转」对照：R=单位阵，仅平移使工件最低点落到 z=0、xy 居中。返回 4×4 变换 T。"""
    import trimesh as _trimesh

    tm = _trimesh.load(obj_fp, force="mesh", process=False)
    V = np.asarray(tm.vertices, dtype=np.float64)
    mn = V.min(axis=0); mx = V.max(axis=0)
    t = np.array([-(mn[0] + mx[0]) / 2.0, -(mn[1] + mx[1]) / 2.0, -mn[2]])
    T = np.eye(4); T[:3, 3] = t
    return T


# ============================================================================
# ② 渲染：z=0 地面 + 坐标架 + 工件，斜上方固定视角/距离，离屏截图
# ============================================================================
def _build_scene(obj_fp, T):
    """返回 [ground, mesh, frame]（open3d 几何）与工件的世界系 AABB（用于摆相机）。"""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(obj_fp)
    if not mesh.has_triangles():
        return None, None
    mesh.transform(T)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.70, 0.75, 0.80])

    aabb = mesh.get_axis_aligned_bounding_box()
    ext_xy = aabb.get_extent()[:2]
    side = max(float(np.linalg.norm(ext_xy)) + 1.0, 2.0)
    th = 0.01
    ground = o3d.geometry.TriangleMesh.create_box(width=side, height=side, depth=th)
    ground.translate([-side / 2.0, -side / 2.0, -th])           # 顶面落在 z=0
    ground.compute_vertex_normals()
    ground.paint_uniform_color([0.85, 0.85, 0.85])

    # 坐标架大小随工件尺度，便于判方向/尺度
    diag = float(np.linalg.norm(aabb.get_extent()))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.3, 0.25 * diag))
    return [ground, mesh, frame], aabb


def _render(geoms, aabb, out_path, azim_deg, elev_deg, dist_factor, width, height):
    """把 geoms 用斜上方相机离屏渲染到 out_path。

    相机方向由方位角 azim（绕 +Z，从 +X 起逆时针）与仰角 elev（相对水平面抬高）确定；
    相机距离 = 工件包围盒对角线 × dist_factor，保证整体入画。

    用 open3d 的 OffscreenRenderer（EGL/filament）离屏渲染：无需窗口/显示器，
    在无头服务器上也能工作（老的 Visualizer.create_window 走 GLFW，无显示时会失败）。
    """
    import open3d as o3d
    from open3d.visualization import rendering

    center = np.asarray(aabb.get_center(), dtype=np.float64)
    diag = max(float(np.linalg.norm(aabb.get_extent())), 1e-3)

    az = np.deg2rad(azim_deg); el = np.deg2rad(elev_deg)
    # front = 由工件中心指向相机的方向；相机位置 eye = center + front * 距离
    front = np.array([np.cos(el) * np.cos(az),
                      np.cos(el) * np.sin(az),
                      np.sin(el)], dtype=np.float64)
    front = front / np.linalg.norm(front)
    eye = center + front * (diag * max(dist_factor, 0.5))

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.scene.set_sun_light([-0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 90000)
    renderer.scene.scene.enable_sun_light(True)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    for i, g in enumerate(geoms):
        renderer.scene.add_geometry(f"g{i}", g, mat)

    # 竖直视场角固定 60°；配合 eye 距离让整体入画
    renderer.setup_camera(60.0, center, eye, np.array([0.0, 0.0, 1.0]))

    img = renderer.render_to_image()
    o3d.io.write_image(out_path, img)
    del renderer
    return out_path


# ============================================================================
# ③ 批处理入口
# ============================================================================
def find_obj_files(root):
    """递归找 *_part_watertight.obj，按路径排序。"""
    pat = os.path.join(root, "**", "*_part_watertight.obj")
    return sorted(glob.glob(pat, recursive=True))


def main():
    ap = argparse.ArgumentParser(
        description="批量把工件平放到地面并从固定视角/距离拍两张对比图（平放 vs 不旋转）。")
    ap.add_argument("input_dir",
                    help="输入根文件夹，其下多个子文件夹各含一个 *_part_watertight.obj")
    ap.add_argument("--out-dir", default=None,
                    help="图片输出根目录；其下自动分 flat/ 与 orig/ 两个子目录。缺省 = 各 obj 同目录下的 flat/ orig/")
    ap.add_argument("--azim", type=float, default=45.0,
                    help="相机方位角(度)，绕 +Z 从 +X 起逆时针；默认 45")
    ap.add_argument("--elev", type=float, default=30.0,
                    help="相机仰角(度)，相对水平面抬高；默认 30（斜上方看，既见整体又见高度）")
    ap.add_argument("--dist-factor", type=float, default=2.0,
                    help="相机距离系数（越大越远/越小画面里工件越小）；默认 2.0")
    ap.add_argument("--width", type=int, default=1280, help="图片宽(px)")
    ap.add_argument("--height", type=int, default=960, help="图片高(px)")
    ap.add_argument("--only-flat", action="store_true",
                    help="只出平放图，不出「不旋转」对照图")
    ap.add_argument("--seam-ratio", type=float, default=0.9,
                    help="焊缝逻辑里「差不多长」的阈值：长度 ≥ ratio·最长焊缝 的那批参与定平面；默认 0.9")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[err] 输入不是文件夹: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    objs = find_obj_files(args.input_dir)
    if not objs:
        print(f"[err] 未找到任何 *_part_watertight.obj：{args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[batch] 找到 {len(objs)} 个工件")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    n_ok = 0
    for idx, obj_fp in enumerate(objs, 1):
        stem = os.path.splitext(os.path.basename(obj_fp))[0]
        # 输出名带上子文件夹名，避免不同子目录同名 obj 覆盖
        sub = os.path.basename(os.path.dirname(obj_fp))
        base = f"{sub}__{stem}" if sub else stem
        # flat 图 → <root>/flat/，orig 图 → <root>/orig/；root = --out-dir 或 obj 同目录
        root = args.out_dir or os.path.dirname(obj_fp)
        flat_dir = os.path.join(root, "flat")
        orig_dir = os.path.join(root, "orig")
        os.makedirs(flat_dir, exist_ok=True)
        os.makedirs(orig_dir, exist_ok=True)

        print(f"\n[batch] ({idx}/{len(objs)}) {obj_fp}")
        try:
            # ① 平放
            T_flat = lay_flat(obj_fp, viz=False, seam_ratio=args.seam_ratio)
            geoms, aabb = _build_scene(obj_fp, T_flat)
            if geoms is None:
                print(f"[warn] open3d 读不到网格，跳过: {obj_fp}")
                continue
            p_flat = os.path.join(flat_dir, f"{base}_flat.png")
            _render(geoms, aabb, p_flat, args.azim, args.elev,
                    args.dist_factor, args.width, args.height)
            print(f"[batch]   -> {p_flat}")

            # ② 不旋转对照
            if not args.only_flat:
                T_orig = ground_align_no_rotate(obj_fp)
                geoms2, aabb2 = _build_scene(obj_fp, T_orig)
                p_orig = os.path.join(orig_dir, f"{base}_orig.png")
                _render(geoms2, aabb2, p_orig, args.azim, args.elev,
                        args.dist_factor, args.width, args.height)
                print(f"[batch]   -> {p_orig}")

            n_ok += 1
        except Exception as e:
            print(f"[err] 处理失败 {obj_fp}: {e}", file=sys.stderr)

    print(f"\n[batch] 完成：{n_ok}/{len(objs)} 个工件成功出图")


if __name__ == "__main__":
    main()
