"""用 Open3D 一屏【平铺】对比各类障碍物「膨胀前 vs 膨胀后」，辅助决定 compute_goal_pose
注入碰撞世界时对障碍用哪种膨胀算法。

背景：STOMP 规划路径对障碍走 gt_gen.obstacle_placement.inflate_prims（基于原语 dims/radius
精确膨胀，Box 每边 +2·buffer、Tube 半径 +buffer）。但 compute_goal_pose 注入的是【trimesh】
（Scene._obstacle_solid_trimeshes：类型2 遮挡板只有多边形棱柱 mesh、无规则 dims），故需另择
一种作用于 mesh 的膨胀算法。本脚本把三种候选算法的结果并排画出来，肉眼确认哪种可用。

覆盖的障碍形态（尽量贴 compute_goal_pose 真实会遇到的 + 全景）：
  · 类型2 遮挡板：plate / triangle / trapezoid（薄棱柱 mesh，_shape_profile+_prism_mesh 造，决策焦点）
  · 类型3      ：open_box（5 块 Box 原语）
  · 类型1      ：gt_gen.obstacles.list_obstacles() 全部原语障碍（--no-type1 可关）

三种膨胀算法（--method）：
  · obb   ：每块 trimesh 的【有向包围盒(OBB)】各维 +2·buffer 重建为盒。plate/open_box 精确；
            triangle/trapezoid 近似成外接矩形盒（偏保守=更安全）。稳健、无自交风险。
  · normal：顶点沿【顶点法向】外移 buffer（mesh dilation）。保留 triangle/trapezoid 真实轮廓，
            但薄/凹结构顶点法向平均后可能轻微自交（cuRobo MESH SDF 一般能容忍）。
  · prims ：有原语(Box/Tube)的走 inflate_prims【精确】膨胀；类型2 板无原语 → 退回 obb 并标注。
  · all   ：默认。同格叠加真实(橙实心)+obb(红线框)+normal(蓝线框)+prims(绿线框)，一屏对比三种。

用法（须在装有 open3d/trimesh 的环境，如 env_isaaclab）：
    conda run -n env_isaaclab python scripts/viz_obstacle_inflate_o3d.py --method all --buffer 0.1
    conda run -n env_isaaclab python scripts/viz_obstacle_inflate_o3d.py --method obb --buffer 0.1
无显示器自检（不弹窗，打印每块膨胀前后 AABB 尺寸 + VIZ_INFLATE_DONE 即算跑通）：
    conda run -n env_isaaclab python scripts/viz_obstacle_inflate_o3d.py --headless
"""
import argparse
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# scene.py 顶层轻量（curobo/torch 惰性导入），直接复用其纯几何造型函数
from gt_gen.scene import (_shape_profile, _prism_mesh, _polygon_mesh_to_trimesh,
                          _box_prim_to_trimesh)  # noqa: E402
from gt_gen import obstacles as ob            # noqa: E402
from gt_gen import obstacle_placement as opl  # noqa: E402

# 颜色（RGB 0~1）
C_REAL = np.array([0.95, 0.55, 0.15])   # 真实尺寸（橙，实心）
C_OBB = np.array([0.85, 0.10, 0.10])    # obb 膨胀（红，线框）
C_NORMAL = np.array([0.15, 0.35, 0.90])  # normal 膨胀（蓝，线框）
C_PRIMS = np.array([0.10, 0.70, 0.20])  # prims 膨胀（绿，线框）
METHOD_COLOR = {"obb": C_OBB, "normal": C_NORMAL, "prims": C_PRIMS}

NCOL = 4          # 平铺网格列数
GAP = 0.35        # 格间间隙（米）


# --------------------------------------------------------------------------- 障碍 → trimesh
def _prim_to_trimesh(prim):
    """Box/Tube 原语 → 实体 trimesh（含 pose 变换）。"""
    import trimesh
    from scipy.spatial.transform import Rotation as Rsp
    pose = np.asarray(prim.pose, float)
    q = pose[3:7]                                    # wxyz
    T = np.eye(4)
    T[:3, :3] = Rsp.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = pose[:3]
    if isinstance(prim, ob.Box):
        return trimesh.creation.box(extents=np.asarray(prim.dims, float), transform=T)
    return trimesh.creation.cylinder(radius=float(prim.radius),
                                     height=float(prim.height), transform=T)


def _make_type2_plate(shape, width=0.30, length=0.30, thickness=0.02):
    """类型2 遮挡板（plate/triangle/trapezoid）→ 单块薄棱柱 trimesh。
    复用 scene._shape_profile + _prism_mesh，anchor=原点、R=单位阵（局部 X=法向,Y=宽,Z=长）。"""
    hw, hl = width / 2.0, length / 2.0
    rng = random.Random(0)
    profile = _shape_profile(shape, hw, hl, apex_sign=1.0, rng=rng)
    mesh = _prism_mesh(np.zeros(3), np.eye(3), profile, thickness, C_REAL.tolist())
    return _polygon_mesh_to_trimesh(mesh)


def build_obstacle_groups(include_type1: bool):
    """返回障碍组列表：每组 dict(name, real_tms=[trimesh...], prims=[原语...]或None)。"""
    groups = []

    # 类型2：遮挡板 3 种形状（决策焦点，无原语）
    for shape in ("plate", "triangle", "trapezoid"):
        tm = _make_type2_plate(shape)
        if tm is not None:
            groups.append(dict(name=f"type2_{shape}", real_tms=[tm], prims=None))

    # 类型3：open_box（5 块 Box 原语）
    prims_ob = ob.build("open_box", [0.0, 0.0, 0.0], size=(0.4, 0.5, 0.4),
                        wall=0.02, open_face="back")
    groups.append(dict(name="type3_open_box",
                       real_tms=[_prim_to_trimesh(p) for p in prims_ob], prims=prims_ob))

    # 类型1：全部原语障碍（全景）
    if include_type1:
        for name in ob.list_obstacles():
            prims = ob.build(name, [0.0, 0.0, 0.0])
            groups.append(dict(name=f"type1_{name}",
                               real_tms=[_prim_to_trimesh(p) for p in prims], prims=prims))
    return groups


# --------------------------------------------------------------------------- 三种膨胀算法
def inflate_obb(tm, buffer):
    """有向包围盒各维 +2·buffer 重建盒（复用 obstacle_placement 单一来源，与 compute_goal_pose 实现一致）。"""
    return opl.inflate_trimesh_obb(tm, buffer)


def inflate_normal(tm, buffer):
    """顶点沿顶点法向外移 buffer。"""
    out = tm.copy()
    out.vertices = np.asarray(out.vertices, float) + np.asarray(out.vertex_normals, float) * buffer
    return out


def inflate_group(group, method, buffer):
    """按 method 膨胀一组障碍，返回 (膨胀 trimesh 列表, note字符串)。
    method='prims' 且该组无原语时退回 obb 并给出 note。"""
    real_tms, prims = group["real_tms"], group["prims"]
    if method == "obb":
        return [inflate_obb(t, buffer) for t in real_tms], ""
    if method == "normal":
        return [inflate_normal(t, buffer) for t in real_tms], ""
    if method == "prims":
        if prims is None:
            return [inflate_obb(t, buffer) for t in real_tms], "无原语→退回obb"
        infl = opl.inflate_prims(prims, buffer)
        return [_prim_to_trimesh(p) for p in infl], ""
    raise ValueError(f"未知 method {method!r}")


# --------------------------------------------------------------------------- Open3D 渲染
def _o3d_mesh(tm, color, offset):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, float) + offset),
        o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
    m.compute_vertex_normals()
    m.paint_uniform_color(np.asarray(color, float))
    return m


def _o3d_wire(tm, color, offset):
    import open3d as o3d
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(_o3d_mesh(tm, color, offset))
    ls.paint_uniform_color(np.asarray(color, float))
    return ls


def _group_aabb(tms):
    v = np.concatenate([np.asarray(t.vertices, float) for t in tms], axis=0)
    return v.min(0), v.max(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["obb", "normal", "prims", "all"], default="all",
                    help="膨胀算法；all=同格叠加三种线框对比（默认）")
    ap.add_argument("--buffer", type=float, default=0.1, help="膨胀量(米)")
    ap.add_argument("--no-type1", action="store_true", help="不画类型1 原语障碍（只看类型2/3）")
    ap.add_argument("--headless", action="store_true", help="不弹窗，打印膨胀前后 AABB + DONE")
    args = ap.parse_args()

    methods = ["obb", "normal", "prims"] if args.method == "all" else [args.method]
    groups = build_obstacle_groups(include_type1=not args.no_type1)

    # 先算每组真实 AABB → 全局最大跨度 → 统一网格间距
    infos = []
    dmax = 0.0
    for g in groups:
        lo, hi = _group_aabb(g["real_tms"])
        ctr = (lo + hi) / 2.0
        span = float(np.max(hi - lo)) + 2.0 * args.buffer   # 含膨胀余量
        dmax = max(dmax, span)
        infos.append((g, ctr))
    pitch = dmax + GAP

    print(f"method={args.method}  buffer={args.buffer:.3f}m  障碍组={len(groups)}  "
          f"网格={NCOL}列  单元跨度≈{dmax:.2f}m")
    print("真实=橙实心；" + "，".join(
        f"{m}={'红' if m=='obb' else '蓝' if m=='normal' else '绿'}线框" for m in methods))

    geoms = []
    for i, (g, ctr) in enumerate(infos):
        col, row = i % NCOL, i // NCOL
        offset = np.array([col * pitch - ctr[0], -row * pitch - ctr[1], -ctr[2]], float)

        for t in g["real_tms"]:                              # 真实（橙实心）
            geoms.append(_o3d_mesh(t, C_REAL, offset))

        notes = []
        for m in methods:                                    # 各方法（线框叠加）
            infl_tms, note = inflate_group(g, m, args.buffer)
            for t in infl_tms:
                geoms.append(_o3d_wire(t, METHOD_COLOR[m], offset))
            lo, hi = _group_aabb(infl_tms)
            rlo, rhi = _group_aabb(g["real_tms"])
            d_real = (rhi - rlo)
            d_infl = (hi - lo)
            tag = f"{m}{'('+note+')' if note else ''}"
            notes.append(f"{tag}:AABB {np.round(d_real,3).tolist()}→{np.round(d_infl,3).tolist()}")
        print(f"  [{i:02d}] 行{row}列{col} {g['name']:<18} 块{len(g['real_tms'])}  " + " | ".join(notes))

    if args.headless:
        print(f"已构建 {len(geoms)} 个 geometry（真实实心 + 膨胀线框）。")
        print("VIZ_INFLATE_DONE")
        return

    import open3d as o3d
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    o3d.visualization.draw_geometries(
        geoms + [frame], window_name=f"障碍膨胀对比 method={args.method} buffer={args.buffer}",
        width=1400, height=900)


if __name__ == "__main__":
    main()
