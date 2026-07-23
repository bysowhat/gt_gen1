#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 USD 场景物体表面上找焊缝直线，并计算焊缝信息（与 ifc_analyzer/step3_vis.py 语义对齐）。

方案见 docs/observeanything/plan.md。仅用 pxr + trimesh + igl，不起 SimulationApp。
虚拟环境：env_isaaclab

流程：
  1. pxr 遍历所有 UsdGeom.Mesh → 世界系(米) trimesh，记录 prim 路径
  2. 【全场景合并成单个 mesh】：concatenate + merge_vertices（--watertight 时再走
       boolean union）。合并后接缝处的面互为相邻，能检出跨 prim 拼接处的焊缝。
  3. 抽特征棱线：相邻面【最小夹角】∈ [angle-min, angle-max]
       最小夹角 = min(夹角, 360-夹角) = π-法向偏差角（立方体棱=90°）
  4. 【不接链】：每条特征边独立作为一条焊缝候选；长度过滤 [min, max]（米）
  5. 每条候选边算焊缝信息(照搬 ifc)：
       两侧面方向 d0/d1 → ±d 贴面探测筛选 → 按绕边角度排序 →
       楔形按绕边角度枚举(含反射角>180°) → 双法向点积判内外 →
       取朝外(空)楔形：bisector 指向外部空间、gap_deg 为朝外张角(立方体棱=270°)
     内外判据：楔形朝外 ⟺ bisector 与两侧面法向点积均 >0（面法向朝材料外、winding
       一致）。取代旧的 fast_winding_number——场景多为钣金开壳，winding 两侧皆≈0
       无法区分，且大半径邻域会把缠绕数整体压成负值，导致朝里楔形被误留。
  6. 只保留朝外焊缝，输出 json（世界系、米）

用法：
  python scripts/find_surface_welds.py --usd scene.usd \
      --min 0.05 --max 2.0 --angle-min 30 --angle-max 150 [--watertight] \
      --crop-radius 0.5 [--crop-watertight] --out welds.json
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import igl
from tqdm import tqdm


def _bootstrap_pxr():
    """env_isaaclab 里 pxr 随 isaacsim 附带，需要动态库路径。找不到就设好路径重启一次解释器。"""
    try:
        import pxr  # noqa: F401
        return
    except Exception:
        pass
    if os.environ.get("_PXR_BOOTSTRAPPED"):
        raise ImportError("pxr 自举失败：已设路径仍无法导入")
    cands = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages",
                                   "isaacsim", "extscache", "omni.usd.libs*"))
    if not cands:
        raise ImportError("找不到 isaacsim 的 pxr（omni.usd.libs*），请确认在 env_isaaclab 内运行")
    pxr_root = cands[0]
    lib = os.pathsep.join([os.path.join(pxr_root, "bin"), os.path.join(sys.prefix, "lib")])
    os.environ["LD_LIBRARY_PATH"] = lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["PYTHONPATH"] = pxr_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["_PXR_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_pxr()
from pxr import Usd, UsdGeom  # noqa: E402


# ===== ifc_analyzer/step3_vis.py 对齐常量 =====
PROBE_DISTS        = np.linspace(0, 0.01, 10)   # 贴面探测：沿方向 0~1cm 取 10 点
ON_FACE_THRESH     = 0.001                       # 到面距离 <1mm 记为贴面
ON_FACE_MIN_COUNT  = 5                           # ≥5 点贴面才算该方向在面上
REGION_PROBE_DISTS = np.linspace(0, 0.2, 5)     # 楔形径向 5 距离
N_REGION_ANGULAR   = 50                           # 楔形角向 5 方向（共 25 采样点）
DEDUP_EPS          = 1e-3                         # 去重端点距离阈值(米)
SIMILAR_LEN_TOL    = 0.05                         # 相近焊缝：长度相对差 ≤5%
SIMILAR_DIR_TOL    = 0.05                         # 相近焊缝：方向差 1-|cos| ≤5%
SIMILAR_DIST_TOL   = 1.0                          # 相近焊缝：中点距离 ≤1m 才算同一条


# ===== 分阶段计时（验证瓶颈用）=====
_T = defaultdict(float)   # 阶段累计耗时(秒)
_C = defaultdict(int)     # 阶段调用次数


class _Timer:
    """with _Timer('winding'): ... 累加到 _T/_C。"""
    __slots__ = ("key", "_t")

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        _T[self.key] += time.perf_counter() - self._t
        _C[self.key] += 1


# ---------------------------------------------------------------------------
# 1. 加载 USD → 世界系(米) trimesh
# ---------------------------------------------------------------------------
def load_scene_meshes(usd_path, verbose=True):
    """遍历 stage 所有 Mesh，烘焙世界变换并换算到米，扇形三角化 → [(prim_path, Trimesh)]。"""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"无法打开 USD: {usd_path}")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue

        # 世界变换（USD 行向量约定：world = local_row @ M）
        T = xc.GetLocalToWorldTransform(prim)
        M = np.array([[T[i][j] for j in range(4)] for i in range(4)], float)
        local = np.array([[p[0], p[1], p[2]] for p in pts], float)
        homo = np.c_[local, np.ones(len(local))]
        world = (homo @ M)[:, :3] * mpu

        # 多边形扇形三角化
        tris, off = [], 0
        for c in counts:
            c = int(c)
            face = [int(idx[off + k]) for k in range(c)]
            off += c
            for k in range(1, c - 1):
                tris.append([face[0], face[k], face[k + 1]])
        if not tris:
            continue
        try:
            tm = trimesh.Trimesh(vertices=world, faces=np.asarray(tris, np.int64),
                                 process=False)
        except Exception:
            continue
        if len(tm.faces) == 0:
            continue
        meshes.append((prim.GetPath().pathString, tm))

    if verbose:
        nf = sum(len(t.faces) for _, t in meshes)
        print(f"[load] {len(meshes)} 个 Mesh prim，共 {nf} 面，metersPerUnit={mpu}")
    return meshes


# ---------------------------------------------------------------------------
# 2. 特征棱线（逐 prim）
# ---------------------------------------------------------------------------
def feature_edges(tm, ang_min_rad, ang_max_rad):
    """返回【最小夹角】∈ [min,max] 的相邻面对：(face_pairs, edge_vids, convex)。

    最小夹角 = 两面所夹的较小张角 = min(夹角, 360°-夹角) = π - 法向偏差角。
    （trimesh 的 face_adjacency_angles 是法向偏差角 α∈[0,π]：0=共面、π=对折；实体侧
      真实二面角 = π-α(凸) 或 π+α(凹)，两者取小恒为 π-α，即此处的最小夹角。
      故立方体棱最小夹角=90°；对应输出楔形朝外 gap_deg=270°，见 enumerate_wedges。）
    """
    adj = tm.face_adjacency
    if len(adj) == 0:
        return (np.empty((0, 2), int), np.empty((0, 2), int), np.empty((0,), bool))
    angs = tm.face_adjacency_angles           # α：相邻面法向偏差角 ∈[0,π]
    # True 代表凸(Convex) False代表凹(Concave)
    convexs = tm.face_adjacency_convex
    edges = tm.face_adjacency_edges
    incl = np.pi - angs                        # 最小夹角(π-α)∈[0,π]
    mask = (incl >= ang_min_rad) & (incl <= ang_max_rad)
    return adj[mask], edges[mask], convexs[mask]

def vis_3d(vertices, edge_vids):
    import open3d as o3d
    import numpy as np
    import random


    # 1. 准备你的数据（替换为你的实际变量）
    # vertices = ... (形状为 [N, 3] 的 NumPy 数组)
    # edge_vids = ... (形状为 [M, 2] 的 NumPy 数组)

    # # 2. 创建 LineSet 对象
    # line_set = o3d.geometry.LineSet()

    # # 3. 赋值顶点和边
    # # open3d 需要用 o3d.utility.Vector3dVector 和 Vector2iVector 包装 numpy 数组
    # line_set.points = o3d.utility.Vector3dVector(vertices)
    # line_set.lines = o3d.utility.Vector2iVector(edge_vids)

    # # 4. 可选：给所有的边设置颜色（例如：红色）
    # colors = np.array([[1, 0, 0] for _ in range(len(edge_vids))])
    # line_set.colors = o3d.utility.Vector3dVector(colors)

    # # 5. 弹出窗口可视化
    # o3d.visualization.draw_geometries([line_set])


    num_edges = len(edge_vids)

    # 2. 构建顶点到边的映射（用于快速寻找共享顶点的相邻边）
    v_to_edges = {}
    for edge_idx, (v1, v2) in enumerate(edge_vids):
        v_to_edges.setdefault(v1, []).append(edge_idx)
        v_to_edges.setdefault(v2, []).append(edge_idx)

    # 3. 贪心算法为边着色：确保共享顶点的边颜色编号不同
    edge_colors_idx = np.full(num_edges, -1, dtype=int)

    for i in range(num_edges):
        v1, v2 = edge_vids[i]
        # 找出所有与当前边相邻的边的颜色
        neighbor_edges = set(v_to_edges[v1] + v_to_edges[v2])
        neighbor_colors = {edge_colors_idx[ne] for ne in neighbor_edges if edge_colors_idx[ne] != -1}
        
        # 分配最小可用的颜色编号
        color = 0
        while color in neighbor_colors:
            color += 1
        edge_colors_idx[i] = color

    # 4. 为每个颜色编号生成随机的 RGB 颜色
    max_color_idx = edge_colors_idx.max()
    unique_rgb = []
    for _ in range(max_color_idx + 1):
        # 生成随机 RGB 值，范围在 [0, 1] 之间
        unique_rgb.append([random.random(), random.random(), random.random()])
    unique_rgb = np.array(unique_rgb)

    # 映射得到最终的边颜色矩阵
    edge_colors = unique_rgb[edge_colors_idx]

    # 5. 装载到 Open3D 中并可视化
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edge_vids)
    line_set.colors = o3d.utility.Vector3dVector(edge_colors)

    o3d.visualization.draw_geometries([line_set])


# ---------------------------------------------------------------------------
# 5. 焊缝信息（照搬 step3_vis.py 语义）
# ---------------------------------------------------------------------------
def probe_on_face_batch(edge_mid, directions, pq):
    """一次判定多个方向是否贴在 mesh 面上（复用 ProximityQuery，批量查询）。

    与逐个 probe_on_face 数值等价：每个方向沿 PROBE_DISTS 取 10 点，到面 <ON_FACE_THRESH
    的点数 ≥ ON_FACE_MIN_COUNT 记为贴面。directions: list[unit vec]，返回等长 list[bool]。
    """
    if len(directions) == 0:
        return []
    D = np.asarray(directions, float)                       # (K,3)
    em = np.asarray(edge_mid, float)
    pts = em[None, None, :] + D[:, None, :] * PROBE_DISTS[None, :, None]   # (K,10,3)
    flat = pts.reshape(-1, 3)
    with _Timer("probe.closest_point"):
        _, dist, _ = pq.on_surface(flat)
    dist = np.asarray(dist).reshape(len(directions), len(PROBE_DISTS))
    cnt = np.sum(dist < ON_FACE_THRESH, axis=1)
    return [int(c) >= ON_FACE_MIN_COUNT for c in cnt]


def two_face_dirs(fi, fj, tm, edge_mid, edge_dir):
    """两侧面各算面内⊥边、朝面内的单位方向 d 及面法向 n。"""
    dirs, normals = [], []
    for f in (fi, fj):
        fn = np.asarray(tm.face_normals[f], float)
        fc = np.asarray(tm.triangles_center[f], float)
        d = np.cross(fn, edge_dir)
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        if float(np.dot(d, fc - edge_mid)) < 0:
            d = -d
        dirs.append(d)
        normals.append(fn.copy())
    return dirs, normals


def sort_dirs_by_angle(dirs, edge_dir, normals):
    """按绕 edge_dir 的角度排序方向，返回 (sorted_dirs, sorted_angles, sorted_normals, u, w)。

    (u, w) 是绕边平面的正交基：u=dirs[0]，w=normalize(edge_dir×u)；任一方向可写作
    cos(θ)·u+sin(θ)·w，θ 即其绕边角度(与返回的 angles 同参考系)。供 enumerate_wedges
    按角度参数化重建方向、正确采样反射角(>180°)楔形。
    """
    if len(dirs) < 2:
        return dirs, [0.0] * len(dirs), normals, None, None
    u = np.asarray(dirs[0], float)
    w = np.cross(edge_dir, u)
    wn = np.linalg.norm(w)
    if wn > 1e-9:
        w = w / wn
    angles = [np.arctan2(float(np.dot(d, w)), float(np.dot(d, u))) for d in dirs]
    order = np.argsort(angles)
    return ([dirs[i] for i in order],
            [angles[i] for i in order],
            [normals[i] for i in order],
            u, w)


def enumerate_wedges(edge_mid, sangles, sdirs, snorms, u, w):
    """列出绕边一圈的所有楔形（含反射角>180°）的几何 + 25 点采样 Q。

    相邻方向对 (i, i+1) 张成一个楔形，全部相加恰覆盖 360°：两个互补楔形都列出，
    由后续 winding 判定留下朝外(空)的那个 → 其 bisector 天然指向外部空间、
    gap_deg 为朝外张角（凸边如立方体棱 = 270°）。

    方向按绕边角度参数化 dir(θ)=cosθ·u+sinθ·w，故反射角楔形沿【大弧】正确采样；
    bisector 取楔形中间角度方向（不再用 d0+d1，避免反射角指反 / 对折退化）。
    返回 [ {bisector, gap_deg, boundary_dirs, boundary_normals, Q(np (M,3))} ]。
    """
    N = len(sdirs)
    wedges = []
    if N < 2 or u is None or w is None:
        return wedges
    u = np.asarray(u, float)
    w = np.asarray(w, float)

    def dir_at(theta):
        return np.cos(theta) * u + np.sin(theta) * w

    for i in range(N):
        j = (i + 1) % N
        gap = sangles[j] - sangles[i]
        if j <= i:
            gap += 2 * np.pi
        if gap <= 1e-6 or gap >= 2 * np.pi - 1e-6:   # 退化/整圈，跳过
            continue

        # 平分线：楔形中间角度方向（指向楔形内部；朝外楔形即指向外部空间）
        b = dir_at(sangles[i] + gap / 2.0)
        bn = np.linalg.norm(b)
        if bn < 1e-9:
            continue
        b = b / bn

        # 楔形扇形采样 25 点：角度从 angle_i 扫到 angle_j（反射角走大弧）
        Q = []
        for ai in range(N_REGION_ANGULAR):
            t = (ai + 0.5) / N_REGION_ANGULAR
            di = dir_at(sangles[i] + t * gap)
            for r in REGION_PROBE_DISTS:
                Q.append(edge_mid + di * r)
        if not Q:
            continue
        wedges.append({
            "bisector": b.tolist(),
            "gap_deg": round(float(np.degrees(gap)), 1),
            "boundary_dirs": [np.asarray(sdirs[i], float).tolist(),
                              np.asarray(sdirs[j], float).tolist()],
            "boundary_normals": [np.asarray(snorms[i], float).tolist(),
                                 np.asarray(snorms[j], float).tolist()],
            "Q": np.asarray(Q, np.float64),
        })
    return wedges


# ---------------------------------------------------------------------------
# 7. 全场景合并成单个 mesh（在合并体外表面找焊缝，避免被别的 prim 遮挡）
# ---------------------------------------------------------------------------
def merge_scene_mesh(meshes, watertight=False, verbose=True):
    """把所有 per-prim trimesh 合并成【一个】 Trimesh 后返回。

    · 默认 concatenate + merge_vertices：焊合重合顶点，使跨 prim 接缝的面互为
      face_adjacency 相邻（能检出拼接处焊缝）；毫秒级、稳。overlap 的内部面仍在，
      靠后续 winding 把朝内楔形剔掉。
    · watertight=True：再走 make_watertight 口径的 boolean union（manifold 引擎）
      成单一 2-流形外表面，真正删除内部/互相嵌入的面 —— 被遮挡的棱彻底消失。
      整场景并集较慢、可能失败，失败则回退到 concatenate 结果。
    """
    tms = [t for _, t in meshes]
    merged = trimesh.util.concatenate(tms)
    merged.merge_vertices()

    if watertight:
        wt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "watertight")
        if wt_dir not in sys.path:
            sys.path.insert(0, wt_dir)
        try:
            import make_watertight as _mw  # noqa: E402
            solids = _mw.components_as_solids(merged)
            union = trimesh.boolean.union(solids, engine="manifold")
            union = _mw.drop_degenerate(union)
            union.merge_vertices()
            merged = union
        except Exception as e:
            if verbose:
                print(f"[merge] boolean union 失败({e})，回退 concatenate")

    if verbose:
        mode = "union(watertight)" if watertight else "concatenate"
        print(f"[merge] {len(tms)} 个 prim → 单 mesh（{mode}）："
              f"{len(merged.vertices)}v/{len(merged.faces)}f")
    return merged


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _aabb_dist_to_point(lo, hi, c):
    """点 c 到轴对齐包围盒 [lo,hi] 的最近距离（点在盒内则 0）。用于「球心 c 半径 R 内整块保留 prim」判定。"""
    d = np.maximum(np.maximum(lo - c, c - hi), 0.0)
    return float(np.linalg.norm(d))


def _neighborhood_mesh(meshes, prim_idxs, crop_watertight=False):
    """把给定 prim 下标集合的 per-prim mesh 拼接成单块，返回 (V, F)。仿 _crop_neighborhood_obj：
    默认 concatenate（每壳闭合、winding 够用）；crop_watertight=True 时走布尔并集成单一 2-流形，
    失败回退拼接。用于 winding 内外判定的「该边邻域局部实体」。"""
    kept = [meshes[i][1] for i in prim_idxs]
    merged = trimesh.util.concatenate(kept)
    if crop_watertight:
        wt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "watertight")
        if wt_dir not in sys.path:
            sys.path.insert(0, wt_dir)
        try:
            import make_watertight as _mw  # noqa: E402
            solids = _mw.components_as_solids(merged)
            union = trimesh.boolean.union(solids, engine="manifold")
            union = _mw.drop_degenerate(union)
            union.merge_vertices()
            merged = union
        except Exception:
            pass
    return (np.asarray(merged.vertices, np.float64),
            np.asarray(merged.faces, np.int64))


def filter_similar_welds(welds, len_tol=SIMILAR_LEN_TOL, dir_tol=SIMILAR_DIR_TOL,
                         dist_tol=SIMILAR_DIST_TOL):
    """全局过滤相近焊缝：长度相对差 ≤len_tol、方向差(1-|cos|) ≤dir_tol
    且中点距离 ≤dist_tol(米) 三者同时满足时，只保留先出现的 1 条（保持原始顺序）。
    （合并成单 mesh 后无 per-prim 概念，故全局去重；再叠加空间邻近约束，
    避免场景中互相远离但恰好等长同向的两条焊缝被误并。）
    """
    reps = []          # [(length, dir_np, mid_np)]
    kept = []
    for w in welds:
        L = float(w["length"])
        d = np.asarray(w["edge_dir"], float)
        mid = (np.asarray(w["corrected_p0"], float) +
               np.asarray(w["corrected_p1"], float)) * 0.5
        dup = False
        for rL, rd, rmid in reps:
            len_diff = abs(L - rL) / max(L, rL, 1e-9)
            dir_diff = 1.0 - abs(float(np.dot(d, rd)))
            dist = float(np.linalg.norm(mid - rmid))
            if len_diff <= len_tol and dir_diff <= dir_tol and dist <= dist_tol:
                dup = True
                break
        if not dup:
            reps.append((L, d, mid))
            kept.append(w)
    return kept


def _wedge_outward_score(bisector, boundary_normals):
    """楔形朝外判据（取代 fast_winding_number 内外判定）。

    返回 (is_outward, score)：
      is_outward = bisector 与【所有】边界面法向点积均 >0，即 bisector 落在各面
                   的材料外侧（面法向 winding 一致且朝材料外，见 winding_consistent）；
      score      = 点积之和，供无严格朝外楔形时兜底 argmax。
    对薄壳/非水密同样成立——场景多为钣金开壳，fast_winding_number 两侧皆≈0 无法区分，
    而面法向方向明确可靠。
    """
    b = np.asarray(bisector, float)
    s = [float(np.dot(b, np.asarray(n, float))) for n in boundary_normals]
    return (all(x > 1e-3 for x in s), float(np.sum(s)))


def find_welds(usd_path, lmin, lmax, ang_min_deg, ang_max_deg,
               watertight=False, crop_radius=0.5, crop_watertight=False,
               similar_len_tol=SIMILAR_LEN_TOL, similar_dir_tol=SIMILAR_DIR_TOL,
               similar_dist_tol=SIMILAR_DIST_TOL, debug_weld=-1, verbose=True):
    t0 = time.time()
    meshes = load_scene_meshes(usd_path, verbose)
    if not meshes:
        return []

    # 全场景合并成单个 mesh：抽特征边（跨 prim 接缝相邻）+ 贴面探测
    tm_all = merge_scene_mesh(meshes, watertight=watertight, verbose=verbose)
    if len(tm_all.faces) == 0:
        return []

    amin, amax = np.radians(ang_min_deg), np.radians(ang_max_deg)
    welds = []

    # ---- Pass A：逐特征边（每条边即一条候选，不接链）收集楔形 ----
    cands = []                                   # 候选，保持原始迭代顺序（去重语义依赖）
    with _Timer("feature_edges"):
        pairs, evids, convexs = feature_edges(tm_all, amin, amax)
    if len(pairs) == 0:
        return []
    pq = trimesh.proximity.ProximityQuery(tm_all)   # 整 mesh 复用一次（贴面探测）

    for ei in tqdm(range(len(pairs))):
        fi, fj = (int(x) for x in pairs[ei])
        a, b = (int(x) for x in evids[ei])
        p0 = np.asarray(tm_all.vertices[a], float)
        p1 = np.asarray(tm_all.vertices[b], float)
        length = float(np.linalg.norm(p1 - p0))
        if length < lmin or length > lmax:
            continue

        edge_mid = (p0 + p1) / 2.0
        ev = p1 - p0
        en = np.linalg.norm(ev)
        if en < 1e-9:
            continue
        edge_dir = ev / en

        dirs, normals = two_face_dirs(fi, fj, tm_all, edge_mid, edge_dir)
        if len(dirs) < 2:
            continue

        # ±d 贴面探测筛选（用合并 mesh），顺序 d0,-d0,d1,-d1 与原实现一致
        cand_dirs = [np.asarray(dirs[0], float), -np.asarray(dirs[0], float),
                     np.asarray(dirs[1], float), -np.asarray(dirs[1], float)]
        cand_nrm = [normals[0], normals[0], normals[1], normals[1]]
        flags = probe_on_face_batch(edge_mid, cand_dirs, pq)
        kept_d = [cand_dirs[i] for i, f in enumerate(flags) if f]
        kept_n = [cand_nrm[i] for i, f in enumerate(flags) if f]
        if len(kept_d) < 2:
            continue

        sdirs, sangles, snorms, ubas, wbas = sort_dirs_by_angle(kept_d, edge_dir, kept_n)
        wedges = enumerate_wedges(edge_mid, sangles, sdirs, snorms, ubas, wbas)
        for w in wedges:
            w.pop("Q", None)                     # 不再用 winding 采样点
        if not wedges:
            continue

        cands.append({"p0": p0, "p1": p1, "length": length, "wedges": wedges})

    # ---- Pass B：原顺序去重(空间哈希) + 双法向点积判朝外 + 组装 ----
    def _cell(p):
        return (int(np.floor(p[0] / DEDUP_EPS)),
                int(np.floor(p[1] / DEDUP_EPS)),
                int(np.floor(p[2] / DEDUP_EPS)))

    def _neigh(c):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield (c[0] + dx, c[1] + dy, c[2] + dz)

    bucket = defaultdict(list)                   # cell -> [seen 下标]
    seen = []                                    # [(p0,p1)]，注册后端点入 bucket

    for c in cands:
        p0, p1 = c["p0"], c["p1"]
        # 收集近邻桶里的 seen 候选，再做精确 allclose（与原 O(n²) 语义等价）
        cand_idx = set()
        for cc in _neigh(_cell(p0)):
            cand_idx.update(bucket.get(cc, ()))
        for cc in _neigh(_cell(p1)):
            cand_idx.update(bucket.get(cc, ()))
        with _Timer("dedup"):
            dup = False
            for si in cand_idx:
                q0, q1 = seen[si]
                if ((np.allclose(p0, q0, atol=DEDUP_EPS) and np.allclose(p1, q1, atol=DEDUP_EPS)) or
                        (np.allclose(p0, q1, atol=DEDUP_EPS) and np.allclose(p1, q0, atol=DEDUP_EPS))):
                    dup = True
                    break
        if dup:
            continue

        # 双法向点积判据：楔形朝外 ⟺ bisector 落在两侧面法向的正侧（材料外）
        scored = [(is_out, s, w) for (is_out, s), w in
                  ((_wedge_outward_score(w["bisector"], w["boundary_normals"]), w)
                   for w in c["wedges"])]
        out_wedges = [(s, w) for is_out, s, w in scored if is_out]
        if not out_wedges:
            # 无严格朝外楔形（非正交/退化折角）→ 兜底取点积和最大的一个
            _, s, w = max(scored, key=lambda t: t[1])
            out_wedges = [(s, w)]

        idx = len(seen)
        seen.append((p0, p1))
        bucket[_cell(p0)].append(idx)
        bucket[_cell(p1)].append(idx)
        overall = p1 - p0
        overall = (overall / np.linalg.norm(overall)).tolist()
        for score, w in out_wedges:
            wd = {
                "prim_path": "merged",
                "corrected_p0": p0.tolist(),
                "corrected_p1": p1.tolist(),
                "length": round(c["length"], 6),
                "edge_dir": overall,
                "bisector": w["bisector"],
                "gap_deg": w["gap_deg"],
                "normal_score": round(float(score), 4),
                "is_inside": False,
                "boundary_dirs": w["boundary_dirs"],
                "boundary_normals": w["boundary_normals"],
            }
            if debug_weld >= 0:
                dbg = []
                for is_out, s, ww in scored:
                    b = np.asarray(ww["bisector"], float)
                    dbg.append({
                        "gap_deg": ww["gap_deg"],
                        "bisector": [round(x, 4) for x in ww["bisector"]],
                        "dots": [round(float(np.dot(b, np.asarray(n, float))), 4)
                                 for n in ww["boundary_normals"]],
                        "score": round(float(s), 4),
                        "outward": bool(is_out),
                    })
                wd["_dbg"] = {
                    "edge_mid": [round(x, 4) for x in ((p0 + p1) / 2.0).tolist()],
                    "n_wedges": len(scored),
                    "wedges": dbg,
                }
            welds.append(wd)

    with _Timer("filter_similar"):
        n_before = len(welds)
        welds = filter_similar_welds(welds, similar_len_tol, similar_dir_tol,
                                     similar_dist_tol)
    if verbose:
        print(f"[filter] 相近焊缝去重(全局, 长度≤{similar_len_tol:.0%} 且 "
              f"方向≤{similar_dir_tol:.0%} 且 中点≤{similar_dist_tol}m): "
              f"{n_before} -> {len(welds)}")

    # ---- 单条焊缝诊断（--debug-weld）：按最终下标打印，随后剥掉 _dbg 再返回 ----
    if debug_weld >= 0:
        if 0 <= debug_weld < len(welds):
            d = welds[debug_weld].get("_dbg", {})
            print(f"\n[debug-weld {debug_weld}] ==============================")
            print(f"  edge_mid={d.get('edge_mid')}  length={welds[debug_weld]['length']}")
            print(f"  该边共 {d.get('n_wedges')} 个楔形（判据: bisector 与两面法向点积均>0=朝外）:")
            for i, wg in enumerate(d.get("wedges", [])):
                tag = "★保留(朝外)" if wg["outward"] else "  丢弃(朝里)"
                print(f"   [{tag}] 楔形{i} gap={wg['gap_deg']}° 点积={wg['dots']} "
                      f"和={wg['score']} bisector={wg['bisector']}")
            print("[debug-weld] ==============================\n")
        else:
            print(f"[debug-weld] 下标 {debug_weld} 越界（共 {len(welds)} 条）")
        for w in welds:
            w.pop("_dbg", None)

    if verbose:
        print(f"[done] {len(welds)} 条朝外焊缝，用时 {time.time() - t0:.1f}s")
        if _T:
            print("[timing] 各阶段累计耗时（占比 / 调用次数）:")
            total = time.time() - t0
            for k in sorted(_T, key=lambda x: -_T[x]):
                print(f"  {k:24s} {_T[k]:8.1f}s  {_T[k]/total*100:5.1f}%  x{_C[k]}")
    return welds


def main():
    ap = argparse.ArgumentParser(description="在 USD 场景物体表面找焊缝直线并算焊缝信息")
    ap.add_argument("--usd", required=True, help="USD 场景路径")
    ap.add_argument("--min", type=float, required=True, dest="lmin", help="最短焊缝长度(米)")
    ap.add_argument("--max", type=float, required=True, dest="lmax", help="最长焊缝长度(米)")
    ap.add_argument("--angle-min", type=float, default=30.0,
                    help="最小夹角下限(度)：两面较小张角 min(夹角,360-夹角)=π-法向偏差角")
    ap.add_argument("--angle-max", type=float, default=150.0,
                    help="最小夹角上限(度)")
    ap.add_argument("--watertight", action="store_true",
                    help="合并整场景(特征检测用)时走 boolean union 成单一 2-流形（较慢/可能失败回退）；"
                         "默认仅 concatenate+merge_vertices")
    ap.add_argument("--crop-radius", type=float, default=0.5,
                    help="[已弃用] 旧 winding 判定的邻域半径；现用双法向点积判内外，此参数不再生效")
    ap.add_argument("--crop-watertight", action="store_true",
                    help="[已弃用] 旧 winding 邻域布尔并集开关；现用双法向点积判内外，不再生效")
    ap.add_argument("--similar-len-tol", type=float, default=SIMILAR_LEN_TOL,
                    help="相近焊缝去重：长度相对差阈值(≤此值算相近)，默认 0.05")
    ap.add_argument("--similar-dir-tol", type=float, default=SIMILAR_DIR_TOL,
                    help="相近焊缝去重：方向差 1-|cos| 阈值(≤此值算相近)，默认 0.05")
    ap.add_argument("--similar-dist-tol", type=float, default=SIMILAR_DIST_TOL,
                    help="相近焊缝去重：中点距离阈值(米)(≤此值才算同一条)，默认 1.0")
    ap.add_argument("--debug-weld", type=int, default=-1,
                    help="诊断单条焊缝：打印其最终下标对应边所有楔形的 bisector 与两侧面法向"
                         "的点积及朝外判定（点积均>0=朝外），用于核对 bisector 朝向")
    ap.add_argument("--out", required=True, help="输出 json 路径")
    args = ap.parse_args()

    welds = find_welds(args.usd, args.lmin, args.lmax,
                       args.angle_min, args.angle_max,
                       watertight=args.watertight,
                       crop_radius=args.crop_radius,
                       crop_watertight=args.crop_watertight,
                       similar_len_tol=args.similar_len_tol,
                       similar_dir_tol=args.similar_dir_tol,
                       similar_dist_tol=args.similar_dist_tol,
                       debug_weld=args.debug_weld)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(welds, f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.out}（{len(welds)} 条）")


if __name__ == "__main__":
    main()
