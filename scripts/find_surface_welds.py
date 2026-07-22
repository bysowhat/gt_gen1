#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 USD 场景物体表面上找焊缝直线，并计算焊缝信息（与 ifc_analyzer/step3_vis.py 语义对齐）。

方案见 docs/observeanything/plan.md。仅用 pxr + trimesh + igl，不起 SimulationApp。
虚拟环境：env_isaaclab

流程：
  1. pxr 遍历所有 UsdGeom.Mesh → 世界系(米) trimesh，记录 prim 路径
  2. 逐 prim 抽特征棱线：相邻面二面角偏差 ∈ [angle-min, angle-max]
  3. 首尾相连且共线(容差 5°)的特征边接成直段
  4. 长度过滤 [min, max]（米）
  5. 每条直段算焊缝信息(照搬 ifc)：
       两侧面方向 d0/d1 → ±d 贴面探测筛选 → 按绕边角度排序 →
       楔形 winding 内外判定 → 取朝外(OUT)楔形的 bisector/gap_deg/boundary_*
     winding 用【全场景合并 mesh】(可按 --nwind 分 N 组)
  6. 只保留朝外焊缝，输出 json（世界系、米）

用法：
  python scripts/find_surface_welds.py --usd scene.usd \
      --min 0.05 --max 2.0 --angle-min 30 --angle-max 150 --nwind 1 \
      --out welds.json
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
REGION_PROBE_DISTS = np.linspace(0, 0.01, 5)     # 楔形径向 5 距离
N_REGION_ANGULAR   = 5                           # 楔形角向 5 方向（共 25 采样点）
COLLINEAR_TOL_DEG  = 5.0                          # 接链共线容差
DEDUP_EPS          = 1e-3                         # 去重端点距离阈值(米)


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
    """返回二面角偏差 ∈ [min,max] 的相邻面对：(face_pairs Nx2, edge_vids Nx2)。"""
    adj = tm.face_adjacency
    if len(adj) == 0:
        return np.empty((0, 2), int), np.empty((0, 2), int)
    angs = tm.face_adjacency_angles
    edges = tm.face_adjacency_edges
    mask = (angs >= ang_min_rad) & (angs <= ang_max_rad)
    return adj[mask], edges[mask]


# ---------------------------------------------------------------------------
# 3. 接链成直段
# ---------------------------------------------------------------------------
def build_straight_chains(vertices, edge_vids, tol_deg):
    """把首尾相连且共线(<tol)的特征边接成直段。返回 [{'vids':[...],'eids':[...]}]。"""
    cos_tol = np.cos(np.radians(tol_deg))
    segs = [(int(a), int(b)) for a, b in edge_vids]
    v2e = defaultdict(list)
    for ei, (a, b) in enumerate(segs):
        v2e[a].append(ei)
        v2e[b].append(ei)
    used = [False] * len(segs)

    def unit(a, b):
        d = vertices[b] - vertices[a]
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else d

    def extend(end, cur_dir):
        """从 end 顶点沿 cur_dir 方向贪心延伸，返回新增顶点(有序)与用掉的边。"""
        added_v, added_e = [], []
        while True:
            nxt = None
            for ej in v2e[end]:
                if used[ej]:
                    continue
                x, y = segs[ej]
                other = y if x == end else (x if y == end else None)
                if other is None:
                    continue
                nd = unit(end, other)
                if float(np.dot(nd, cur_dir)) >= cos_tol:
                    nxt = (ej, other, nd)
                    break
            if nxt is None:
                break
            ej, other, nd = nxt
            used[ej] = True
            added_v.append(other)
            added_e.append(ej)
            end, cur_dir = other, nd
        return added_v, added_e

    chains = []
    for ei in range(len(segs)):
        if used[ei]:
            continue
        a, b = segs[ei]
        used[ei] = True
        fv, fe = extend(b, unit(a, b))          # 前向
        bv, be = extend(a, unit(b, a))          # 后向
        vids = list(reversed(bv)) + [a, b] + fv
        eids = list(reversed(be)) + [ei] + fe
        chains.append({"vids": vids, "eids": eids})
    return chains


# ---------------------------------------------------------------------------
# 5. 焊缝信息（照搬 step3_vis.py 语义）
# ---------------------------------------------------------------------------
def probe_on_face(edge_mid, direction, mesh):
    """从 edge_mid 沿 direction 取点，判断是否贴在 mesh 面上。"""
    pts = np.array([edge_mid + direction * d for d in PROBE_DISTS])
    _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
    return int(np.sum(dist < ON_FACE_THRESH)) >= ON_FACE_MIN_COUNT


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
    """按绕 edge_dir 的角度排序方向，返回 (sorted_dirs, sorted_angles, sorted_normals)。"""
    if len(dirs) < 2:
        return dirs, [0.0] * len(dirs), normals
    u = np.asarray(dirs[0], float)
    w = np.cross(edge_dir, u)
    wn = np.linalg.norm(w)
    if wn > 1e-9:
        w = w / wn
    angles = [np.arctan2(float(np.dot(d, w)), float(np.dot(d, u))) for d in dirs]
    order = np.argsort(angles)
    return ([dirs[i] for i in order],
            [angles[i] for i in order],
            [normals[i] for i in order])


def classify_out_regions(edge_mid, sdirs, sangles, snorms, Vw, Fw):
    """相邻方向围成的楔形，winding 判内外，只返回朝外(OUT)区域的焊缝量。"""
    N = len(sdirs)
    out = []
    for i in range(N):
        j = (i + 1) % N
        d0, d1 = np.asarray(sdirs[i], float), np.asarray(sdirs[j], float)
        gap = sangles[j] - sangles[i]
        if j <= i:
            gap += 2 * np.pi
        if gap < 0:
            gap += 2 * np.pi
        if gap > np.pi:                          # 反射角侧，跳过
            continue
        b = d0 + d1
        bn = np.linalg.norm(b)
        if bn < 1e-9:
            continue
        b = b / bn

        # 楔形扇形采样 25 点
        Q = []
        for ai in range(N_REGION_ANGULAR):
            t = (ai + 0.5) / N_REGION_ANGULAR
            di = (1 - t) * d0 + t * d1
            n = np.linalg.norm(di)
            if n < 1e-9:
                continue
            di = di / n
            for r in REGION_PROBE_DISTS:
                Q.append(edge_mid + di * r)
        if not Q:
            continue
        wn = igl.fast_winding_number(Vw, Fw, np.asarray(Q, np.float64))
        ratio = float(np.sum(wn > 0.5)) / len(Q)
        if ratio > 0.5:                          # inside → 丢弃
            continue
        out.append({
            "bisector": b.tolist(),
            "gap_deg": round(float(np.degrees(gap)), 1),
            "inside_ratio": round(ratio, 4),
            "is_inside": False,
            "boundary_dirs": [d0.tolist(), d1.tolist()],
            "boundary_normals": [snorms[i].tolist(), snorms[j].tolist()],
        })
    return out


# ---------------------------------------------------------------------------
# 7. winding 分组网格
# ---------------------------------------------------------------------------
def merge_meshes(tms):
    """把若干 Trimesh 顶点/面拼成 (V, F)。"""
    V_list, F_list, off = [], [], 0
    for tm in tms:
        V_list.append(np.asarray(tm.vertices, np.float64))
        F_list.append(np.asarray(tm.faces, np.int64) + off)
        off += len(tm.vertices)
    if not V_list:
        return np.empty((0, 3)), np.empty((0, 3), np.int64)
    return np.vstack(V_list), np.vstack(F_list)


def build_winding_groups(meshes, nwind):
    """返回 (prim_path -> group_id, [ (V,F) ... ])。nwind=1 全场景一个大 mesh。"""
    paths = [p for p, _ in meshes]
    tms = [t for _, t in meshes]
    n = min(max(1, nwind), len(tms))
    if n == 1:
        V, F = merge_meshes(tms)
        return {p: 0 for p in paths}, [(V, F)]

    from sklearn.cluster import KMeans
    centroids = np.array([tm.vertices.mean(axis=0) for tm in tms])
    labels = KMeans(n_clusters=n, n_init=10, random_state=0).fit_predict(centroids)
    groups = [[] for _ in range(n)]
    for i, lb in enumerate(labels):
        groups[int(lb)].append(tms[i])
    VF = [merge_meshes(g) for g in groups]
    return {paths[i]: int(labels[i]) for i in range(len(paths))}, VF


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def find_welds(usd_path, lmin, lmax, ang_min_deg, ang_max_deg, nwind, verbose=True):
    t0 = time.time()
    meshes = load_scene_meshes(usd_path, verbose)
    if not meshes:
        return []

    path2grp, VF = build_winding_groups(meshes, nwind)
    if verbose:
        print(f"[winding] 分 {len(VF)} 组")

    amin, amax = np.radians(ang_min_deg), np.radians(ang_max_deg)
    welds = []
    seen = []                                    # 去重用端点

    for prim_path, tm in tqdm(meshes):
        # if len(welds) > 5:
        #     break
        pairs, evids = feature_edges(tm, amin, amax)
        if len(pairs) == 0:
            continue
        chains = build_straight_chains(tm.vertices, evids, COLLINEAR_TOL_DEG)
        Vw, Fw = VF[path2grp[prim_path]]

        for ch in chains:
            vids = ch["vids"]
            p0 = np.asarray(tm.vertices[vids[0]], float)
            p1 = np.asarray(tm.vertices[vids[-1]], float)
            length = float(np.linalg.norm(p1 - p0))
            if length < lmin or length > lmax:
                continue

            # 去重（端点近重复）
            dup = any(
                (np.allclose(p0, q0, atol=DEDUP_EPS) and np.allclose(p1, q1, atol=DEDUP_EPS)) or
                (np.allclose(p0, q1, atol=DEDUP_EPS) and np.allclose(p1, q0, atol=DEDUP_EPS))
                for q0, q1 in seen)
            if dup:
                continue

            # 代表边（链中点处）算焊缝量，忠于 ifc 的逐边语义
            rep = ch["eids"][len(ch["eids"]) // 2]
            fi, fj = (int(x) for x in pairs[rep])
            a, b = (int(x) for x in evids[rep])
            edge_mid = (tm.vertices[a] + tm.vertices[b]) / 2.0
            ev = tm.vertices[b] - tm.vertices[a]
            en = np.linalg.norm(ev)
            if en < 1e-9:
                continue
            edge_dir = ev / en

            dirs, normals = two_face_dirs(fi, fj, tm, edge_mid, edge_dir)
            if len(dirs) < 2:
                continue

            # ±d 贴面探测筛选（用本 prim 网格）
            kept_d, kept_n = [], []
            for d, nrm in zip(dirs, normals):
                for cand in (d, -d):
                    if probe_on_face(edge_mid, cand, tm):
                        kept_d.append(np.asarray(cand, float))
                        kept_n.append(nrm)
            if len(kept_d) < 2:
                continue

            sdirs, sangles, snorms = sort_dirs_by_angle(kept_d, edge_dir, kept_n)
            regions = classify_out_regions(edge_mid, sdirs, sangles, snorms, Vw, Fw)
            if not regions:
                continue

            seen.append((p0.copy(), p1.copy()))
            overall = p1 - p0
            overall = (overall / np.linalg.norm(overall)).tolist()
            for r in regions:
                welds.append({
                    "prim_path": prim_path,
                    "p0": p0.tolist(),
                    "p1": p1.tolist(),
                    "length": round(length, 6),
                    "edge_dir": overall,
                    **r,
                })

    if verbose:
        print(f"[done] {len(welds)} 条朝外焊缝，用时 {time.time() - t0:.1f}s")
    return welds


def main():
    ap = argparse.ArgumentParser(description="在 USD 场景物体表面找焊缝直线并算焊缝信息")
    ap.add_argument("--usd", required=True, help="USD 场景路径")
    ap.add_argument("--min", type=float, required=True, dest="lmin", help="最短焊缝长度(米)")
    ap.add_argument("--max", type=float, required=True, dest="lmax", help="最长焊缝长度(米)")
    ap.add_argument("--angle-min", type=float, default=30.0, help="二面角偏差下限(度)")
    ap.add_argument("--angle-max", type=float, default=150.0, help="二面角偏差上限(度)")
    ap.add_argument("--nwind", type=int, default=1, help="winding 分组数(默认1=全场景一个大mesh)")
    ap.add_argument("--out", required=True, help="输出 json 路径")
    args = ap.parse_args()

    welds = find_welds(args.usd, args.lmin, args.lmax,
                       args.angle_min, args.angle_max, args.nwind)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(welds, f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.out}（{len(welds)} 条）")


if __name__ == "__main__":
    main()
