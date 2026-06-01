"""M1.6 视觉验证: 围 P_weld 撒候选 viewpoint, 算 gain, 可视化分布.

输出:
    Open3D 3D 视图: 候选位置按 gain 染色 (蓝低 红高)
    matplotlib 2D: vol_gain vs target_bias 散点 (color=total)
    PNG snapshot:  按 gain 排序的 top-N candidates 列表

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_gain.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.gain import GainConfig, gain, target_bias, volumetric_gain
from phase1.mapping import VoxelMap
from phase1.observation_check import (
    Viewpoint,
    check_distance,
    check_in_frustum,
    check_line_of_sight,
    check_seam_wedge,
)
from phase1.tests._viz_helpers import load_seam_data, look_at, make_seam_geoms


def _find_cjk_font():
    cands = [f.name for f in fm.fontManager.ttflist]
    for kw in ["noto sans cjk", "source han", "yahei", "wenquanyi", "fallback"]:
        for n in cands:
            if kw in n.lower():
                return n
    return None
_cjk = _find_cjk_font()
if _cjk:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_cjk] + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    _default_out = (Path(__file__).resolve().parent.parent.parent
                    / "tmp" / "m1_6_gain")
    parser.add_argument("--output-dir", default=str(_default_out))
    parser.add_argument("--n-candidates", type=int, default=120)
    parser.add_argument("--radius-min", type=float, default=0.2)
    parser.add_argument("--radius-max", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-vol", type=float, default=1.0)
    parser.add_argument("--w-target", type=float, default=20.0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--d-min", type=float, default=0.30,
                        help="合格观测距离下限 (M1.3 用)")
    parser.add_argument("--d-max", type=float, default=0.60,
                        help="合格观测距离上限")
    parser.add_argument("--no-validity-filter", action="store_true",
                        help="不过 M1.3 合格观测过滤 (调试用; 默认会过滤)")
    parser.add_argument("--gt-occupancy", action="store_true", default=True,
                        help="用 mesh 真值填 voxel 占据 (推荐; 否则 LOS 在 unknown 处会漏)")
    parser.add_argument("--no-gt-occupancy", dest="gt_occupancy", action="store_false")
    parser.add_argument("--strict-los", action="store_true",
                        help="LOS 把 unknown 也当障碍. 默认 optimistic.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载 ----
    K = CameraIntrinsics.from_fov(86.0, 57.0, 240, 320)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K, device=device)
    seam = load_seam_data(args.pkl)
    print(f"[seam] {seam.label}")

    # ---- 建 voxel map (8 视角) ----
    bounds = np.stack([sim.mesh.bounds[0] - 0.5, sim.mesh.bounds[1] + 0.5])
    vm = VoxelMap(bounds=bounds, resolution=0.02, device=device, max_range=2.0)
    for i in range(8):
        ang = 2 * np.pi * i / 8
        cp = seam.p_weld + np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang), 0.3])
        d = sim.render(look_at(cp, seam.p_weld))
        vm.integrate(d, K, look_at(cp, seam.p_weld))
    counts = vm.num_voxels_by_state()
    print(f"[map raycast] occupied={counts['occupied']}, free={counts['free']}, "
          f"unknown={counts['unknown']}")

    # ---- 可选: 用 mesh 真值填 voxel 占据 ----
    # mesh 不 watertight 时, trimesh.contains 不可靠, 改用 libigl fast_winding_number
    # (Generalized Winding Number, Jacobson et al. 2013): wn > 0.5 即为内部.
    # 重要: 覆盖任何非 occupied 状态 (unknown OR free), 因为非封闭 mesh 的"漏洞"
    # 会让 raycast 把内部体素错误标为 free, 单填 unknown 不够.
    if args.gt_occupancy:
        print("[gt_occupancy] 用 libigl fast_winding_number 填墙体内部 (避免 LOS 漏)...")
        import igl
        from phase1.mapping import STATE_OCCUPIED
        V = np.asarray(sim.mesh.vertices, dtype=np.float64)
        F = np.asarray(sim.mesh.faces, dtype=np.int32)
        # 非 occupied 体素 (unknown 或 free, 都可能误标)
        non_occ_idx = (vm.state != STATE_OCCUPIED).nonzero(as_tuple=False)
        if len(non_occ_idx) > 0:
            non_occ_centers = vm.index_to_world(non_occ_idx).cpu().numpy().astype(np.float64)
            CHUNK = 50000
            inside_mask = np.zeros(len(non_occ_centers), dtype=bool)
            for s in range(0, len(non_occ_centers), CHUNK):
                e = min(s + CHUNK, len(non_occ_centers))
                wn = igl.fast_winding_number(V, F, non_occ_centers[s:e])
                inside_mask[s:e] = wn > 0.5
            n_inside = int(inside_mask.sum())
            if n_inside:
                inside_idx = non_occ_idx[inside_mask]
                vm._state[inside_idx[:, 0], inside_idx[:, 1], inside_idx[:, 2]] = \
                    STATE_OCCUPIED
            print(f"  补 {n_inside} 个 mesh 内体素为 occupied (libigl, 含 raycast 误标 free)")
        counts = vm.num_voxels_by_state()
        print(f"[map gt]      occupied={counts['occupied']}, "
              f"free={counts['free']}, unknown={counts['unknown']}")

    # ---- 撒候选位置 (全球面, 不是只上半球: seam_limits 可能朝任何方向) ----
    radii = rng.uniform(args.radius_min, args.radius_max, args.n_candidates)
    u01 = rng.uniform(-1.0, 1.0, args.n_candidates)   # cos(theta), 全球面
    v01 = rng.uniform(0, 1, args.n_candidates)
    theta = np.arccos(u01)                              # full sphere
    phi = 2 * np.pi * v01
    offsets = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1) * radii[:, None]
    positions = seam.p_weld + offsets

    # ---- 算 gain (分解三项) ----
    cfg = GainConfig(w_vol=args.w_vol, w_target=args.w_target,
                     n_rays_h=8, n_rays_v=8, max_range=2.0)
    print(f"[cfg] w_vol={cfg.w_vol}, w_target={cfg.w_target}")

    # 一半候选正对 P_weld (典型 M1.7 用法), 一半随机朝向 (展示 target_bias 动态范围)
    half = args.n_candidates // 2
    aim_at_target = np.zeros(args.n_candidates, dtype=bool)
    aim_at_target[:half] = True

    t0 = time.time()
    vol_gains = np.zeros(args.n_candidates)
    tgt_biases = np.zeros(args.n_candidates)
    totals = np.zeros(args.n_candidates)
    cam_dirs = np.zeros((args.n_candidates, 3))
    valid_flags = np.zeros(args.n_candidates, dtype=bool)
    fail_reasons = ["" for _ in range(args.n_candidates)]
    for i, pos in enumerate(positions):
        if aim_at_target[i]:
            cam_dir = (seam.p_weld - pos)
            cam_dir /= np.linalg.norm(cam_dir)
        else:
            d = rng.normal(size=3)
            d[2] = abs(d[2])
            cam_dir = d / np.linalg.norm(d)
        cam_dirs[i] = cam_dir
        cam_up = np.array([0, 0, 1.0])
        vol_gains[i] = volumetric_gain(pos, cam_dir, cam_up, vm, cfg)
        tgt_biases[i] = target_bias(pos, cam_dir, seam.p_weld)
        totals[i] = cfg.w_vol * vol_gains[i] + cfg.w_target * tgt_biases[i]

        # M1.3 合格观测检查
        # viz 用 mesh ray-cast 做 ground-truth LOS (绕过体素离散化精度问题)
        # M1.7 实战会用 voxel LOS, 这里 viz 看几何真实
        if not args.no_validity_filter:
            vp = Viewpoint(pos=pos, dir=cam_dir, up=cam_up)
            d_ok, _ = check_distance(vp, seam.p_weld, args.d_min, args.d_max)
            f_ok = check_in_frustum(vp, seam.p_weld, K) if d_ok else False
            # mesh-based LOS: 视线到 P_weld 之前不能撞 mesh
            l_ok = False
            if d_ok and f_ok:
                rel = seam.p_weld - pos
                dist_full = np.linalg.norm(rel)
                dir_unit = rel / dist_full
                locs, _, _ = sim.mesh.ray.intersects_location(
                    ray_origins=pos.reshape(1, 3),
                    ray_directions=dir_unit.reshape(1, 3),
                    multiple_hits=True,
                )
                # 留 5mm 余量, 避免把 P_weld 自己的表面 hit 当障碍
                n_blockers = sum(
                    1 for loc in locs
                    if np.linalg.norm(loc - pos) < dist_full - 0.005
                )
                l_ok = (n_blockers == 0)
            # ④ wedge: 用 seam.seam_limits[mid] + tangent
            if d_ok and f_ok and l_ok:
                slim_mid = seam.seam_limits[seam.p_weld_idx]
                w_ok, _ = check_seam_wedge(vp, seam.p_weld, slim_mid,
                                            tangent=seam.tangent)
            else:
                w_ok = False
            valid_flags[i] = d_ok and f_ok and l_ok and w_ok
            if not valid_flags[i]:
                if not d_ok: fail_reasons[i] = "distance"
                elif not f_ok: fail_reasons[i] = "frustum"
                elif not l_ok: fail_reasons[i] = "line_of_sight"
                else: fail_reasons[i] = "wedge"
        else:
            valid_flags[i] = True
    elapsed = time.time() - t0
    print(f"[gain] {args.n_candidates} candidates "
          f"({half} 朝目标 + {args.n_candidates - half} 随机朝向) "
          f"in {elapsed:.2f} s ({elapsed / args.n_candidates * 1000:.1f} ms/each)")
    print(f"[validity] {valid_flags.sum()}/{args.n_candidates} 通过 合格观测检查")
    if not args.no_validity_filter:
        from collections import Counter
        c = Counter([r for r in fail_reasons if r])
        for reason, n in c.most_common():
            print(f"  ✗ {reason:<14s} {n}")

    # ---- 排序统计: 只在 valid 候选里挑 top ----
    valid_idx = np.where(valid_flags)[0]
    if len(valid_idx) == 0:
        print("\n[WARN] 0 个候选通过合格观测! Open3D 不画 top-1, 见 invalid 灰色 ×.")
        order_in_valid = np.array([], dtype=int)
        order = np.argsort(-totals)
    else:
        order_in_valid = valid_idx[np.argsort(-totals[valid_idx])]
        order = np.argsort(-totals)        # 全集排序 (展示用)

    print("\n[top 5 valid]")
    print(f"  {'idx':>3s}  {'total':>8s}  {'vol':>8s}  {'tgt_bias':>9s}  pos (world)")
    for k in range(min(5, len(order_in_valid))):
        idx = int(order_in_valid[k])
        p = positions[idx]
        print(f"  {idx:>3d}  {totals[idx]:>8.2f}  {vol_gains[idx]:>8.0f}  "
              f"{tgt_biases[idx]:>+9.3f}  ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    # ---- 诊断: 重跑 top-1 的 5 条检查, 每一项都打印 ----
    if len(order_in_valid) > 0 and not args.no_validity_filter:
        idx = int(order_in_valid[0])
        pos = positions[idx]
        cam_dir_dbg = cam_dirs[idx]
        vp_dbg = Viewpoint(pos=pos, dir=cam_dir_dbg, up=np.array([0, 0, 1.0]))
        print(f"\n[DEBUG top-1 #{idx}] pos={pos.round(4)}, "
              f"cam_dir={cam_dir_dbg.round(3)}")
        d_ok_, d_ = check_distance(vp_dbg, seam.p_weld, args.d_min, args.d_max)
        f_ok_ = check_in_frustum(vp_dbg, seam.p_weld, K)
        l_ok_strict = check_line_of_sight(vp_dbg, seam.p_weld, vm, True)
        l_ok_opt = check_line_of_sight(vp_dbg, seam.p_weld, vm, False)
        slim_dbg = seam.seam_limits[seam.p_weld_idx]
        w_ok_, mdot_ = check_seam_wedge(vp_dbg, seam.p_weld, slim_dbg,
                                         tangent=seam.tangent)
        # 也算 gap_deg 给用户看
        d1_n = slim_dbg[0] / np.linalg.norm(slim_dbg[0])
        d2_n = slim_dbg[1] / np.linalg.norm(slim_dbg[1])
        gap_deg = float(np.degrees(np.arccos(np.clip(np.dot(d1_n, d2_n), -1, 1))))
        # 用 trimesh 直接做地面真值 ray-mesh
        rel = seam.p_weld - pos
        dist_full = np.linalg.norm(rel)
        dir_unit = rel / dist_full
        locs, _, _ = sim.mesh.ray.intersects_location(
            ray_origins=pos.reshape(1, 3),
            ray_directions=dir_unit.reshape(1, 3),
            multiple_hits=True,
        )
        n_hits_before_target = sum(
            1 for loc in locs
            if np.linalg.norm(loc - pos) < dist_full - 0.005
        )
        print(f"  ① distance:    ok={d_ok_}  d={d_:.3f}  range=[{args.d_min},{args.d_max}]")
        print(f"  ② frustum:     ok={f_ok_}")
        print(f"  ③ LOS strict:     ok={l_ok_strict}")
        print(f"     LOS optimistic: ok={l_ok_opt}  (viz 用这个)")
        print(f"     trimesh ray-mesh: {len(locs)} hits, {n_hits_before_target} 在 P_weld 之前")
        print(f"  ④ wedge:       ok={w_ok_}  cos_to_bis - cos_half={mdot_:+.4f}")
        print(f"     gap_deg = {gap_deg:.1f}° (90°=L 形, <90°=锐角槽, >90°=钝角)")
        print(f"     d1={slim_dbg[0].round(3)}  d2={slim_dbg[1].round(3)}")

    # ---- 1. matplotlib: vol_gain vs target_bias 散点 (color=total, valid/invalid 区分) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(valid_idx) > 0:
        sc = ax.scatter(vol_gains[valid_idx], tgt_biases[valid_idx],
                        c=totals[valid_idx], cmap="viridis",
                        s=80, alpha=0.85, edgecolors="black", linewidths=0.5,
                        label="✓ valid")
        plt.colorbar(sc, ax=ax, label="total gain (valid only)")
    invalid_idx = np.where(~valid_flags)[0]
    if len(invalid_idx) > 0:
        ax.scatter(vol_gains[invalid_idx], tgt_biases[invalid_idx],
                   c="lightgray", marker="x", s=40, alpha=0.5,
                   label=f"✗ invalid ({len(invalid_idx)} 个)")
    if len(order_in_valid) > 0:
        best_i = int(order_in_valid[0])
        ax.scatter([vol_gains[best_i]], [tgt_biases[best_i]],
                   s=350, marker="*", c="red", edgecolors="black",
                   linewidths=1.5, zorder=10, label=f"top-1 valid (#{best_i})")
    ax.set_xlabel("volumetric_gain")
    ax.set_ylabel("target_bias")
    ax.set_title(f"M1.6 gain 分布 (含 M1.3 合格观测过滤)  |  {seam.label}\n"
                 f"w_vol={cfg.w_vol}, w_target={cfg.w_target}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(1, color="green", linestyle=":", alpha=0.4)
    ax.axhline(-1, color="red", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    out_scatter = out_dir / "01_gain_scatter.png"
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_scatter}")

    # ---- 2. matplotlib: top-N 候选的 gain 分解条形图 (只用 valid) ----
    n_show = min(15, len(order_in_valid))
    if n_show > 0:
        top_idx = order_in_valid[:n_show]
        top_total = totals[top_idx]
        top_vol = cfg.w_vol * vol_gains[top_idx]
        top_tgt = cfg.w_target * tgt_biases[top_idx]
        labels = [f"#{i}" for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(n_show)
        ax.bar(x, top_vol, color="#3070C0", label=f"w_vol·vol_gain")
        ax.bar(x, top_tgt, bottom=top_vol, color="#E67E22",
               label=f"w_target·target_bias")
        ax.plot(x, top_total, "ro-", label="total", linewidth=2, markersize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel("gain")
        ax.set_title(f"Top-{n_show} VALID 候选 gain 分解  |  {seam.label}")
        ax.legend(); ax.grid(True, alpha=0.3)
        out_bar = out_dir / "02_top_breakdown.png"
        plt.tight_layout()
        plt.savefig(out_bar, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_bar}")

    # ---- 3. Open3D 3D: 候选位置按 total 染色 (区分 valid / invalid) ----
    geoms = []
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)
    geoms.extend(make_seam_geoms(seam))

    # valid 候选: viridis 染色
    if len(valid_idx) > 0:
        v_pos = positions[valid_idx]
        v_total = totals[valid_idx]
        t_min, t_max = float(v_total.min()), float(v_total.max())
        norm_v = (v_total - t_min) / max(t_max - t_min, 1e-6)
        import matplotlib.cm as cm
        v_colors = cm.viridis(norm_v)[:, :3]
        pcd_v = o3d.geometry.PointCloud()
        pcd_v.points = o3d.utility.Vector3dVector(v_pos)
        pcd_v.colors = o3d.utility.Vector3dVector(v_colors)
        geoms.append(pcd_v)

    # invalid 候选: 灰色小点
    if len(invalid_idx) > 0:
        inv_pos = positions[invalid_idx]
        pcd_inv = o3d.geometry.PointCloud()
        pcd_inv.points = o3d.utility.Vector3dVector(inv_pos)
        pcd_inv.paint_uniform_color([0.55, 0.55, 0.55])
        geoms.append(pcd_inv)

    # 在 top-1 valid 候选位置画大球 + 朝向箭头
    # 0 valid 时不画 (避免误导: fallback 是 invalid 候选, 视线可能穿工件)
    if len(valid_idx) > 0 and len(order_in_valid) > 0:
        best_idx = int(order_in_valid[0])
        best_pos = positions[best_idx]
        best_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025, resolution=14)
        best_sphere.translate(best_pos)
        best_sphere.paint_uniform_color([1.0, 1.0, 0.0])
        best_sphere.compute_vertex_normals()
        geoms.append(best_sphere)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.vstack([best_pos, seam.p_weld]))
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geoms.append(line)
    else:
        print("[!!!] 0 个 valid 候选; 不画 top-1 (避免误导)")

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=seam.p_weld))

    print()
    print("Legend (Open3D):")
    print("  灰 mesh         — 工件")
    print("  品红圆柱        — 焊缝, 绿球起点 红球终点 橙球 P_weld")
    print("  彩色点云        — ✓ valid 候选, viridis 色 (黄=高 gain, 紫=低)")
    print(f"  灰色 ×          — ✗ invalid 候选 ({len(invalid_idx)} 个), 4 条合格观测某条挂了")
    print("  黄大球          — top-1 VALID best candidate")
    print("  黄线            — best 朝 P_weld 的视线")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"M1.6 Gain  |  {seam.label}",
            width=1280, height=720,
        )

    print()
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
