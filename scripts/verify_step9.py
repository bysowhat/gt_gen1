"""Step 9 验证：特权 NBV 打分与选择。参考 verify_step7/8，每个 verify_* 验证 nbv.py 的一块。

复用 verify_step8.build_scene 的【冷启动 seam_22】场景（圆柱初始 FREE、cur_cfg=retract、真值 P*、阻塞段 B）。

被测：
  - verify_raycast_reveal        → raycast_reveal（假设性 raycast 出"将确定"的体素；与 B 求交得 gain）
  - verify_score_and_argmax      → score_candidate（gain=|reveal∩B|, score=gain-λ·cost）+ argmax 选最优
  - verify_best_next_view_oracle → best_next_view_using_oracle（P*→B→候选→打分→argmax，端到端一轮 NBV）

运行：conda run -n env_isaaclab python scripts/verify_step9.py [--viz]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from verify_step8 import (build_scene, SEAM, _arm_mesh, _work_mesh, _cells_mesh, _draw,
                          _roi_and_base, _ball, _lines, _fov_frustum)


def _make_B_and_cands(ctx):
    """从冷启动场景算 B + 一批候选（供组件验证复用同一份固定数据）。"""
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.candidates import generate_candidates
    h, vm = ctx["h"], ctx["vm"]
    k = ctx["cfg"].params["nbv"]["k_lookahead"]
    B = compute_blocking_B(h, vm, ctx["P"], ctx["reach_idx"], k)
    cands = generate_candidates(h, vm, B, ctx["cam"], ctx["cur_cfg"])
    return B, cands


def verify_raycast_reveal(ctx, B, cands, viz=False):
    """验证 raycast_reveal：假设性 raycast 出"将确定"的体素。

    判据：① 每个候选 reveal 非空且全在 voxmap 界内；② 至少一个候选揭开的体素与 B 有交(gain>0)
         ——说明候选确实朝 B 看、能确认那片未知。
    可视化（--viz）：对【每个候选】各弹一个窗口，展示 raycast_reveal 的结果 + gain 的算法：
       整臂@cfg(绿) + 相机坐标轴 + FOV视锥 + 视线→T(红) + 该视点 reveal 全体(青半透明,降采样) +
       B 中【被揭开 reveal∩B=绿】/【没揭开=橙】，标题写 gain=绿/总B。
    """
    from gt_gen import nbv
    h, vm, cam, scene = ctx["h"], ctx["vm"], ctx["cam"], ctx["scene"]

    print("\n== verify raycast_reveal ==")
    gains, reveals = [], []
    for c in cands:
        reveal = nbv.raycast_reveal(vm, c.cam_pose, cam, scene)
        assert reveal.shape[0] > 0, "假设性 raycast 没揭开任何体素"
        assert bool(vm.in_bounds(reveal).all()), "reveal 含越界体素"
        gains.append(nbv._count_intersection(reveal, B)); reveals.append(reveal)
    gains = np.array(gains)
    print(f"  候选数={len(cands)}  各候选 reveal∩B(gain) = {gains.tolist()}")
    assert gains.max() > 0, "没有候选能揭开 B（朝向/遮挡有问题）"
    print(f"  ✓ 每个候选 reveal 非空且在界内；最佳候选揭开 B 体素数={int(gains.max())}>0")

    if viz:
        import open3d as o3d
        from gt_gen.voxmap import FREE
        Bw = vm.voxel_to_world(B); N = len(cands)
        for i, (c, reveal, gain) in enumerate(zip(cands, reveals, gains)):
            seen = np.array([tuple(b) in set(map(tuple, reveal)) for b in B], bool)  # B 是否被这次 reveal 命中
            g = [("arm", _arm_mesh(h, c.config), "lit", None),
                 ("work", _work_mesh(scene), "lit", None)]
            fc = vm.state_centers(FREE)
            if fc.shape[0]:
                g.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.08]))
            rv = reveal if reveal.shape[0] <= 600 else reveal[::int(np.ceil(reveal.shape[0] / 600))]
            g.append(("reveal", _cells_mesh(vm, vm.voxel_to_world(rv)), "fill", [0.0, 0.8, 0.8, 0.10]))  # reveal 全体 青
            if (~seen).any():
                m = _cells_mesh(vm, Bw[~seen]); m.paint_uniform_color([1.0, 0.55, 0.0])     # B 没揭开 橙
                g.append(("B_miss", m, "lit", None))
            if seen.any():
                m = _cells_mesh(vm, Bw[seen]); m.paint_uniform_color([0.1, 0.85, 0.2])      # B 被揭开 绿
                g.append(("B_seen", m, "lit", None))
            g.append(("T", _ball(c.target, 0.04, [0.9, 0.1, 0.9]), "lit", None))
            depth = float(np.linalg.norm(c.target - c.cam_pose[:3, 3]))
            edges, cone = _fov_frustum(c.cam_pose, cam, depth)
            fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12); fr.transform(c.cam_pose)
            g += [("cam", fr, "lit", None), ("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.18]),
                  ("fov_edges", edges, "line", None),
                  ("ray", _lines([(c.cam_pose[:3, 3], c.target)], [0.9, 0.1, 0.1]), "line", None)]
            g += _roi_and_base(vm)
            _draw(g, f"step9 raycast_reveal 候选{i+1}/{N}: reveal全体(青) gain=reveal∩B(绿)={int(gain)}/{len(B)} 未揭开(橙) FOV(紫)")
    return gains


def verify_score_and_argmax(ctx, B, cands, gains, viz=False):
    """验证 score_candidate + argmax：score=gain−λ·cost；λ=0 时 score==gain，argmax 即揭开 B 最多者。

    命令行逐候选打印 gain / path_cost / score(=减 path_cost 后的分)。
    可视化（--viz）：每个候选一窗（同 raycast_reveal）：reveal∩B(绿)/未揭开(橙)+reveal全体(青)+FOV，
       标题写 gain、path_cost、score，并标出 argmax 选中者(★)。
    """
    from gt_gen import nbv
    h, vm, cam, scene, cur = ctx["h"], ctx["vm"], ctx["cam"], ctx["scene"], ctx["cur_cfg"]
    lam = float(ctx["cfg"].params["nbv"].get("lambda_cost", 0.0))

    print("\n== verify score_candidate + argmax ==")
    print(f"  lambda_cost(default.yaml)={lam}")
    rows = []                                                # (gain, path_cost, score, reveal)
    for c in cands:
        g, s, reveal = nbv.score_candidate(vm, c, B, scene, cam, cur, lambda_cost=lam)
        pc = float(np.linalg.norm(np.asarray(c.config, float) - np.asarray(cur, float)))
        rows.append((g, pc, s, reveal))
    g_arr = np.array([r[0] for r in rows]); s_arr = np.array([r[2] for r in rows])
    best_i = int(np.argmax(s_arr))
    for i, (g, pc, s, _) in enumerate(rows):
        mark = " ★argmax" if i == best_i else ""
        print(f"    候选#{i}: gain={int(g):3d}  path_cost={pc:.3f}  score=gain−{lam}·cost={s:.2f}{mark}")
    if lam == 0.0:
        assert np.allclose(g_arr, s_arr), "λ=0 时 score 应等于 gain"
    assert g_arr[best_i] == g_arr.max(), "argmax(score) 在 λ=0 时应同时是 max(gain)"
    assert g_arr[best_i] > 0, "选中视点应能揭开 B"
    print(f"  ✓ score=gain−λ·cost；argmax → 候选#{best_i} gain={int(g_arr[best_i])} score={s_arr[best_i]:.2f}")

    if viz:
        import open3d as o3d
        from gt_gen.voxmap import FREE
        Bw = vm.voxel_to_world(B); N = len(cands)
        for i, (c, (g, pc, s, reveal)) in enumerate(zip(cands, rows)):
            seen = np.array([tuple(b) in set(map(tuple, reveal)) for b in B], bool)
            geo = [("arm", _arm_mesh(h, c.config), "lit", None),
                   ("work", _work_mesh(scene), "lit", None)]
            fc = vm.state_centers(FREE)
            if fc.shape[0]:
                geo.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.08]))
            rv = reveal if reveal.shape[0] <= 600 else reveal[::int(np.ceil(reveal.shape[0] / 600))]
            geo.append(("reveal", _cells_mesh(vm, vm.voxel_to_world(rv)), "fill", [0.0, 0.8, 0.8, 0.10]))
            if (~seen).any():
                m = _cells_mesh(vm, Bw[~seen]); m.paint_uniform_color([1.0, 0.55, 0.0])
                geo.append(("B_miss", m, "lit", None))
            if seen.any():
                m = _cells_mesh(vm, Bw[seen]); m.paint_uniform_color([0.1, 0.85, 0.2])
                geo.append(("B_seen", m, "lit", None))
            geo.append(("T", _ball(c.target, 0.04, [0.9, 0.1, 0.9]), "lit", None))
            depth = float(np.linalg.norm(c.target - c.cam_pose[:3, 3]))
            edges, cone = _fov_frustum(c.cam_pose, cam, depth)
            fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12); fr.transform(c.cam_pose)
            geo += [("cam", fr, "lit", None), ("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.18]),
                    ("fov_edges", edges, "line", None),
                    ("ray", _lines([(c.cam_pose[:3, 3], c.target)], [0.9, 0.1, 0.1]), "line", None)]
            geo += _roi_and_base(vm)
            star = " ★argmax" if i == best_i else ""
            _draw(geo, f"step9 score 候选{i+1}/{N}: gain={int(g)}(绿) path_cost={pc:.3f} "
                       f"score=gain−{lam}·cost={s:.2f}{star}")
    return best_i


def verify_best_next_view_oracle(ctx):
    """端到端验证 best_next_view_using_oracle：P*→reach_pt/B→候选→打分→argmax，返回下一视点。

    判据：status=ok；选中 cfg 可行(check_state)、从当前构型经自由区可达(motion_stays_in_free)、gain>0。
    注：传入 build_scene 已算好的 P*（p_star=），避免 NBV 内部重规划的随机抖动 → 与上面固定数据一致。
    """
    from gt_gen import nbv, curobo_iface as ci
    from gt_gen.swept import motion_stays_in_free
    h, vm, scene, cur = ctx["h"], ctx["vm"], ctx["scene"], ctx["cur_cfg"]

    print("\n== verify best_next_view_using_oracle ==")
    r = nbv.best_next_view_using_oracle(h, cur, vm, scene, ctx["goal_pose"], p_star=ctx["P"])
    print(f"  status={r.status}  reach_idx={r.reach_idx}  |B|={r.n_B}  候选={r.n_candidates}  "
          f"gain={r.gain:.0f}  score={r.score:.2f}")
    assert r.status == "ok", f"应选出视点，实际 status={r.status}"
    assert r.gain > 0, "选中视点应能揭开 B"
    feas, _ = ci.check_state(h, r.cfg)
    ok, nnf = motion_stays_in_free(h, vm, cur, r.cfg)
    print(f"  选中 cfg：自碰撞OK={feas}  可达={ok}(非FREE={nnf})  目标T={np.round(r.target,2)}")
    assert feas and ok, "选中视点应可行且可达"
    print("  ✓ oracle 选出一个揭开 B>0、可行、可达的下一视点")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=SEAM)
    ap.add_argument("--viz", action="store_true", help="弹 open3d 看 raycast_reveal/选中视点（需显示器）")
    args = ap.parse_args()

    ctx = build_scene(args)
    B, cands = _make_B_and_cands(ctx)
    print(f"\n固定数据：|B|={B.shape[0]}  候选={len(cands)}")
    assert len(cands) >= 1, "无候选——先确保 Step 8 在该场景能产候选"

    gains = verify_raycast_reveal(ctx, B, cands, viz=False)#viz=args.viz
    verify_score_and_argmax(ctx, B, cands, gains, viz=False)#viz=args.viz
    verify_best_next_view_oracle(ctx)

    print("\nVERIFY_STEP9_OK [raycast_reveal, score_candidate, best_next_view_using_oracle]")


if __name__ == "__main__":
    main()
