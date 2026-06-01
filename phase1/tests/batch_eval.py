"""批量评估 Phase1 在所有焊缝上的成功率, 并保存可直接可视化的规划路线.

特点:
  - 跑 segment_sub_output 下所有工件的所有 seam_*.pkl (可 --limit 限制条数)
  - 每完成 1 条焊缝就立刻更新 summary.json (成功率) + 写该条的路线 .npz
  - 可续跑: 已有路线 .npz 的焊缝默认跳过 (--redo 强制重跑)

输出 (tmp/batch_eval/):
  summary.json              累计统计 + 每条结果一行
  summary.csv               同上 CSV
  routes/<part>__<seam>.npz 每条焊缝的规划路线 (供 visualize_route.py 可视化)

跑法 (建议后台):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  /home/a/miniforge3/envs/env_isaaclab/bin/python phase1/tests/batch_eval.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import igl

from phase1.config import INITIAL_JOINT_ANGLES, Phase1Config
from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.exploration import run_phase1, compute_camera_pose
from phase1.geodesic import workspace_bounds
from phase1.arm_kin import ArmKin
from phase1.mapping import VoxelMap, STATE_FREE, STATE_OCCUPIED
from phase1.tests._viz_helpers import load_seam_data


ROOT = "/media/a/新加卷/hanfeng/segment_sub_output"
ROBOT_YML = "/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml"
OUTDIR = Path("/home/a/Projects/Github/gt_gen_hanfeng/tmp/batch_eval")
ROUTEDIR = OUTDIR / "routes"


def find_obj(part_dir: str) -> str | None:
    objs = glob.glob(os.path.join(part_dir, "*.obj"))
    return objs[0] if objs else None


def build_gt_map(sim: RaycastSim, seam, cfg: Phase1Config) -> VoxelMap:
    """工件网格填成 occupied 的工作空间体素图 (碰撞 ground-truth)."""
    bounds = workspace_bounds(seam.robot_pose, cfg.arm_reach, cfg.workspace_pad)
    vm = VoxelMap(bounds=bounds, resolution=0.05, max_range=2.0)
    V = np.asarray(sim.mesh.vertices, np.float64)
    F = np.asarray(sim.mesh.faces, np.int32)
    ai = (vm.state != STATE_OCCUPIED).nonzero(as_tuple=False)
    cc = vm.index_to_world(ai).cpu().numpy().astype(np.float64)
    wn = np.zeros(len(cc))
    for s in range(0, len(cc), 50000):
        e = min(s + 50000, len(cc))
        wn[s:e] = igl.fast_winding_number(V, F, cc[s:e])
    inside = wn > 0.5
    vm._state[:] = STATE_FREE
    if inside.any():
        ii = ai[inside]
        vm._state[ii[:, 0], ii[:, 1], ii[:, 2]] = STATE_OCCUPIED
    return vm


def save_route(path: Path, part: str, seam_name: str, obj: str,
               seam, res, cfg: Phase1Config):
    """保存一条焊缝的规划路线 (供 visualize_route.py)."""
    st = res.final_state
    ee_path = (np.stack(st.ee_pos_history) if st.ee_pos_history
               else np.zeros((0, 3)))
    ee_poses = (np.stack(st.ee_pose_history) if st.ee_pose_history
                else np.zeros((0, 4, 4)))           # 每路点完整相机位姿 4×4
    theta_path = (np.stack(st.theta_history) if st.theta_history
                  else np.zeros((0, 6)))            # 每路点关节角
    graph_pts = np.stack([n.ee_pos_world for n in st.graph.all_nodes()]) \
        if len(st.graph) else np.zeros((0, 3))
    np.savez(
        path,
        part=part, seam=seam_name, obj=obj,
        success=bool(res.success),
        switch_at=(-1 if res.phase_switch_at is None else res.phase_switch_at),
        fail_reason=(res.fail_reason or ""),
        observed=np.array(sorted(st.p_observed), dtype=np.int64),
        cycles=st.cycle_idx,
        solver_calls=st.solver_calls,
        ee_path=ee_path,                       # (N,3) 规划路线位置
        ee_poses=ee_poses,                     # (N,4,4) 每点相机位姿 (含朝向)
        theta_path=theta_path,                 # (N,6) 每点关节角 (可复现机械臂构型)
        graph_pts=graph_pts,                   # (M,3) 探索图节点
        final_theta=(st.theta_now if st.theta_now is not None
                     else np.zeros(6)),
        cam_fov=np.array([cfg.cam_fov_h_deg, cfg.cam_fov_v_deg]),
        obs_d=np.array([cfg.obs_d_min, cfg.obs_d_max]),
        seam_line=seam.seam_line,
        seam_tangent=seam.seam_tangent,
        seam_limits=seam.seam_limits,
        p_weld=seam.p_weld,
        p_weld_idx=seam.p_weld_idx,
        robot_pose=np.asarray(seam.robot_pose),
    )


def update_summary(rows: list[dict], t0: float):
    n = len(rows)
    n_ok = sum(r["success"] for r in rows)
    summary = {
        "total": n,
        "success": n_ok,
        "success_rate": round(100.0 * n_ok / n, 1) if n else 0.0,
        "elapsed_s": round(time.time() - t0, 0),
        "rows": rows,
    }
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    # CSV
    lines = ["part,seam,success,switch_at,cycles,min_d2P,solver_calls,sec,fail_reason"]
    for r in rows:
        lines.append(f'{r["part"]},{r["seam"]},{int(r["success"])},'
                     f'{r["switch_at"]},{r["cycles"]},{r["min_d2P"]:.3f},'
                     f'{r["solver_calls"]},{r["sec"]:.0f},{r["fail_reason"]}')
    (OUTDIR / "summary.csv").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="最多跑几条 (0=全部)")
    ap.add_argument("--max-cycles", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=150)
    ap.add_argument("--solver-budget", type=int, default=12000)
    ap.add_argument("--redo", action="store_true", help="重跑已有结果")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ROUTEDIR.mkdir(parents=True, exist_ok=True)

    # 收集所有 (part_dir, obj, pkl)
    tasks = []
    for part_dir in sorted(glob.glob(os.path.join(ROOT, "*/"))):
        obj = find_obj(part_dir)
        if obj is None:
            continue
        part = Path(part_dir.rstrip("/")).name
        pkls = sorted(glob.glob(os.path.join(part_dir, "seam_*.pkl")),
                      key=lambda p: int(p.split("seam_")[1].split(".")[0]))
        for pkl in pkls:
            tasks.append((part, obj, pkl))
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"共 {len(tasks)} 条焊缝待评估", flush=True)

    # 每个 obj 只建一次 RaycastSim (缓存)
    sim_cache: dict[str, RaycastSim] = {}
    K = CameraIntrinsics.from_fov(86, 57, 240, 320)

    rows = []
    t0 = time.time()
    for i, (part, obj, pkl) in enumerate(tasks):
        seam_name = Path(pkl).stem
        route_path = ROUTEDIR / f"{part}__{seam_name}.npz"
        if route_path.exists() and not args.redo:
            # 续跑: 读回已存结果计入统计
            d = np.load(route_path, allow_pickle=True)
            rows.append(dict(part=part, seam=seam_name,
                             success=bool(d["success"]),
                             switch_at=int(d["switch_at"]),
                             cycles=int(d["cycles"]),
                             min_d2P=float("nan"),
                             solver_calls=int(d["solver_calls"]),
                             sec=0.0, fail_reason=str(d["fail_reason"])))
            update_summary(rows, t0)
            print(f"[{i+1}/{len(tasks)}] {part}/{seam_name} (已存, 跳过) "
                  f"成功={bool(d['success'])}", flush=True)
            continue
        try:
            if obj not in sim_cache:
                sim_cache[obj] = RaycastSim(mesh_paths=[obj], intrinsics=K)
            sim = sim_cache[obj]
            seam = load_seam_data(pkl)
            cfg = Phase1Config(max_cycles=args.max_cycles, n_samples=args.n_samples,
                               solver_budget=args.solver_budget, solver_max_calls=1)
            arm = ArmKin(robot_yml_path=ROBOT_YML, robot_pose_world=seam.robot_pose)
            vm = build_gt_map(sim, seam, cfg)
            ts = time.time()
            res = run_phase1(theta_init=INITIAL_JOINT_ANGLES,
                             seam_points=seam.seam_line, seam_tangents=seam.seam_tangent,
                             seam_limits=seam.seam_limits, p_target=seam.p_weld,
                             arm_kin=arm, depth_source=sim, intrinsics=K,
                             voxel_map=vm, cfg=cfg, rng=np.random.default_rng(0))
            st = res.final_state
            ds = [float(np.linalg.norm(ee - seam.p_weld)) for ee in st.ee_pos_history]
            sec = time.time() - ts
            save_route(route_path, part, seam_name, obj, seam, res, cfg)
            rows.append(dict(part=part, seam=seam_name, success=bool(res.success),
                             switch_at=(-1 if res.phase_switch_at is None
                                        else res.phase_switch_at),
                             cycles=st.cycle_idx, min_d2P=(min(ds) if ds else -1),
                             solver_calls=st.solver_calls, sec=sec,
                             fail_reason=(res.fail_reason or "")))
            update_summary(rows, t0)
            n_ok = sum(r["success"] for r in rows)
            print(f"[{i+1}/{len(tasks)}] {part}/{seam_name} "
                  f'{"OK " if res.success else "FAIL"} switch@{res.phase_switch_at} '
                  f"cyc={st.cycle_idx} minD={min(ds) if ds else -1:.2f} "
                  f"solver={st.solver_calls} {sec:.0f}s | 累计 {n_ok}/{len(rows)} "
                  f"= {100*n_ok/len(rows):.0f}%", flush=True)
            del arm, vm
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append(dict(part=part, seam=seam_name, success=False, switch_at=-1,
                             cycles=-1, min_d2P=-1, solver_calls=0, sec=0,
                             fail_reason=f"EXC:{e}"))
            update_summary(rows, t0)
            print(f"[{i+1}/{len(tasks)}] {part}/{seam_name} ERROR {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()

    n_ok = sum(r["success"] for r in rows)
    print(f"\n===== 最终成功率 {n_ok}/{len(rows)} = "
          f"{100*n_ok/len(rows):.1f}%  总耗时 {time.time()-t0:.0f}s =====",
          flush=True)
    print(f"结果: {OUTDIR}/summary.json  路线: {ROUTEDIR}/", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
