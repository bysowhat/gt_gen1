"""M1.8 视觉验证: 跑完整 Phase 1, 在 Open3D 里画出整条探索轨迹.

输出:
  灰 mesh           — 工件
  品红圆柱+端球    — 焊缝 (绿=start, 红=end)
  橙球              — P_weld
  黑球+黑线         — init pose (从这里出发)
  蓝小球 (链状)     — 每 cycle 的 EE 位置
  红色折线          — EE 探索轨迹 (按 cycle 序连起来)
  黄大球            — 触发 phase switch 的那一帧 EE 位置 (若成功)
  紫色球            — 所有 graph 节点 (大小∝last_gain)

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \\
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_multi_cycle.py \\
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \\
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_24.pkl" \\
        --max-cycles 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d

from phase1.arm_kin import ArmKin
from phase1.config import INITIAL_JOINT_ANGLES, Phase1Config
from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.exploration import run_phase1
from phase1.geodesic import workspace_bounds
from phase1.mapping import STATE_FREE, STATE_OCCUPIED, VoxelMap
from phase1.tests._viz_helpers import load_seam_data, make_seam_geoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--robot-yml",
        default="/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml")
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="vm 全 free (不喂 mesh 当障碍, 让 collision 总过)")
    parser.add_argument("--no-seam-limits", action="store_true",
                        help="关掉 wedge 检查 + 中线对齐 (默认开)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- 加载 ----
    seam = load_seam_data(args.pkl)
    print(f"[seam] {seam.label}")
    K = CameraIntrinsics.from_fov(86.0, 57.0, 240, 320)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K)
    print(f"[mesh] {sim.num_triangles} triangles")
    arm_kin = ArmKin(robot_yml_path=args.robot_yml,
                     robot_pose_world=seam.robot_pose)

    # ---- cfg + voxel map (网格 = 机械臂工作空间 base±(reach+pad)) ----
    cfg = Phase1Config(
        max_cycles=args.max_cycles,
        n_samples=args.n_samples,
    )
    bounds = workspace_bounds(seam.robot_pose, cfg.arm_reach, cfg.workspace_pad)
    vm = VoxelMap(bounds=bounds, resolution=0.05, max_range=2.0)
    print(f"[map] workspace bounds={bounds.round(2).tolist()}, grid={vm.shape}")
    if args.no_obstacles:
        vm._state[:] = STATE_FREE
        print("[map] free-world (no collision)")
    else:
        import igl
        V = np.asarray(sim.mesh.vertices, dtype=np.float64)
        F = np.asarray(sim.mesh.faces, dtype=np.int32)
        all_idx = (vm.state != STATE_OCCUPIED).nonzero(as_tuple=False)
        all_centers = vm.index_to_world(all_idx).cpu().numpy().astype(np.float64)
        wn = np.zeros(len(all_centers))
        for s in range(0, len(all_centers), 50000):
            e = min(s + 50000, len(all_centers))
            wn[s:e] = igl.fast_winding_number(V, F, all_centers[s:e])
        inside = wn > 0.5
        vm._state[:] = STATE_FREE
        if inside.any():
            iidx = all_idx[inside]
            vm._state[iidx[:,0], iidx[:,1], iidx[:,2]] = STATE_OCCUPIED
        c = vm.num_voxels_by_state()
        print(f"[map] free={c['free']}, occupied={c['occupied']}")

    # ---- 跑 phase 1 ----
    print(f"[cfg] max_cycles={cfg.max_cycles}, n_samples={cfg.n_samples}, "
          f"patience={cfg.patience}")
    print()
    seam_lim = None if args.no_seam_limits else seam.seam_limits
    if args.no_seam_limits:
        print("[note] seam_limits=None (wedge 跳过, 不做中线对齐)")
    else:
        print("[note] seam_limits 启用: wedge 检查 + 中线对齐评分")
    print()

    t0 = time.time()
    result = run_phase1(
        theta_init=INITIAL_JOINT_ANGLES,
        seam_points=seam.seam_line,
        seam_tangents=seam.seam_tangent,
        seam_limits=seam_lim,
        p_target=seam.p_weld,
        arm_kin=arm_kin,
        depth_source=sim,
        intrinsics=K,
        voxel_map=vm,
        cfg=cfg,
        rng=rng,
    )
    elapsed = time.time() - t0

    state = result.final_state
    print(f"=== Phase 1 done in {elapsed:.1f} s ===")
    print(f"  success: {result.success}")
    print(f"  cycles: {state.cycle_idx} / {cfg.max_cycles}")
    print(f"  graph: {len(state.graph)} nodes, {state.graph.num_edges} edges")
    print(f"  global steps: {len(state.global_cycle_indices)} "
          f"(cycles {state.global_cycle_indices})")
    if result.success:
        print(f"  phase_switch_at: cycle {result.phase_switch_at}")
        print(f"  first observed P_i: index {result.p_observed_first}")
        print(f"  total observed: {sorted(state.p_observed)}")
    else:
        print(f"  fail_reason: {result.fail_reason}")
    print(f"  EE 位置链长: {len(state.ee_pos_history)}")

    # ---- viz ----
    geoms = []
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)
    geoms.extend(make_seam_geoms(seam))

    # init pose (黑大球)
    if state.ee_pos_history:
        init_pos = state.ee_pos_history[0]
        s_init = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=16)
        s_init.translate(init_pos)
        s_init.paint_uniform_color([0.0, 0.0, 0.0])
        s_init.compute_vertex_normals()
        geoms.append(s_init)

    # 每个 cycle 的 EE (蓝小球 + 红折线)
    if len(state.ee_pos_history) >= 2:
        pts = np.stack(state.ee_pos_history)                   # (N, 3)
        # 折线
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(pts)
        line.lines = o3d.utility.Vector2iVector(
            [[i, i+1] for i in range(len(pts) - 1)],
        )
        line.colors = o3d.utility.Vector3dVector(
            [[1.0, 0.2, 0.2]] * (len(pts) - 1),
        )
        geoms.append(line)
        # 中间小球 (跳过首尾, 已有标注)
        for i in range(1, len(pts) - 1):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=8)
            s.translate(pts[i])
            s.paint_uniform_color([0.2, 0.5, 1.0])
            s.compute_vertex_normals()
            geoms.append(s)

    # 终点 (黄大球, 触发 switch 或 max_cycles 时的位置)
    if state.ee_pos_history:
        end_pos = state.ee_pos_history[-1]
        s_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=16)
        s_end.translate(end_pos)
        c = [1.0, 1.0, 0.0] if result.success else [0.5, 0.0, 0.5]
        s_end.paint_uniform_color(c)
        s_end.compute_vertex_normals()
        geoms.append(s_end)

    # graph 节点 (紫色球, 半径∝gain)
    for node in state.graph.all_nodes():
        if node.node_id == "init":
            continue
        r = 0.012 + 0.025 * min(node.last_gain / 5000.0, 1.0)
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
        s.translate(node.ee_pos_world)
        s.paint_uniform_color([0.6, 0.2, 0.8])
        s.compute_vertex_normals()
        geoms.append(s)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=seam.p_weld))

    # ---- wedge 中线 (青) + 两条边界 dir (灰) + 黄→中点连线 (黄) ----
    # 让用户直观判断 "黄球→焊缝中点" 是否落在 wedge 中央.
    from phase1.observation_check import wedge_bisector
    if seam_lim is not None:
        L = 0.55     # 参考线长度 (米)
        bis = wedge_bisector(seam.seam_limits[seam.p_weld_idx],
                             tangent=seam.tangent)

        def _ray(origin, direction, color, length=L):
            d = np.asarray(direction, dtype=np.float64)
            d = d / (np.linalg.norm(d) + 1e-12)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(
                np.vstack([origin, origin + d * length]))
            ls.lines = o3d.utility.Vector2iVector([[0, 1]])
            ls.colors = o3d.utility.Vector3dVector([color])
            return ls

        # 两条 boundary dir (浅灰) — wedge 的两条边
        sl = seam.seam_limits[seam.p_weld_idx]
        geoms.append(_ray(seam.p_weld, sl[0], [0.6, 0.6, 0.6]))
        geoms.append(_ray(seam.p_weld, sl[1], [0.6, 0.6, 0.6]))
        # 中线 (青) — 理想观测方向
        if bis is not None:
            geoms.append(_ray(seam.p_weld, bis, [0.0, 0.9, 0.9]))
        # 实际 黄球→焊缝中点 方向 (黄, 反向画成 中点→黄球便于对比中线)
        if result.success and state.ee_pos_history:
            ee = state.ee_pos_history[-1]
            rel = ee - seam.p_weld
            d2p = np.linalg.norm(rel)
            geoms.append(_ray(seam.p_weld, rel, [1.0, 0.85, 0.0], length=d2p))
            if bis is not None and d2p > 1e-9:
                cosb = float(np.dot(rel / d2p, bis))
                ang = np.degrees(np.arccos(np.clip(cosb, -1, 1)))
                print(f"[wedge] 黄球→焊缝中点 与 中线夹角 = {ang:.1f}° "
                      f"(0°=完美居中, cos={cosb:.3f})")

    print()
    print("Legend:")
    print("  灰 mesh         — 工件")
    print("  品红圆柱+端球   — 焊缝 (绿=start, 红=end)")
    print("  橙球             — P_weld 中点")
    print("  黑大球          — init pose (出发点)")
    color = "黄" if result.success else "紫"
    print(f"  {color}大球          — 最后一帧 EE 位置 "
          f"({'触发 switch' if result.success else '失败终止'})")
    print("  红折线 + 蓝小球 — cycle 间 EE 轨迹")
    print("  紫色球 (×N)     — graph 节点, 半径∝gain")
    if seam_lim is not None:
        print("  青线            — wedge 中线 (理想观测方向)")
        print("  灰线 ×2         — wedge 两条边界")
        print("  黄线            — 实际 焊缝中点→黄球 (应贴近青线)")

    title = (f"M1.8 MultiCycle | {seam.label} | "
             f"{'OK@' + str(result.phase_switch_at) if result.success else 'FAIL'} "
             f"| {elapsed:.1f}s")
    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms, window_name=title, width=1280, height=720,
        )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
