"""M1.7 视觉验证: 跑一次 single cycle, 在 Open3D 里画候选 + 决策.

输出:
  灰 mesh           — 工件
  品红圆柱 + 端球   — 焊缝
  橙球              — P_weld
  青大球              — 当前 EE 位置
  浅灰小点          — 全部 N 个采样候选
  浅蓝小点          — 通过 IK 过滤的
  绿色小点          — 通过 IK + 碰撞过滤的
  黄大球 + 黄线     — top-1 best (next_theta 对应 EE)
  灰色 bbox         — VoxelMap 边界

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_single_cycle.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_24.pkl"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d
import torch

from phase1.arm_kin import ArmKin
from phase1.config import INITIAL_JOINT_ANGLES, Phase1Config
from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.exploration import (
    filter_by_collision,
    filter_by_ik,
    run_one_cycle,
    sample_candidate_ee_poses,
    score_candidates,
)
from phase1.graph import ExplorationGraph, GraphNode
from phase1.mapping import STATE_FREE, STATE_OCCUPIED, VoxelMap
from phase1.tests._viz_helpers import load_seam_data, look_at, make_seam_geoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--robot-yml",
        default="/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--free-world", action="store_true",
                        help="不喂 mesh 占据 (世界全 free, 调试用)")
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
    print(f"[arm] dof={arm_kin.dof}")

    # ---- voxel map: 用 mesh 做 ground-truth 占据 ----
    bounds = np.stack([sim.mesh.bounds[0] - 0.5, sim.mesh.bounds[1] + 0.5])
    vm = VoxelMap(bounds=bounds, resolution=0.03, max_range=2.0)
    if args.free_world:
        vm._state[:] = STATE_FREE
        print("[map] free-world mode (全 free, 不会撞)")
    else:
        # 用 libigl 直接填 mesh 内部为 occupied
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
        counts = vm.num_voxels_by_state()
        print(f"[map] free={counts['free']}, occupied={counts['occupied']} "
              f"(libigl mesh-fill)")

    # ---- 初始 graph: 起点 = 固定 INITIAL_JOINT_ANGLES (= ur12e retract_config) ----
    # 注意: pkl 里 joint_angles 是焊接到位姿 (EE 贴 P_weld), 不是初始位姿!
    theta_init = INITIAL_JOINT_ANGLES
    g = ExplorationGraph()
    fk_init = arm_kin.fk(torch.tensor(theta_init, dtype=torch.float32,
                                       device="cuda"))
    init_node = GraphNode(
        node_id="init",
        theta=theta_init.copy(),
        ee_pos_world=fk_init["ee_pos_world"][0].cpu().numpy(),
        ee_dir_world=fk_init["ee_quat_world"][0].cpu().numpy()[:3],
        cycle_added=0,
    )
    g.add_node(init_node)
    ee_pos_now = init_node.ee_pos_world
    print(f"[init] theta_init = retract_config (Phase 1 起点)")
    print(f"       EE at {ee_pos_now.round(3)}, distance to P_weld = "
          f"{np.linalg.norm(ee_pos_now - seam.p_weld):.3f} m")

    # ---- 跑 single cycle ----
    cfg = Phase1Config(n_samples=args.n_samples)
    print(f"[cfg] n_samples={cfg.n_samples}, sample_box_half={cfg.sample_box_half}")
    print()
    t0 = time.time()
    result = run_one_cycle(
        theta_now=theta_init.copy(),
        voxel_map=vm,
        graph=g,
        arm_kin=arm_kin,
        seam_points=seam.seam_line,
        seam_tangents=seam.seam_tangent,
        seam_limits=seam.seam_limits,
        p_target=seam.p_weld,
        cycle_idx=1,
        cfg=cfg,
        intrinsics=K,
        rng=rng,
    )
    elapsed = time.time() - t0

    print(f"[cycle] success={result.success}, took {elapsed:.2f} s")
    print(f"  candidates: {result.n_candidates_total} → "
          f"IK pass {result.n_candidates_after_ik} → "
          f"collision pass {result.n_candidates_after_collision}")
    if result.success:
        print(f"  best_score: {result.best_score:.2f}  "
              f"(gain {result.best_gain:.0f} - "
              f"λ·cost {cfg.lambda_cost*result.best_traj_cost:.2f})")
        print(f"  next_theta: {np.array(result.next_theta).round(3).tolist()}")
        print(f"  next_ee_pos: {result.next_ee_pose_world[:3, 3].round(3)}")
        print(f"  phase_switch: {result.phase_switch_triggered}, "
              f"observed P_i: {result.p_seam_observed}")

    # ---- Open3D viz ----
    # 重新跑一次采样 + 过滤, 拿到中间结果 (因为 run_one_cycle 不返回这些)
    rng2 = np.random.default_rng(args.seed)
    arm_kin.update_world(vm)
    poses_all = sample_candidate_ee_poses(ee_pos_now, seam.p_weld, cfg,
                                            cfg.n_samples, rng=rng2)
    poses_ik, thetas_ik = filter_by_ik(poses_all, arm_kin)
    poses_col, thetas_col = filter_by_collision(thetas_ik, poses_ik, arm_kin)

    geoms = []
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)
    geoms.extend(make_seam_geoms(seam))

    def _pcd(positions, color):
        if len(positions) == 0:
            return None
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(positions)
        p.paint_uniform_color(color)
        return p

    # 全部采样 (浅灰)
    p_all = _pcd(poses_all[:, :3, 3], [0.75, 0.75, 0.75])
    if p_all: geoms.append(p_all)
    # IK pass (浅蓝)
    p_ik = _pcd(poses_ik[:, :3, 3], [0.4, 0.6, 1.0])
    if p_ik: geoms.append(p_ik)
    # collision pass (绿)
    p_col = _pcd(poses_col[:, :3, 3], [0.2, 0.85, 0.2])
    if p_col: geoms.append(p_col)

    # 当前 EE (青大球)
    cur = o3d.geometry.TriangleMesh.create_sphere(radius=0.04, resolution=16)
    cur.translate(ee_pos_now); cur.paint_uniform_color([0.0, 1.0, 1.0])
    cur.compute_vertex_normals()
    geoms.append(cur)

    # top-1 (黄大 + 黄线)
    if result.success and result.next_ee_pose_world is not None:
        bp = result.next_ee_pose_world[:3, 3]
        bs = o3d.geometry.TriangleMesh.create_sphere(radius=0.025, resolution=16)
        bs.translate(bp); bs.paint_uniform_color([1.0, 1.0, 0.0])
        bs.compute_vertex_normals()
        geoms.append(bs)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.vstack([bp, seam.p_weld]))
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geoms.append(line)
        # 当前 EE → top-1 也画一条线 (运动方向)
        line2 = o3d.geometry.LineSet()
        line2.points = o3d.utility.Vector3dVector(np.vstack([ee_pos_now, bp]))
        line2.lines = o3d.utility.Vector2iVector([[0, 1]])
        line2.colors = o3d.utility.Vector3dVector([[0.5, 0.0, 0.5]])  # 紫
        geoms.append(line2)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=seam.p_weld))

    print()
    print("Legend (Open3D):")
    print("  灰 mesh          — 工件")
    print("  品红圆柱+端球    — 焊缝")
    print(f"  橙球              — P_weld")
    print(f"  青大球            — 当前 EE 位置")
    print(f"  浅灰点 (×{cfg.n_samples}) — 全部采样候选")
    print(f"  浅蓝点 (×{len(poses_ik)})  — 通过 IK")
    print(f"  绿色点 (×{len(poses_col)})  — 通过 IK + 碰撞")
    print(f"  黄大球+黄线        — top-1 best (next move)")
    print(f"  紫色线            — 当前 EE → top-1 运动方向")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"M1.7 SingleCycle  |  {seam.label}",
            width=1280, height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
