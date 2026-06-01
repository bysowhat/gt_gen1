"""M1.8 单元 + 集成测试.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_multi_cycle.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from phase1.arm_kin import ArmKin
from phase1.config import INITIAL_JOINT_ANGLES, Phase1Config
from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.exploration import (
    Phase1Result,
    Phase1State,
    compute_camera_pose,
    do_global_step,
    is_local_stuck,
    pick_frontier_node,
    reevaluate_node_gains,
    run_phase1,
)
from phase1.graph import ExplorationGraph, GraphNode
from phase1.mapping import STATE_FREE, VoxelMap
from phase1.tests._viz_helpers import load_seam_data


ROBOT_YML = "/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml"
BEAM_OBJ = ("/media/a/新加卷/hanfeng/segment_sub_output/"
            "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/"
            "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj")
SEAM_PKL = ("/media/a/新加卷/hanfeng/segment_sub_output/"
            "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_24.pkl")


# ---------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def seam():
    return load_seam_data(SEAM_PKL)


@pytest.fixture(scope="module")
def arm_kin(seam):
    return ArmKin(robot_yml_path=ROBOT_YML, robot_pose_world=seam.robot_pose)


@pytest.fixture(scope="module")
def intrin():
    return CameraIntrinsics.from_fov(86.0, 57.0, 240, 320)


@pytest.fixture(scope="module")
def sim(intrin):
    return RaycastSim(mesh_paths=[BEAM_OBJ], intrinsics=intrin)


def make_free_voxel_map(sim):
    bounds = np.stack([sim.mesh.bounds[0] - 0.5, sim.mesh.bounds[1] + 0.5])
    vm = VoxelMap(bounds=bounds, resolution=0.05, max_range=2.0)
    vm._state[:] = STATE_FREE
    return vm


# ---------------------------------------------------------------------- compute_camera_pose


def test_compute_camera_pose_shape(arm_kin):
    pose = compute_camera_pose(INITIAL_JOINT_ANGLES, arm_kin)
    assert pose.shape == (4, 4)
    assert np.allclose(pose[3], [0, 0, 0, 1])
    R = pose[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)         # 正交
    assert abs(np.linalg.det(R) - 1.0) < 1e-5                  # 旋转 (非反射)


def test_compute_camera_pose_matches_fk(arm_kin):
    """compute_camera_pose 的 t 应等于 arm_kin.fk(theta).ee_pos_world."""
    theta = INITIAL_JOINT_ANGLES
    fk = arm_kin.fk(torch.tensor(theta, dtype=torch.float32, device="cuda"))
    pos = fk["ee_pos_world"][0].cpu().numpy()
    pose = compute_camera_pose(theta, arm_kin)
    assert np.allclose(pose[:3, 3], pos, atol=1e-6)


# ---------------------------------------------------------------------- is_local_stuck


def test_is_local_stuck_below_patience():
    cfg = Phase1Config(patience=3)
    state = Phase1State(cfg=cfg, stuck_count=2, last_global_cycle=-100,
                        cycle_idx=5)
    assert not is_local_stuck(state)


def test_is_local_stuck_meets_patience():
    cfg = Phase1Config(patience=3)
    state = Phase1State(cfg=cfg, stuck_count=3, last_global_cycle=-100,
                        cycle_idx=5)
    assert is_local_stuck(state)


def test_is_local_stuck_holds_off_after_global():
    """global step 刚做过 (last_global_cycle == cycle_idx), 不能立刻再来."""
    cfg = Phase1Config(patience=3)
    state = Phase1State(cfg=cfg, stuck_count=3, last_global_cycle=5,
                        cycle_idx=5)
    assert not is_local_stuck(state)
    # cycle_idx 推 2 步以后, 才允许再 stuck
    state.cycle_idx = 7
    assert is_local_stuck(state)


# ---------------------------------------------------------------------- pick_frontier_node


def test_pick_frontier_picks_high_gain_close():
    """两个 frontier, 远的 gain 高一点应 win (跟设计文档对齐)."""
    g = ExplorationGraph()
    nodes = []
    for i in range(5):
        n = GraphNode(node_id=f"n{i}",
                      theta=np.full(6, i * 0.1),
                      ee_pos_world=np.array([i * 0.1, 0, 0]),
                      ee_dir_world=np.array([0, 0, 1.0]),
                      cycle_added=0)
        n.last_gain = 0.0
        g.add_node(n)
        nodes.append(n)
        if i > 0:
            edge_w = float(np.linalg.norm(nodes[i].theta - nodes[i-1].theta))
            g.add_edge(f"n{i-1}", f"n{i}", weight=edge_w)
    nodes[1].last_gain = 20.0
    nodes[4].last_gain = 100.0

    cfg = Phase1Config(global_lambda_cost=0.5, g_low=2.0,
                        global_frontier_gain_threshold=4.0)
    f = pick_frontier_node(g, theta_now=np.zeros(6), cfg=cfg)
    assert f is not None
    assert f.node_id == "n4"


def test_pick_frontier_no_frontier_returns_none():
    g = ExplorationGraph()
    g.add_node(GraphNode(node_id="n0",
                          theta=np.zeros(6),
                          ee_pos_world=np.zeros(3),
                          ee_dir_world=np.array([0, 0, 1.0]),
                          cycle_added=0))
    cfg = Phase1Config(g_low=10.0, global_frontier_gain_threshold=20.0)
    assert pick_frontier_node(g, theta_now=np.zeros(6), cfg=cfg) is None


def test_pick_frontier_skips_current_node():
    """当前节点本身 gain 高也不能选 (cur_node 跳过)."""
    g = ExplorationGraph()
    cur = GraphNode(node_id="cur",
                     theta=np.zeros(6),
                     ee_pos_world=np.zeros(3),
                     ee_dir_world=np.array([0, 0, 1.0]),
                     cycle_added=0)
    cur.last_gain = 999.0
    g.add_node(cur)
    other = GraphNode(node_id="other",
                       theta=np.full(6, 0.5),
                       ee_pos_world=np.array([0.5, 0, 0]),
                       ee_dir_world=np.array([0, 0, 1.0]),
                       cycle_added=0)
    other.last_gain = 50.0
    g.add_node(other)
    g.add_edge("cur", "other", weight=float(np.linalg.norm(other.theta - cur.theta)))

    cfg = Phase1Config(global_lambda_cost=0.5, g_low=10.0,
                        global_frontier_gain_threshold=20.0)
    f = pick_frontier_node(g, theta_now=np.zeros(6), cfg=cfg)
    assert f is not None
    assert f.node_id == "other"


# ---------------------------------------------------------------------- reevaluate_node_gains


def test_reevaluate_node_gains_updates_in_place(seam, sim):
    """vm 全 free + n_rays 默认: 节点 gain 应 > 0 (有体积+目标偏置)."""
    vm = make_free_voxel_map(sim)
    g = ExplorationGraph()
    n = GraphNode(node_id="n",
                   theta=np.zeros(6),
                   ee_pos_world=seam.p_weld + np.array([0.0, 0.4, 0.0]),
                   ee_dir_world=(seam.p_weld
                                  - (seam.p_weld + np.array([0.0, 0.4, 0.0]))),
                   cycle_added=0)
    n.last_gain = 0.0
    g.add_node(n)
    cfg = Phase1Config()
    reevaluate_node_gains(g, vm, seam.p_weld, cfg)
    assert g.get_node("n").last_gain > 0.0


# ---------------------------------------------------------------------- do_global_step (功能性)


def test_do_global_step_no_frontier_returns_false(arm_kin, seam, sim, intrin):
    """图里没 frontier (gain 全是 0), do_global_step 应返回 False."""
    vm = make_free_voxel_map(sim)
    g = ExplorationGraph()
    g.add_node(GraphNode(node_id="init",
                          theta=INITIAL_JOINT_ANGLES.copy(),
                          ee_pos_world=np.zeros(3),
                          ee_dir_world=np.array([0, 0, 1.0]),
                          cycle_added=0))
    cfg = Phase1Config(g_low=1e6,                       # 没人能达到
                        global_frontier_gain_threshold=1e6)
    state = Phase1State(
        cycle_idx=5, theta_now=INITIAL_JOINT_ANGLES.copy(),
        voxel_map=vm, graph=g, arm_kin=arm_kin,
        seam_points=seam.seam_line, seam_tangents=seam.seam_tangent,
        seam_limits=None, p_target=seam.p_weld,
        cfg=cfg, intrinsics=intrin,
    )
    ok = do_global_step(state, depth_source=sim)
    assert not ok


# ---------------------------------------------------------------------- run_phase1 集成


def test_run_phase1_max_cycles_termination(arm_kin, seam, sim, intrin):
    """目标点放到臂展外 → 永远无合格观测 → 跑满 max_cycles 后 fail 退出."""
    vm = make_free_voxel_map(sim)
    cfg = Phase1Config(max_cycles=3, n_samples=20)
    far_target = seam.p_weld + np.array([10.0, 10.0, 10.0])   # 10m 外, 够不到
    far_points = seam.seam_line + np.array([10.0, 10.0, 10.0])
    res = run_phase1(
        theta_init=INITIAL_JOINT_ANGLES,
        seam_points=far_points,
        seam_tangents=seam.seam_tangent,
        seam_limits=None,
        p_target=far_target,
        arm_kin=arm_kin,
        depth_source=sim,
        intrinsics=intrin,
        voxel_map=vm,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )
    assert isinstance(res, Phase1Result)
    assert not res.success
    assert "max_cycles" in res.fail_reason
    assert res.final_state.cycle_idx == cfg.max_cycles


def test_run_phase1_records_cycle_history(arm_kin, seam, sim, intrin):
    """cycle_history 应非空 (除非第一帧就触发切换 — 但 init 远离 P_weld 不会)."""
    vm = make_free_voxel_map(sim)
    cfg = Phase1Config(max_cycles=3, n_samples=20)
    res = run_phase1(
        theta_init=INITIAL_JOINT_ANGLES,
        seam_points=seam.seam_line, seam_tangents=seam.seam_tangent,
        seam_limits=None, p_target=seam.p_weld,
        arm_kin=arm_kin, depth_source=sim, intrinsics=intrin,
        voxel_map=vm, cfg=cfg, rng=np.random.default_rng(0),
    )
    # init pose 离 P_weld 1.29m, 第 0 个 cycle 的 detect_phase_switch
    # 因距离>d_max=0.6 全失败, 所以会进入 run_one_cycle
    assert len(res.final_state.cycle_history) >= 1
    # init + 至少一次 cycle 加点
    assert len(res.final_state.graph) >= 2
