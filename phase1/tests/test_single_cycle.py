"""M1.7 单元测试: 单 cycle 探索.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_single_cycle.py -v
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from phase1.arm_kin import ArmKin
from phase1.config import INITIAL_JOINT_ANGLES, Phase1Config
from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.exploration import (
    CycleResult,
    detect_phase_switch,
    filter_by_collision,
    filter_by_ik,
    run_one_cycle,
    sample_candidate_ee_poses,
    score_candidates,
)
from phase1.graph import ExplorationGraph, GraphNode
from phase1.mapping import STATE_FREE, VoxelMap
from phase1.observation_check import Viewpoint
from phase1.tests._viz_helpers import load_seam_data, look_at


ROBOT_YML = "/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml"
BEAM_OBJ = ("/media/a/新加卷/hanfeng/segment_sub_output/"
            "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/"
            "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj")
SAMPLE_PKL = ("/media/a/新加卷/hanfeng/segment_sub_output/"
              "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl")


# ---------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def seam():
    return load_seam_data(SAMPLE_PKL)


@pytest.fixture(scope="module")
def kin(seam):
    return ArmKin(robot_yml_path=ROBOT_YML, robot_pose_world=seam.robot_pose)


@pytest.fixture(scope="module")
def intrin():
    return CameraIntrinsics.from_fov(86.0, 57.0, 360, 640)


@pytest.fixture(scope="module")
def vm(seam):
    """Free voxel map covering BEAM workspace."""
    bounds = np.array([[19.0, -1.0, 10.5], [29.0, 1.5, 13.0]])
    vm = VoxelMap(bounds=bounds, resolution=0.05, max_range=2.0)
    vm._state[:] = STATE_FREE
    return vm


# ---------------------------------------------------------------------- sample


def test_sample_count_and_shape(seam):
    cfg = Phase1Config(sample_box_half=0.15, n_samples=50)
    poses = sample_candidate_ee_poses(seam.p_weld, seam.p_weld + np.array([0,1,0]),
                                       cfg, n=50, rng=np.random.default_rng(0))
    assert poses.shape == (50, 4, 4)


def test_sample_position_in_box(seam):
    cfg = Phase1Config(sample_box_half=0.15)
    poses = sample_candidate_ee_poses(seam.p_weld, seam.p_weld + np.array([0,1,0]),
                                       cfg, n=200, rng=np.random.default_rng(0))
    rel = poses[:, :3, 3] - seam.p_weld
    assert (np.abs(rel) <= 0.15 + 1e-6).all()


def test_sample_homogeneous_row(seam):
    cfg = Phase1Config()
    poses = sample_candidate_ee_poses(seam.p_weld, seam.p_weld + np.array([1,0,0]),
                                       cfg, n=10, rng=np.random.default_rng(0))
    # 最后一行应该是 [0, 0, 0, 1]
    last_row = poses[:, 3, :]
    assert np.allclose(last_row, [0, 0, 0, 1])


def test_sample_orientation_biased_to_target(seam):
    """无抖动 (yaw=pitch=0) 时, z_axis 应直接指向 target."""
    cfg = Phase1Config(yaw_jitter_deg=0, pitch_jitter_deg=0,
                        sample_box_half=0.05)
    target = seam.p_weld + np.array([0.5, 0, 0])
    poses = sample_candidate_ee_poses(seam.p_weld, target,
                                       cfg, n=10, rng=np.random.default_rng(0))
    for T in poses:
        z_world = T[:3, 2]
        to_tgt = target - T[:3, 3]
        to_tgt_n = to_tgt / np.linalg.norm(to_tgt)
        assert np.dot(z_world, to_tgt_n) > 0.99   # 几乎完全对齐


# ---------------------------------------------------------------------- IK filter


def test_filter_by_ik_drops_unreachable(kin):
    """工作空间外的 pose 全部 IK 失败."""
    poses = np.tile(np.eye(4), (10, 1, 1))
    poses[:, :3, 3] = np.array([100, 100, 100])     # 远到不可能 IK
    pp, tt = filter_by_ik(poses, kin)
    assert len(pp) == 0
    assert tt.shape == (0, kin.dof)


def test_filter_by_ik_keeps_reachable(kin, seam):
    """从 pkl 的 home pose 算 ee, 应该能 IK 反推."""
    theta = torch.tensor(seam.joint_angles, dtype=torch.float32, device="cuda")
    fk = kin.fk(theta)
    pos = fk["ee_pos_world"][0].cpu().numpy()
    quat_wxyz = fk["ee_quat_world"][0].cpu().numpy()

    # 构造 (1, 4, 4) pose
    from scipy.spatial.transform import Rotation as ScipyRot
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = ScipyRot.from_quat(quat_xyzw).as_matrix()
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = pos
    pp, tt = filter_by_ik(pose[None], kin)
    assert len(pp) == 1


# ---------------------------------------------------------------------- collision filter


def test_filter_by_collision_empty_world_passes(kin, seam, vm):
    """全 free world, 多个安全姿态都不撞."""
    arm_kin = kin
    arm_kin.update_world(vm)
    theta = torch.tensor(seam.joint_angles, dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    poses = np.tile(np.eye(4), (1, 1, 1))           # dummy
    pp, tt = filter_by_collision(theta, poses, arm_kin)
    assert len(pp) == 1


# ---------------------------------------------------------------------- score


def test_score_shape(kin, seam, vm):
    """K 个候选 → (K,) 分数."""
    cfg = Phase1Config()
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, 3] = seam.p_weld + np.array([0.1, 0, 0.1])
    poses[0, :3, 3] = seam.p_weld + np.array([0.1, 0.1, 0])
    thetas = torch.zeros((3, kin.dof), dtype=torch.float32, device="cuda")
    theta_now = np.array(seam.joint_angles)

    scores, gains, costs = score_candidates(
        poses, thetas, theta_now, vm, seam.p_weld, kin, cfg,
    )
    assert scores.shape == (3,)
    assert gains.shape == (3,)
    assert costs.shape == (3,)


def test_score_empty_input(kin, seam, vm):
    """K=0, 返回 0 长向量."""
    cfg = Phase1Config()
    poses = np.zeros((0, 4, 4))
    thetas = torch.zeros((0, kin.dof), device="cuda")
    s, g, c = score_candidates(poses, thetas, np.array(seam.joint_angles),
                                vm, seam.p_weld, kin, cfg)
    assert len(s) == 0


# ---------------------------------------------------------------------- detect_phase_switch


def test_detect_phase_switch_returns_indices(seam, vm, intrin):
    """全 free voxel map + 朝 P_weld 视角. 至少有 P_weld 自己合格 (距离 0)?
    实际上 距离 = 0 不在 [0.3, 0.6], 会失败. 所以应返回 []."""
    cfg = Phase1Config()
    vp = Viewpoint(pos=seam.p_weld, dir=np.array([1, 0, 0]),
                   up=np.array([0, 0, 1]))
    obs = detect_phase_switch(vm, seam.seam_line, seam.seam_tangent,
                               seam.seam_limits, vp, cfg, intrin)
    assert isinstance(obs, list)
    # 距离=0 < d_min, 都失败
    assert obs == []


def test_detect_phase_switch_finds_valid_view(seam, vm, intrin):
    """构造一个 viewpoint, 距离 ~0.45m 朝 P_weld, 应找到至少 1 个 P_i 合格.
    (vm 全 free, seam_limits 可能为 0 → wedge 跳过)."""
    cfg = Phase1Config(obs_los_unknown_as_block=False)
    cam_pos = seam.p_weld + np.array([0.0, 0.45, 0.0])
    cam_dir = (seam.p_weld - cam_pos)
    cam_dir /= np.linalg.norm(cam_dir)
    vp = Viewpoint(pos=cam_pos, dir=cam_dir, up=np.array([0, 0, 1]))
    # seam_limits 全 0 时 wedge 检查会出问题, 这里手动设 None
    obs = detect_phase_switch(vm, seam.seam_line, seam.seam_tangent,
                               None, vp, cfg, intrin)
    assert len(obs) > 0


# ---------------------------------------------------------------------- 主函数


def test_run_one_cycle_returns_action(kin, seam, vm, intrin):
    """完整 cycle, free world, 应返回 success=True + 非空 next_theta."""
    cfg = Phase1Config(n_samples=30, max_cycles=10)
    graph = ExplorationGraph()
    init_node = GraphNode(
        node_id="init",
        theta=INITIAL_JOINT_ANGLES.copy(),
        ee_pos_world=np.zeros(3),
        ee_dir_world=np.array([0, 0, 1.0]),
        cycle_added=0,
    )
    graph.add_node(init_node)

    result = run_one_cycle(
        theta_now=INITIAL_JOINT_ANGLES.copy(),
        voxel_map=vm,
        graph=graph,
        arm_kin=kin,
        seam_points=seam.seam_line,
        seam_tangents=seam.seam_tangent,
        seam_limits=None,
        p_target=seam.p_weld,
        cycle_idx=1,
        cfg=cfg,
        intrinsics=intrin,
        rng=np.random.default_rng(0),
    )
    assert result.success
    assert result.next_theta is not None
    assert result.next_theta.shape == (kin.dof,)
    assert result.n_candidates_total == cfg.n_samples
    assert result.n_candidates_after_ik > 0
    assert result.n_candidates_after_collision > 0
    # graph 应有 init + new 两个节点
    assert len(graph) == 2


def test_run_one_cycle_collision_world_returns_failure(kin, seam, intrin):
    """全 occupied 世界, 没法不撞, 应返回 success=False.

    bounds 必须罩住 init EE 周围整个采样盒 (box=0.30), 否则盒外=free 会漏过碰撞.
    """
    import torch
    fk = kin.fk(torch.tensor(INITIAL_JOINT_ANGLES, dtype=torch.float32,
                              device="cuda"))
    ee0 = fk["ee_pos_world"][0].cpu().numpy()
    pad = 0.6                                       # > sample_box_half, 留余量
    bounds = np.stack([ee0 - pad, ee0 + pad])
    vm_blocked = VoxelMap(bounds=bounds, resolution=0.05, max_range=2.0)
    from phase1.mapping import STATE_OCCUPIED
    vm_blocked._state[:] = STATE_OCCUPIED          # 全堵死

    cfg = Phase1Config(n_samples=20)
    graph = ExplorationGraph()
    graph.add_node(GraphNode(node_id="init",
                              theta=INITIAL_JOINT_ANGLES.copy(),
                              ee_pos_world=np.zeros(3),
                              ee_dir_world=np.array([0, 0, 1]),
                              cycle_added=0))

    result = run_one_cycle(
        theta_now=INITIAL_JOINT_ANGLES.copy(),
        voxel_map=vm_blocked,
        graph=graph,
        arm_kin=kin,
        seam_points=seam.seam_line,
        seam_tangents=seam.seam_tangent,
        seam_limits=None,
        p_target=seam.p_weld,
        cycle_idx=1,
        cfg=cfg,
        intrinsics=intrin,
        rng=np.random.default_rng(0),
    )
    assert not result.success
    assert result.next_theta is None
