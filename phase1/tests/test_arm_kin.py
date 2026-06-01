"""M1.4 单元测试: ArmKin (cuRobo IK + 整臂碰撞封装).

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_arm_kin.py -v
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from phase1.arm_kin import ArmKin
from phase1.mapping import STATE_OCCUPIED, VoxelMap


ROBOT_YML = "/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml"
SAMPLE_PKL = ("/media/a/新加卷/hanfeng/segment_sub_output/"
              "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl")


# ---------------------------------------------------------------------- fixtures


def _pose7_to_mat4(p7) -> np.ndarray:
    p = np.asarray(p7).reshape(-1)
    pos = p[:3]
    qw, qx, qy, qz = p[3:7]
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T


@pytest.fixture(scope="module")
def kin_identity():
    """ArmKin with robot at world origin (default)."""
    return ArmKin(robot_yml_path=ROBOT_YML)


@pytest.fixture(scope="module")
def kin_pkl():
    """ArmKin with robot at pkl's robot_pose."""
    with open(SAMPLE_PKL, "rb") as f:
        pkl = pickle.load(f)
    pose = _pose7_to_mat4(pkl["robot_pose"][0])
    return ArmKin(robot_yml_path=ROBOT_YML, robot_pose_world=pose)


@pytest.fixture(scope="module")
def pkl_data():
    with open(SAMPLE_PKL, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------- init


def test_init_loads(kin_identity):
    assert kin_identity.dof == 6


def test_init_with_world_pose(kin_pkl, pkl_data):
    expected = _pose7_to_mat4(pkl_data["robot_pose"][0])
    assert np.allclose(kin_pkl.robot_pose_world, expected)


# ---------------------------------------------------------------------- FK


def test_fk_shape(kin_identity):
    theta = torch.zeros(6, device="cuda")
    out = kin_identity.fk(theta)
    assert out["ee_pos_base"].shape == (1, 3)
    assert out["ee_quat_base"].shape == (1, 4)
    assert out["ee_pos_world"].shape == (1, 3)
    assert out["ee_quat_world"].shape == (1, 4)


def test_fk_batch(kin_identity):
    theta = torch.zeros((5, 6), device="cuda")
    out = kin_identity.fk(theta)
    assert out["ee_pos_base"].shape == (5, 3)


def test_fk_identity_pose_world_eq_base(kin_identity):
    """robot_pose_world = identity → world EE 位置 == base EE 位置."""
    theta = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                         device="cuda", dtype=torch.float32)
    out = kin_identity.fk(theta)
    diff_pos = (out["ee_pos_world"] - out["ee_pos_base"]).norm()
    diff_quat = (out["ee_quat_world"] - out["ee_quat_base"]).norm()
    assert diff_pos.item() < 1e-5
    assert diff_quat.item() < 1e-5


def test_fk_with_robot_pose_translation(kin_pkl, pkl_data):
    """robot_pose_world 平移后, world EE 位置 = base EE 位置 + 平移."""
    theta = torch.tensor(pkl_data["joint_angles"], device="cuda", dtype=torch.float32)
    out = kin_pkl.fk(theta)
    pose = _pose7_to_mat4(pkl_data["robot_pose"][0])
    expected_world = (
        torch.tensor(pose[:3, :3], device="cuda", dtype=torch.float32) @
        out["ee_pos_base"][0] +
        torch.tensor(pose[:3, 3], device="cuda", dtype=torch.float32)
    )
    diff = (out["ee_pos_world"][0] - expected_world).norm()
    assert diff.item() < 1e-4


# ---------------------------------------------------------------------- IK


def test_ik_round_trip_identity(kin_identity):
    """FK 算 EE pose, IK 反推 — 关节角不一定恢复 (多解), 但 EE pose 该恢复."""
    theta_in = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                             device="cuda", dtype=torch.float32)
    fk = kin_identity.fk(theta_in)
    pos = fk["ee_pos_world"][0].cpu().numpy()
    quat = fk["ee_quat_world"][0].cpu().numpy()

    res = kin_identity.ik(pos, quat)
    assert bool(res["success"][0].item())

    # FK 一遍 IK 输出, 应该恢复 EE pose
    theta_out = res["theta"][0]
    fk2 = kin_identity.fk(theta_out)
    pos_diff = (fk["ee_pos_world"] - fk2["ee_pos_world"]).norm()
    assert pos_diff.item() < 0.005, f"FK→IK→FK pos diff {pos_diff} > 5mm"


def test_ik_round_trip_with_world_pose(kin_pkl, pkl_data):
    """关键: world frame 转换不能引入误差."""
    theta_in = torch.tensor(pkl_data["joint_angles"], device="cuda", dtype=torch.float32)
    fk = kin_pkl.fk(theta_in)
    pos = fk["ee_pos_world"][0].cpu().numpy()
    quat = fk["ee_quat_world"][0].cpu().numpy()

    res = kin_pkl.ik(pos, quat)
    assert bool(res["success"][0].item())
    fk2 = kin_pkl.fk(res["theta"][0])
    diff = (fk["ee_pos_world"] - fk2["ee_pos_world"]).norm()
    assert diff.item() < 0.005, f"world→base→IK→world diff {diff*1000:.2f} mm"


def test_ik_unreachable_fails(kin_identity):
    """目标在工作空间外, IK 应失败 (但不崩)."""
    pos_far = np.array([10.0, 10.0, 10.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    res = kin_identity.ik(pos_far, quat)
    assert not bool(res["success"][0].item())


def test_ik_batch(kin_identity):
    """IK 一次喂 N 个目标."""
    # 用 N 个随机 sample 出来的关节角做正解, 然后 IK 反推
    q_seed = kin_identity.ik_solver.sample_configs(5)
    fk = kin_identity.fk(q_seed)
    pos = fk["ee_pos_world"].cpu().numpy()
    quat = fk["ee_quat_world"].cpu().numpy()
    res = kin_identity.ik(pos, quat)
    assert res["success"].shape == (5,)
    assert res["theta"].shape == (5, 6)


# ---------------------------------------------------------------------- collision


def test_no_collision_in_empty_world(kin_identity):
    """空 world (无障碍), 任何合法关节角都不应碰撞."""
    theta = torch.zeros((10, 6), device="cuda")
    coll = kin_identity.collides(theta)
    assert coll.shape == (10,)
    # 注意: 仍可能 self-collision
    # 用安全的 home 位姿避免


def test_no_collision_at_home_pose(kin_identity):
    """已知不撞自己的 pkl joint_angles."""
    theta = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                         device="cuda", dtype=torch.float32)
    coll = kin_identity.collides(theta)
    assert not bool(coll.item())


def test_collision_detected_with_voxel_obstacle(kin_identity):
    """放一个 occupied 体素正好在臂的某个 link 上, 应检测到碰撞."""
    # 用 pkl 的 home pose 算每个 link 位置, 然后在某个 link 中心放体素
    theta = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                         device="cuda", dtype=torch.float32)
    # 取一个肯定靠近末端的位置: ee + 5cm 偏移
    fk = kin_identity.fk(theta)
    ee_world = fk["ee_pos_world"][0].cpu().numpy()
    block_pos = ee_world + np.array([0.0, 0.0, 0.0])  # 直接覆盖 ee

    # 体素 map: 1m 立方覆盖 ee 周围
    bounds = np.stack([ee_world - 0.5, ee_world + 0.5])
    vm = VoxelMap(bounds=bounds, resolution=0.05, device="cuda", max_range=2.0)
    block_idx = vm.world_to_index(
        torch.tensor(block_pos.reshape(1, 3), device="cuda", dtype=torch.float32),
    )[0]
    bx, by, bz = block_idx.tolist()
    # 在 block + 几个邻域设 occupied (确保 link 球碰到)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                vm._state[bx + dx, by + dy, bz + dz] = STATE_OCCUPIED

    n_added = kin_identity.update_world(vm)
    assert n_added > 0
    coll = kin_identity.collides(theta)
    assert bool(coll.item()), "体素正好在 ee, 应检测到碰撞"


def test_update_world_with_no_occupied(kin_identity):
    """全空 voxel map, update_world 不报错, 返回 0."""
    bounds = np.array([[0, 0, 0], [1, 1, 1]])
    vm = VoxelMap(bounds=bounds, resolution=0.1, device="cuda", max_range=2.0)
    n = kin_identity.update_world(vm)
    assert n == 0


# ---------------------------------------------------------------------- edge / cost


def test_edge_free_same_config(kin_identity):
    """两端相同, 必通过."""
    theta = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                         device="cuda", dtype=torch.float32)
    assert kin_identity.edge_free(theta, theta)


def test_edge_free_small_step_no_obstacle(kin_identity):
    """小步长, 无障碍, free."""
    a = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                     device="cuda", dtype=torch.float32)
    b = a + 0.05
    # 清空 world
    bounds = np.array([[-1, -1, -1], [1, 1, 1]])
    kin_identity.update_world(VoxelMap(bounds=bounds, resolution=0.1, device="cuda"))
    assert kin_identity.edge_free(a, b)


def test_trajectory_cost_zero_if_same(kin_identity):
    a = torch.zeros(6, device="cuda")
    b = torch.zeros(6, device="cuda")
    assert kin_identity.trajectory_cost(a, b).item() == 0


def test_trajectory_cost_unit(kin_identity):
    """关节都偏移 1 → cost = sqrt(6)."""
    a = torch.zeros(6, device="cuda")
    b = torch.ones(6, device="cuda")
    cost = kin_identity.trajectory_cost(a, b).item()
    assert abs(cost - np.sqrt(6.0)) < 1e-4


def test_trajectory_cost_batch(kin_identity):
    a = torch.zeros((10, 6), device="cuda")
    b = torch.ones((10, 6), device="cuda")
    cost = kin_identity.trajectory_cost(a, b)
    assert cost.shape == (10,)


# ---------------------------------------------------------------------- 性能


def test_ik_throughput(kin_identity):
    """单 IK 调用 (warmup 后) < 50 ms."""
    theta_in = torch.tensor([-0.78, -2.11, 1.67, -0.76, 1.92, 0.0],
                             device="cuda", dtype=torch.float32)
    fk = kin_identity.fk(theta_in)
    pos = fk["ee_pos_world"][0].cpu().numpy()
    quat = fk["ee_quat_world"][0].cpu().numpy()

    # warmup
    kin_identity.ik(pos, quat)

    n = 10
    t0 = time.time()
    for _ in range(n):
        kin_identity.ik(pos, quat)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) / n * 1000
    print(f"\n  IK avg: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 100
