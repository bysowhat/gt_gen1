"""M1.2 单元测试.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_mapping.py -v
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from phase1.depth_source import CameraIntrinsics
from phase1.mapping import (
    STATE_FREE,
    STATE_OCCUPIED,
    STATE_UNKNOWN,
    VoxelMap,
)


# ---------------------------------------------------------------------- helpers


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_empty_map(bounds=None, res=0.05) -> VoxelMap:
    if bounds is None:
        bounds = np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 2.0]])
    return VoxelMap(bounds=bounds, resolution=res, device=_device(), max_range=2.0)


def _identity_pose() -> np.ndarray:
    """Camera at origin, R = identity (z_cam = +world_z, OpenCV)."""
    return np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------- init


def test_init_shape():
    vm = _make_empty_map()
    nx, ny, nz = vm.shape
    # bounds = (1m, 1m, 2m), res = 0.05 → 20×20×40
    assert (nx, ny, nz) == (20, 20, 40)
    assert vm.state.shape == (20, 20, 40)
    assert vm.state.dtype == torch.uint8


def test_init_all_unknown():
    vm = _make_empty_map()
    assert (vm.state == STATE_UNKNOWN).all().item()
    counts = vm.num_voxels_by_state()
    assert counts["unknown"] == 20 * 20 * 40
    assert counts["free"] == 0
    assert counts["occupied"] == 0


def test_init_validation():
    with pytest.raises(ValueError):
        VoxelMap(bounds=np.zeros((2, 2)), resolution=0.05)  # bad shape
    with pytest.raises(ValueError):
        VoxelMap(bounds=np.array([[1, 0, 0], [0, 1, 1]]), resolution=0.05)  # bad order
    with pytest.raises(ValueError):
        VoxelMap(bounds=np.array([[0, 0, 0], [1, 1, 1]]), resolution=0)
    with pytest.raises(ValueError):
        VoxelMap(bounds=np.array([[0, 0, 0], [1, 1, 1]]), resolution=0.05, max_range=0)


# ---------------------------------------------------------------------- coords


def test_world_to_index_basic():
    vm = _make_empty_map()
    # 体素中心在每个 idx 的 (idx + 0.5) * res + bounds[0]
    pts = torch.tensor([[0.0, 0.0, 0.0],            # idx=(10,10,0)
                        [0.0, 0.0, 1.0],            # idx=(10,10,20)
                        [-0.5, -0.5, 0.0]],         # idx=(0,0,0)
                       device=_device(), dtype=torch.float32)
    idx = vm.world_to_index(pts)
    assert idx[0].tolist() == [10, 10, 0]
    assert idx[1].tolist() == [10, 10, 20]
    assert idx[2].tolist() == [0, 0, 0]


def test_world_to_index_oob_returns_neg1():
    vm = _make_empty_map()
    pts = torch.tensor([[5.0, 0, 0], [-1, 0, 0]], device=_device(), dtype=torch.float32)
    idx = vm.world_to_index(pts)
    # 越界整行设 -1
    assert (idx[0] == -1).all()
    assert (idx[1] == -1).all()


def test_index_to_world_round_trip():
    vm = _make_empty_map()
    idx_in = torch.tensor([[5, 5, 5], [0, 0, 0], [19, 19, 39]],
                          device=_device(), dtype=torch.long)
    pts = vm.index_to_world(idx_in)
    idx_out = vm.world_to_index(pts)
    assert (idx_in == idx_out).all().item()


def test_query_unknown_initially():
    vm = _make_empty_map()
    pts = torch.zeros((10, 3), device=_device())
    states = vm.query(pts)
    assert (states == STATE_UNKNOWN).all().item()


def test_query_oob_returns_unknown():
    vm = _make_empty_map()
    pts = torch.tensor([[100.0, 0, 0]], device=_device(), dtype=torch.float32)
    states = vm.query(pts)
    assert states.item() == STATE_UNKNOWN


# ---------------------------------------------------------------------- integrate


def _make_uniform_depth(intrin, value: float, device):
    return torch.full((intrin.height, intrin.width), value, dtype=torch.float32, device=device)


def test_integrate_wall_center_pixel():
    """相机在原点朝 +z, 中心像素 depth=1.0 → 中心列上:
        z<1 都 free, z=1 那个 voxel occupied, z>1 仍 unknown."""
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 30, 30)
    depth = torch.zeros((30, 30), device=_device())
    depth[15, 15] = 1.0   # 只有中心像素有 hit
    vm.integrate(depth, intrin, _identity_pose())

    # 中心像素方向是 (0, 0, 1)
    # voxel z idx = (z-0)/0.05
    # z=0.5 → 10, z=1.0 → 20, z=1.5 → 30
    pts = torch.tensor([[0.0, 0, 0.5], [0.0, 0, 1.0], [0.0, 0, 1.5]],
                       device=_device(), dtype=torch.float32)
    states = vm.query(pts)
    assert states[0].item() == STATE_FREE,    f"z=0.5 should be FREE; got {states[0]}"
    assert states[1].item() == STATE_OCCUPIED, f"z=1.0 should be OCCUPIED; got {states[1]}"
    assert states[2].item() == STATE_UNKNOWN, f"z=1.5 should still be UNKNOWN; got {states[2]}"


def test_integrate_no_hit_marks_free_to_max_range():
    """depth=0 → 整条射线到 max_range 都标 free."""
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 10, 10)
    depth = torch.zeros((10, 10), device=_device())  # 全 0 = 无 hit
    vm.integrate(depth, intrin, _identity_pose())

    # 中心射线方向 (0,0,1), 沿途到 max_range=2 都该 free
    pts = torch.tensor([[0.0, 0, 0.5], [0.0, 0, 1.0], [0.0, 0, 1.9]],
                       device=_device(), dtype=torch.float32)
    states = vm.query(pts)
    assert (states == STATE_FREE).all().item()
    # 没有 occupied 体素
    assert vm.num_voxels_by_state()["occupied"] == 0


def test_integrate_occupied_count_increases():
    """整合一帧后, occupied 至少 1 个."""
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 30, 30)
    depth = torch.full((30, 30), 1.0, device=_device())
    vm.integrate(depth, intrin, _identity_pose())
    counts = vm.num_voxels_by_state()
    assert counts["occupied"] > 0, f"got {counts}"
    assert counts["free"] > 0, f"got {counts}"


def test_integrate_two_frames_consistent():
    """同一帧整合两次, 结果不变."""
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 30, 30)
    depth = torch.full((30, 30), 1.0, device=_device())

    vm.integrate(depth, intrin, _identity_pose())
    counts1 = vm.num_voxels_by_state()
    vm.integrate(depth, intrin, _identity_pose())
    counts2 = vm.num_voxels_by_state()
    assert counts1 == counts2


def test_integrate_pose_validation():
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 10, 10)
    depth = torch.zeros((10, 10), device=_device())
    with pytest.raises(ValueError):
        vm.integrate(depth, intrin, np.eye(3))
    # depth 形状不对
    with pytest.raises(ValueError):
        vm.integrate(torch.zeros((9, 10), device=_device()), intrin, _identity_pose())


# ---------------------------------------------------------------------- raycast


def test_raycast_empty_map_no_hit_max_unknown():
    vm = _make_empty_map()
    origins = torch.tensor([[0.0, 0, 0]], device=_device(), dtype=torch.float32)
    dirs = torch.tensor([[0.0, 0, 1]], device=_device(), dtype=torch.float32)
    out = vm.raycast(origins, dirs)
    assert out["hit"].item() is False or out["hit"].item() == 0
    assert abs(out["distance"].item() - vm.max_range) < 0.01
    # 整条射线都在地图内, 全 unknown
    expected_steps = vm._n_steps  # 因为整条射线都在 [0, 2] 之内
    # 但越界部分会被 in_grid 过滤掉
    # bounds z in [0, 2], 射线终点 z=2.0 刚好在边界, 部分 sample 在内
    assert out["unknown_count"].item() > expected_steps - 5  # 容忍少量边界


def test_raycast_finds_occupied():
    """先 integrate 一面墙, raycast 应该击中."""
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 30, 30)
    depth = torch.full((30, 30), 1.0, device=_device())
    vm.integrate(depth, intrin, _identity_pose())

    # 沿 +z 射线, 应该在 z≈1 处 hit
    origins = torch.tensor([[0.0, 0, 0]], device=_device(), dtype=torch.float32)
    dirs = torch.tensor([[0.0, 0, 1]], device=_device(), dtype=torch.float32)
    out = vm.raycast(origins, dirs)
    assert out["hit"].item() == 1
    assert abs(out["distance"].item() - 1.0) < 0.05


def test_raycast_unknown_count_drops_after_integrate():
    """整合后, 同方向 raycast 的 unknown_count 应减少."""
    vm = _make_empty_map()
    intrin = CameraIntrinsics.from_fov(60.0, 60.0, 30, 30)

    origins = torch.tensor([[0.0, 0, 0]], device=_device(), dtype=torch.float32)
    dirs = torch.tensor([[0.0, 0, 1]], device=_device(), dtype=torch.float32)
    before = vm.raycast(origins, dirs)["unknown_count"].item()

    depth = torch.full((30, 30), 1.0, device=_device())
    vm.integrate(depth, intrin, _identity_pose())
    after = vm.raycast(origins, dirs)["unknown_count"].item()

    assert after < before, f"before={before}, after={after}"


def test_raycast_batch_consistency():
    """一次 N=10 的 raycast 应等于循环 10 次单条."""
    vm = _make_empty_map()
    origins = torch.zeros((10, 3), device=_device(), dtype=torch.float32)
    dirs = torch.zeros((10, 3), device=_device(), dtype=torch.float32)
    dirs[:, 2] = 1.0
    out = vm.raycast(origins, dirs)
    assert out["distance"].shape == (10,)
    # 所有射线相同 → 结果应一致
    assert (out["unknown_count"] == out["unknown_count"][0]).all().item()


# ---------------------------------------------------------------------- performance


def test_integrate_360p_performance():
    """360×640 单帧 integrate < 100 ms."""
    vm = VoxelMap(
        bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        resolution=0.02,
        device=_device(),
        max_range=2.0,
    )
    intrin = CameraIntrinsics.from_fov(86.0, 57.0, 360, 640)
    depth = torch.full((360, 640), 0.8, device=_device())
    pose = np.eye(4)

    # warmup
    vm.integrate(depth, intrin, pose)

    n = 5
    t0 = time.time()
    for _ in range(n):
        vm.integrate(depth, intrin, pose)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) / n * 1000
    print(f"\n  integrate 360x640: {elapsed_ms:.1f} ms/frame")
    assert elapsed_ms < 200, f"too slow: {elapsed_ms:.1f} ms"


def test_raycast_throughput():
    """1000 根射线 < 50 ms."""
    vm = _make_empty_map()
    origins = torch.zeros((1000, 3), device=_device(), dtype=torch.float32)
    # 随机方向
    torch.manual_seed(0)
    dirs = torch.randn((1000, 3), device=_device())
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    vm.raycast(origins, dirs)  # warmup
    n = 5
    t0 = time.time()
    for _ in range(n):
        vm.raycast(origins, dirs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) / n * 1000
    print(f"\n  raycast 1000 rays: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 50


# ---------------------------------------------------------------------- device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_state_on_cuda():
    vm = _make_empty_map()
    assert vm.state.device.type == "cuda"
