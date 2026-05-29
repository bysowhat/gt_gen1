"""M1.3 单元测试: 合格观测检查器.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_observation_check.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from phase1.depth_source import CameraIntrinsics
from phase1.mapping import STATE_FREE, STATE_OCCUPIED, VoxelMap
from phase1.observation_check import (
    ObservationResult,
    Viewpoint,
    check_distance,
    check_in_frustum,
    check_incidence,
    check_line_of_sight,
    is_valid_observation,
)


# ---------------------------------------------------------------------- fixtures


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _intrin_default():
    """常见焊缝相机内参: 86° × 57°, 480 × 640."""
    return CameraIntrinsics.from_fov(86.0, 57.0, 480, 640)


def _empty_free_map(bounds=None, res=0.05):
    """整个工作区初始化成 free, 没有任何障碍."""
    if bounds is None:
        bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    vm = VoxelMap(bounds=bounds, resolution=res, device=_device())
    vm._state[:] = STATE_FREE
    return vm


def _seam_at_origin():
    """焊缝点放原点, 切线沿 +x."""
    return np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------- viewpoint dataclass


def test_viewpoint_normalizes_dir():
    vp = Viewpoint(pos=[0, 0, 0], dir=[0, 0, 5])
    assert np.allclose(vp.dir, [0, 0, 1])


def test_viewpoint_zero_dir_raises():
    with pytest.raises(ValueError):
        Viewpoint(pos=[0, 0, 0], dir=[0, 0, 0])


# ---------------------------------------------------------------------- ① 距离


def test_distance_pass():
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    p_seam = np.array([0, 0, 0])
    ok, d = check_distance(vp, p_seam, d_min=0.3, d_max=0.6)
    assert ok
    assert abs(d - 0.4) < 1e-9


def test_distance_too_close():
    vp = Viewpoint(pos=[0, 0.1, 0], dir=[0, -1, 0])
    ok, d = check_distance(vp, np.zeros(3), 0.3, 0.6)
    assert not ok and abs(d - 0.1) < 1e-9


def test_distance_too_far():
    vp = Viewpoint(pos=[0, 1.0, 0], dir=[0, -1, 0])
    ok, d = check_distance(vp, np.zeros(3), 0.3, 0.6)
    assert not ok and abs(d - 1.0) < 1e-9


# ---------------------------------------------------------------------- ② 视锥


def test_frustum_in_center():
    """相机正对 P_i, 中心像素必命中."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0], up=[0, 0, 1])
    K = _intrin_default()
    assert check_in_frustum(vp, np.zeros(3), K)


def test_frustum_behind_camera():
    """P_i 在相机背后, 不在视锥."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, +1, 0], up=[0, 0, 1])  # 朝远离 P_i 方向
    K = _intrin_default()
    assert not check_in_frustum(vp, np.zeros(3), K)


def test_frustum_lateral_outside():
    """P_i 在视锥侧外."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0], up=[0, 0, 1])
    K = CameraIntrinsics.from_fov(20.0, 20.0, 100, 100)  # 窄视场
    p_far_to_side = np.array([2.0, 0, 0])  # 远离光轴
    assert not check_in_frustum(vp, p_far_to_side, K)


# ---------------------------------------------------------------------- ③ 视线无遮挡


def test_los_clear_in_free_map():
    """全 free 地图, 视线必通."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    vm = _empty_free_map()
    assert check_line_of_sight(vp, np.zeros(3), vm)


def test_los_blocked_by_occupied():
    """中间放一面 occupied 体素, 视线被挡."""
    vm = _empty_free_map()
    # 在 (0, 0.2, 0) 附近一个体素标 occupied
    block_idx = vm.world_to_index(
        torch.tensor([[0.0, 0.2, 0.0]], device=vm.device, dtype=torch.float32),
    )[0]
    bx, by, bz = block_idx.tolist()
    vm._state[bx, by, bz] = STATE_OCCUPIED

    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    assert not check_line_of_sight(vp, np.zeros(3), vm)


def test_los_blocked_by_unknown_default():
    """unknown 默认当障碍."""
    vm = VoxelMap(bounds=np.array([[-1, -1, -1], [1, 1, 1]]),
                  resolution=0.05, device=_device())
    # 默认全 unknown
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    assert not check_line_of_sight(vp, np.zeros(3), vm, unknown_as_block=True)


def test_los_unknown_allowed_when_disabled():
    """unknown_as_block=False 时, 全 unknown 也算通过."""
    vm = VoxelMap(bounds=np.array([[-1, -1, -1], [1, 1, 1]]),
                  resolution=0.05, device=_device())
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    assert check_line_of_sight(vp, np.zeros(3), vm, unknown_as_block=False)


# ---------------------------------------------------------------------- ④ 入射角


def test_incidence_perpendicular_pass():
    """视线垂直于切线 (90°), 入射角最优."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    p_seam = np.zeros(3)
    tangent = np.array([1.0, 0, 0])
    ok, ang = check_incidence(vp, p_seam, tangent, 30, 90)
    assert ok and abs(ang - 90) < 0.5


def test_incidence_parallel_fail():
    """视线和切线平行 (0°), 失败."""
    vp = Viewpoint(pos=[-0.4, 0, 0], dir=[1, 0, 0])
    p_seam = np.zeros(3)
    tangent = np.array([1.0, 0, 0])
    ok, ang = check_incidence(vp, p_seam, tangent, 30, 90)
    assert not ok and ang < 1


def test_incidence_45_pass_in_range():
    """45°, 在 [30, 90] 内通过."""
    vp = Viewpoint(pos=[-0.4, 0.4, 0], dir=[1, -1, 0])
    p_seam = np.zeros(3)
    tangent = np.array([1.0, 0, 0])
    ok, ang = check_incidence(vp, p_seam, tangent, 30, 90)
    assert ok and 44 < ang < 46


def test_incidence_uses_absolute():
    """切线反向, 结果不变."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0])
    p_seam = np.zeros(3)
    _, ang_pos = check_incidence(vp, p_seam, np.array([1.0, 0, 0]), 30, 90)
    _, ang_neg = check_incidence(vp, p_seam, np.array([-1.0, 0, 0]), 30, 90)
    assert abs(ang_pos - ang_neg) < 1e-6


# ---------------------------------------------------------------------- 主函数


def test_valid_all_pass():
    """4 条全过."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0], up=[0, 0, 1])
    p_seam, tangent = _seam_at_origin()
    vm = _empty_free_map()
    K = _intrin_default()
    r = is_valid_observation(vp, p_seam, tangent, vm, K)
    assert r.valid, f"failed at: {r.fail_reason}"
    assert r.fail_reason is None
    assert r.distance_ok and r.in_frustum_ok
    assert r.line_of_sight_ok and r.incidence_ok


def test_short_circuit_distance():
    """距离失败, 后面三条不查 (节省时间)."""
    vp = Viewpoint(pos=[0, 0.1, 0], dir=[0, -1, 0])  # 太近
    p_seam, tangent = _seam_at_origin()
    vm = _empty_free_map()
    K = _intrin_default()
    r = is_valid_observation(vp, p_seam, tangent, vm, K)
    assert not r.valid
    assert r.fail_reason == "distance"
    assert not r.distance_ok
    assert not r.in_frustum_ok          # 没查
    assert not r.line_of_sight_ok        # 没查


def test_fail_at_frustum():
    """距离过, 视锥失败."""
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, +1, 0], up=[0, 0, 1])  # 朝错方向
    p_seam, tangent = _seam_at_origin()
    vm = _empty_free_map()
    K = _intrin_default()
    r = is_valid_observation(vp, p_seam, tangent, vm, K)
    assert not r.valid
    assert r.fail_reason == "frustum"
    assert r.distance_ok and not r.in_frustum_ok


def test_fail_at_line_of_sight():
    """前两条过, 视线失败."""
    vm = _empty_free_map()
    block_idx = vm.world_to_index(
        torch.tensor([[0.0, 0.2, 0.0]], device=vm.device, dtype=torch.float32),
    )[0]
    bx, by, bz = block_idx.tolist()
    vm._state[bx, by, bz] = STATE_OCCUPIED

    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0], up=[0, 0, 1])
    p_seam, tangent = _seam_at_origin()
    K = _intrin_default()
    r = is_valid_observation(vp, p_seam, tangent, vm, K)
    assert not r.valid
    assert r.fail_reason == "line_of_sight"
    assert r.distance_ok and r.in_frustum_ok and not r.line_of_sight_ok


def test_fail_at_incidence():
    """前三条过, 入射角失败 (相机沿切线方向看)."""
    vp = Viewpoint(pos=[-0.4, 0, 0], dir=[1, 0, 0], up=[0, 0, 1])
    p_seam = np.zeros(3)
    tangent = np.array([1.0, 0, 0])  # 与视线平行 → 入射角 ≈ 0°
    vm = _empty_free_map()
    K = _intrin_default()
    r = is_valid_observation(vp, p_seam, tangent, vm, K)
    assert not r.valid
    assert r.fail_reason == "incidence"
    assert r.distance_ok and r.in_frustum_ok and r.line_of_sight_ok


# ---------------------------------------------------------------------- 性能


def test_many_calls_throughput():
    """1000 次 is_valid_observation < 1 s (单次 < 1 ms)."""
    import time
    vp = Viewpoint(pos=[0, 0.4, 0], dir=[0, -1, 0], up=[0, 0, 1])
    p_seam, tangent = _seam_at_origin()
    vm = _empty_free_map()
    K = _intrin_default()

    is_valid_observation(vp, p_seam, tangent, vm, K)  # warmup

    n = 1000
    t0 = time.time()
    for _ in range(n):
        is_valid_observation(vp, p_seam, tangent, vm, K)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n  1000 calls: {elapsed_ms:.0f} ms total ({elapsed_ms/n:.2f} ms/call)")
    assert elapsed_ms < 2000, f"too slow: {elapsed_ms} ms total"
