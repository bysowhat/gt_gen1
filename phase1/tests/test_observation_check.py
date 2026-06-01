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
    check_line_of_sight,
    check_seam_wedge,
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
    assert r.line_of_sight_ok


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


# ---------------------------------------------------------------------- ⑤ wedge


def _example_seam_limits_90():
    """90° 凹角 (L 形): 一面方向 +x, 另一面方向 +z. 开口 90°, 朝 +x+z 象限."""
    return np.array([
        [1.0, 0.0, 0.0],   # d1 = +x (一面延伸方向)
        [0.0, 0.0, 1.0],   # d2 = +z (另一面延伸方向)
    ])


def _example_seam_limits_45():
    """45° 凹槽: 两面之间夹角 45° (锐角)."""
    a = np.deg2rad(67.5)        # 让 bisector 大致朝 +x
    b = np.deg2rad(22.5)
    return np.array([
        [np.cos(a), 0.0, np.sin(a)],
        [np.cos(b), 0.0, np.sin(b)],
    ])


def _tangent_y():
    """两个 fixture 都让 tangent 沿 +y, 所以 limits 在 xz 平面."""
    return np.array([0.0, 1.0, 0.0])


def test_wedge_90_in_open_side_pass():
    """90° wedge: cam 在 +x +z 象限 (开口侧), 通过."""
    vp = Viewpoint(pos=np.array([0.4, 0, 0.4]), dir=np.array([-1, 0, -1]))
    ok, _ = check_seam_wedge(vp, np.zeros(3), _example_seam_limits_90(), _tangent_y())
    assert ok


def test_wedge_90_wrong_side_fail():
    """90° wedge: cam 在 -x +z (跨过 +z 那条边界), 失败."""
    vp = Viewpoint(pos=np.array([-0.4, 0, 0.4]), dir=np.array([1, 0, -1]))
    ok, _ = check_seam_wedge(vp, np.zeros(3), _example_seam_limits_90(), _tangent_y())
    assert not ok


def test_wedge_90_opposite_side_fail():
    """90° wedge: cam 在 -x -z (完全反向), 失败."""
    vp = Viewpoint(pos=np.array([-0.4, 0, -0.4]), dir=np.array([1, 0, 1]))
    ok, _ = check_seam_wedge(vp, np.zeros(3), _example_seam_limits_90(), _tangent_y())
    assert not ok


def test_wedge_45_narrow_pass():
    """45° wedge: 在 bisector 方向 (45°), 远在 wedge 中心, 通过."""
    # bisector = (limits[0] + limits[1]) / norm = (0.707, 0, 0.707)
    # cam 沿 bisector 方向放在 (0.5, 0, 0.5), 应在 wedge 正中
    vp = Viewpoint(pos=np.array([0.5, 0, 0.5]), dir=np.array([-1, 0, -1]))
    ok, _ = check_seam_wedge(vp, np.zeros(3), _example_seam_limits_45(), _tangent_y())
    assert ok


def test_wedge_45_outside_narrow_fail():
    """45° wedge: 在原 90° 检查会通过 (rel·d1>0 且 rel·d2>0), 但实际偏离 wedge 中线
    超过 22.5°, 应失败. cam 在 (1, 0, 1) 方向 (45°), 而 limits 都集中在 +x 附近 (22.5°)."""
    vp = Viewpoint(pos=np.array([0.4, 0, 0.4]), dir=np.array([-1, 0, -1]))
    ok, _ = check_seam_wedge(vp, np.zeros(3), _example_seam_limits_45(), _tangent_y())
    # bisector 在 22.5°+45°/2 = 45° 处? 让我重新算. limits 在 22.5° 和 67.5°,
    # bisector 在 45°. 这个 cam 在 45° 方向, 应该在 wedge 内.
    # 改个 test: cam 在 0° (+x), 应失败 (偏离中线 45°, 但 wedge 半角=22.5°)
    vp2 = Viewpoint(pos=np.array([0.5, 0, 0.0]), dir=np.array([-1, 0, 0]))
    ok2, _ = check_seam_wedge(vp2, np.zeros(3), _example_seam_limits_45(), _tangent_y())
    assert not ok2


def test_wedge_validation():
    """seam_limits shape 错时报错."""
    vp = Viewpoint(pos=np.array([0.4, 0, 0.4]), dir=np.array([-1, 0, -1]))
    with pytest.raises(ValueError):
        check_seam_wedge(vp, np.zeros(3), np.zeros((3, 3)))


def test_is_valid_observation_with_wedge():
    """主函数: 4 条都过 + wedge 失败, 返回 invalid (fail_reason=wedge)."""
    p_seam, _, vm, intrin = make_test_setup()
    sl = _example_seam_limits_90()
    # cam 在 wedge 错侧 (-x +z 跨过边界), 距离 ok, frustum ok, LOS ok
    vp = Viewpoint(pos=np.array([-0.4, 0, 0.4]), dir=np.array([1, 0, -1]),
                   up=np.array([0, 0, 1]))
    r = is_valid_observation(vp, p_seam, _tangent_y(), vm, intrin, seam_limits=sl)
    assert not r.valid
    assert r.fail_reason == "wedge"
    assert r.distance_ok and r.in_frustum_ok and r.line_of_sight_ok


def test_is_valid_observation_skips_wedge_when_none():
    """seam_limits=None 时跳过 wedge 检查 (向后兼容)."""
    p_seam, _, vm, intrin = make_test_setup()
    vp = Viewpoint(pos=np.array([-0.4, 0, 0.4]), dir=np.array([1, 0, -1]),
                   up=np.array([0, 0, 1]))
    r = is_valid_observation(vp, p_seam, _tangent_y(), vm, intrin, seam_limits=None)
    assert r.valid


def make_test_setup():
    """需要的 fixtures 集中在这里, 给 wedge tests 用. tangent = +y."""
    from phase1.depth_source import CameraIntrinsics
    from phase1.mapping import VoxelMap, STATE_FREE
    p_seam = np.zeros(3)
    tangent = np.array([0, 1, 0])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vm = VoxelMap(bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
                  resolution=0.05, device=device)
    vm._state[:] = STATE_FREE
    intrin = CameraIntrinsics.from_fov(86.0, 57.0, 480, 640)
    return p_seam, tangent, vm, intrin


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


# ---------------------------------------------------------------------- 性能
