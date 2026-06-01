"""M1.6 单元测试: gain (体积增益 + 目标偏置).

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_gain.py -v
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from phase1.gain import (
    GainConfig,
    gain,
    gain_batch,
    target_bias,
    volumetric_gain,
)
from phase1.mapping import STATE_FREE, STATE_OCCUPIED, VoxelMap


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _empty_unknown_map(bounds=None, res=0.05) -> VoxelMap:
    if bounds is None:
        bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    return VoxelMap(bounds=bounds, resolution=res, device=_device(), max_range=2.0)


# ---------------------------------------------------------------------- target_bias


def test_target_bias_aligned():
    """正对目标 → +1."""
    b = target_bias(
        cam_pos=np.array([0, 0, 0]),
        cam_dir=np.array([0, 0, 1]),
        p_target=np.array([0, 0, 1]),
    )
    assert abs(b - 1.0) < 1e-6


def test_target_bias_opposite():
    """背对目标 → -1."""
    b = target_bias(
        cam_pos=np.array([0, 0, 0]),
        cam_dir=np.array([0, 0, -1]),
        p_target=np.array([0, 0, 1]),
    )
    assert abs(b - (-1.0)) < 1e-6


def test_target_bias_orthogonal():
    """侧向 → 0."""
    b = target_bias(
        cam_pos=np.array([0, 0, 0]),
        cam_dir=np.array([1, 0, 0]),
        p_target=np.array([0, 0, 1]),
    )
    assert abs(b) < 1e-6


def test_target_bias_zero_distance():
    """目标 = 相机位置 → 不该崩, 返回 0."""
    b = target_bias(
        cam_pos=np.array([5, 5, 5]),
        cam_dir=np.array([0, 0, 1]),
        p_target=np.array([5, 5, 5]),
    )
    assert b == 0.0


def test_target_bias_normalizes_cam_dir():
    """cam_dir 不需要预先归一."""
    b = target_bias(
        cam_pos=np.zeros(3),
        cam_dir=np.array([0, 0, 5]),
        p_target=np.array([0, 0, 1]),
    )
    assert abs(b - 1.0) < 1e-6


# ---------------------------------------------------------------------- volumetric_gain


def test_volumetric_gain_empty_map():
    """全 unknown 地图: 任何朝向 gain 都很大."""
    vm = _empty_unknown_map()
    cfg = GainConfig(n_rays_h=8, n_rays_v=8, max_range=1.0)
    g = volumetric_gain(
        cam_pos=np.array([0, 0, 0]),
        cam_dir=np.array([0, 0, 1]),
        cam_up=np.array([0, -1, 0]),
        voxel_map=vm, cfg=cfg,
    )
    # 64 射线, 每根 1m / 0.025 步 (res/2) = 40 步, 全 unknown → ~ 64*40 = 2560
    # 实际有些射线穿过 bounds 外被剪短, 取个保守下限
    assert g > 1500, f"expected > 1500, got {g}"


def test_volumetric_gain_all_free_zero():
    """全 free 地图, gain = 0."""
    vm = _empty_unknown_map()
    vm._state[:] = STATE_FREE
    g = volumetric_gain(
        cam_pos=np.zeros(3),
        cam_dir=np.array([0, 0, 1]),
        cam_up=np.array([0, -1, 0]),
        voxel_map=vm,
        cfg=GainConfig(n_rays_h=4, n_rays_v=4, max_range=1.0),
    )
    assert g == 0.0


def test_volumetric_gain_blocked_drops():
    """前面 0.2m 处一面 occupied 墙, gain 应明显小于全 unknown."""
    vm_open = _empty_unknown_map()  # 全 unknown
    vm_blocked = _empty_unknown_map()
    # 在 z = 0.2 那一层 (z_idx = (0.2-(-1))/0.05 = 24) 全标 occupied
    z_idx = int((0.2 - (-1.0)) / 0.05)
    vm_blocked._state[:, :, z_idx] = STATE_OCCUPIED

    cfg = GainConfig(n_rays_h=4, n_rays_v=4, max_range=1.0)
    g_open = volumetric_gain(np.zeros(3), np.array([0, 0, 1]),
                             np.array([0, -1, 0]), vm_open, cfg)
    g_blocked = volumetric_gain(np.zeros(3), np.array([0, 0, 1]),
                                np.array([0, -1, 0]), vm_blocked, cfg)
    assert g_blocked < g_open / 2, f"blocked={g_blocked}, open={g_open}"


def test_volumetric_gain_orientation_dependent():
    """Half-space 已知 (z<0 全 free), 朝 +z (unknown 区) 应 >> 朝 -z."""
    vm = _empty_unknown_map()
    # 把 z<0 的体素标 free (相当于"已经探完"下半空间)
    nz_half = vm.shape[2] // 2  # bounds z=-1..1, 中点 z=0 对应 idx=nz/2
    vm._state[:, :, :nz_half] = STATE_FREE

    cfg = GainConfig(n_rays_h=8, n_rays_v=8, max_range=1.0)
    g_up = volumetric_gain(np.zeros(3), np.array([0, 0, 1]),
                            np.array([0, -1, 0]), vm, cfg)
    g_down = volumetric_gain(np.zeros(3), np.array([0, 0, -1]),
                              np.array([0, -1, 0]), vm, cfg)
    assert g_up > g_down * 5, f"up={g_up}, down={g_down}"


# ---------------------------------------------------------------------- gain (合成)


def test_gain_combines_with_weights():
    """w_target 主导 vs w_vol 主导, 得分应明显不同."""
    vm = _empty_unknown_map()
    p_target = np.array([0, 0, 1])
    cam_pos = np.zeros(3)
    cam_dir = np.array([0, 0, 1])
    cam_up = np.array([0, -1, 0])

    cfg_vol = GainConfig(w_vol=1.0, w_target=0.0,
                          n_rays_h=4, n_rays_v=4, max_range=1.0)
    cfg_tgt = GainConfig(w_vol=0.0, w_target=100.0,
                          n_rays_h=4, n_rays_v=4, max_range=1.0)

    g_vol_only = gain(cam_pos, cam_dir, cam_up, vm, p_target, cfg_vol)
    g_tgt_only = gain(cam_pos, cam_dir, cam_up, vm, p_target, cfg_tgt)

    assert g_vol_only > 100        # 全 unknown 的体积增益很大
    assert abs(g_tgt_only - 100.0) < 1.0  # target_bias = 1.0, * 100 = 100


def test_gain_target_facing_higher():
    """同位置, 一个朝目标一个背对, 朝目标的 score 高 (target_bias 主导)."""
    vm = _empty_unknown_map()
    vm._state[:] = STATE_FREE  # 关掉体积部分, 只比 target_bias
    p_target = np.array([0, 0, 1])
    cam_pos = np.zeros(3)
    cfg = GainConfig(w_vol=1.0, w_target=10.0, max_range=0.5)

    g_face = gain(cam_pos, np.array([0, 0, 1]), np.array([0, -1, 0]),
                  vm, p_target, cfg)
    g_back = gain(cam_pos, np.array([0, 0, -1]), np.array([0, -1, 0]),
                  vm, p_target, cfg)
    assert g_face > g_back


# ---------------------------------------------------------------------- batch


def test_gain_batch_shape():
    vm = _empty_unknown_map()
    p_target = np.array([0, 0, 1])
    poses = np.tile(np.eye(4), (5, 1, 1))
    out = gain_batch(poses, vm, p_target,
                     GainConfig(n_rays_h=4, n_rays_v=4, max_range=0.5))
    assert out.shape == (5,)


def test_gain_batch_pose_validation():
    vm = _empty_unknown_map()
    with pytest.raises(ValueError):
        gain_batch(np.eye(4), vm, np.zeros(3))  # 缺 batch 维


def test_gain_batch_consistent_with_single():
    """batch 5 等于循环 5 次."""
    vm = _empty_unknown_map()
    p_target = np.array([0, 0, 1])
    cfg = GainConfig(n_rays_h=4, n_rays_v=4, max_range=0.5)

    poses = []
    for i in range(5):
        T = np.eye(4)
        T[:3, 3] = [0, 0, i * 0.1]
        poses.append(T)
    poses = np.stack(poses)

    batch_out = gain_batch(poses, vm, p_target, cfg)

    # 单次循环
    expected = []
    for T in poses:
        cam_pos = T[:3, 3]
        cam_dir = T[:3, 2]
        cam_up = -T[:3, 1]
        expected.append(gain(cam_pos, cam_dir, cam_up, vm, p_target, cfg))
    expected = torch.tensor(expected, device=vm.device, dtype=torch.float32)
    assert torch.allclose(batch_out, expected, atol=1e-3)


# ---------------------------------------------------------------------- 性能


def test_volumetric_gain_throughput():
    """100 次 8x8 射线 < 5 s."""
    vm = _empty_unknown_map()
    cfg = GainConfig(n_rays_h=8, n_rays_v=8, max_range=2.0)
    # warmup
    volumetric_gain(np.zeros(3), np.array([0, 0, 1]), np.array([0, -1, 0]), vm, cfg)
    n = 100
    t0 = time.time()
    for _ in range(n):
        volumetric_gain(np.zeros(3), np.array([0, 0, 1]),
                        np.array([0, -1, 0]), vm, cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n * 1000
    print(f"\n  volumetric_gain 8x8 rays: {elapsed:.2f} ms/call")
    assert elapsed < 50
