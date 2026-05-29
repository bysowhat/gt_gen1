"""M1.1 单元测试.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_depth_source.py -v
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from phase1.depth_source import CameraIntrinsics, RaycastSim


# ---------------------------------------------------------------------- helpers


def _cube_obj(extents=(1.0, 1.0, 1.0), out_path="/tmp/test_cube.obj") -> str:
    cube = trimesh.primitives.Box(extents=extents)
    cube.export(out_path)
    return out_path


def _cube_sim(intrin=None) -> RaycastSim:
    if intrin is None:
        intrin = CameraIntrinsics.from_fov(90.0, 90.0, 100, 100)
    return RaycastSim(mesh_paths=[_cube_obj()], intrinsics=intrin)


def _cam_pose_looking_down(z: float = 2.0) -> np.ndarray:
    """Camera at (0,0,z), z_cam = -world_z (相机朝下).

    R 列向量: x_cam, y_cam, z_cam (in world).
        x_cam = +world_x      (向右)
        y_cam = -world_y      (camera y 向下, 在 world 里也朝负 y)
        z_cam = -world_z      (相机光轴朝下)
    """
    R = np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]],
        dtype=np.float64,
    )
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[2, 3] = z
    return pose


# ---------------------------------------------------------------------- intrinsics


def test_intrinsics_from_fov_square():
    """90° × 90°, 100×100: fx = fy = 50."""
    K = CameraIntrinsics.from_fov(h_fov_deg=90.0, v_fov_deg=90.0, height=100, width=100)
    assert abs(K.fx - 50.0) < 1e-6
    assert abs(K.fy - 50.0) < 1e-6
    assert K.cx == 50.0
    assert K.cy == 50.0
    assert K.width == 100
    assert K.height == 100


def test_intrinsics_post_init_coerces_int():
    K = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, height=100.0, width=100.0)
    assert isinstance(K.height, int)
    assert isinstance(K.width, int)


def test_intrinsics_validation():
    with pytest.raises(ValueError):
        CameraIntrinsics(fx=-1, fy=100, cx=50, cy=50, height=100, width=100)
    with pytest.raises(ValueError):
        CameraIntrinsics(fx=100, fy=100, cx=50, cy=50, height=0, width=100)


# ---------------------------------------------------------------------- raycast core


def test_unit_cube_center_pixel_distance():
    """1×1×1 立方体中心在原点, 相机在 (0,0,2) 朝 -z 看. 中心像素深度 = 1.5."""
    sim = _cube_sim()
    depth = sim.render(_cam_pose_looking_down(2.0))
    assert depth.shape == (100, 100)
    # 中心 (50, 50) 应命中立方体顶面 z=0.5, 距离 = 2 - 0.5 = 1.5
    cy, cx = 50, 50
    d = float(depth[cy, cx])
    assert abs(d - 1.5) < 1e-3, f"center depth = {d:.4f}, expected 1.5"


def test_unit_cube_offset_camera():
    """相机平移到 (0.2, 0.0, 2.0), 中心像素仍命中立方体顶面同一深度 1.5."""
    sim = _cube_sim()
    pose = _cam_pose_looking_down(2.0)
    pose[0, 3] = 0.2
    depth = sim.render(pose)
    d = float(depth[50, 50])
    assert abs(d - 1.5) < 1e-3, f"got {d:.4f}"


def test_no_hit_returns_zero():
    """相机朝 +z 看 (远离立方体), 全部像素深度 = 0."""
    sim = _cube_sim()
    pose = np.eye(4)
    pose[2, 3] = 2.0
    # 默认 R = identity 即 z_cam = +world z, 远离立方体
    depth = sim.render(pose)
    assert depth.shape == (100, 100)
    assert float(depth.max()) == 0.0


def test_corner_pixels_either_miss_or_farther():
    """角落像素要么打不到 (0), 要么距离 ≥ 中心像素."""
    sim = _cube_sim()
    depth = sim.render(_cam_pose_looking_down(2.0))
    center_d = float(depth[50, 50])
    for y, x in [(0, 0), (0, 99), (99, 0), (99, 99)]:
        d = float(depth[y, x])
        if d > 0:
            assert d >= center_d - 1e-3, f"corner ({y},{x})={d}, center={center_d}"


def test_two_cubes_concatenation():
    """两个立方体, 各在 (0,0,0) 和 (3,0,0). 用宽 FOV 应同时看到."""
    p1 = _cube_obj(out_path="/tmp/test_cube_a.obj")
    p2 = _cube_obj(out_path="/tmp/test_cube_b.obj")
    pose1 = np.eye(4)
    pose2 = np.eye(4)
    pose2[0, 3] = 3.0
    sim = RaycastSim(
        mesh_paths=[p1, p2],
        mesh_poses=[pose1, pose2],
        intrinsics=CameraIntrinsics.from_fov(120.0, 90.0, 100, 200),
    )
    # 工厂方法等价: assert
    assert sim.num_triangles > 0
    pose = _cam_pose_looking_down(2.0)
    depth = sim.render(pose)
    n_hit = int((depth > 0).sum())
    # 至少有一个立方体顶面被打到
    assert n_hit > 50, f"only {n_hit} hits"


def test_pose_shape_validation():
    sim = _cube_sim()
    with pytest.raises(ValueError):
        sim.render(np.eye(3))
    with pytest.raises(ValueError):
        sim.render(np.zeros((4,)))


def test_mesh_path_does_not_exist():
    with pytest.raises(FileNotFoundError):
        RaycastSim(mesh_paths=["/tmp/this_does_not_exist.obj"])


def test_mesh_pose_count_mismatch():
    with pytest.raises(ValueError):
        RaycastSim(
            mesh_paths=[_cube_obj()],
            mesh_poses=[np.eye(4), np.eye(4)],
        )


# ---------------------------------------------------------------------- device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_output_on_cuda():
    sim = _cube_sim()
    depth = sim.render(_cam_pose_looking_down(2.0))
    assert depth.device.type == "cuda"


def test_output_dtype_float32():
    sim = _cube_sim()
    depth = sim.render(_cam_pose_looking_down(2.0))
    assert depth.dtype == torch.float32


# ---------------------------------------------------------------------- batch


def test_render_batch_matches_single():
    sim = _cube_sim()
    pose = _cam_pose_looking_down(2.0)
    d_single = sim.render(pose)
    d_batch = sim.render_batch(pose[None].repeat(3, axis=0))
    assert d_batch.shape == (3, 100, 100)
    for i in range(3):
        assert torch.allclose(d_batch[i], d_single)


def test_render_batch_pose_shape_validation():
    sim = _cube_sim()
    with pytest.raises(ValueError):
        sim.render_batch(np.eye(4))  # 缺 N 维


# ---------------------------------------------------------------------- performance


def test_performance_warmup():
    """Warmup 后 100×100 单帧 < 50 ms."""
    sim = _cube_sim()
    pose = _cam_pose_looking_down(2.0)
    sim.render(pose)  # warmup BVH

    n = 10
    t0 = time.time()
    for _ in range(n):
        sim.render(pose)
    elapsed_ms = (time.time() - t0) / n * 1000
    print(f"\n  100x100 raycast: {elapsed_ms:.2f} ms/frame")
    assert elapsed_ms < 50.0, f"too slow: {elapsed_ms:.2f} ms/frame"
