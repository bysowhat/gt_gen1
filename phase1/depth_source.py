"""M1.1 RaycastSim — 用 trimesh + embreex 把 OBJ 网格当 ground truth 模拟深度相机.

抽象接口 DepthSource:
    .render(camera_pose: (4,4) np.ndarray) -> torch.Tensor (H, W) float32 in meters

实现 RaycastSim 用 BVH 求交; M1.9 会再加一个 IsaacSimDepth 共享接口.

相机系约定 (OpenCV / Isaac Sim):
    +z 朝前 (光轴),  +x 右,  +y 下

camera_pose 是 4x4 齐次, 相机在 world 中的位姿.
    R = pose[:3, :3]: 列分别是 x_cam, y_cam, z_cam 在 world 中的方向
    t = pose[:3,  3]: 相机在 world 中的位置

像素 (u, v) → camera-frame 射线方向:
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    z_c = 1
    dir_cam = normalize([x_c, y_c, z_c])
    dir_world = R @ dir_cam
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch
import trimesh


class DepthSource(Protocol):
    """统一的深度来源接口. M1.1 RaycastSim, M1.9 IsaacSimDepth 都实现它."""

    def render(self, camera_pose: np.ndarray) -> torch.Tensor:
        """camera_pose: (4,4). 返回 (H, W) float32 on configured device.
        没 hit 的像素深度 = 0.
        """
        ...


@dataclass
class CameraIntrinsics:
    """针孔相机内参 + 图像尺寸."""

    fx: float
    fy: float
    cx: float
    cy: float
    height: int
    width: int

    def __post_init__(self) -> None:
        # 确保 dtype, 防止从 yaml 传 float 进来
        self.height = int(self.height)
        self.width = int(self.width)
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"fx, fy must be > 0; got fx={self.fx}, fy={self.fy}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"image size must be > 0; got {self.width}x{self.height}")

    @classmethod
    def from_fov(
        cls,
        h_fov_deg: float,
        v_fov_deg: float,
        height: int,
        width: int,
    ) -> "CameraIntrinsics":
        """从对称视场角构造. 默认相机中心在图像正中."""
        fx = 0.5 * width / np.tan(np.radians(h_fov_deg) / 2.0)
        fy = 0.5 * height / np.tan(np.radians(v_fov_deg) / 2.0)
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   height=height, width=width)


class RaycastSim:
    """基于 trimesh + embreex 的 ground-truth 深度模拟器.

    Args:
        mesh_paths: OBJ/STL/PLY 等网格文件路径列表.
        mesh_poses: 每个网格的 (4,4) world 系初始位姿. None = 全部 identity.
        intrinsics: 相机内参. None = 默认 D455 360p.
        device:    输出深度图所在 device.
    """

    def __init__(
        self,
        mesh_paths: Iterable[str | Path],
        mesh_poses: list[np.ndarray] | None = None,
        intrinsics: CameraIntrinsics | None = None,
        device: str = "cuda:0",
    ) -> None:
        mesh_paths = [Path(p) for p in mesh_paths]
        if not mesh_paths:
            raise ValueError("at least one mesh path required")
        if mesh_poses is None:
            mesh_poses = [np.eye(4) for _ in mesh_paths]
        if len(mesh_poses) != len(mesh_paths):
            raise ValueError(
                f"mesh_poses length {len(mesh_poses)} != mesh_paths length {len(mesh_paths)}"
            )

        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(86.0, 57.0, 360, 640)
        self.device = device

        loaded: list[trimesh.Trimesh] = []
        for path, pose in zip(mesh_paths, mesh_poses):
            if not path.exists():
                raise FileNotFoundError(path)
            m = trimesh.load(str(path), force="mesh")
            if not isinstance(m, trimesh.Trimesh):
                raise ValueError(f"loaded {path} is not a Trimesh (got {type(m).__name__})")
            pose = np.asarray(pose, dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"mesh pose must be (4,4), got {pose.shape}")
            m.apply_transform(pose)
            loaded.append(m)

        self.mesh: trimesh.Trimesh = trimesh.util.concatenate(loaded)
        # 触发 BVH 缓存; 第一次访问会建索引
        _ = self.mesh.ray
        # 预计算像素射线方向 (camera frame)
        self._dirs_cam = self._make_pixel_dirs()  # (H*W, 3) float64

    # ------------------------------------------------------------------
    def _make_pixel_dirs(self) -> np.ndarray:
        K = self.intrinsics
        u = np.arange(K.width)
        v = np.arange(K.height)
        uu, vv = np.meshgrid(u, v, indexing="xy")  # (H, W)
        x = (uu - K.cx) / K.fx
        y = (vv - K.cy) / K.fy
        z = np.ones_like(x)
        dirs = np.stack([x, y, z], axis=-1).astype(np.float64)  # (H, W, 3)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs.reshape(-1, 3)

    # ------------------------------------------------------------------
    def render(self, camera_pose: np.ndarray) -> torch.Tensor:
        """主接口. camera_pose: (4,4). 返回 (H, W) torch.float32, 单位米."""
        K = self.intrinsics
        camera_pose = np.asarray(camera_pose, dtype=np.float64)
        if camera_pose.shape != (4, 4):
            raise ValueError(f"camera_pose must be (4,4), got {camera_pose.shape}")

        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

        # camera dir → world: R 的列是基向量, dirs_world = R @ dirs_cam
        # 等价 dirs_world[i] = (R @ dirs_cam[i]) = dirs_cam[i] @ R.T
        dirs_world = self._dirs_cam @ R.T                       # (H*W, 3)
        origins_world = np.broadcast_to(t, dirs_world.shape).copy()

        # intersects_first 只算 hit 的三角形 id, 比 intersects_location 快 4-5 倍.
        # 然后我们自己用 ray-plane 算距离.
        tri_ids = self.mesh.ray.intersects_first(
            ray_origins=origins_world,
            ray_directions=dirs_world,
        )                                                        # (H*W,) int, -1 = no hit
        hit_mask = tri_ids >= 0

        depth_flat = np.zeros(K.height * K.width, dtype=np.float32)
        if hit_mask.any():
            # ray-plane intersection: 用 hit 三角形的法线 + 任一顶点
            hit_tris = self.mesh.faces[tri_ids[hit_mask]]        # (M, 3) 顶点 idx
            v0 = self.mesh.vertices[hit_tris[:, 0]]              # (M, 3) 三角形一个顶点
            normals = self.mesh.face_normals[tri_ids[hit_mask]]  # (M, 3)
            o = origins_world[hit_mask]                          # (M, 3)
            d = dirs_world[hit_mask]                             # (M, 3)
            # 平面方程: (P - v0) · n = 0; P = o + t·d
            # → t = (v0 - o) · n / (d · n)
            num = np.sum((v0 - o) * normals, axis=1)
            den = np.sum(d * normals, axis=1)
            # den 不会是 0 (否则不会 hit), 但留一个保护
            t_hit = np.where(np.abs(den) > 1e-12, num / den, 0.0)
            depth_flat[hit_mask] = t_hit.astype(np.float32)

        depth = depth_flat.reshape(K.height, K.width)
        return torch.from_numpy(depth).to(self.device)

    # ------------------------------------------------------------------
    def render_batch(self, camera_poses: np.ndarray) -> torch.Tensor:
        """(N, 4, 4) → (N, H, W). 当前是简单循环, 后续可向量化."""
        camera_poses = np.asarray(camera_poses)
        if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
            raise ValueError(
                f"camera_poses must be (N,4,4), got {camera_poses.shape}"
            )
        return torch.stack([self.render(p) for p in camera_poses])

    # ------------------------------------------------------------------
    @property
    def num_triangles(self) -> int:
        return int(len(self.mesh.faces))


__all__ = ["DepthSource", "CameraIntrinsics", "RaycastSim"]
