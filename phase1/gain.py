"""M1.6 增益函数 — 体积增益 + 目标偏置.

主要给 M1.7 单 cycle 评分用:
    score = w_vol · volumetric_gain + w_target · target_bias

volumetric_gain:
    在相机视锥里撒 n_h × n_v 根稀疏射线 (默认 8×8 = 64),
    每条射线在 voxel_map 上 raycast, 累加沿途 unknown 体素数.

target_bias:
    cosine(angle(cam_dir, p_target - cam_pos)). 范围 [-1, +1]:
        +1 = 相机正对目标
         0 = 侧向
        -1 = 背对
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from phase1.mapping import VoxelMap


# ---------------------------------------------------------------------- config


@dataclass
class GainConfig:
    """增益参数. 可改, 但默认对应 D455 (86° × 57°)."""

    w_vol: float = 1.0
    w_target: float = 0.5
    n_rays_h: int = 8
    n_rays_v: int = 8
    fov_h_deg: float = 86.0
    fov_v_deg: float = 57.0
    max_range: float = 2.0


# ---------------------------------------------------------------------- helpers


def _camera_basis(cam_dir: np.ndarray, cam_up: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 cam_dir 和 cam_up 算相机系基向量在 world 中的表示.

    OpenCV 系: x 右, y 下, z 前.
        z = cam_dir
        x = (up × z) / |...|
        y = z × x
    """
    z = np.asarray(cam_dir, dtype=np.float64).reshape(3)
    z = z / np.linalg.norm(z)

    up = np.asarray(cam_up, dtype=np.float64).reshape(3) if cam_up is not None \
        else np.array([0.0, 0.0, 1.0])

    if abs(float(np.dot(z, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(z, up))) > 0.999:
            up = np.array([1.0, 0.0, 0.0])

    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def _frustum_dirs_world(
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    cam_up: np.ndarray | None,
    cfg: GainConfig,
) -> np.ndarray:
    """生成 n_h × n_v 根射线的方向 (world 单位向量), shape (n_h*n_v, 3)."""
    half_h = np.radians(cfg.fov_h_deg / 2.0)
    half_v = np.radians(cfg.fov_v_deg / 2.0)
    th_h = np.linspace(-half_h, half_h, cfg.n_rays_h)
    th_v = np.linspace(-half_v, half_v, cfg.n_rays_v)
    th_v_g, th_h_g = np.meshgrid(th_v, th_h, indexing="ij")  # (V, H)

    # camera frame 方向
    dx = np.tan(th_h_g)
    dy = np.tan(th_v_g)
    dz = np.ones_like(dx)
    dirs_cam = np.stack([dx, dy, dz], axis=-1)               # (V, H, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam = dirs_cam.reshape(-1, 3)                        # (V*H, 3)

    # camera → world: R 列是 (x_c, y_c, z_c)
    x_c, y_c, z_c = _camera_basis(cam_dir, cam_up)
    R = np.stack([x_c, y_c, z_c], axis=1)                     # (3, 3)
    dirs_world = dirs_cam @ R.T                                # (V*H, 3)
    return dirs_world


# ---------------------------------------------------------------------- 三个增益


def target_bias(
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    p_target: np.ndarray,
) -> float:
    """cosine(angle(cam_dir, p_target - cam_pos)). [-1, +1]."""
    cam_dir = np.asarray(cam_dir, dtype=np.float64).reshape(3)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    rel = np.asarray(p_target, dtype=np.float64).reshape(3) - \
          np.asarray(cam_pos, dtype=np.float64).reshape(3)
    n = np.linalg.norm(rel)
    if n < 1e-9:
        return 0.0
    return float(np.dot(cam_dir, rel) / n)


def volumetric_gain(
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    cam_up: np.ndarray | None,
    voxel_map: "VoxelMap",
    cfg: GainConfig | None = None,
) -> float:
    """从 cam_pos 朝 cam_dir 撒 n_h × n_v 根射线, 累加每条 raycast 沿途 unknown 体素数."""
    if cfg is None:
        cfg = GainConfig()

    cam_pos_arr = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    dirs = _frustum_dirs_world(cam_pos_arr, cam_dir, cam_up, cfg)        # (N, 3)
    N = dirs.shape[0]

    origins_t = torch.from_numpy(
        np.broadcast_to(cam_pos_arr, dirs.shape).copy()
    ).to(voxel_map.device).float()
    dirs_t = torch.from_numpy(dirs).to(voxel_map.device).float()

    out = voxel_map.raycast(origins_t, dirs_t, max_range=cfg.max_range)
    return float(out["unknown_count"].sum().item())


def gain(
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    cam_up: np.ndarray | None,
    voxel_map: "VoxelMap",
    p_target: np.ndarray,
    cfg: GainConfig | None = None,
) -> float:
    """合成评分 = w_vol·volumetric_gain + w_target·target_bias."""
    if cfg is None:
        cfg = GainConfig()
    g_vol = volumetric_gain(cam_pos, cam_dir, cam_up, voxel_map, cfg)
    g_tgt = target_bias(cam_pos, cam_dir, p_target)
    return cfg.w_vol * g_vol + cfg.w_target * g_tgt


def gain_batch(
    cam_poses: np.ndarray,                   # (N, 4, 4)
    voxel_map: "VoxelMap",
    p_target: np.ndarray,
    cfg: GainConfig | None = None,
) -> torch.Tensor:
    """批量打分. 返回 (N,) tensor on voxel_map.device.

    cam_pose 约定: world 中相机位姿 (R 列是 x_c, y_c, z_c 在 world).
    """
    if cfg is None:
        cfg = GainConfig()
    cam_poses = np.asarray(cam_poses, dtype=np.float64)
    if cam_poses.ndim != 3 or cam_poses.shape[-2:] != (4, 4):
        raise ValueError(f"cam_poses must be (N, 4, 4), got {cam_poses.shape}")
    N = cam_poses.shape[0]
    out = torch.zeros(N, device=voxel_map.device, dtype=torch.float32)
    for i in range(N):
        T = cam_poses[i]
        cam_pos = T[:3, 3]
        cam_dir = T[:3, 2]
        cam_up = -T[:3, 1]      # OpenCV: y 朝下 → up = -y
        out[i] = gain(cam_pos, cam_dir, cam_up, voxel_map, p_target, cfg)
    return out


__all__ = ["GainConfig", "target_bias", "volumetric_gain", "gain", "gain_batch"]
