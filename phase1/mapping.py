"""M1.2 PyTorch 体素栅格 — 三态在线建图.

数据结构: dense 3D tensor (Nx, Ny, Nz) uint8.
状态码:   0 = unknown, 1 = free, 2 = occupied.

接口:
    VoxelMap(bounds, resolution, max_range, device)
    .integrate(depth, intrinsics, camera_pose)   把一帧 RGB-D 写入地图
    .raycast(origins, directions, max_range)     批量射线查询 (返回 hit 距离 + unknown 计数)
    .query(pts)                                   单点状态查询
    .world_to_index(pts)                          坐标转换
    .index_to_world(idx)
    .num_voxels_by_state()                        debug 计数
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # 避免循环依赖
    from phase1.depth_source import CameraIntrinsics


# 状态码常量
STATE_UNKNOWN = 0
STATE_FREE = 1
STATE_OCCUPIED = 2


class VoxelMap:
    """三态稠密体素栅格.

    Args:
        bounds:     (2, 3) [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        resolution: 体素边长 (米). 默认 0.02.
        device:     'cuda:0' / 'cpu'.
        max_range:  raycast 最远距离 (米). 默认 2.0.
        chunk_pixels: integrate 每次处理多少像素, 控制 GPU 内存. 默认 8000.
    """

    def __init__(
        self,
        bounds: np.ndarray | torch.Tensor,
        resolution: float = 0.02,
        device: str = "cuda:0",
        max_range: float = 2.0,
        chunk_pixels: int = 8000,
    ):
        bounds = np.asarray(bounds, dtype=np.float32)
        if bounds.shape != (2, 3):
            raise ValueError(f"bounds must be (2, 3), got {bounds.shape}")
        if (bounds[1] <= bounds[0]).any():
            raise ValueError(f"bounds[1] must be > bounds[0]; got {bounds}")
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0; got {resolution}")
        if max_range <= 0:
            raise ValueError(f"max_range must be > 0; got {max_range}")

        self.device = device
        self.res = float(resolution)
        self.max_range = float(max_range)
        self.chunk_pixels = int(chunk_pixels)

        self._bounds = torch.from_numpy(bounds).to(device)  # (2, 3)
        nx = int(np.ceil((bounds[1, 0] - bounds[0, 0]) / resolution))
        ny = int(np.ceil((bounds[1, 1] - bounds[0, 1]) / resolution))
        nz = int(np.ceil((bounds[1, 2] - bounds[0, 2]) / resolution))
        self._shape = (nx, ny, nz)
        self._shape_t = torch.tensor(self._shape, device=device, dtype=torch.long)
        self._state = torch.zeros(self._shape, dtype=torch.uint8, device=device)

        # 预生成沿射线的采样距离 (linspace)
        # 步长 = res/2 保证不漏穿薄物
        self._sample_step = self.res / 2.0
        n_steps = int(self.max_range / self._sample_step) + 1
        self._sample_d = torch.linspace(
            self._sample_step, self.max_range, n_steps, device=device, dtype=torch.float32,
        )  # (S,)
        self._n_steps = n_steps

    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def state(self) -> torch.Tensor:
        return self._state

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    # ------------------------------------------------------------------
    def world_to_index(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) world coord → (..., 3) int64 voxel index. 越界的 idx 整行设为 -1."""
        rel = (pts - self._bounds[0]) / self.res
        idx = rel.to(torch.long)
        valid = ((idx >= 0) & (idx < self._shape_t)).all(dim=-1)
        # 越界整行设 -1, 后续 valid mask 用 idx>=0 判断
        idx = torch.where(valid.unsqueeze(-1), idx, torch.full_like(idx, -1))
        return idx

    def index_to_world(self, idx: torch.Tensor) -> torch.Tensor:
        """(..., 3) int idx → (..., 3) 体素中心 world coord."""
        return self._bounds[0] + (idx.to(torch.float32) + 0.5) * self.res

    def query(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) → (...) uint8. 越界返回 STATE_UNKNOWN."""
        idx = self.world_to_index(pts)
        valid = (idx >= 0).all(dim=-1)
        out = torch.zeros(pts.shape[:-1], dtype=torch.uint8, device=pts.device)
        if valid.any():
            v_idx = idx[valid]
            out[valid] = self._state[v_idx[..., 0], v_idx[..., 1], v_idx[..., 2]]
        return out

    # ------------------------------------------------------------------
    def integrate(
        self,
        depth: torch.Tensor,
        intrinsics: "CameraIntrinsics",
        camera_pose: np.ndarray,
    ) -> None:
        """把一帧 depth 写入地图. depth.shape = (H, W), 单位米, 0 = no hit.

        每条射线沿 z_camera 方向:
          - sample_d < depth - res/2 → 标记为 free
          - |sample_d - depth| ≤ res/2 → 标记为 occupied (hit 体素)
          - sample_d > depth + res/2 → 不写入 (保留 unknown)
        depth = 0 (no hit) 时, 整条射线到 max_range 都标 free.
        """
        H, W = int(intrinsics.height), int(intrinsics.width)
        if depth.shape != (H, W):
            raise ValueError(f"depth shape {tuple(depth.shape)} != ({H}, {W})")

        depth = depth.to(self.device, dtype=torch.float32)

        # ---- 一次性算所有像素的 ray 方向 (camera frame) ----
        K = intrinsics
        u = torch.arange(W, device=self.device, dtype=torch.float32)
        v = torch.arange(H, device=self.device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        x_c = (uu - K.cx) / K.fx
        y_c = (vv - K.cy) / K.fy
        z_c = torch.ones_like(x_c)
        dirs_cam = torch.stack([x_c, y_c, z_c], dim=-1)  # (H, W, 3)
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)
        dirs_cam_flat = dirs_cam.reshape(-1, 3)  # (N=H*W, 3)

        # ---- camera → world ----
        pose_t = torch.as_tensor(camera_pose, dtype=torch.float32, device=self.device)
        if pose_t.shape != (4, 4):
            raise ValueError(f"camera_pose must be (4,4), got {tuple(pose_t.shape)}")
        R = pose_t[:3, :3]
        t = pose_t[:3, 3]
        dirs_world_flat = dirs_cam_flat @ R.T  # (N, 3)

        depth_flat = depth.reshape(-1)  # (N,)

        # ---- 分块写入 ----
        N = depth_flat.shape[0]
        for start in range(0, N, self.chunk_pixels):
            end = min(start + self.chunk_pixels, N)
            self._integrate_chunk(
                t,                              # 所有射线共享同一个 origin
                dirs_world_flat[start:end],     # (n, 3)
                depth_flat[start:end],          # (n,)
            )

    def _integrate_chunk(
        self,
        origin: torch.Tensor,         # (3,) 共享起点
        dirs: torch.Tensor,           # (n, 3) 单位向量
        depths: torch.Tensor,         # (n,) 0 = no hit
    ) -> None:
        n = dirs.shape[0]
        S = self._n_steps
        sample_d = self._sample_d                   # (S,)
        step = self._sample_step

        # 沿每条射线的所有采样点: (n, S, 3)
        # points = origin + sample_d[s] * dirs[i]
        pts = origin.view(1, 1, 3) + sample_d.view(1, S, 1) * dirs.view(n, 1, 3)

        # 分类: free / occupied / unknown(不写)
        no_hit = depths == 0.0                          # (n,)
        eff_d = torch.where(no_hit, torch.full_like(depths, self.max_range), depths)  # (n,)
        d_thr = eff_d.view(n, 1)                        # (n, 1)
        sd = sample_d.view(1, S)                        # (1, S)

        is_free = (sd < (d_thr - step)) | no_hit.view(n, 1)
        is_occ = (~no_hit.view(n, 1)) & (torch.abs(sd - d_thr) <= step)

        # world → idx
        idx = self.world_to_index(pts)              # (n, S, 3); 越界 = -1
        in_grid = (idx >= 0).all(dim=-1)            # (n, S)

        # 写入 (occupied 写在 free 之后, 让 occupied 优先)
        free_mask = is_free & in_grid
        occ_mask = is_occ & in_grid

        if free_mask.any():
            f_idx = idx[free_mask]
            self._state[f_idx[..., 0], f_idx[..., 1], f_idx[..., 2]] = STATE_FREE

        if occ_mask.any():
            o_idx = idx[occ_mask]
            self._state[o_idx[..., 0], o_idx[..., 1], o_idx[..., 2]] = STATE_OCCUPIED

    # ------------------------------------------------------------------
    def raycast(
        self,
        origins: torch.Tensor,      # (N, 3) world
        directions: torch.Tensor,   # (N, 3) world, 应单位向量
        max_range: float | None = None,
    ) -> dict:
        """对 N 根射线一起做 raycast.

        Returns:
            'hit':            (N,) bool, 是否击中 occupied 体素
            'distance':       (N,) float, 第一个 occupied 的距离, 没击中 = max_range
            'unknown_count':  (N,) int, 沿射线穿过的 unknown 体素数 (击中后停止)
            'first_unknown':  (N,) float, 第一个 unknown 体素的距离 (没有 = max_range)
        """
        if origins.shape != directions.shape or origins.shape[-1] != 3:
            raise ValueError(f"origins/directions shape mismatch: "
                             f"{tuple(origins.shape)} vs {tuple(directions.shape)}")

        max_range = float(max_range) if max_range is not None else self.max_range
        N = origins.shape[0]
        device = self.device

        origins = origins.to(device, dtype=torch.float32)
        directions = directions.to(device, dtype=torch.float32)

        # 用 _sample_d 截到 max_range 以内
        sample_d = self._sample_d[self._sample_d <= max_range + 1e-6]  # (S,)
        S = sample_d.shape[0]

        # (N, S, 3)
        pts = origins.view(N, 1, 3) + sample_d.view(1, S, 1) * directions.view(N, 1, 3)
        idx = self.world_to_index(pts)
        in_grid = (idx >= 0).all(dim=-1)

        # 取每个采样点的状态 (越界 = unknown)
        states = torch.zeros((N, S), dtype=torch.uint8, device=device)
        if in_grid.any():
            v_idx = idx[in_grid]
            states[in_grid] = self._state[v_idx[..., 0], v_idx[..., 1], v_idx[..., 2]]

        is_occ = states == STATE_OCCUPIED        # (N, S)
        is_unk = states == STATE_UNKNOWN

        # 第一个 occupied 的索引
        any_occ = is_occ.any(dim=1)              # (N,)
        # argmax 在全 False 时返回 0; 用 any_occ 区分
        first_occ_idx = is_occ.float().argmax(dim=1)  # (N,)
        first_occ_dist = torch.where(
            any_occ,
            sample_d[first_occ_idx],
            torch.full((N,), max_range, dtype=torch.float32, device=device),
        )

        # 第一个 unknown 的索引
        any_unk = is_unk.any(dim=1)
        first_unk_idx = is_unk.float().argmax(dim=1)
        first_unk_dist = torch.where(
            any_unk,
            sample_d[first_unk_idx],
            torch.full((N,), max_range, dtype=torch.float32, device=device),
        )

        # unknown 计数 (击中前的部分)
        # mask: 对每条射线, 未击中前的步骤为 True
        step_idx = torch.arange(S, device=device).view(1, S)  # (1, S)
        all_true = torch.ones((N, S), dtype=torch.bool, device=device)
        before_hit = step_idx < first_occ_idx.view(N, 1)
        mask_before_hit = torch.where(any_occ.view(N, 1), before_hit, all_true)
        unknown_count = (is_unk & mask_before_hit).sum(dim=1)

        return {
            "hit": any_occ,
            "distance": first_occ_dist,
            "unknown_count": unknown_count,
            "first_unknown": first_unk_dist,
        }

    # ------------------------------------------------------------------
    def num_voxels_by_state(self) -> dict[str, int]:
        """统计三态各多少, 用于 debug."""
        st = self._state
        return {
            "unknown": int((st == STATE_UNKNOWN).sum().item()),
            "free": int((st == STATE_FREE).sum().item()),
            "occupied": int((st == STATE_OCCUPIED).sum().item()),
        }


__all__ = [
    "VoxelMap",
    "STATE_UNKNOWN",
    "STATE_FREE",
    "STATE_OCCUPIED",
]
