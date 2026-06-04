"""Step 2: 三态体素地图（FREE / OCCUPIED / UNKNOWN）。

见 docs/gt-generation-curobo-implementation.md §2.1、privileged-nbv.md §4.5。

约定：
- 坐标系 = 机械臂 **base_link**（与 cuRobo 规划世界一致 → Step 5 同步最省事）。
- `origin` = 体素 [0,0,0] 的**最小角**（lower corner），单位米。
- `voxel_to_world(idx)` 返回体素**中心**坐标。
- 下标用整型 (i,j,k)，i↔x、j↔y、k↔z；内部用 numpy uint8 网格，order='C'。
- 所有查询/设置方法都支持单个 (3,) 或批量 (N,3) 下标/点。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# 三态常量
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

STATE_NAMES = {UNKNOWN: "UNKNOWN", FREE: "FREE", OCCUPIED: "OCCUPIED"}


class ThreeStateVoxelMap:
    """ROI 包围盒内的三态占据栅格；初始全 UNKNOWN。

    职责：世界(base)坐标 <-> 体素下标互转、查询/设置、导出"非 FREE"掩码（供碰撞同步）。
    """

    def __init__(self, origin: Sequence[float], size_xyz: Sequence[float], voxel_size: float):
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.voxel_size = float(voxel_size)
        size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
        # 体素数：四舍五入，至少 1（避免退化）
        self.dims = np.maximum(np.round(size / self.voxel_size).astype(np.int64), 1)  # (nx,ny,nz)
        self.grid = np.full(tuple(self.dims), UNKNOWN, dtype=np.uint8)

    # ---- 形状/几何信息 ----
    @property
    def shape(self):
        return tuple(int(x) for x in self.dims)

    @property
    def num_voxels(self) -> int:
        return int(self.grid.size)

    @property
    def extent(self) -> np.ndarray:
        """物理尺寸 (米)，= dims * voxel_size。"""
        return self.dims * self.voxel_size

    @property
    def center(self) -> np.ndarray:
        """ROI 包围盒中心（base 坐标）；供 Step 5 映射到 cuRobo 的 center-based VoxelGrid。"""
        return self.origin + 0.5 * self.extent

    @property
    def upper(self) -> np.ndarray:
        """最大角（base 坐标）。"""
        return self.origin + self.extent

    # ---- 坐标互转 ----
    def world_to_voxel(self, p) -> np.ndarray:
        """base 坐标 -> 体素下标（向下取整）。p:(3,)->（3,)；(N,3)->(N,3)，dtype=int。

        不做边界裁剪——可能返回越界下标，用 in_bounds() 过滤。
        """
        p = np.asarray(p, dtype=np.float64)
        idx = np.floor((p - self.origin) / self.voxel_size).astype(np.int64)
        return idx

    def voxel_to_world(self, idx) -> np.ndarray:
        """体素下标 -> 体素中心 base 坐标。idx:(3,)->（3,)；(N,3)->(N,3)。"""
        idx = np.asarray(idx, dtype=np.float64)
        return self.origin + (idx + 0.5) * self.voxel_size

    def in_bounds(self, idx) -> np.ndarray:
        """下标是否在网格内。idx:(3,)->标量 bool；(N,3)->(N,) bool。"""
        idx = np.asarray(idx)
        ok = (idx >= 0) & (idx < self.dims)
        return np.all(ok, axis=-1)

    # ---- 查询 / 设置 ----
    def get(self, idx):
        """取状态。idx:(3,)->标量 int；(N,3)->(N,) uint8。越界返回 UNKNOWN（保守、仍属非 FREE）。"""
        idx = np.asarray(idx, dtype=np.int64)
        single = idx.ndim == 1
        idx2 = idx.reshape(1, 3) if single else idx
        inb = self.in_bounds(idx2)
        out = np.full(idx2.shape[0], UNKNOWN, dtype=np.uint8)
        if inb.any():
            ii = idx2[inb]
            out[inb] = self.grid[ii[:, 0], ii[:, 1], ii[:, 2]]
        return int(out[0]) if single else out

    def set_many(self, indices, state: int):
        """把一批体素设为某状态；越界下标静默跳过。indices:(3,) 或 (N,3)。"""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx.reshape(1, 3)
        inb = self.in_bounds(idx)
        if not inb.any():
            return 0
        ii = idx[inb]
        self.grid[ii[:, 0], ii[:, 1], ii[:, 2]] = np.uint8(state)
        return int(inb.sum())

    def set_world(self, points, state: int):
        """便捷：直接用 base 坐标点批量设置（内部转下标）。points:(3,) 或 (N,3)。"""
        return self.set_many(self.world_to_voxel(points), state)

    def fill(self, state: int):
        """整图置为某状态。"""
        self.grid[...] = np.uint8(state)

    # ---- 掩码 / 统计 / 导出 ----
    def non_free_mask(self) -> np.ndarray:
        """返回 OCCUPIED ∪ UNKNOWN 的占据掩码（bool, 形状=dims），喂给 cuRobo 碰撞世界。"""
        return self.grid != FREE

    def counts(self) -> dict:
        """各状态体素数 {UNKNOWN:.., FREE:.., OCCUPIED:..}。"""
        return {s: int((self.grid == s).sum()) for s in (UNKNOWN, FREE, OCCUPIED)}

    def state_centers(self, state: int) -> np.ndarray:
        """返回某状态全部体素的中心 base 坐标，形状 (M,3)；用于可视化/调试。"""
        ii = np.argwhere(self.grid == state)
        if ii.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return self.voxel_to_world(ii)


def build_roi_voxmap(config, margin_voxels: int = 1) -> ThreeStateVoxelMap:
    """按 config.roi（单一 ROI 来源：center/dims/voxel_size_m）构建空地图（全 UNKNOWN）。

    与 init_curobo 的 cuRobo voxel 世界**同框**（同 center/dims/voxel）。额外向外扩 margin_voxels
    个体素，使 voxmap 比 cuRobo 网格稍大，包住其每轴 `1+floor(dim/voxel)` 多出的"+1"边界层
    （否则那层落在 voxmap 外 → 越界判 UNKNOWN→占据，在 ROI 边界形成多余"墙"）。
    """
    vs = config.voxel_size_m
    center = np.asarray(config.roi_center, dtype=np.float64).reshape(3)
    dims = np.asarray(config.roi_dims, dtype=np.float64).reshape(3)
    size = dims + 2.0 * margin_voxels * vs
    origin = center - 0.5 * size
    return ThreeStateVoxelMap(origin=origin, size_xyz=size, voxel_size=vs)
