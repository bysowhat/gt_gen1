"""Step 2: 三态体素地图（FREE / OCCUPIED / UNKNOWN）。

见 docs/gt-generation-curobo-implementation.md §2.1、privileged-nbv.md §4.5。
"""
from __future__ import annotations

# 三态常量
UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class ThreeStateVoxelMap:
    """ROI 包围盒内的三态占据栅格；初始全 UNKNOWN。

    职责：世界坐标 <-> 体素下标互转、查询/设置、导出"非 FREE"掩码（供碰撞同步）。
    """

    def __init__(self, origin, size_xyz, voxel_size):
        raise NotImplementedError("Step 2")

    def world_to_voxel(self, p):
        raise NotImplementedError("Step 2")

    def voxel_to_world(self, idx):
        raise NotImplementedError("Step 2")

    def get(self, idx):
        raise NotImplementedError("Step 2")

    def set_many(self, indices, state):
        raise NotImplementedError("Step 2")

    def non_free_mask(self):
        """返回 OCCUPIED ∪ UNKNOWN 的占据掩码（喂给 cuRobo 碰撞世界）。"""
        raise NotImplementedError("Step 2")


def build_roi_voxmap(config, reach_radius):
    """按 ROI = 可达范围 + expand_m、分辨率 voxel_size_m 构建空地图。"""
    raise NotImplementedError("Step 2")
