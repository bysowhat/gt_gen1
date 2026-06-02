"""Step 5: voxmap -> cuRobo 碰撞世界同步（未知=障碍）。

见 docs/gt-generation-curobo-implementation.md §4.2。
关键：cuRobo 默认乐观（只把已知占据当障碍）；这里把"非 FREE"全部当占据。
"""
from __future__ import annotations


def sync_collision_world(handle, voxmap):
    """把 voxmap 的 (OCCUPIED ∪ UNKNOWN) 灌进 cuRobo 的 voxel 碰撞世界。"""
    raise NotImplementedError("Step 5")
