"""Step 4: 观测更新——把 raycast 结果合并进三态图。

见 privileged-nbv.md §4.5 ⑥。⑥（实拍，提交进 voxmap） vs §4.5 ④（假设性，仅打分）。

合并策略（OCCUPIED 粘滞，保守避障）：
- occ 点 → 无条件置 OCCUPIED（允许 UNKNOWN→OCC、FREE→OCC）；
- free 点 → 只把【当前非 OCCUPIED】体素置 FREE（绝不把已知障碍降级成可通行）；
- 单次观测内先写 free 再写 occ，保证命中体素最终为 OCCUPIED。
离散化下边界体素可能被不同射线判定冲突，粘滞策略保证「宁可多障碍、绝不少障碍」。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def commit_observation(voxmap, free_points, occ_points, sticky_occupied: bool = True) -> dict:
    """把一次观测的 free/occ 点体素化、按策略写进 voxmap。返回 {'free':.., 'occ':..} 生效体素数。"""
    from gt_gen.voxmap import UNKNOWN, FREE, OCCUPIED  # noqa: F401

    nf = no = 0
    fp = np.asarray(free_points)
    if fp.shape[0]:
        idx = voxmap.world_to_voxel(fp)
        idx = idx[voxmap.in_bounds(idx)]
        if sticky_occupied and idx.shape[0]:
            idx = idx[voxmap.get(idx) != OCCUPIED]   # 不降级已知障碍
        nf = voxmap.set_many(idx, FREE) if idx.shape[0] else 0

    op = np.asarray(occ_points)
    if op.shape[0]:
        no = voxmap.set_many(voxmap.world_to_voxel(op), OCCUPIED)  # 无条件，且在 free 之后 → 覆盖
    return {"free": int(nf), "occ": int(no)}


def observe_and_update(voxmap, camera_pose, camera_model, truth_scene, max_depth,
                       pixel_stride: int = 16, free_step: Optional[float] = None,
                       sticky_occupied: bool = True) -> dict:
    """实拍一次：raycast → 穿过的体素标 FREE、命中点标 OCCUPIED，提交进 voxmap。

    free_step 默认取 voxmap.voxel_size（沿射线约每体素一采样）。返回生效体素数 dict。
    """
    from gt_gen.sensor import raycast_observe

    if free_step is None:
        free_step = voxmap.voxel_size
    free_pts, occ_pts = raycast_observe(
        camera_pose, camera_model, truth_scene, max_depth,
        pixel_stride=pixel_stride, free_step=free_step)
    return commit_observation(voxmap, free_pts, occ_pts, sticky_occupied=sticky_occupied)
