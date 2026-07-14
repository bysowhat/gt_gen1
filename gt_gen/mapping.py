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


def _commit_carve(voxmap, free_idx, occ_idx, sticky_occupied: bool = True) -> dict:
    """把一对 (free_idx, occ_idx) 裸下标按粘滞策略写进【单张】ThreeStateVoxelMap 子网格。"""
    from gt_gen.voxmap import FREE, OCCUPIED

    nf = no = 0
    if free_idx.shape[0]:
        if sticky_occupied:
            free_idx = free_idx[voxmap.get(free_idx) != OCCUPIED]   # 不降级已知障碍
        if free_idx.shape[0]:
            nf = voxmap.set_many(free_idx, FREE)
    if occ_idx.shape[0]:
        no = voxmap.set_many(occ_idx, OCCUPIED)                     # 无条件、free 之后 → 覆盖
    return {"free": int(nf), "occ": int(no)}


def observe_and_update(voxmap, camera_pose, camera_model, truth_scene, max_depth,
                       pixel_stride: Optional[int] = None, free_step: Optional[float] = None,
                       sticky_occupied: bool = True) -> dict:
    """实拍一次（warp 视锥体素雕刻）：视锥内体素连相机中心判遮挡 → 提交进 voxmap。

    走 sensor.carve_observe 拿 (free_idx, occ_idx) 体素下标，按粘滞策略写状态：
      free → 只写当前非 OCCUPIED 的格（不降级已知障碍）；occ → 无条件、且在 free 之后 → 覆盖。
    产实心 FREE（扛得住 sync inflate=1），优于旧 trimesh 稀疏射线。返回生效体素数 dict。

    多分辨率（MultiResVoxelMap）：对 fine（盒内全部）+ coarse（挖空 fine 盒）两子网格各雕各写回，
    合并计数。coarse 用 exclude_box=fine 盒排除盒内体素（那块交 fine 判），与壳 in_fine_box 同一
    半开判据 → 边界不留缝、不重复（方案 Row 4）。

    pixel_stride / free_step：已废弃（旧 trimesh 路径形参），保留仅为兼容调用方，忽略。
    """
    from gt_gen.sensor import carve_observe
    from gt_gen.voxmap import MultiResVoxelMap

    if isinstance(voxmap, MultiResVoxelMap):
        box = (voxmap.fine_min, voxmap.fine_max)
        total = {"free": 0, "occ": 0}
        for sub, excl in ((voxmap.fine, None), (voxmap.coarse, box)):
            fi, oi = carve_observe(sub, camera_pose, camera_model, truth_scene, max_depth,
                                   exclude_box=excl)
            r = _commit_carve(sub, fi, oi, sticky_occupied)
            total["free"] += r["free"]
            total["occ"] += r["occ"]
        return total

    free_idx, occ_idx = carve_observe(voxmap, camera_pose, camera_model, truth_scene, max_depth)
    return _commit_carve(voxmap, free_idx, occ_idx, sticky_occupied)
