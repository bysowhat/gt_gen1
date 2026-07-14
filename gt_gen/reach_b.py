"""Step 7: reach_pt 与阻塞段 B。

见 privileged-nbv.md §4.5 ②(2)(3)、§8 顶层伪代码。

- reach_pt：沿 P*(真值最优路径) 逐点向前扫，整臂扫掠体积 ⊆ FREE 则前进，
  首次碰到非 FREE 体素的那一段停下，其前一个点 = reach_pt（能保守走到的最远点）。
- B（阻塞段）：reach_pt 前方一小段(k 个路点)扫掠体积里【仍 UNKNOWN】的体素集合
  ——即"卡住下一步、唯一原因是还没看过"的格子（不含 FREE/OCCUPIED）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_reach_pt(handle, voxmap, p_star, resolution: Optional[float] = None) -> int:
    """沿 P* 逐点向前扫：整臂扫掠 ⊆ FREE 则前进，首个越界处停。返回 reach_idx。"""
    from gt_gen.swept import motion_stays_in_free

    p_star = np.asarray(p_star, float)
    reach_idx = 0
    for i in range(len(p_star) - 1):
        ok, _ = motion_stays_in_free(handle, voxmap, p_star[i], p_star[i + 1], resolution=resolution)
        if ok:
            reach_idx = i + 1
        else:
            break
    return reach_idx


def compute_blocking_B(handle, voxmap, p_star, reach_idx, k_lookahead,
                       resolution: Optional[float] = None) -> np.ndarray:
    """reach_pt 前方 k 个路点扫掠体积里仍 UNKNOWN 的体素集合 = B。返回 (M,3) 下标。

    多分辨率：B 是【体素下标集合】，统一在 coarse 索引空间算（index_grid）——NBV 视点选择用，见
    voxmap.index_grid。GT 闸门（compute_reach_pt 的 motion_stays_in_free）仍走整壳、fine 精度。
    """
    from gt_gen.swept import swept_volume
    from gt_gen.voxmap import UNKNOWN, index_grid

    g = index_grid(voxmap)                                    # 多分辨率壳 → coarse；单图 → 自身
    p_star = np.asarray(p_star, float)
    B = set()
    end = min(reach_idx + int(k_lookahead), len(p_star) - 1)
    for i in range(reach_idx, end):
        cells = swept_volume(handle, g, p_star[i], p_star[i + 1], resolution=resolution)
        if cells.shape[0] == 0:
            continue
        st = np.asarray(g.get(cells))
        for v in map(tuple, cells[st == UNKNOWN]):
            B.add(v)
    if not B:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(sorted(B), dtype=np.int64)
