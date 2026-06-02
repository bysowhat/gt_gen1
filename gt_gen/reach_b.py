"""Step 7: reach_pt 与阻塞段 B。

见 privileged-nbv.md §4.5 ②(2)(3)、§8 顶层伪代码。
"""
from __future__ import annotations


def compute_reach_pt(handle, voxmap, p_star):
    """沿 P* 逐点向前扫：整臂扫掠体积 ⊆ FREE 则前进，首个越界处停。

    返回 reach_idx（P* 上能保守走到的最远下标）。
    """
    raise NotImplementedError("Step 7")


def compute_blocking_B(handle, voxmap, p_star, reach_idx, k_lookahead):
    """reach_pt 前方 k 个路点扫掠体积里仍 UNKNOWN 的体素集合 = B。"""
    raise NotImplementedError("Step 7")
