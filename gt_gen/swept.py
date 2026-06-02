"""Step 6: 整臂扫掠体积 + 是否 ⊆ FREE。

见 privileged-nbv.md §4.5 ②(2)。决策 A = whole_arm：整条臂的碰撞球扫掠体积。
"""
from __future__ import annotations


def swept_volume(handle, q_from, q_to):
    """一段运动 q_from -> q_to 的整臂扫掠体素集合（碰撞球近似 + 细插值）。"""
    raise NotImplementedError("Step 6")


def motion_stays_in_free(handle, voxmap, q_from, q_to):
    """该段扫掠体积是否全部落在 FREE 体素内。"""
    raise NotImplementedError("Step 6")
