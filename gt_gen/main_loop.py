"""Step 10: 主循环编排（准备阶段 + ①~⑦）+ 冷启动 + 卡住处理。

见 privileged-nbv.md §4.5 完整主循环。
"""
from __future__ import annotations


def look_around(handle, voxmap, cur_cfg, camera_model, truth_scene, max_depth):
    """冷启动 / 卡住兜底：在当前自由区内小幅摆动相机，扫开周边未知。"""
    raise NotImplementedError("Step 10")


def handle_stuck(handle, voxmap, cur_cfg, truth_scene):
    """就近揭示最多任意未知的可达视点；连续无进展则判场景卡死。"""
    raise NotImplementedError("Step 10")


def generate_gt(handle, voxmap, truth_scene, retract_config, goal_pose, params):
    """完整 ①~⑦ 主循环，返回到达目标的整条关节轨迹（=GT）及状态。"""
    raise NotImplementedError("Step 10")
