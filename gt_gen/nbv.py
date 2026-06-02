"""Step 9: 特权 NBV 打分与选择。

见 privileged-nbv.md §4.5 ④、§5（fallback）、§8 顶层伪代码。
"""
from __future__ import annotations


def raycast_reveal(camera_pose, camera_model, truth_scene, max_depth):
    """假设性 raycast：算该视点"将确定"的体素（不改地图，仅用于打分）。"""
    raise NotImplementedError("Step 9")


def score_candidate(handle, cfg, B, truth_scene, camera_model, cur_cfg, lambda_cost):
    """gain = |reveal ∩ B|；score = gain - lambda * path_cost。"""
    raise NotImplementedError("Step 9")


def best_next_view_using_oracle(handle, cur_cfg, voxmap, truth_scene, goal_cfg, params):
    """顶层：P* -> reach_pt/B -> 候选 -> 打分 -> argmax。返回 (best_cfg, status)。"""
    raise NotImplementedError("Step 9")


def best_next_view_line_weighted(handle, cur_cfg, voxmap, truth_scene, goal_pose, params):
    """v1 fallback：按"离 start->goal 直线距离"反比加权的目标导向 NBV。"""
    raise NotImplementedError("Step 9")
