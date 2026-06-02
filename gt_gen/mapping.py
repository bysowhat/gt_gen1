"""Step 4: 观测更新——把 raycast 结果合并进三态图。

见 privileged-nbv.md §4.5 ⑥。⑥（实拍，提交进 voxmap） vs §4.5 ④（假设性，仅打分）。
"""
from __future__ import annotations


def observe_and_update(voxmap, camera_pose, camera_model, truth_scene, max_depth):
    """实拍一次：穿过的体素标 FREE、命中点标 OCCUPIED，提交进 voxmap。"""
    raise NotImplementedError("Step 4")
