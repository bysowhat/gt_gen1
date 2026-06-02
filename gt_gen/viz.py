"""可视化辅助（穿插在各步验证中使用）。"""
from __future__ import annotations


def show_voxmap(voxmap, **kw):
    raise NotImplementedError("viz")


def show_rays(camera_pose, hits, **kw):
    raise NotImplementedError("viz")


def show_candidates(B, candidates, **kw):
    raise NotImplementedError("viz")
