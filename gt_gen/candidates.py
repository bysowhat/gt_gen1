"""Step 8: 候选视点生成。

见 privileged-nbv.md §4.5 ③(1)(2)(3)。
"""
from __future__ import annotations


def cluster_centroids(B):
    """对阻塞集 B 空间聚类，取簇心当"要看的目标点" T。"""
    raise NotImplementedError("Step 8")


def standoff_poses_looking_at(T, voxmap, camera_model):
    """对目标点 T，按 (几档距离 × 若干方向) 生成朝向 T、站位在 FREE 区的相机位姿。"""
    raise NotImplementedError("Step 8")


def generate_candidates(handle, voxmap, B, camera_model, cur_cfg):
    """聚类 -> standoff 位姿 -> IK -> 保守可达性过滤，返回候选关节构型列表。"""
    raise NotImplementedError("Step 8")
