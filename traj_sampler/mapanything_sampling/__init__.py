"""
MapAnything风格的共视检测和分段采样核心模块

这个模块包含：
1. 基于深度重投影验证的共视检测（MapAnything方法）
2. SE(3)运动度量
3. 分段关键帧采样算法

主要模块：
- geometry_ops: GPU几何运算模块（3D点生成、投影、深度采样）
- covisibility_gpu: 共视性校验核心（深度一致性检查、共视率计算）
- action_metrics: SE(3)运动度量模块（SE(3)距离计算）
- sampling_pipeline: 分段采样策略（整合硬性节点约束和双重几何条件）
"""

from .geometry_ops import (
    unproject_depth_to_world,
    project_world_to_pixel,
    sample_depths_at_reproj,
    inverse_transform_4x4
)

from .covisibility_gpu import (
    check_depth_consistency,
    compute_score_for_pair,
    in_image_mask
)

from .action_metrics import (
    compute_se3_distance,
    build_action_matrix
)

from .sampling_pipeline import (
    build_overlap_matrix,
    get_segmented_keyframes,
    get_adaptive_indices
)

__all__ = [
    # geometry_ops
    'unproject_depth_to_world',
    'project_world_to_pixel',
    'sample_depths_at_reproj',
    'inverse_transform_4x4',
    # covisibility_gpu
    'check_depth_consistency',
    'compute_score_for_pair',
    'in_image_mask',
    # action_metrics
    'compute_se3_distance',
    'build_action_matrix',
    # sampling_pipeline
    'build_overlap_matrix',
    'get_segmented_keyframes',
    'get_adaptive_indices',
]

