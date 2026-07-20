"""
按照空间位姿采样核心模块

这个模块包含：
2. SE(3)运动度量
3. 分段关键帧采样算法

主要模块：
- action_metrics: SE(3)运动度量模块（SE(3)距离计算）
- sampling_pipeline: 分段采样策略（整合硬性节点约束和双重几何条件）
"""


from .action_metrics import (
    compute_se3_distance,
    build_action_matrix
)

from .sampling import (
    get_segmented_keyframes,
    sample_keyframes_single,
)

from .kinematics import UR12e_t, CameraFKConfig

__all__ = [
    # action_metrics
    'compute_se3_distance',
    'build_action_matrix',

    # sampling_pipeline
    'get_segmented_keyframes',
    'sample_keyframes_single',

    # kinematics
    'UR12e_t',
    'CameraFKConfig',
]