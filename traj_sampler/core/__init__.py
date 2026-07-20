"""
向后兼容性：从core模块导入mapanything_sampling的功能
"""
# 向后兼容：允许从core模块导入mapanything_sampling的功能
# 使用延迟导入避免循环导入问题
def __getattr__(name):
    """延迟导入mapanything_sampling的功能"""
    if name in [
        'unproject_depth_to_world',
        'project_world_to_pixel',
        'sample_depths_at_reproj',
        'inverse_transform_4x4',
        'check_depth_consistency',
        'compute_score_for_pair',
        'in_image_mask',
        'compute_se3_distance',
        'build_action_matrix',
        'build_overlap_matrix',
        'get_segmented_keyframes',
        'get_adaptive_indices',
    ]:
        from mapanything_sampling import __dict__ as ms_dict
        return ms_dict[name]
    raise AttributeError(f"module 'core' has no attribute '{name}'")

__all__ = [
    'unproject_depth_to_world',
    'project_world_to_pixel',
    'sample_depths_at_reproj',
    'inverse_transform_4x4',
    'check_depth_consistency',
    'compute_score_for_pair',
    'in_image_mask',
    'compute_se3_distance',
    'build_action_matrix',
    'build_overlap_matrix',
    'get_segmented_keyframes',
    'get_adaptive_indices',
]
