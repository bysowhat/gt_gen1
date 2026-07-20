"""
基于重叠率的自适应采样
"""
from samplers.base_sampler import BaseSampler
from core.covisibility import PoseCovisibilityChecker
import numpy as np


class AdaptiveSampler(BaseSampler):
    """基于重叠率的自适应采样器"""
    
    def __init__(self, poses, intrinsics, img_h, img_w, 
                 min_overlap=0.3, max_overlap=0.8, **kwargs):
        """
        初始化自适应采样器
        
        Args:
            poses: 位姿列表
            intrinsics: 相机内参矩阵
            img_h: 图像高度
            img_w: 图像宽度
            min_overlap: 最小重叠率，低于此值则必须采样
            max_overlap: 最大重叠率，高于此值则跳过采样
        """
        super().__init__(poses, intrinsics, img_h, img_w, **kwargs)
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        device = kwargs.get('device', None)
        self.covisibility_checker = PoseCovisibilityChecker(
            intrinsics, img_h, img_w, threshold=0.1, device=device
        )
    
    def sample(self):
        """
        执行自适应采样
        
        Returns:
            selected_indices: list of int, 选中的帧索引
        """
        selected_indices = [0]  # 总是包含第一帧
        last_selected_idx = 0
        
        for i in range(1, self.num_frames):
            # 计算与上一个选中帧的重叠率
            overlap_ratio = self.covisibility_checker.compute_overlap_ratio(
                self.get_pose(last_selected_idx),
                self.get_pose(i)
            )
            
            # 如果重叠率太低，必须采样
            if overlap_ratio < self.min_overlap:
                selected_indices.append(i)
                last_selected_idx = i
            # # 如果重叠率太高，跳过（不采样）
            # elif overlap_ratio > self.max_overlap:
            #     continue
            # # 在中间范围内，根据策略决定是否采样
            # # 这里采用简单策略：如果重叠率在合理范围内，采样
            # else:
            #     # 可以添加更复杂的策略，比如基于距离的加权
            #     selected_indices.append(i)
            #     last_selected_idx = i
        
        return selected_indices

