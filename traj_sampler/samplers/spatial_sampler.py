"""
固定距离/角度采样
"""
from samplers.base_sampler import BaseSampler
import numpy as np


class SpatialSampler(BaseSampler):
    """固定距离或角度采样器"""
    
    def __init__(self, poses, intrinsics, img_h, img_w, 
                 distance_threshold=None, angle_threshold=None, **kwargs):
        """
        初始化空间采样器
        
        Args:
            poses: 位姿列表
            intrinsics: 相机内参矩阵
            img_h: 图像高度
            img_w: 图像宽度
            distance_threshold: 距离阈值（米），当移动距离超过此值时采样
            angle_threshold: 角度阈值（度），当旋转角度超过此值时采样
        """
        super().__init__(poses, intrinsics, img_h, img_w, **kwargs)
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        
        if distance_threshold is None and angle_threshold is None:
            raise ValueError("至少需要指定 distance_threshold 或 angle_threshold 之一")
    
    def sample(self):
        """
        执行空间采样
        
        Returns:
            selected_indices: list of int, 选中的帧索引
        """
        selected_indices = [0]  # 总是包含第一帧
        last_selected_idx = 0
        
        for i in range(1, self.num_frames):
            should_sample = False
            
            # 检查距离阈值
            if self.distance_threshold is not None:
                distance = self.compute_distance(last_selected_idx, i)
                if distance >= self.distance_threshold:
                    should_sample = True
            
            # 检查角度阈值
            if self.angle_threshold is not None:
                angle = self.compute_angle(last_selected_idx, i)
                if angle >= self.angle_threshold:
                    should_sample = True
            
            if should_sample:
                selected_indices.append(i)
                last_selected_idx = i
        
        return selected_indices

