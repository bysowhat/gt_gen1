"""
基于SE(3)动作幅度的采样器（支持GPU）
计算两个位姿之间的混合距离（平移+旋转），作为动作幅度的代理
"""
from samplers.base_sampler import BaseSampler
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from core.geometry import _is_tensor


class SE3ActionSampler(BaseSampler):
    """
    基于SE(3)动作幅度的采样器
    
    动作幅度 = ||ΔT|| + λ·||ΔR||
    其中：
    - ||ΔT|| 是平移距离（米）
    - ||ΔR|| 是旋转角度（弧度）
    - λ 是旋转权重
    """
    
    def __init__(self, poses, intrinsics, img_h, img_w,
                 action_threshold=0.1, rot_weight=1.0, 
                 use_accumulated=True, **kwargs):
        """
        初始化SE(3)动作采样器
        
        Args:
            poses: 位姿列表
            intrinsics: 相机内参矩阵
            img_h: 图像高度
            img_w: 图像宽度
            action_threshold: 动作幅度阈值
                - 如果use_accumulated=True，这是累积动作幅度的阈值
                - 如果use_accumulated=False，这是单步动作幅度的阈值
            rot_weight: 旋转权重λ，用于平衡平移和旋转
            use_accumulated: 是否使用累积动作幅度模式
                - True: 累积从上一个采样点到当前点的动作幅度
                - False: 计算相邻帧之间的动作幅度
        """
        super().__init__(poses, intrinsics, img_h, img_w, **kwargs)
        self.action_threshold = action_threshold
        self.rot_weight = rot_weight
        self.use_accumulated = use_accumulated
    
    def _compute_se3_delta(self, pose1, pose2):
        """
        计算两个位姿之间的SE(3)相对运动量（支持GPU）
        
        Args:
            pose1: 第一个位姿 (4x4矩阵)
            pose2: 第二个位姿 (4x4矩阵)
        
        Returns:
            translation_dist: 平移距离（米）
            rotation_angle: 旋转角度（弧度）
            action_magnitude: 混合动作幅度 = translation_dist + rot_weight * rotation_angle
        """
        use_torch = TORCH_AVAILABLE and _is_tensor(pose1)
        
        # 平移距离 (L2 norm)
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        
        if use_torch:
            translation_dist = torch.norm(pos2 - pos1)
            translation_dist = translation_dist.item() if translation_dist.numel() == 1 else float(translation_dist)
        else:
            translation_dist = np.linalg.norm(pos2 - pos1)
        
        # 旋转距离 (Axis-Angle 的角度部分)
        rot1 = pose1[:3, :3]
        rot2 = pose2[:3, :3]
        # 相对旋转 R_rel = R1^(-1) @ R2 = R1.T @ R2
        r_rel = rot1.T @ rot2
        
        # trace = 1 + 2cos(theta)
        # theta = arccos((trace - 1) / 2)
        if use_torch:
            trace = torch.trace(r_rel)
            trace = torch.clamp(trace, -1.0, 3.0)  # 数值稳定性
            rotation_angle = torch.acos((trace - 1) / 2.0)
            rotation_angle = rotation_angle.item() if rotation_angle.numel() == 1 else float(rotation_angle)
        else:
            trace = np.trace(r_rel)
            trace = np.clip(trace, -1.0, 3.0)  # 数值稳定性
            rotation_angle = np.arccos((trace - 1) / 2.0)
        
        # 混合动作幅度
        action_magnitude = translation_dist + self.rot_weight * rotation_angle
        
        return translation_dist, rotation_angle, action_magnitude
    
    def sample(self):
        """
        执行基于动作幅度的采样
        
        Returns:
            selected_indices: list of int, 选中的帧索引
        """
        selected_indices = [0]  # 总是包含第一帧
        last_selected_idx = 0
        
        if self.use_accumulated:
            # 累积动作幅度模式
            # 计算从上一个采样点到当前帧的累积动作幅度
            # 注意：这里直接计算相对变换，本身就是累积的（从last_selected到i的总动作）
            for i in range(1, self.num_frames):
                # 计算从上一个采样点到当前帧的动作幅度
                _, _, action_mag = self._compute_se3_delta(
                    self.get_pose(last_selected_idx),
                    self.get_pose(i)
                )
                
                # 如果累积动作幅度超过阈值，采样当前帧
                if action_mag >= self.action_threshold:
                    selected_indices.append(i)
                    last_selected_idx = i
        else:
            # 单步动作幅度模式（计算相邻帧之间的动作）
            for i in range(1, self.num_frames):
                # 计算相邻帧之间的动作幅度
                _, _, action_mag = self._compute_se3_delta(
                    self.get_pose(i - 1),
                    self.get_pose(i)
                )
                
                # 如果动作幅度超过阈值，采样当前帧
                if action_mag >= self.action_threshold:
                    selected_indices.append(i)
                    last_selected_idx = i
        
        return selected_indices
    
    def compute_action_magnitude(self, idx1, idx2):
        """
        计算两个帧之间的动作幅度（用于分析和调试）
        
        Args:
            idx1: 第一个帧索引
            idx2: 第二个帧索引
        
        Returns:
            translation_dist: 平移距离（米）
            rotation_angle: 旋转角度（弧度）
            action_magnitude: 混合动作幅度
        """
        return self._compute_se3_delta(
            self.get_pose(idx1),
            self.get_pose(idx2)
        )

