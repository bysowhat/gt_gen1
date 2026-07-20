"""
基于Pose的共视检测（支持GPU）
"""
import numpy as np
from core.geometry import get_frustum_samples, project_points, _is_tensor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class PoseCovisibilityChecker:
    def __init__(self, intrinsics, img_h, img_w, threshold=0.5, 
                 near=0.01, far=1.3, num_samples_per_plane=10, device=None):
        """
        初始化共视检测器（支持GPU）
        
        Args:
            intrinsics: 相机内参矩阵 (3, 3) (numpy array 或 torch tensor)
            img_h: 图像高度
            img_w: 图像宽度
            threshold: 覆盖率阈值，至少要有多少比例的点在视野内（默认0.5）
            near: 近平面距离（默认0.01m，基于实际深度数据）
            far: 远平面距离（默认1.3m，基于实际深度数据）
            num_samples_per_plane: 每个平面上采样的点数（默认10，会生成10x10=100个点）
            device: torch device (cpu/cuda)，如果为None则自动检测
        """
        self.K = intrinsics
        self.H = img_h
        self.W = img_w
        self.threshold = threshold
        self.near = near
        self.far = far
        self.num_samples_per_plane = num_samples_per_plane
        self.device = device
        if device is None and _is_tensor(intrinsics):
            self.device = intrinsics.device

    def _compute_unidirectional_overlap(self, pose_ref, pose_target):
        """
        计算单向重叠率：将target的视锥体采样点投影到ref的像素平面（支持GPU）
        
        Args:
            pose_ref: 参考位姿的4x4变换矩阵（相机到世界）
            pose_target: 目标位姿的4x4变换矩阵（相机到世界）
        
        Returns:
            overlap_ratio: float, 重叠率（在视野内的点比例）
        """
        # 1. 获取 Target 视锥体上的采样点 (在世界坐标系下)
        target_samples_3d = get_frustum_samples(
            pose_target, self.K, self.H, self.W, 
            self.near, self.far, self.num_samples_per_plane, device=self.device
        )

        # 2. 将这些点投影到 Ref 的像素坐标系
        points_2d, depths, valid_mask = project_points(
            target_samples_3d, pose_ref, self.K, device=self.device
        )

        # 3. 检查有多少点在图像范围内
        use_torch = TORCH_AVAILABLE and _is_tensor(points_2d)
        
        if use_torch:
            in_bounds = (
                (points_2d[:, 0] >= 0) & 
                (points_2d[:, 0] < self.W) &
                (points_2d[:, 1] >= 0) & 
                (points_2d[:, 1] < self.H) &
                valid_mask &
                (depths > 0)
            )
            overlap_ratio = torch.sum(in_bounds).float() / len(target_samples_3d)
            # 转换为Python float
            overlap_ratio = overlap_ratio.item()
        else:
            in_bounds = (
                (points_2d[:, 0] >= 0) & 
                (points_2d[:, 0] < self.W) &
                (points_2d[:, 1] >= 0) & 
                (points_2d[:, 1] < self.H) &
                valid_mask &
                (depths > 0)
            )
            overlap_ratio = np.sum(in_bounds) / len(target_samples_3d)
        
        return overlap_ratio

    def check(self, pose_ref, pose_target):
        """
        判断两个位姿是否共视（双向检查）
        方法：将两个视锥体的采样点分别投影到对方的像素平面，计算双向重叠率
        
        Args:
            pose_ref: 参考位姿的4x4变换矩阵（相机到世界）
            pose_target: 目标位姿的4x4变换矩阵（相机到世界）
        
        Returns:
            is_covisibile: bool, 是否共视
            overlap_ratio: float, 重叠率（双向重叠率的平均值）
        """
        # 计算双向重叠率
        overlap_ref_to_target = self._compute_unidirectional_overlap(pose_ref, pose_target)
        overlap_target_to_ref = self._compute_unidirectional_overlap(pose_target, pose_ref)
        
        # 取双向重叠率的平均值（更稳定）
        overlap_ratio = (overlap_ref_to_target + overlap_target_to_ref) / 2.0
        is_covisibile = overlap_ratio >= self.threshold

        return is_covisibile, overlap_ratio

    def compute_overlap_ratio(self, pose_ref, pose_target):
        """
        计算两个位姿之间的重叠率（双向）
        
        Args:
            pose_ref: 参考位姿的4x4变换矩阵（相机到世界）
            pose_target: 目标位姿的4x4变换矩阵（相机到世界）
        
        Returns:
            overlap_ratio: float, 重叠率 [0, 1]（双向重叠率的平均值）
        """
        _, overlap_ratio = self.check(pose_ref, pose_target)
        return overlap_ratio

