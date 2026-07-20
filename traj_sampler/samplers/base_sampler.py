"""
抽象基类：所有采样器的接口（支持GPU）
"""
from abc import ABC, abstractmethod
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from core.geometry import _is_tensor


class BaseSampler(ABC):
    """采样器抽象基类（支持GPU）"""
    
    def __init__(self, poses, intrinsics, img_h, img_w, device=None, **kwargs):
        """
        初始化采样器
        
        Args:
            poses: 位姿列表，每个元素是4x4变换矩阵（相机到世界）
                   可以是numpy数组或torch tensor
            intrinsics: 相机内参矩阵 (3, 3) (numpy array 或 torch tensor)
            img_h: 图像高度
            img_w: 图像宽度
            device: torch device (cpu/cuda)，如果为None则自动检测
        """
        self.poses = poses
        self.intrinsics = intrinsics
        self.img_h = img_h
        self.img_w = img_w
        self.num_frames = len(poses)
        self.device = device
        if device is None and len(poses) > 0 and _is_tensor(poses[0]):
            self.device = poses[0].device
    
    @abstractmethod
    def sample(self):
        """
        执行采样，返回选中的帧索引列表
        
        Returns:
            selected_indices: list of int, 选中的帧索引
        """
        pass
    
    def get_pose(self, idx):
        """获取指定索引的位姿"""
        return self.poses[idx]
    
    def compute_distance(self, idx1, idx2):
        """计算两个帧之间的平移距离（支持GPU）"""
        pose1 = self.get_pose(idx1)
        pose2 = self.get_pose(idx2)
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]
        
        if TORCH_AVAILABLE and _is_tensor(pose1):
            dist = torch.norm(t2 - t1)
            return dist.item() if dist.numel() == 1 else float(dist)
        else:
            return np.linalg.norm(t2 - t1)
    
    def compute_angle(self, idx1, idx2):
        """计算两个帧之间的旋转角度（度，支持GPU）"""
        pose1 = self.get_pose(idx1)
        pose2 = self.get_pose(idx2)
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        R_rel = R2 @ R1.T
        
        use_torch = TORCH_AVAILABLE and _is_tensor(pose1)
        
        if use_torch:
            trace = torch.trace(R_rel)
            angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
            angle_deg = torch.rad2deg(angle_rad)
            return angle_deg.item() if angle_deg.numel() == 1 else float(angle_deg)
        else:
            trace = np.trace(R_rel)
            angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            angle_deg = np.degrees(angle_rad)
            return angle_deg

