"""
共视-运动混合采样器（支持GPU）
以共视采样为主，当共视率低于阈值时必须采样，
当共视率高于阈值时，如果动作幅度过大也需要采样

支持两种共视率检测方法：
1. 基于Pose的共视检测（默认，不需要深度图）
2. 基于深度重投影验证的共视检测（MapAnything方法，需要深度图）
"""
from samplers.base_sampler import BaseSampler
from core.covisibility import PoseCovisibilityChecker
from mapanything_sampling import compute_score_for_pair
import numpy as np
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from core.geometry import _is_tensor


class HybridSampler(BaseSampler):
    """
    共视-运动混合采样器
    
    采样策略：
    1. 如果共视率 < min_overlap：必须采样（视野变化大）
    2. 如果共视率 >= min_overlap 且 <= max_overlap：
       - 如果动作幅度 >= action_threshold：采样（动作大）
       - 否则：跳过（共视好且动作小）
    3. 如果共视率 > max_overlap：跳过（视野重叠太多，太相似）
    """
    
    def __init__(self, poses, intrinsics, img_h, img_w,
                 min_overlap=0.3, max_overlap=0.7,
                 action_threshold=0.1, rot_weight=1.0,
                 use_depth_reprojection=False, depth_threshold=0.01,
                 depth_assoc_rel_error_thres=0.0,
                 depth_assoc_error_temp=0.0,
                 data_root=None, **kwargs):
        """
        初始化混合采样器
        
        Args:
            poses: 位姿列表
            intrinsics: 相机内参矩阵
            img_h: 图像高度
            img_w: 图像宽度
            min_overlap: 最小重叠率，低于此值必须采样
            max_overlap: 最大重叠率，高于此值跳过采样
            action_threshold: 动作幅度阈值（当共视率在合理范围内时使用）
            rot_weight: 旋转权重λ，用于计算SE(3)动作幅度
            use_depth_reprojection: 是否使用深度重投影验证方法（MapAnything方法）
            depth_threshold: 深度误差阈值（米），用于深度重投影验证（绝对误差阈值）
            depth_assoc_rel_error_thres: 深度关联相对误差阈值，默认0.0（固定阈值）。设置>0启用动态阈值
            depth_assoc_error_temp: 深度关联误差温度参数，默认0.0
            data_root: 数据根目录路径，用于加载深度图（如果use_depth_reprojection=True）
        """
        super().__init__(poses, intrinsics, img_h, img_w, **kwargs)
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.action_threshold = action_threshold
        self.rot_weight = rot_weight
        self.use_depth_reprojection = use_depth_reprojection
        self.data_root = data_root
        
        # 初始化共视检测器
        device = kwargs.get('device', None)
        if use_depth_reprojection:
            # 使用深度重投影验证方法（MapAnything方法，使用新的模块化代码）
            if data_root is None:
                raise ValueError("data_root must be provided when use_depth_reprojection=True")
            # 保存参数用于新的模块化代码
            self.depth_threshold = depth_threshold
            self.depth_assoc_rel_error_thres = depth_assoc_rel_error_thres
            self.depth_assoc_error_temp = depth_assoc_error_temp
            self.device = device
            # 预加载所有深度图（可选，按需加载）
            self.depth_maps = None  # 延迟加载
        else:
            # 使用基于Pose的共视检测（默认方法）
            self.covisibility_checker = PoseCovisibilityChecker(
                intrinsics, img_h, img_w, threshold=0.1, device=device
            )
            self.depth_maps = None
    
    def _compute_se3_action_magnitude(self, pose1, pose2):
        """
        计算两个位姿之间的SE(3)动作幅度（支持GPU）
        
        Args:
            pose1: 第一个位姿 (4x4矩阵)
            pose2: 第二个位姿 (4x4矩阵)
        
        Returns:
            action_magnitude: 混合动作幅度 = ||ΔT|| + λ·||ΔR||
        """
        use_torch = TORCH_AVAILABLE and _is_tensor(pose1)
        
        # 平移距离
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        
        if use_torch:
            translation_dist = torch.norm(pos2 - pos1)
            translation_dist = translation_dist.item() if translation_dist.numel() == 1 else float(translation_dist)
        else:
            translation_dist = np.linalg.norm(pos2 - pos1)
        
        # 旋转角度
        rot1 = pose1[:3, :3]
        rot2 = pose2[:3, :3]
        r_rel = rot1.T @ rot2
        
        if use_torch:
            trace = torch.trace(r_rel)
            trace = torch.clamp(trace, -1.0, 3.0)
            rotation_angle = torch.acos((trace - 1) / 2.0)
            rotation_angle = rotation_angle.item() if rotation_angle.numel() == 1 else float(rotation_angle)
        else:
            trace = np.trace(r_rel)
            trace = np.clip(trace, -1.0, 3.0)
            rotation_angle = np.arccos((trace - 1) / 2.0)
        
        # 混合动作幅度
        action_magnitude = translation_dist + self.rot_weight * rotation_angle
        
        return action_magnitude
    
    def _load_depth_map(self, frame_idx):
        """加载指定帧的深度图（延迟加载）"""
        if self.depth_maps is None:
            # 延迟加载：按需加载深度图
            from utils.io import load_depth_map
            return load_depth_map(self.data_root, frame_idx, device=self.device)
        else:
            return self.depth_maps[frame_idx]
    
    def sample(self):
        """
        执行混合采样
        
        Returns:
            selected_indices: list of int, 选中的帧索引
        """
        selected_indices = [0]  # 总是包含第一帧
        last_selected_idx = 0
        
        for i in range(1, self.num_frames):
            # 计算与上一个选中帧的共视率
            if self.use_depth_reprojection:
                # 使用深度重投影验证方法（新的模块化代码）
                depth_ref = self._load_depth_map(last_selected_idx)
                depth_target = self._load_depth_map(i)
                overlap_ratio = compute_score_for_pair(
                    depth_ref, self.get_pose(last_selected_idx),
                    depth_target, self.get_pose(i),
                    self.intrinsics, self.img_h, self.img_w,
                    depth_assoc_error_thres=self.depth_threshold,
                    depth_assoc_rel_error_thres=self.depth_assoc_rel_error_thres,
                    depth_assoc_error_temp=self.depth_assoc_error_temp,
                    device=self.device
                )
            else:
                # 使用基于Pose的共视检测
                overlap_ratio = self.covisibility_checker.compute_overlap_ratio(
                    self.get_pose(last_selected_idx),
                    self.get_pose(i)
                )
            
            should_sample = False
            
            # 策略1：共视率太低，必须采样
            if overlap_ratio < self.min_overlap:
                should_sample = True
            
            # 策略2：共视率在合理范围内，检查动作幅度
            elif overlap_ratio >= self.min_overlap and overlap_ratio <= self.max_overlap:
                # 计算动作幅度
                action_mag = self._compute_se3_action_magnitude(
                    self.get_pose(last_selected_idx),
                    self.get_pose(i)
                )
                
                # 如果动作幅度超过阈值，采样
                if action_mag >= self.action_threshold:
                    should_sample = True
            
            # 策略3：共视率太高（> max_overlap），跳过（不采样）
            # else: should_sample remains False
            
            if should_sample:
                selected_indices.append(i)
                last_selected_idx = i
        
        return selected_indices

