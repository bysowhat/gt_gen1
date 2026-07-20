"""
阶段 II：共视性校验核心
封装 MapAnything 的核心几何一致性检查逻辑
"""
import numpy as np
import math
from .geometry_ops import (
    unproject_depth_to_world,
    project_world_to_pixel,
    sample_depths_at_reproj
)
from core.geometry import _is_tensor, _to_tensor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def in_image_mask(P_2d, img_h, img_w, min_depth=0.0):
    """
    检查2D点是否在图像范围内（参考MapAnything的in_image函数）
    
    Args:
        P_2d: 像素坐标 (M, 2) 或 (M, 3) 或 (H, W, 2) 或 (H, W, 3)
        img_h: 图像高度
        img_w: 图像宽度
        min_depth: 最小有效深度
    
    Returns:
        valid_mask: 有效点掩码，形状与P_2d的前两维相同
    """
    use_torch = TORCH_AVAILABLE and _is_tensor(P_2d)
    
    if use_torch:
        # 检查所有坐标 >= 0
        in_bounds = torch.all(P_2d >= 0, dim=-1)
        # 检查 u < img_w, v < img_h
        in_bounds = in_bounds & (P_2d[..., 0] < img_w) & (P_2d[..., 1] < img_h)
        # 如果有深度信息，检查深度 > min_depth
        if P_2d.shape[-1] == 3:
            in_bounds = in_bounds & (P_2d[..., 2] > min_depth)
    else:
        in_bounds = np.all(P_2d >= 0, axis=-1)
        in_bounds = in_bounds & (P_2d[..., 0] < img_w) & (P_2d[..., 1] < img_h)
        if P_2d.shape[-1] == 3:
            in_bounds = in_bounds & (P_2d[..., 2] > min_depth)
    
    return in_bounds


def check_depth_consistency(D_expected, D_sampled, M_valid, 
                           depth_assoc_error_thres=0.01,
                           depth_assoc_rel_error_thres=0.0,
                           depth_assoc_error_temp=0.0):
    """
    指令 II-1: 检查深度一致性（计算重投影误差并应用动态阈值）
    
    Args:
        D_expected: 期望深度值，任意形状
        D_sampled: 采样得到的深度值，形状与D_expected相同
        M_valid: 有效点掩码（像素范围掩码），形状与D_expected相同
        depth_assoc_error_thres: 深度关联绝对误差阈值（米）
        depth_assoc_rel_error_thres: 深度关联相对误差阈值（相对于期望深度）
        depth_assoc_error_temp: 深度关联误差温度参数
    
    Returns:
        M_inlier: 内点掩码（布尔），形状与D_expected相同
    """
    use_torch = TORCH_AVAILABLE and _is_tensor(D_expected)
    
    # 计算重投影误差
    if use_torch:
        E_reproj = torch.abs(D_expected - D_sampled)
        # 动态深度关联阈值（参考MapAnything）
        depth_assoc_thres = (
            depth_assoc_error_thres
            + depth_assoc_rel_error_thres * D_expected
            - math.log(0.5) * depth_assoc_error_temp
        )
        M_inlier = (E_reproj < depth_assoc_thres) & M_valid
    else:
        E_reproj = np.abs(D_expected - D_sampled)
        depth_assoc_thres = (
            depth_assoc_error_thres
            + depth_assoc_rel_error_thres * D_expected
            - math.log(0.5) * depth_assoc_error_temp
        )
        M_inlier = (E_reproj < depth_assoc_thres) & M_valid
    
    return M_inlier


def compute_score_for_pair(depth_ref, pose_ref, depth_target, pose_target,
                          K, img_h, img_w,
                          depth_assoc_error_thres=0.01,
                          depth_assoc_rel_error_thres=0.0,
                          depth_assoc_error_temp=0.0,
                          denominator_mode="valid_target_depth",
                          min_depth=0.04,
                          world_pts3d_ref=None,
                          device=None):
    """
    指令 II-2: 总控函数，计算两个帧之间的共视率分数
    
    按照 MapAnything 的四阶段逻辑：
    1. 数据准备与3D点云生成
    2. 几何转换与深度采样
    3. 几何一致性检查
    4. 最终得分聚合
    
    Args:
        depth_ref: 参考帧的深度图 (H, W)
        pose_ref: 参考帧的位姿 (4, 4) 相机到世界坐标系
        depth_target: 目标帧的深度图 (H, W)
        pose_target: 目标帧的位姿 (4, 4) 相机到世界坐标系
        K: 相机内参矩阵 (3, 3)
        img_h: 图像高度
        img_w: 图像宽度
        depth_assoc_error_thres: 深度关联绝对误差阈值（米）
        depth_assoc_rel_error_thres: 深度关联相对误差阈值
        depth_assoc_error_temp: 深度关联误差温度参数
        denominator_mode: 分母模式，"valid_target_depth" 或 "full"
        min_depth: 最小有效深度（米）
        world_pts3d_ref: 预计算的世界坐标点 (H*W, 3)，如果为None则实时计算
        device: torch device
    
    Returns:
        O_score: float [0, 1]，共视率分数
    """
    use_torch = TORCH_AVAILABLE and _is_tensor(depth_ref)
    
    # ========== 阶段一：数据准备与3D点云生成 ==========
    if world_pts3d_ref is None:
        # 实时计算世界坐标点
        P_world, M_ref = unproject_depth_to_world(
            K, pose_ref, depth_ref, device=device
        )
        # 如果返回的是 (H, W, 3)，需要展平为点云格式
        if P_world.ndim == 3:
            H, W = P_world.shape[:2]
            P_world = P_world.reshape(H * W, 3)  # (H*W, 3)
            M_ref = M_ref.flatten()  # (H*W,)
    else:
        # 使用预计算的世界坐标点
        world_pts3d_ref = _to_tensor(world_pts3d_ref, device=device)
        M_ref = depth_ref.flatten() > 1e-6
        if _is_tensor(world_pts3d_ref):
            P_world = world_pts3d_ref[M_ref]
        else:
            P_world = world_pts3d_ref[M_ref]
    
    if len(P_world) == 0:
        return 0.0
    
    # ========== 阶段二：几何转换与深度采样 ==========
    # 2.1 转换到目标相机系并投影到像素平面
    P_2d, D_expected = project_world_to_pixel(
        K, pose_target, P_world, device=device
    )
    
    # 2.2 图像边界过滤
    # 将深度信息添加到P_2d以便in_image检查
    if use_torch:
        if P_2d.dim() == 1:
            # 如果P_2d是1D的，需要reshape
            P_2d = P_2d.unsqueeze(0) if P_2d.shape[0] == 2 else P_2d
        P_2d_with_depth = torch.cat([P_2d, D_expected.unsqueeze(-1)], dim=-1)  # (M, 3)
    else:
        if P_2d.ndim == 1:
            P_2d = P_2d.reshape(1, -1) if P_2d.shape[0] == 2 else P_2d.reshape(-1, 2)
        P_2d_with_depth = np.concatenate([P_2d, D_expected[..., np.newaxis]], axis=-1)  # (M, 3)
    
    M_valid_in_image = in_image_mask(P_2d_with_depth, img_h, img_w, min_depth=min_depth)
    
    if not M_valid_in_image.any():
        return 0.0
    
    # 2.3 采样目标深度
    D_sampled = sample_depths_at_reproj(
        depth_target, P_2d, img_h, img_w, device=device
    )
    
    # 确保 D_sampled 和 D_expected 的形状一致
    if use_torch:
        # 如果 D_sampled 是 (1, M) 形状，需要squeeze
        if D_sampled.dim() == 2 and D_sampled.shape[0] == 1:
            D_sampled = D_sampled.squeeze(0)
        # 如果 D_expected 是 (1, M) 形状，需要squeeze
        if D_expected.dim() == 2 and D_expected.shape[0] == 1:
            D_expected = D_expected.squeeze(0)
    
    # 只使用有效掩码的点
    if use_torch:
        D_expected_valid = D_expected[M_valid_in_image]
        D_sampled_valid = D_sampled[M_valid_in_image]
        # M_valid 应该与 D_expected_valid 和 D_sampled_valid 的形状相同
        M_valid = torch.ones_like(D_expected_valid, dtype=torch.bool)  # 所有通过图像边界检查的点都是有效的
    else:
        D_expected_valid = D_expected[M_valid_in_image]
        D_sampled_valid = D_sampled[M_valid_in_image]
        M_valid = np.ones_like(D_expected_valid, dtype=bool)
    
    if len(D_expected_valid) == 0:
        return 0.0
    
    # ========== 阶段三：几何一致性检查 ==========
    M_inlier = check_depth_consistency(
        D_expected_valid, D_sampled_valid, M_valid,
        depth_assoc_error_thres=depth_assoc_error_thres,
        depth_assoc_rel_error_thres=depth_assoc_rel_error_thres,
        depth_assoc_error_temp=depth_assoc_error_temp
    )
    
    # ========== 阶段四：最终得分聚合 ==========
    # 确保 M_inlier 和 depth_target 是相同类型
    if use_torch:
        if not _is_tensor(M_inlier):
            M_inlier = _to_tensor(M_inlier, device=device)
        if not _is_tensor(depth_target):
            depth_target = _to_tensor(depth_target, device=device)
        
        N_inlier = torch.sum(M_inlier).item()
        
        if denominator_mode == "valid_target_depth":
            N_valid_target = torch.sum(depth_target > 1e-6).item()
            O_score = N_inlier / max(N_valid_target, 1)
            O_score = min(O_score, 1.0)  # clamp to [0, 1]
        elif denominator_mode == "full":
            O_score = N_inlier / (img_h * img_w)
        else:
            raise ValueError(f"Unknown denominator_mode: {denominator_mode}")
    else:
        if _is_tensor(M_inlier):
            M_inlier = M_inlier.cpu().numpy() if hasattr(M_inlier, 'cpu') else np.array(M_inlier)
        if _is_tensor(depth_target):
            depth_target = depth_target.cpu().numpy() if hasattr(depth_target, 'cpu') else np.array(depth_target)
        
        N_inlier = np.sum(M_inlier)
        
        if denominator_mode == "valid_target_depth":
            N_valid_target = np.sum(depth_target > 1e-6)
            O_score = N_inlier / max(N_valid_target, 1)
            O_score = min(O_score, 1.0)
        elif denominator_mode == "full":
            O_score = N_inlier / (img_h * img_w)
        else:
            raise ValueError(f"Unknown denominator_mode: {denominator_mode}")
    
    return float(O_score)

