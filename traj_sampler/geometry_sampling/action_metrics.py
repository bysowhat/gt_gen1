"""
阶段 I：SE(3) 运动度量模块
计算帧之间的SE(3)距离，用于评估运动成本
"""
import numpy as np
import math

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from core.geometry import _is_tensor, _to_tensor


def build_action_matrix(T_all, w_trans=1.0, w_rot=1.0):
    """
    指令 I-B: 构建所有帧对的SE(3)距离矩阵（GPU并行优化）
    
    使用PyTorch的广播机制，并行计算所有帧对的SE(3)距离。
    
    Args:
        T_all: (B, N, 4, 4) 张量
        w_trans: 平移权重（默认1.0）
        w_rot: 旋转权重（默认1.0）
    
    Returns:
        M_action: (B, N, N) 矩阵，M_action[i, j] 表示从帧i到帧j的SE(3)累计距离
    """
    device = T_all.device
    dtype = T_all.dtype
    
    # 转换为张量
    if T_all.dim() == 2:
        T_all = T_all.unsqueeze(0).unsqueeze(0)  # (4, 4) -> (1, 1, 4, 4)
    
    assert T_all.dim() == 4, "T_all size doesn't match and must be equal to 4"

    B, N = T_all.shape[:2]

    # 计算所有逆矩阵
    T_all_inv = torch.inverse(T_all)  # (B, N, 4, 4)


    # 计算相邻点的相对变换
    T_rel_n = T_all_inv[..., :-1, :, :] @ T_all[..., 1:, :, :]      # (B, N-1, 4, 4)

    # 提取平移向量 (B, N-1, 3)
    t_all = T_rel_n[..., :3, 3]
    
    # 提取旋转矩阵 (B, N-1, 3, 3)
    R_all = T_rel_n[..., :3, :3]
    
    # 计算平移距离 (B, N-1)
    trans_distances = torch.norm(t_all, dim=-1)  # (B, N-1)
    
    # 计算旋转角度 (B, N-1)
    # 对于每个 (i, j)，计算 R_all[i, j] 的旋转角度
    trace = torch.diagonal(R_all, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N-1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    theta = torch.acos(cos_theta)  # (B, N-1)
    rot_distances = torch.abs(theta)  # (B, N-1)
    
    # 计算相邻点的加权SE(3)距离
    d_n = w_trans * trans_distances + w_rot * rot_distances  # (B, N-1)

    # 计算每个点之间的距离矩阵
    d_sum = torch.cat((torch.zeros((B, 1), dtype=dtype, device=device),
                       torch.cumsum(d_n, dim=-1)), dim=-1)      # (B, N)
    M_action =  -(d_sum.unsqueeze(-1) - d_sum.unsqueeze(-2))    # (B, N, N)

    return M_action


def rotation_matrix_to_axis_angle(R):
    """
    将旋转矩阵转换为轴角表示
    
    Args:
        R: 旋转矩阵 (3, 3) 或 (N, 3, 3)
    
    Returns:
        theta: 旋转角度（弧度）标量或 (N,)
    """
    use_torch = TORCH_AVAILABLE and _is_tensor(R)
    
    if use_torch:
        # 计算旋转矩阵的迹
        if R.dim() == 2:
            trace = torch.trace(R)
        else:
            trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
        
        # 使用迹计算角度：trace(R) = 1 + 2*cos(theta)
        # theta = arccos((trace - 1) / 2)
        cos_theta = (trace - 1.0) / 2.0
        # 限制在[-1, 1]范围内
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
        theta = torch.acos(cos_theta)
        
        return theta
    else:
        if R.ndim == 2:
            trace = np.trace(R)
        else:
            trace = np.trace(R, axis1=-2, axis2=-1)
        
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        return theta


def compute_se3_distance(T_i, T_j, w_trans=1.0, w_rot=1.0, device=None):
    """
    指令 I-A: 计算两个位姿之间的SE(3)距离
    
    SE(3)距离 = w_trans * ||t|| + w_rot * |theta|
    其中 t 是相对变换的平移向量，theta 是旋转角度
    
    Args:
        T_i: 位姿矩阵 (4, 4) - 相机到世界坐标系
        T_j: 位姿矩阵 (4, 4) - 相机到世界坐标系
        w_trans: 平移权重（默认1.0）
        w_rot: 旋转权重（默认1.0，用于平衡平移和旋转）
        device: torch device
    
    Returns:
        distance: SE(3)距离（标量）
    """
    T_i = _to_tensor(T_i, device=device)
    T_j = _to_tensor(T_j, device=device)
    use_torch = TORCH_AVAILABLE and _is_tensor(T_i)
    
    # 计算相对变换：T_rel = T_i^{-1} @ T_j
    # 这表示从T_i到T_j的变换
    if use_torch:
        T_i_inv = torch.inverse(T_i)
        T_rel = T_i_inv @ T_j
    else:
        T_i_inv = np.linalg.inv(T_i)
        T_rel = T_i_inv @ T_j
    
    # 提取平移向量 t (前3个元素，最后一列)
    t = T_rel[:3, 3]
    
    # 提取旋转矩阵 R (前3x3)
    R = T_rel[:3, :3]
    
    # 计算平移距离
    if use_torch:
        trans_distance = torch.norm(t)
    else:
        trans_distance = np.linalg.norm(t)
    
    # 计算旋转角度
    theta = rotation_matrix_to_axis_angle(R)
    if use_torch:
        rot_distance = torch.abs(theta)
    else:
        rot_distance = np.abs(theta)
    
    # 计算加权SE(3)距离
    se3_distance = w_trans * trans_distance + w_rot * rot_distance
    
    # 转换为标量
    if use_torch:
        return se3_distance.item()
    else:
        return float(se3_distance)