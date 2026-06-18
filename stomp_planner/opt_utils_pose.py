import torch
import numpy as np


def create_R_matrix(alpha: torch.Tensor):
    cos_alpha = torch.cos(alpha)    # (..., 1)
    sin_alpha = torch.sin(alpha)    # (..., 1)
    row0 = torch.cat((cos_alpha, -sin_alpha), dim=-1)       # (..., 2)
    row1 = torch.cat((sin_alpha,  cos_alpha), dim=-1)       # (..., 2)
    R_matix = torch.stack((row0, row1), dim=-2)             # (..., 2, 2)
    return R_matix


def normal_vector(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor):
    return torch.cross(p2 - p1, p3 - p1, dim=-1)


def compute_normal(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, 
                   p5: torch.Tensor, p6: torch.Tensor, p7: torch.Tensor, p8: torch.Tensor):
    "Compute normal vectors of all surfaces of a hexahedral space for 1d input"
    n1 = normal_vector(p1, p4, p2)
    n1 = (n1 / torch.norm(n1, p=2, dim=-1))
    n2 = normal_vector(p5, p8, p1)
    n2 = (n2 / torch.norm(n2, p=2, dim=-1))
    n3 = normal_vector(p6, p5, p2)
    n3 = (n3 / torch.norm(n3, p=2, dim=-1))
    n4 = normal_vector(p7, p6, p3)
    n4 = (n4 / torch.norm(n4, p=2, dim=-1))
    n5 = normal_vector(p8, p7, p4)
    n5 = (n5 / torch.norm(n5, p=2, dim=-1))
    n6 = normal_vector(p5, p6, p8)
    n6 = (n6 / torch.norm(n6, p=2, dim=-1))
    
    return torch.stack((n1, n2, n3, n4, n5, n6), dim=0)


def matrix_from_vectors(x_axis, y_axis, z_axis, orthogonalize=True):
    """
    根据旋转后的三个坐标轴向量，构造旋转矩阵
    Args:
        x_axis: (B, 3) 批量旋转后的 x 轴向量
        y_axis: (B, 3) 批量旋转后的 y 轴向量
        z_axis: (B, 3) 批量旋转后的 z 轴向量
        orthogonalize: 是否正交化(保证结果是SO(3)矩阵）
    Returns:
        R: (B, 3, 3) 批量旋转矩阵
    """
    # 堆叠成 (B, 3, 3)，每一列是一个基向量
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, 3, 3)

    if orthogonalize:
        # 使用 SVD 保证正交性
        U, _, Vt = torch.linalg.svd(R)
        R = U @ Vt
        # 确保 det(R)=+1
        det = torch.det(R)
        mask = det < 0
        if mask.any():
            U[mask, :, -1] *= -1
            R = U @ Vt

    return R


def matrix_from_so3(delta: torch.Tensor) -> torch.Tensor:
    """
    将so(3)增量向量转换为旋转矩阵 (Rodrigues公式)
    Args:
        delta: (..., 3) 增量向量
    Returns:
        R: (..., 3, 3) 旋转矩阵
    """
    theta = torch.norm(delta, dim=-1, keepdim=True)  # (..., 1)

    def hat(delta: torch.Tensor) -> torch.Tensor:
        """
        so(3) hat算子
        Args:
            delta: (..., 3)
        Returns:
            delta_hat: (..., 3, 3)
        """
        wx, wy, wz = delta.unbind(dim=-1)
        O = torch.zeros_like(wx)
        return torch.stack([
            torch.stack([ O, -wz,  wy], dim=-1),
            torch.stack([ wz,  O, -wx], dim=-1),
            torch.stack([-wy, wx,  O], dim=-1),
        ], dim=-2)
    
    delta_hat = hat(delta)  # (..., 3, 3)

    # 防止除0
    theta = theta.clamp(min=1e-8)

    A = torch.sin(theta) / theta
    B = (1 - torch.cos(theta)) / (theta ** 2)

    I = torch.eye(3, device=delta.device, dtype=delta.dtype).expand(delta.shape[:-1] + (3, 3))
    R = I + A[..., None] * delta_hat + B[..., None] * (delta_hat @ delta_hat)

    return R


def matrix_from_single_vector(v: torch.Tensor) -> torch.Tensor:
    """
    返回一个 3x3 旋转矩阵 R,使得 R @ [0,0,1] = normalized(v)
    Args:
        v: (B, 3)
    """
    B = v.shape[0]
    device = v.device
    dtype = v.dtype

    # 归一化目标向量
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v = v / v_norm      # (B, 3)

    z = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).unsqueeze(-1).repeat(B, 1)  # (B, 3)
    c = torch.sum(v * z, dim=-1)    # (B,), cos(theta)

    # 创建旋转矩阵
    rot = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)     # (B, 3, 3)

    # 反方向 (180 度旋转),默认绕x轴旋转180度
    oppo_mask = (torch.abs(c + 1) < 1e-6)
    rot[oppo_mask, 1, 1] = -1
    rot[oppo_mask, 2, 2] = -1

    # 一般情况：Rodrigues 简化公式 R = I + K + K^2 / (1+c)
    mask = (torch.abs(c + 1) >= 1e-6) & (torch.abs(c - 1) >= 1e-6)
    k = torch.cross(z[mask], v[mask], dim=-1)       # (M, 3)
    K = torch.zeros((k.shape[0], 3, 3), dtype=dtype, device=device)      # (M, 3, 3)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] =  k[:, 1]
    K[:, 1, 0] =  k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] =  k[:, 0]
    rot[mask] = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) + K + K @ K / (1.0 + c[mask])
    
    return rot


def slerp_vector(A: torch.Tensor, B: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    对两个向量之间进行是slerp插值
    Args:
        A: (N, 3) tensor
        B: (N, 3) tensor
        t: (N, M) tensor, values in [0, 1]
    Returns:
        (N, M, 3) interpolated vectors
    """
    if A.shape != B.shape or A.shape[-1] != 3:
        raise ValueError("A and B must have shape (N, 3)")
    if t.dim() != 2 or t.shape[0] != A.shape[0]:
        raise ValueError("t must have shape (N, M)")

    # 归一化输入向量
    A_norm = A / A.norm(dim=-1, keepdim=True)
    B_norm = B / B.norm(dim=-1, keepdim=True)

    # cos(theta)
    dot = (A_norm * B_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (N, 1)
    theta = torch.acos(dot)  # (N, 1)

    # 处理几乎重合的情况，避免除零
    eps = 1e-8
    sin_theta = torch.sin(theta).clamp(min=eps)

    # 扩展维度以匹配 t (N, M)
    t_exp = t.unsqueeze(-1)  # (N, M, 1)
    A_exp = A_norm.unsqueeze(1)  # (N, 1, 3)
    B_exp = B_norm.unsqueeze(1)  # (N, 1, 3)
    theta_exp = theta.unsqueeze(1)  # (N, 1, 1)
    sin_theta_exp = sin_theta.unsqueeze(1)  # (N, 1, 1)

    # slerp 公式
    coeff_A = torch.sin((1 - t_exp) * theta_exp) / sin_theta_exp
    coeff_B = torch.sin(t_exp * theta_exp) / sin_theta_exp
    result = coeff_A * A_exp + coeff_B * B_exp  # (N, M, 3)

    # 保持单位长度
    result = result / (result.norm(dim=-1, keepdim=True) + 1e-8)
    return result