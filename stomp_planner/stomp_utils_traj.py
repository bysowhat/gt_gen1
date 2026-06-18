"""
utilities for stomp.py
version: v0
date: 25.08.20
"""

import torch
import numpy as np

from enum import IntEnum

# --- 常量定义 ---
FINITE_CENTRAL_DIFF_COEFFS = (
    ( 0, 0, 0, 1, 0, 0, 0 ),                                                  # position
    ( 0, 1.0 / 12.0, -2.0 / 3.0, 0, 2.0 / 3.0, -1.0 / 12.0, 0 ),              # velocity
    ( 0, -1 / 12.0, 16 / 12.0, -30 / 12.0, 16 / 12.0, -1 / 12.0, 0 ),         # acceleration (five point stencil)
    ( 0, 1 / 12.0, -17 / 12.0, 46 / 12.0, -46 / 12.0, 17 / 12.0, -1 / 12.0 )  # jerk
)
DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT = 1.0          # 用于在计算概率时调整不同轨迹的贡献。
MIN_COST_DIFFERENCE = 1e-8
MIN_CONTROL_COST_WEIGHT = 1e-8
FINITE_DIFF_RULE_LENGTH = 7                         # 整个差分规则使用了7个点（7点模板 / stencil）


# --- 枚举和配置 ---
class TrajectoryInitializations(IntEnum):
    LINEAR_INTERPOLATION = 1
    CUBIC_POLYNOMIAL_INTERPOLATION = 2
    MININUM_CONTROL_COST = 3


class DerivativeOrder(IntEnum):
    STOMP_POSITION = 0         # Calculate position using finite differentiation
    STOMP_VELOCITY = 1         # Calculate velocity using finite differentiation
    STOMP_ACCELERATION = 2     # Calculate acceleration using finite differentiation
    STOMP_JERK = 3             # Calculate jerk using finite differentiation


def computeLinearInterpolation(first: torch.Tensor,
                               last: torch.Tensor,
                               num_timesteps: int):
    """
    计算线性插值轨迹
    Args:
        first: (B, D)
        last: (B, D)
        trajectory_joints: (B, D, T)
    """
    dtheta = (last - first) / (num_timesteps - 1)   # (B, D)
    timesteps = torch.arange(num_timesteps, dtype=first.dtype, device=first.device)
    # (B, D, T) = (B, D, 1) + (B, D, 1) * (1, 1, T)
    trajectory_joints = first.unsqueeze(-1) + dtheta.unsqueeze(-1) * timesteps.unsqueeze(0).unsqueeze(0)
    return trajectory_joints


def computeCubicInterpolation(first: torch.Tensor,
                              last: torch.Tensor,
                              num_timesteps: int,
                              dt: float) -> torch.Tensor:
    """计算三次插值轨迹
    Args:
        first: (B, D)
        last: (B, D)
        trajectory_joints: (B, D, T)
    """
    total_time = (num_timesteps - 1) * dt
    coeffs_0 = first
    coeffs_2 = (3 / (total_time ** 2)) * (last - first)
    coeffs_3 = (-2 / (total_time ** 3)) * (last - first)

    timesteps = torch.arange(num_timesteps, dtype=first.dtype, device=first.device) * dt
    t_squared = timesteps ** 2
    t_cubed = timesteps ** 3
    # (B, D, T) = (B, D, 1) + (B, D, 1) * (1, 1, T) + (B, D, 1) * (1, 1, T)
    trajectory_joints = (
        coeffs_0.unsqueeze(-1) + 
        coeffs_2.unsqueeze(-1) * t_squared.unsqueeze(0).unsqueeze(0) + 
        coeffs_3.unsqueeze(-1) * t_cubed.unsqueeze(0).unsqueeze(0)
    )
    return trajectory_joints


def computeMinCostTrajectory(first: torch.Tensor,
                             last: torch.Tensor,
                             control_cost_matrix_R_padded: torch.Tensor,
                             inv_control_cost_matrix_R: torch.Tensor) -> torch.Tensor:
    """计算最小控制成本轨迹
    Args:
        first: (B, D)
        last: (B, D)
        control_cost_matrix_R_padded: (T+L, T+L), L = FINITE_DIFF_RULE_LENGTH - 1
        inv_control_cost_matrix_R: (T, T)
        trajectory_joints: (B, D, T)
    """
    timesteps = control_cost_matrix_R_padded.shape[0] - 2 * (FINITE_DIFF_RULE_LENGTH - 1)
    start_index_padded = FINITE_DIFF_RULE_LENGTH - 1
    end_index_padded = start_index_padded + timesteps
    
    # Create vectors of ones
    ones_vec = torch.ones((FINITE_DIFF_RULE_LENGTH - 1,), dtype=first.dtype, device=first.device)   # (L,)
    
    # Extract blocks
    block1 = control_cost_matrix_R_padded[0:FINITE_DIFF_RULE_LENGTH - 1, start_index_padded:end_index_padded]   # (L, T)
    block2_start_row = control_cost_matrix_R_padded.shape[0] - (FINITE_DIFF_RULE_LENGTH - 1)
    block2 = control_cost_matrix_R_padded[block2_start_row:block2_start_row + (FINITE_DIFF_RULE_LENGTH - 1), 
                                          start_index_padded:end_index_padded]   # (L, T)

    # Compute linear control cost for all dimensions at once
    # (B, D, 1) * (1, 1, T)
    linear_control_cost_part1 = first.unsqueeze(-1) * (ones_vec.unsqueeze(0) @ block1).unsqueeze(0)     # (B, D, T)
    linear_control_cost_part2 = last.unsqueeze(-1) * (ones_vec.unsqueeze(0) @ block2).unsqueeze(0)      # (B, D, T)
    linear_control_cost = 2.0 * (linear_control_cost_part1 + linear_control_cost_part2)      # (B, D, T)

    # Compute trajectory: -0.5 * inv_R * linear_control_cost.T
    trajectory_joints = (-0.5 * (inv_control_cost_matrix_R.unsqueeze(0).unsqueeze(0) @ linear_control_cost.unsqueeze(-1)).squeeze(-1))
    
    # Set boundary conditions
    trajectory_joints[..., 0] = first
    trajectory_joints[..., -1] = last
    return trajectory_joints


def computeParametersControlCosts(rollouts: torch.Tensor,
                                  dt: float,
                                  control_cost_weight: float,
                                  control_cost_matrix_R: torch.Tensor) -> torch.Tensor:
    """计算参数的控制成本
    Args:
        rollouts: (B, R, D, T)
        control_cost_matrix_R: (T, T)
    """
    num_timesteps = rollouts.shape[-1]
    costs_per_step = 0.5 * (1.0 / dt) * (rollouts.unsqueeze(-2) @ control_cost_matrix_R.unsqueeze(0).unsqueeze(0).unsqueeze(0) 
                      @ rollouts.unsqueeze(-1)).squeeze(-1).squeeze(-1)     # (B, R, D)
    max_val = torch.max(costs_per_step, dim=-1, keepdim=True)[0]
    max_val = torch.where(max_val > 1e-8, max_val, 1)
    costs_per_step /= max_val
    control_costs = costs_per_step.unsqueeze(-1).repeat(1, 1, 1, num_timesteps)
    control_costs *= control_cost_weight

    return control_costs


def computeParametersControlCosts_1(rollouts: torch.Tensor,
                                  dt: float,
                                  control_cost_weight: float) -> torch.Tensor:
    """计算参数的控制成本,对加速度计算成本
    Args:
        rollouts: (B, R, D, T)
        control_cost_matrix_R: (T, T)
    """
    device = rollouts.device
    num_timesteps = rollouts.shape[-1]

    rollouts_padded = torch.cat((rollouts[..., 0].unsqueeze(-1).repeat(1, 1, 1, FINITE_DIFF_RULE_LENGTH - 1),
                                 rollouts,
                                 rollouts[..., -1].unsqueeze(-1).repeat(1, 1, 1, FINITE_DIFF_RULE_LENGTH - 1),), dim=-1)    # (B, R, D, NP)

    num_timesteps_padded = num_timesteps + 2 * (FINITE_DIFF_RULE_LENGTH - 1)
    finite_diff_matrix_A_padded = generateFiniteDifferenceMatrix(
        num_timesteps_padded, DerivativeOrder.STOMP_ACCELERATION, dt, device)  # (NP, NP)
    
    finite_diff_matrix_A_padded = finite_diff_matrix_A_padded.unsqueeze(0).unsqueeze(0)    # (1, 1, NP, NP)
    acc = (finite_diff_matrix_A_padded @ rollouts_padded.transpose(-2, -1)).transpose(-2, -1)   # (B, R, D, NP) 
    # costs_per_step = 0.5 * (1.0 / dt) * torch.sum(
    #     (acc[..., (FINITE_DIFF_RULE_LENGTH - 1):-(FINITE_DIFF_RULE_LENGTH - 1)])**2, dim=-1)    # (B, R, D)
    costs_per_step = 0.5 * torch.sum(
        (acc[..., (FINITE_DIFF_RULE_LENGTH - 1):-(FINITE_DIFF_RULE_LENGTH - 1)])**2, dim=-1)    # (B, R, D)
    # max_val = torch.max(costs_per_step, dim=-1, keepdim=True)[0]
    # max_val = torch.where(max_val > 1e-8, max_val, 1)
    # costs_per_step /= max_val
    control_costs = costs_per_step.unsqueeze(-1).repeat(1, 1, 1, num_timesteps)
    control_costs *= control_cost_weight

    return control_costs


def computeParametersControlCosts_2(rollouts: torch.Tensor,
                                  dt: float,
                                  control_cost_weight: float) -> torch.Tensor:
    """计算参数的控制成本,对速度计算成本
    Args:
        rollouts: (B, R, D, T)
        control_cost_matrix_R: (T, T)
    """
    device = rollouts.device
    num_timesteps = rollouts.shape[-1]

    rollouts_padded = torch.cat((rollouts[..., 0].unsqueeze(-1).repeat(1, 1, 1, FINITE_DIFF_RULE_LENGTH - 1),
                                 rollouts,
                                 rollouts[..., -1].unsqueeze(-1).repeat(1, 1, 1, FINITE_DIFF_RULE_LENGTH - 1),), dim=-1)    # (B, R, D, NP)

    num_timesteps_padded = num_timesteps + 2 * (FINITE_DIFF_RULE_LENGTH - 1)
    finite_diff_matrix_A_padded = generateFiniteDifferenceMatrix(
        num_timesteps_padded, DerivativeOrder.STOMP_VELOCITY, dt, device)  # (NP, NP)
    
    finite_diff_matrix_A_padded = finite_diff_matrix_A_padded.unsqueeze(0).unsqueeze(0)    # (1, 1, NP, NP)
    vel = (finite_diff_matrix_A_padded @ rollouts_padded.transpose(-2, -1)).transpose(-2, -1)   # (B, R, D, NP) 
    # costs_per_step = 0.5 * (1.0 / dt) * torch.sum(
    #     (acc[..., (FINITE_DIFF_RULE_LENGTH - 1):-(FINITE_DIFF_RULE_LENGTH - 1)])**2, dim=-1)    # (B, R, D)
    # costs_per_step = torch.sqrt(torch.sum(
    #     (vel[..., (FINITE_DIFF_RULE_LENGTH - 1):-(FINITE_DIFF_RULE_LENGTH - 1)])**2, dim=-1))    # (B, R, D) 
    # max_val = torch.max(costs_per_step, dim=-1, keepdim=True)[0]
    # max_val = torch.where(max_val > 1e-8, max_val, 1)
    # costs_per_step /= max_val
    costs_per_step = torch.sum(
    (vel[..., (FINITE_DIFF_RULE_LENGTH - 1):-(FINITE_DIFF_RULE_LENGTH - 1)])**2, dim=-1)     # (B, R, D) 
    control_costs = costs_per_step.unsqueeze(-1).repeat(1, 1, 1, num_timesteps)
    control_costs *= control_cost_weight

    return control_costs


# 生成有限差分矩阵
def generateFiniteDifferenceMatrix(num_time_steps: int, order: DerivativeOrder, dt: float, device):
    """生成有限差分矩阵"""
    diff_matrix = torch.zeros((num_time_steps, num_time_steps), device=device)
    multiplier = 1.0 / (dt ** int(order))
    half_length = FINITE_DIFF_RULE_LENGTH // 2
    for i in range(num_time_steps):
        for j in range(-half_length, half_length + 1):
            index = i + j
            # 处理边界条件：如果索引超出范围，则跳过（不设置矩阵元素）
            if index < 0 or index >= num_time_steps:
                continue
            coeff_index = j + half_length
            diff_matrix[i, index] = multiplier * FINITE_CENTRAL_DIFF_COEFFS[order][coeff_index]

    return diff_matrix


def generateControlCostMatrix(num_timesteps: int, delta_t: float, padding: bool, device):
    if padding:
        start_index_padded = FINITE_DIFF_RULE_LENGTH - 1
        num_timesteps_padded = num_timesteps + 2 * (FINITE_DIFF_RULE_LENGTH - 1)
        finite_diff_matrix_A_padded = generateFiniteDifferenceMatrix(
            num_timesteps_padded, DerivativeOrder.STOMP_ACCELERATION, delta_t, device)  # (NP, NP)
        # 生成 R_padded 矩阵 (R = A_transpose * A)，控制成本矩阵
        control_cost_matrix_R_padded = ((delta_t**(int(DerivativeOrder.STOMP_ACCELERATION)*2-1)) * 
                                            (finite_diff_matrix_A_padded.t() @ finite_diff_matrix_A_padded))
        # 提取 R 矩阵
        start_idx = start_index_padded
        end_idx = start_idx + num_timesteps
        control_cost_matrix_R = control_cost_matrix_R_padded[start_idx:end_idx, start_idx:end_idx]
    else:
        finite_diff_matrix_A = generateFiniteDifferenceMatrix(
            num_timesteps, DerivativeOrder.STOMP_ACCELERATION, delta_t, device)  # (NP, NP)
        # 生成 R 矩阵 (R = A_transpose * A)，控制成本矩阵
        control_cost_matrix_R = ((delta_t**(int(DerivativeOrder.STOMP_ACCELERATION)*2-1)) * 
                                            (finite_diff_matrix_A.t() @ finite_diff_matrix_A))
    # R 矩阵求逆
    try:
        # Use LU decomposition for better numerical stability if needed
        inv_control_cost_matrix_R = torch.inverse(control_cost_matrix_R)
    except:
        print("Error: Control cost matrix R is singular and cannot be inverted.")
    # 缩放使得max(R^-1)==1,用来计算生成的噪音
    max_val = torch.max(torch.abs(inv_control_cost_matrix_R))
    if padding:
        control_cost_matrix_R_padded *= max_val
    else:
        control_cost_matrix_R_padded = torch.empty(0)
    control_cost_matrix_R *= max_val
    inv_control_cost_matrix_R /= max_val    # (T, T)

    diff = (inv_control_cost_matrix_R - inv_control_cost_matrix_R.T).abs().max()
    if diff > 1e-8:
        inv_control_cost_matrix_R = (inv_control_cost_matrix_R + inv_control_cost_matrix_R.T) / 2

    # float32 求逆病态矩阵会留微小负特征值 → 协方差非严格正定，新版 torch 的
    # MultivariateNormal(PositiveDefinite/Cholesky) 会报错。按最小特征值抬升对角保证 PD：
    # 抖动量 ~1e-6·最大特征值，远小于噪声尺度，对 STOMP 采样/轨迹无实质影响。
    evals = torch.linalg.eigvalsh(inv_control_cost_matrix_R)
    floor = 1e-6 * evals.max().clamp_min(1e-12)
    if evals.min() < floor:
        n = inv_control_cost_matrix_R.shape[0]
        inv_control_cost_matrix_R = inv_control_cost_matrix_R + (floor - evals.min()) * torch.eye(
            n, device=device, dtype=inv_control_cost_matrix_R.dtype)

    return control_cost_matrix_R_padded, control_cost_matrix_R, inv_control_cost_matrix_R



# TODO:需要修改,该方法是stomp原始算法,不能插入不更新的中间点
def generateSmoothingMatrix_original(matrix: torch.Tensor):
    """
    对R_inv矩阵每列进行缩放,使对角线上数值为1,矩阵起到滤波效果

    """
    projection = matrix.clone()
    num_timesteps = matrix.shape[0]
    for t in range(num_timesteps):
        max_val = projection[t, t]  # 获取对角线元素
        # 对第 t 列进行缩放
        projection[:, t] *= (1.0 / max_val)
        # projection[:, t] *= (1.0 / (num_timesteps * max_val))
    return projection
    

def generateSmoothingMatrix(R_inv_matrix: torch.Tensor, num_transfer: int, num_timesteps_next: int):
    """
    根据R_inv矩阵拟合,拓展到多段轨迹上,矩阵起到滤波效果
    Args:
        matrix: R_inv矩阵
        num_transfer: 观测位姿间轨迹的数量
        num_timesteps_next: 观测位姿间轨迹的步长
    """
    device = R_inv_matrix.device

    # 计算五次多项式的系数
    idx_flat = torch.argmax(R_inv_matrix)
    idx_array = torch.unravel_index(idx_flat, R_inv_matrix.shape)
    pts = R_inv_matrix[:idx_array[0]+1, idx_array[1]].clone()
    idx_mid = int(idx_array[0] / 2 + 1)
    dt = 1 / (len(pts) - 1)
    pos = torch.zeros((3,), dtype=torch.float, device=device)
    vel = torch.zeros((3,), dtype=torch.float, device=device)
    pos_mid = pts[idx_mid].clone()
    vel_mid = torch.sum(pts[idx_mid - 3: idx_mid + 4] * torch.tensor(
        FINITE_CENTRAL_DIFF_COEFFS[DerivativeOrder.STOMP_VELOCITY], dtype=torch.float, device=device)) / (dt**1)
    pos[1] = pos_mid
    vel[1] = vel_mid
    pos[2] = 1
    t_norm = torch.tensor([0, 0.5, 1], dtype=torch.float, device=device)
    coeffs = fit_quintic(t_norm, pos, vel)

    # 构造滤波矩阵
    # version 1:
    def generateCurve(idx, num_points, coeffs):
        """
        生成单点对应的滤波曲线,即矩阵的列
        """
        idx1 = idx
        idx2 = num_points - 1 - idx
        def computePoints(idx, coeffs):
            x = torch.linspace(0, 1, idx+1, device=device).unsqueeze(1)
            p = torch.arange(6, device=device).unsqueeze(0)
            return (x ** p) @ coeffs
        front = computePoints(idx1, coeffs)
        back = computePoints(idx2, coeffs)[:-1]
        back = back.flip(0)
        return torch.cat((front, back), dim=-1)
    # 构建滤波矩阵，无缩放
    R_inv_matrix = generateSmoothingMatrix_original(R_inv_matrix)
    T = R_inv_matrix.shape[0]
    num_timesteps_all = T + num_transfer * (num_timesteps_next - 1)
    ratio = int((T - 1)/(num_timesteps_next - 1))

    mat_R = torch.zeros((num_timesteps_all, T), dtype=torch.float, device=device)
    for i in range(T):
        pts = R_inv_matrix[:, i]
        idx = torch.argmax(pts)
        max = pts[idx]
        array = generateCurve(idx, len(pts), coeffs) * max
        array_new = array[0::ratio]
        for _ in range(num_transfer):
            array_new = array_new.flip(0) * (-1)
            array = torch.cat((array, array_new[1:]), dim=-1)
        mat_R[:, i] = array
    mat_N = mat_R[:, 0::ratio]
    TN = mat_N.shape[1]
    for i in range(num_transfer):
        mat_N = mat_N.flip(1) * (-1)
        mat_R = torch.cat((mat_R, mat_N[:, 1:]), dim=-1)
    # 进行缩放
    scale = torch.ones_like(mat_R, dtype=torch.float, device=device)
    scale[T:, :T] = 1/ratio
    for i in range(num_transfer):
        col_start_idx = T + i * (TN - 1)
        col_end_idx = T + (i + 1) * (TN - 1)
        row_front_end_idx = T + i * (TN - 1)
        row_back_start_idx = T + (i + 1) * (TN - 1)
        scale[:row_front_end_idx, col_start_idx:col_end_idx] = 1/ratio
        scale[row_back_start_idx:, col_start_idx:col_end_idx] = 1/ratio
    
    return mat_R * scale / num_timesteps_all


def generateSmoothingMatrix_1(R_inv_matrix: torch.Tensor, num_transfer: int, num_timesteps_next: int, damping: float):
    """
    根据R_inv矩阵拟合,拓展到多段轨迹上,矩阵起到滤波效果
    Args:
        matrix: R_inv矩阵
        num_transfer: 观测位姿间轨迹的数量
        num_timesteps_next: 观测位姿间轨迹的步长
    """
    device = R_inv_matrix.device

    # 计算五次多项式的系数
    idx_flat = torch.argmax(R_inv_matrix)
    idx_array = torch.unravel_index(idx_flat, R_inv_matrix.shape)
    pts = R_inv_matrix[:idx_array[0]+1, idx_array[1]].clone()
    idx_mid = int(idx_array[0] / 2 + 1)
    dt = 1 / (len(pts) - 1)
    pos = torch.zeros((3,), dtype=torch.float, device=device)
    vel = torch.zeros((3,), dtype=torch.float, device=device)
    pos_mid = pts[idx_mid].clone()
    vel_mid = torch.sum(pts[idx_mid - 3: idx_mid + 4] * torch.tensor(
        FINITE_CENTRAL_DIFF_COEFFS[DerivativeOrder.STOMP_VELOCITY], dtype=torch.float, device=device)) / (dt**1)
    pos[1] = pos_mid
    vel[1] = vel_mid
    pos[2] = 1
    t_norm = torch.tensor([0, 0.5, 1], dtype=torch.float, device=device)
    coeffs = fit_quintic(t_norm, pos, vel)

    # 构造滤波矩阵
    # version 1:
    def generateCurve(idx, num_points, coeffs):
        """
        生成单点对应的滤波曲线,即矩阵的列
        """
        device = coeffs.device
        if idx == num_points - 1:
            idx1 = idx
            idx2 = num_points - 1 - idx
            def computePoints(idx, coeffs):
                x = torch.linspace(0, 1, idx+1, device=device).unsqueeze(1)
                p = torch.arange(6, device=device).unsqueeze(0)
                return (x ** p) @ coeffs
            front = computePoints(idx1, coeffs)[:-1]
            back = computePoints(idx2, coeffs)
            back = back.flip(0)
            return torch.cat((front, back), dim=-1)
        else:
            idx1 = idx
            idx2 = num_points - 1 - idx
            def computePoints(idx, coeffs):
                x = torch.linspace(0, 1, idx+1, device=device).unsqueeze(1)
                p = torch.arange(6, device=device).unsqueeze(0)
                return (x ** p) @ coeffs
            front = computePoints(idx1, coeffs)
            back = computePoints(idx2, coeffs)[:-1]
            back = back.flip(0)
            return torch.cat((front, back), dim=-1)
    # 构建滤波矩阵，无缩放
    T = R_inv_matrix.shape[0]
    num_timesteps_all = T + num_transfer * (num_timesteps_next - 1)
    ratio = int((T - 1)/(num_timesteps_next - 1))

    mat_R = torch.zeros((num_timesteps_all, T), dtype=torch.float, device=device)
    for i in range(T):
        array = generateCurve(i, T, coeffs)
        array_new = array[0::ratio]
        for _ in range(num_transfer):
            array_new = array_new.flip(0) * (-1)
            array = torch.cat((array, array_new[1:]), dim=-1)
        mat_R[:, i] = array
    mat_N = mat_R[:, 0::ratio]
    TN = mat_N.shape[1]
    for i in range(num_transfer):
        mat_N = mat_N.flip(1) * (-1)
        mat_R = torch.cat((mat_R, mat_N[:, 1:]), dim=-1)
    # 进行缩放
    scale = torch.ones_like(mat_R, dtype=torch.float, device=device)

    for i in range(num_transfer):
        col_start_idx = T + i * (TN - 1)
        # col_end_idx = T + (i + 1) * (TN - 1)
        if i == 0:
            row_start_idx = 0
            row_end_idx = T
        else:
            row_start_idx = T + (i - 1) * (TN - 1)
            row_end_idx = T + i * (TN - 1)
        for j in range(num_transfer - i):
            col_start_idx += j * (TN - 1)
            scale[row_start_idx:row_end_idx, col_start_idx:] *= 1/damping
    scale = torch.min(scale, scale.T)
    
    return mat_R * scale / T


def fit_quintic(t_norm: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor):
    """
    拟合5次多项式系数,约束为3个点的位置+速度
    """
    X = pos
    V = vel
    device = pos.device

    # 构造矩阵 A 和向量 b
    A = torch.zeros((6, 6), dtype=torch.float, device=device)
    b = torch.zeros(6, dtype=torch.float, device=device)
    # 位置约束
    for i in range(3):
        t = t_norm[i]
        A[i, :] = torch.tensor([1, t, t**2, t**3, t**4, t**5], dtype=torch.float)
        b[i] = X[i]
    # 速度约束
    for i in range(3):
        t = t_norm[i]
        A[3+i, :] = torch.tensor([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4], dtype=torch.float)
        b[3+i] = V[i]
    # 求解 A a = b
    coeffs_norm = torch.linalg.solve(A, b)  # 在归一化时间下的系数

    return coeffs_norm


def polyfit(x, y, degree, device):
    """
    实现的多项式最小二乘拟合
    Args:
        x: (N,)
        y: (N,)
        degree: 多项式阶数

        coef_norm: 拟合系数 (a0, a1, ...)
        predict_fn: 预测函数
        (x_mean, x_std): 归一化参数
    """
    x = torch.as_tensor(x, dtype=torch.float, device=device)
    y = torch.as_tensor(y, dtype=torch.float, device=device)

    # 归一化
    x_mean = x.mean()
    x_std = x.std()
    x_norm = (x - x_mean) / x_std
    y_norm = y / torch.max(y)

    # X矩阵
    powers = torch.arange(degree + 1, device=device, dtype=torch.float)
    X = x_norm.unsqueeze(1) ** powers.unsqueeze(0)  # (N, degree+1)

    # 最小二乘法求解
    coef_norm = torch.linalg.lstsq(X, y_norm).solution

    return coef_norm