"""
阶段 I：GPU 几何运算模块
提供所有必要的、可并行化的几何原语，所有函数均以 PyTorch/Tensor 形式编写
参考 MapAnything 的实现逻辑
"""
import numpy as np
from core.geometry import _is_tensor, _to_tensor

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None


def unproject_depth_to_world(K, T_cam_to_world, depth_map, valid_mask=None, device=None):
    """
    指令 I-1: 将深度图反投影到3D世界坐标系（对应 MapAnything 的 m_unproject 逻辑）
    
    参考 traj_visualize.py 中的正确逻辑：
    - pose 是 T_cam_to_world（相机到世界坐标系）
    - points_world = T_cam_to_world @ points_cam
    
    Args:
        K: 相机内参矩阵 (3, 3) 或 (N, 3, 3)
        T_cam_to_world: 相机到世界坐标系的变换矩阵 (4, 4) 或 (N, 4, 4)
        depth_map: 深度图 (H, W) 或 (N, H, W)
        valid_mask: 有效深度掩码 (H, W) 或 (N, H, W)，如果为None则自动生成
        device: torch device
    
    Returns:
        P_world: 世界坐标点云 (H, W, 3) 或 (N, H, W, 3)
        M_valid: 有效点掩码 (H, W) 或 (N, H, W)
    """
    K = _to_tensor(K, device=device)
    T_cam_to_world = _to_tensor(T_cam_to_world, device=device)
    depth_map = _to_tensor(depth_map, device=device)
    
    if device is None and _is_tensor(depth_map):
        device = depth_map.device
    
    use_torch = TORCH_AVAILABLE and _is_tensor(depth_map)
    
    # 处理批次维度
    is_batch = depth_map.dim() == 3 if use_torch else depth_map.ndim == 3
    
    if is_batch:
        N, H, W = depth_map.shape
    else:
        H, W = depth_map.shape[-2:]
        N = 1
        if use_torch:
            depth_map = depth_map.unsqueeze(0)
            K = K.unsqueeze(0) if K.dim() == 2 else K
            T_cam_to_world = T_cam_to_world.unsqueeze(0) if T_cam_to_world.dim() == 2 else T_cam_to_world
    
    # 生成有效深度掩码
    if valid_mask is None:
        if use_torch:
            valid_mask = depth_map > 1e-6
        else:
            valid_mask = depth_map > 1e-6
    else:
        valid_mask = _to_tensor(valid_mask, device=device)
        if not is_batch and use_torch:
            valid_mask = valid_mask.unsqueeze(0)
    
    # 创建像素坐标网格（参考 traj_visualize.py 的正确逻辑）
    # 注意：u 对应宽度（列），v 对应高度（行）
    if use_torch:
        u_coords = torch.arange(W, device=device, dtype=depth_map.dtype)  # 宽度方向
        v_coords = torch.arange(H, device=device, dtype=depth_map.dtype)  # 高度方向
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')  # 使用 'xy' 以匹配 traj_visualize.py
        # 扩展批次维度
        if is_batch:
            u_grid = u_grid.unsqueeze(0).expand(N, -1, -1)
            v_grid = v_grid.unsqueeze(0).expand(N, -1, -1)
    else:
        u_coords = np.arange(W)  # 宽度方向
        v_coords = np.arange(H)  # 高度方向
        u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing='xy')  # 使用 'xy' 以匹配 traj_visualize.py
        if is_batch:
            u_grid = np.tile(u_grid, (N, 1, 1))
            v_grid = np.tile(v_grid, (N, 1, 1))
    
    # 提取内参
    if use_torch:
        if K.dim() == 2:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
            fx = fx.view(N, 1, 1)
            fy = fy.view(N, 1, 1)
            cx = cx.view(N, 1, 1)
            cy = cy.view(N, 1, 1)
    else:
        if K.ndim == 2:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
            fx = fx[:, np.newaxis, np.newaxis]
            fy = fy[:, np.newaxis, np.newaxis]
            cx = cx[:, np.newaxis, np.newaxis]
            cy = cy[:, np.newaxis, np.newaxis]
    
    # 反投影到相机坐标系
    if use_torch:
        x_cam = (u_grid - cx) * depth_map / fx
        y_cam = (v_grid - cy) * depth_map / fy
        z_cam = depth_map
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, H, W, 3)
    else:
        x_cam = (u_grid - cx) * depth_map / fx
        y_cam = (v_grid - cy) * depth_map / fy
        z_cam = depth_map
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, H, W, 3)
    
    # 转换到世界坐标系
    if use_torch:
        # 转换为齐次坐标
        ones = torch.ones((N, H, W, 1), device=device, dtype=points_cam.dtype)
        points_cam_homo = torch.cat([points_cam, ones], dim=-1)  # (N, H, W, 4)
        
        # 应用变换矩阵：T_cam_to_world @ P_cam = P_world（参考 traj_visualize.py）
        if T_cam_to_world.dim() == 2:
            T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(N, -1, -1)
        
        # 重塑为批量矩阵乘法
        # points_cam_homo: (N, H, W, 4) -> (N, H*W, 4)
        points_cam_homo_flat = points_cam_homo.reshape(N, H * W, 4)  # (N, H*W, 4)
        # 批量矩阵乘法: (N, 4, 4) @ (N, 4, H*W) -> (N, 4, H*W)
        # T_cam_to_world @ P_cam = P_world
        points_world_homo_flat = torch.bmm(
            T_cam_to_world.reshape(N, 4, 4),
            points_cam_homo_flat.permute(0, 2, 1)  # (N, 4, H*W)
        )  # (N, 4, H*W)
        points_world_homo_flat = points_world_homo_flat.permute(0, 2, 1)  # (N, H*W, 4)
        points_world_homo = points_world_homo_flat.reshape(N, H, W, 4)
        points_world = points_world_homo[..., :3]  # (N, H, W, 3)
    else:
        ones = np.ones((N, H, W, 1))
        points_cam_homo = np.concatenate([points_cam, ones], axis=-1)  # (N, H, W, 4)
        
        if T_cam_to_world.ndim == 2:
            T_cam_to_world = np.tile(T_cam_to_world, (N, 1, 1))
        
        points_world = np.zeros((N, H, W, 3))
        for i in range(N):
            points_cam_homo_i = points_cam_homo[i].reshape(H * W, 4).T  # (4, H*W)
            # T_cam_to_world @ P_cam = P_world
            points_world_homo_i = T_cam_to_world[i] @ points_cam_homo_i  # (4, H*W)
            points_world[i] = points_world_homo_i[:3].T.reshape(H, W, 3)
    
    # 移除批次维度（如果输入是单帧）
    if not is_batch:
        if use_torch:
            points_world = points_world.squeeze(0)
            valid_mask = valid_mask.squeeze(0)
        else:
            points_world = points_world[0]
            valid_mask = valid_mask[0]
    
    return points_world, valid_mask


def inverse_transform_4x4(T, device=None):
    """
    指令 I-2: 计算4x4变换矩阵的逆矩阵
    
    Args:
        T: 变换矩阵 (4, 4) 或 (N, 4, 4)
        device: torch device
    
    Returns:
        T_inv: 逆矩阵 (4, 4) 或 (N, 4, 4)
    """
    T = _to_tensor(T, device=device)
    
    if _is_tensor(T):
        if T.dim() == 2:
            return torch.inverse(T)
        else:
            return torch.inverse(T)
    else:
        if T.ndim == 2:
            return np.linalg.inv(T)
        else:
            return np.linalg.inv(T)


def project_world_to_pixel(K, T_cam_to_world, P_world, device=None):
    """
    指令 I-3: 将世界点投影到目标相机的像素平面（对应 MapAnything 的 m_project 逻辑）
    
    Args:
        K: 相机内参矩阵 (3, 3) 或 (N, 3, 3)
        T_cam_to_world: 相机到世界坐标系的变换矩阵 (4, 4) 或 (N, 4, 4)
        P_world: 世界坐标点 (H, W, 3) 或 (N, H, W, 3) 或 (M, 3)
        device: torch device
    
    Returns:
        P_2d: 像素坐标 (H, W, 2) 或 (N, H, W, 2) 或 (M, 2)
        D_expected: 期望深度值 (H, W) 或 (N, H, W) 或 (M,)
    """
    K = _to_tensor(K, device=device)
    T_cam_to_world = _to_tensor(T_cam_to_world, device=device)
    P_world = _to_tensor(P_world, device=device)
    
    if device is None and _is_tensor(P_world):
        device = P_world.device
    
    use_torch = TORCH_AVAILABLE and _is_tensor(P_world)
    
    # 处理不同的输入形状
    if use_torch:
        if P_world.dim() == 2 and P_world.shape[1] == 3:  # (M, 3) - 点云格式
            M = P_world.shape[0]
            is_point_cloud = True
        elif P_world.dim() == 3 and P_world.shape[2] == 3:  # (H, W, 3)
            H, W = P_world.shape[:2]
            is_point_cloud = False
            P_world = P_world.unsqueeze(0)
        elif P_world.dim() == 4:  # (N, H, W, 3)
            N, H, W = P_world.shape[:3]
            is_point_cloud = False
        else:
            raise ValueError(f"Unexpected P_world shape: {P_world.shape}")
    else:
        if P_world.ndim == 2 and P_world.shape[1] == 3:  # (M, 3)
            M = P_world.shape[0]
            is_point_cloud = True
        elif P_world.ndim == 3 and P_world.shape[2] == 3:  # (H, W, 3)
            H, W = P_world.shape[:2]
            is_point_cloud = False
            P_world = P_world[np.newaxis, ...]
        elif P_world.ndim == 4:  # (N, H, W, 3)
            N, H, W = P_world.shape[:3]
            is_point_cloud = False
        else:
            raise ValueError(f"Unexpected P_world shape: {P_world.shape}")
    
    # 转换到相机坐标系
    T_world_to_cam = inverse_transform_4x4(T_cam_to_world, device=device)
    
    if use_torch:
        if is_point_cloud:
            # (M, 3) -> (M, 4) 齐次坐标
            ones = torch.ones((M, 1), device=device, dtype=P_world.dtype)
            P_world_homo = torch.cat([P_world, ones], dim=1)  # (M, 4)
            
            if T_world_to_cam.dim() == 2:
                # 单个变换矩阵: (4, 4) -> (1, 4, 4)
                T_world_to_cam = T_world_to_cam.unsqueeze(0)
            
            # 矩阵乘法: (1, 4, 4) @ (4, M) -> (1, 4, M) -> (M, 4)
            P_cam_homo = (T_world_to_cam @ P_world_homo.T).squeeze(0).T  # (M, 4)
            P_cam = P_cam_homo[:, :3]  # (M, 3)
        else:
            # (N, H, W, 3) -> (N, H, W, 4)
            N = P_world.shape[0]
            ones = torch.ones((N, H, W, 1), device=device, dtype=P_world.dtype)
            P_world_homo = torch.cat([P_world, ones], dim=-1)  # (N, H, W, 4)
            
            if T_world_to_cam.dim() == 2:
                T_world_to_cam = T_world_to_cam.unsqueeze(0).expand(N, -1, -1)
            
            # 重塑为批量矩阵乘法
            P_world_homo_flat = P_world_homo.reshape(N, H * W, 4)  # (N, H*W, 4)
            # 批量矩阵乘法: (N, 4, 4) @ (N, 4, H*W) -> (N, 4, H*W)
            P_cam_homo_flat = torch.bmm(
                T_world_to_cam.reshape(N, 4, 4),
                P_world_homo_flat.permute(0, 2, 1)  # (N, 4, H*W)
            )  # (N, 4, H*W)
            P_cam_homo_flat = P_cam_homo_flat.permute(0, 2, 1)  # (N, H*W, 4)
            P_cam_homo = P_cam_homo_flat.reshape(N, H, W, 4)
            P_cam = P_cam_homo[..., :3]  # (N, H, W, 3)
    else:
        if is_point_cloud:
            ones = np.ones((M, 1))
            P_world_homo = np.concatenate([P_world, ones], axis=1)  # (M, 4)
            P_cam_homo = (T_world_to_cam @ P_world_homo.T).T  # (M, 4)
            P_cam = P_cam_homo[:, :3]  # (M, 3)
        else:
            N = P_world.shape[0]
            ones = np.ones((N, H, W, 1))
            P_world_homo = np.concatenate([P_world, ones], axis=-1)  # (N, H, W, 4)
            P_cam = np.zeros((N, H, W, 3))
            for i in range(N):
                P_world_homo_i = P_world_homo[i].reshape(H * W, 4).T  # (4, H*W)
                P_cam_homo_i = T_world_to_cam[i] @ P_world_homo_i  # (4, H*W)
                P_cam[i] = P_cam_homo_i[:3].T.reshape(H, W, 3)
    
    # 提取深度
    D_expected = P_cam[..., 2]
    
    # 投影到像素坐标系
    if use_torch:
        if K.dim() == 2:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
            if not is_point_cloud:
                fx = fx.view(N, 1, 1)
                fy = fy.view(N, 1, 1)
                cx = cx.view(N, 1, 1)
                cy = cy.view(N, 1, 1)
        
        valid_mask = D_expected > 1e-6
        
        if is_point_cloud:
            u = torch.zeros(M, device=device, dtype=P_cam.dtype)
            v = torch.zeros(M, device=device, dtype=P_cam.dtype)
            u[valid_mask] = fx * P_cam[valid_mask, 0] / D_expected[valid_mask] + cx
            v[valid_mask] = fy * P_cam[valid_mask, 1] / D_expected[valid_mask] + cy
            P_2d = torch.stack([u, v], dim=1)  # (M, 2)
        else:
            u = torch.zeros(N, H, W, device=device, dtype=P_cam.dtype)
            v = torch.zeros(N, H, W, device=device, dtype=P_cam.dtype)
            u[valid_mask] = fx * P_cam[valid_mask, 0] / D_expected[valid_mask] + cx
            v[valid_mask] = fy * P_cam[valid_mask, 1] / D_expected[valid_mask] + cy
            P_2d = torch.stack([u, v], dim=-1)  # (N, H, W, 2)
            
            # 移除批次维度（如果输入是单帧）
            if N == 1:
                P_2d = P_2d.squeeze(0)
                D_expected = D_expected.squeeze(0)
    else:
        if K.ndim == 2:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
            if not is_point_cloud:
                fx = fx[:, np.newaxis, np.newaxis]
                fy = fy[:, np.newaxis, np.newaxis]
                cx = cx[:, np.newaxis, np.newaxis]
                cy = cy[:, np.newaxis, np.newaxis]
        
        valid_mask = D_expected > 1e-6
        
        if is_point_cloud:
            u = np.zeros(M)
            v = np.zeros(M)
            u[valid_mask] = fx * P_cam[valid_mask, 0] / D_expected[valid_mask] + cx
            v[valid_mask] = fy * P_cam[valid_mask, 1] / D_expected[valid_mask] + cy
            P_2d = np.stack([u, v], axis=1)  # (M, 2)
        else:
            N = P_cam.shape[0]
            u = np.zeros((N, H, W))
            v = np.zeros((N, H, W))
            u[valid_mask] = fx * P_cam[valid_mask, 0] / D_expected[valid_mask] + cx
            v[valid_mask] = fy * P_cam[valid_mask, 1] / D_expected[valid_mask] + cy
            P_2d = np.stack([u, v], axis=-1)  # (N, H, W, 2)
            
            if N == 1:
                P_2d = P_2d[0]
                D_expected = D_expected[0]
    
    return P_2d, D_expected


def sample_depths_at_reproj(depth_map, P_2d, img_h, img_w, device=None):
    """
    指令 I-4: 在重投影点处采样深度值（使用 torch.nn.functional.grid_sample 逻辑）
    
    Args:
        depth_map: 目标深度图 (H, W) 或 (N, H, W)
        P_2d: 像素坐标 (M, 2) 或 (H, W, 2) 或 (N, H, W, 2)
        img_h: 图像高度
        img_w: 图像宽度
        device: torch device
    
    Returns:
        D_sampled: 采样得到的深度值，形状与P_2d的前两维相同
    """
    depth_map = _to_tensor(depth_map, device=device)
    P_2d = _to_tensor(P_2d, device=device)
    
    if device is None and _is_tensor(depth_map):
        device = depth_map.device
    
    use_torch = TORCH_AVAILABLE and _is_tensor(depth_map)
    
    if not use_torch:
        # NumPy实现：使用最近邻采样
        if P_2d.ndim == 2 and P_2d.shape[1] == 2:  # (M, 2)
            u = np.clip(P_2d[:, 0].astype(np.int32), 0, img_w - 1)
            v = np.clip(P_2d[:, 1].astype(np.int32), 0, img_h - 1)
            return depth_map[v, u]
        elif P_2d.ndim == 3:  # (H, W, 2)
            u = np.clip(P_2d[:, :, 0].astype(np.int32), 0, img_w - 1)
            v = np.clip(P_2d[:, :, 1].astype(np.int32), 0, img_h - 1)
            return depth_map[v, u]
        else:  # (N, H, W, 2)
            N = P_2d.shape[0]
            result = np.zeros((N, P_2d.shape[1], P_2d.shape[2]))
            for i in range(N):
                u = np.clip(P_2d[i, :, :, 0].astype(np.int32), 0, img_w - 1)
                v = np.clip(P_2d[i, :, :, 1].astype(np.int32), 0, img_h - 1)
                result[i] = depth_map[i][v, u]
            return result
    
    # PyTorch实现：使用grid_sample
    # 归一化坐标：2 * coord / (size - 1) - 1
    # grid_sample 期望的格式是 (x, y)，其中 x 对应宽度（列，u），y 对应高度（行，v）
    # P_2d 是 (u, v) 格式，直接对应 (x, y)
    # 归一化时：x (u) 除以 img_w - 1，y (v) 除以 img_h - 1
    normalized_pts = (
        2 * P_2d[..., [0, 1]] /  # (u, v) -> (x, y)
        torch.tensor([img_w - 1, img_h - 1], device=device, dtype=P_2d.dtype)  # [x_scale, y_scale]
        - 1
    )
    normalized_pts = torch.clamp(normalized_pts, min=-1.0, max=1.0)
    
    # 处理不同的输入形状
    # 优化：统一处理逻辑，确保所有形状都能正确采样
    if P_2d.dim() == 2:  # (M, 2) - 点云格式
        # 确保 normalized_pts 的形状为 (1, 1, M, 2) 用于 grid_sample
        normalized_pts = normalized_pts.unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)
        if depth_map.dim() == 2:
            depth_batch = depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            depth_batch = depth_map.unsqueeze(1)  # (N, 1, H, W)
        depth_sampled = F.grid_sample(
            depth_batch,
            normalized_pts,
            mode="nearest",
            align_corners=True,
            padding_mode='zeros'
        )
        # 确保正确移除维度
        if depth_sampled.dim() == 4:
            depth_sampled = depth_sampled.squeeze(0).squeeze(0)  # (M,)
        elif depth_sampled.dim() == 3:
            depth_sampled = depth_sampled.squeeze(0)  # (M,)
        else:
            depth_sampled = depth_sampled.squeeze()  # (M,) 或 (N, M)
    elif P_2d.dim() == 3:  # (H, W, 2)
        normalized_pts = normalized_pts.unsqueeze(0)  # (1, H, W, 2)
        if depth_map.dim() == 2:
            depth_batch = depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            depth_batch = depth_map.unsqueeze(1)  # (N, 1, H, W)
        depth_sampled = F.grid_sample(
            depth_batch,
            normalized_pts,
            mode="nearest",
            align_corners=True,
            padding_mode='zeros'
        )
        depth_sampled = depth_sampled.squeeze().squeeze()  # (H, W) 或 (N, H, W)
    else:  # (N, H, W, 2)
        N = P_2d.shape[0]
        if depth_map.dim() == 2:
            depth_batch = depth_map.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1)  # (N, 1, H, W)
        else:
            depth_batch = depth_map.unsqueeze(1)  # (N, 1, H, W)
        depth_sampled = F.grid_sample(
            depth_batch,
            normalized_pts,
            mode="nearest",
            align_corners=True,
            padding_mode='zeros'
        )
        depth_sampled = depth_sampled.squeeze(1)  # (N, H, W)
    
    return depth_sampled

