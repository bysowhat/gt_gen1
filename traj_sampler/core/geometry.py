"""
几何变换和投影辅助函数（支持GPU）
"""
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# 向后兼容：如果没有torch，使用scipy
if not TORCH_AVAILABLE:
    from scipy.spatial.transform import Rotation


def _is_tensor(x):
    """检查输入是否为torch tensor"""
    return TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def _to_tensor(x, device=None):
    """将输入转换为tensor（如果是numpy则转换，如果已经是tensor则返回）"""
    if _is_tensor(x):
        if device is not None and x.device != device:
            return x.to(device)
        return x
    elif TORCH_AVAILABLE:
        return torch.tensor(x, device=device, dtype=torch.float32)
    else:
        return np.array(x)


def _quaternion_to_rotation_matrix(quaternion, device=None):
    """
    将四元数转换为旋转矩阵（支持GPU）
    
    Args:
        quaternion: 四元数 [w, x, y, z] 格式（从transform_7_to_4x4传入）
        device: torch device (cpu/cuda)
    
    Returns:
        rotation_matrix: 3x3旋转矩阵
    """
    # 如果是numpy数组，先处理
    if isinstance(quaternion, np.ndarray):
        if TORCH_AVAILABLE:
            quaternion = torch.from_numpy(quaternion).float()
            if device is not None:
                quaternion = quaternion.to(device)
        else:
            # 回退到scipy
            # 输入是 [w, x, y, z]，scipy需要 [x, y, z, w]
            quaternion_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
            rotation = Rotation.from_quat(quaternion_xyzw)
            return rotation.as_matrix()
    
    # 确保是tensor
    if not _is_tensor(quaternion):
        quaternion = _to_tensor(quaternion, device=device)
    
    # 输入格式是 [w, x, y, z]
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    # 使用torch实现四元数到旋转矩阵的转换
    # R = [[1-2(y^2+z^2), 2(xy-wz), 2(xz+wy)],
    #      [2(xy+wz), 1-2(x^2+z^2), 2(yz-wx)],
    #      [2(xz-wy), 2(yz+wx), 1-2(x^2+y^2)]]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)
    
    return R


def transform_7_to_4x4(pose_7d, device=None):
    """
    将7维位姿 [x, y, z, qw, qx, qy, qz] 转换为4x4变换矩阵（支持GPU）
    
    Args:
        pose_7d: 7维位姿向量 [x, y, z, qw, qx, qy, qz] (numpy array 或 torch tensor)
        device: torch device (cpu/cuda)，如果为None则自动检测
    
    Returns:
        transform: 4x4变换矩阵 (相机到世界坐标系)
    """
    # 转换为tensor
    pose_7d = _to_tensor(pose_7d, device=device)
    if device is None and _is_tensor(pose_7d):
        device = pose_7d.device
    
    cam_position_in_world = pose_7d[:3]  # 相机在世界坐标系中的位置
    quaternion = pose_7d[3:]  # [w, x, y, z] 格式，表示"相机到世界"的旋转
    
    # 转换为旋转矩阵
    if TORCH_AVAILABLE and _is_tensor(pose_7d):
        R_cam_to_world = _quaternion_to_rotation_matrix(quaternion, device=device)
        # 构建4x4变换矩阵
        transform = torch.eye(4, device=device, dtype=pose_7d.dtype)
        transform[:3, :3] = R_cam_to_world
        transform[:3, 3] = cam_position_in_world
    else:
        # 回退到numpy/scipy
        quaternion_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        rotation = Rotation.from_quat(quaternion_xyzw)
        R_cam_to_world = rotation.as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = R_cam_to_world
        transform[:3, 3] = cam_position_in_world
    
    return transform


def project_points(points_3d, pose_cam_to_world, intrinsics, device=None):
    """
    将3D点从世界坐标系投影到相机像素坐标系（支持GPU）
    
    Args:
        points_3d: 3D点 (N, 3) 在世界坐标系下 (numpy array 或 torch tensor)
        pose_cam_to_world: 4x4变换矩阵，相机到世界坐标系
        intrinsics: 相机内参矩阵 (3, 3)
        device: torch device (cpu/cuda)，如果为None则自动检测
    
    Returns:
        points_2d: 2D像素坐标 (N, 2)
        depths: 深度值 (N,)
        valid_mask: 有效点掩码 (N,)
    """
    # 转换为tensor
    points_3d = _to_tensor(points_3d, device=device)
    pose_cam_to_world = _to_tensor(pose_cam_to_world, device=device)
    intrinsics = _to_tensor(intrinsics, device=device)
    
    if device is None and _is_tensor(points_3d):
        device = points_3d.device
    
    # 确定使用torch还是numpy
    use_torch = TORCH_AVAILABLE and _is_tensor(points_3d)
    
    # 转换为齐次坐标
    if points_3d.shape[-1] == 3:
        if use_torch:
            ones = torch.ones((points_3d.shape[0], 1), device=device, dtype=points_3d.dtype)
            points_3d_homo = torch.cat([points_3d, ones], dim=1)
        else:
            points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    else:
        points_3d_homo = points_3d
    
    # 转换到相机坐标系 (世界到相机 = 相机到世界的逆)
    if use_torch:
        pose_world_to_cam = torch.inverse(pose_cam_to_world)
        points_cam_homo = (pose_world_to_cam @ points_3d_homo.T).T
    else:
        pose_world_to_cam = np.linalg.inv(pose_cam_to_world)
        points_cam_homo = (pose_world_to_cam @ points_3d_homo.T).T
    
    points_cam = points_cam_homo[:, :3]
    
    # 提取深度
    depths = points_cam[:, 2]
    
    # 投影到像素坐标系
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # 避免除零
    if use_torch:
        valid_mask = depths > 1e-6
        u = torch.zeros(points_3d.shape[0], device=device, dtype=points_3d.dtype)
        v = torch.zeros(points_3d.shape[0], device=device, dtype=points_3d.dtype)
        u[valid_mask] = fx * points_cam[valid_mask, 0] / depths[valid_mask] + cx
        v[valid_mask] = fy * points_cam[valid_mask, 1] / depths[valid_mask] + cy
        points_2d = torch.stack([u, v], dim=1)
    else:
        valid_mask = depths > 1e-6
        u = np.zeros(points_3d.shape[0])
        v = np.zeros(points_3d.shape[0])
        u[valid_mask] = fx * points_cam[valid_mask, 0] / depths[valid_mask] + cx
        v[valid_mask] = fy * points_cam[valid_mask, 1] / depths[valid_mask] + cy
        points_2d = np.stack([u, v], axis=1)
    
    return points_2d, depths, valid_mask


def get_frustum_corners(pose_cam_to_world, intrinsics, img_h, img_w, 
                        near=0.1, far=10.0, device=None):
    """
    获取相机视锥体的8个角点（在世界坐标系下，支持GPU）
    
    Args:
        pose_cam_to_world: 4x4变换矩阵，相机到世界坐标系
        intrinsics: 相机内参矩阵 (3, 3)
        img_h: 图像高度
        img_w: 图像宽度
        near: 近平面距离
        far: 远平面距离
        device: torch device (cpu/cuda)，如果为None则自动检测
    
    Returns:
        corners: 8个角点 (8, 3) 在世界坐标系下
    """
    pose_cam_to_world = _to_tensor(pose_cam_to_world, device=device)
    intrinsics = _to_tensor(intrinsics, device=device)
    
    if device is None and _is_tensor(pose_cam_to_world):
        device = pose_cam_to_world.device
    
    use_torch = TORCH_AVAILABLE and _is_tensor(pose_cam_to_world)
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # 图像四个角在像素坐标系
    if use_torch:
        corners_2d = torch.tensor([
            [0, 0],           # 左上
            [img_w, 0],       # 右上
            [img_w, img_h],   # 右下
            [0, img_h],       # 左下
        ], device=device, dtype=pose_cam_to_world.dtype)
    else:
        corners_2d = np.array([
            [0, 0],           # 左上
            [img_w, 0],       # 右上
            [img_w, img_h],   # 右下
            [0, img_h],       # 左下
        ])
    
    # 转换为相机坐标系下的3D点（在近平面和远平面上）
    corners_3d = []
    for z in [near, far]:
        for i in range(len(corners_2d)):
            u, v = corners_2d[i, 0], corners_2d[i, 1]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            corners_3d.append([x, y, z])
    
    if use_torch:
        corners_3d = torch.tensor(corners_3d, device=device, dtype=pose_cam_to_world.dtype)
    else:
        corners_3d = np.array(corners_3d)
    
    # 转换到世界坐标系
    if use_torch:
        ones = torch.ones((8, 1), device=device, dtype=pose_cam_to_world.dtype)
        corners_3d_homo = torch.cat([corners_3d, ones], dim=1)
        corners_world_homo = (pose_cam_to_world @ corners_3d_homo.T).T
    else:
        corners_3d_homo = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
        corners_world_homo = (pose_cam_to_world @ corners_3d_homo.T).T
    
    corners_world = corners_world_homo[:, :3]
    
    return corners_world


def get_frustum_samples(pose_cam_to_world, intrinsics, img_h, img_w,
                        near=0.1, far=10.0, num_samples_per_plane=10, device=None):
    """
    获取相机视锥体上的采样点（在世界坐标系下，支持GPU）
    在近平面和远平面上均匀采样网格点，用于更准确的共视检测
    
    Args:
        pose_cam_to_world: 4x4变换矩阵，相机到世界坐标系
        intrinsics: 相机内参矩阵 (3, 3)
        img_h: 图像高度
        img_w: 图像宽度
        near: 近平面距离
        far: 远平面距离
        num_samples_per_plane: 每个平面上采样的点数（会生成 num_samples_per_plane^2 个点）
        device: torch device (cpu/cuda)，如果为None则自动检测
    
    Returns:
        samples: 采样点 (N, 3) 在世界坐标系下，N = 2 * num_samples_per_plane^2
    """
    pose_cam_to_world = _to_tensor(pose_cam_to_world, device=device)
    intrinsics = _to_tensor(intrinsics, device=device)
    
    if device is None and _is_tensor(pose_cam_to_world):
        device = pose_cam_to_world.device
    
    use_torch = TORCH_AVAILABLE and _is_tensor(pose_cam_to_world)
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # 在图像平面上生成均匀采样网格
    if use_torch:
        u_samples = torch.linspace(0, img_w, num_samples_per_plane, device=device, dtype=pose_cam_to_world.dtype)
        v_samples = torch.linspace(0, img_h, num_samples_per_plane, device=device, dtype=pose_cam_to_world.dtype)
        u_grid, v_grid = torch.meshgrid(u_samples, v_samples, indexing='ij')
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
    else:
        u_samples = np.linspace(0, img_w, num_samples_per_plane)
        v_samples = np.linspace(0, img_h, num_samples_per_plane)
        u_grid, v_grid = np.meshgrid(u_samples, v_samples)
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
    
    # 在近平面和远平面上生成采样点
    samples_3d = []
    for z in [near, far]:
        for i in range(len(u_flat)):
            u, v = u_flat[i], v_flat[i]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            samples_3d.append([x, y, z])
    
    if use_torch:
        samples_3d = torch.tensor(samples_3d, device=device, dtype=pose_cam_to_world.dtype)
    else:
        samples_3d = np.array(samples_3d)
    
    # 转换到世界坐标系
    if use_torch:
        ones = torch.ones((samples_3d.shape[0], 1), device=device, dtype=pose_cam_to_world.dtype)
        samples_3d_homo = torch.cat([samples_3d, ones], dim=1)
        samples_world_homo = (pose_cam_to_world @ samples_3d_homo.T).T
    else:
        samples_3d_homo = np.concatenate([samples_3d, np.ones((samples_3d.shape[0], 1))], axis=1)
        samples_world_homo = (pose_cam_to_world @ samples_3d_homo.T).T
    
    samples_world = samples_world_homo[:, :3]
    
    return samples_world

