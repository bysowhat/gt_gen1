"""
加载 Pose GT 数据（支持GPU）
"""
import os
import numpy as np
from core.geometry import transform_7_to_4x4, _is_tensor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def load_pose_data(data_root, device=None, use_gpu=False):
    """
    加载位姿数据（支持GPU）
    
    Args:
        data_root: 数据根目录路径（包含info.npy的目录）
        device: torch device (cpu/cuda)，如果为None则根据use_gpu决定
        use_gpu: 是否使用GPU（如果device为None，则根据此参数决定）
    
    Returns:
        poses: 位姿列表，每个元素是4x4变换矩阵（相机到世界）
               如果use_gpu=True且torch可用，则返回torch tensor列表
        intrinsics: 相机内参矩阵 (3, 3) (numpy array 或 torch tensor)
        img_h: 图像高度
        img_w: 图像宽度
        num_frames: 总帧数
    """
    info_path = os.path.join(data_root, 'info.npy')
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"找不到info.npy文件: {info_path}")
    
    data_info = np.load(info_path, allow_pickle=True).item()
    
    cam_pose_list = data_info['cam_pose_list']
    cam_quat_list = data_info['cam_quat_list']
    cam_intrinsic = data_info['cam_intrinsic']
    
    num_frames = len(cam_pose_list)
    
    # 确定是否使用GPU
    if device is None:
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device('cuda')
        elif TORCH_AVAILABLE:
            device = torch.device('cpu')
        else:
            device = None
    
    # 转换位姿为4x4矩阵
    poses = []
    for i in range(num_frames):
        pose_7d = np.concatenate([cam_pose_list[i], cam_quat_list[i]])
        pose_4x4 = transform_7_to_4x4(pose_7d, device=device)
        poses.append(pose_4x4)
    
    # 转换内参
    if device is not None and TORCH_AVAILABLE:
        intrinsics = torch.from_numpy(cam_intrinsic).float().to(device)
    else:
        intrinsics = cam_intrinsic
    
    # 从内参推断图像尺寸（假设内参的cx, cy在图像中心附近）
    # 或者可以从实际图像文件读取
    if TORCH_AVAILABLE and _is_tensor(intrinsics):
        cx = intrinsics[0, 2].item()
        cy = intrinsics[1, 2].item()
    else:
        cx = cam_intrinsic[0, 2]
        cy = cam_intrinsic[1, 2]
    
    # 通常图像尺寸是内参中心坐标的2倍
    img_w = int(cx * 2)
    img_h = int(cy * 2)
    
    # 尝试从实际图像文件获取尺寸
    rgb_path = os.path.join(data_root, '0_rgb.jpg')
    if os.path.exists(rgb_path):
        import cv2
        img = cv2.imread(rgb_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
    
    return poses, intrinsics, img_h, img_w, num_frames


def load_depth_map(data_root, frame_idx, device=None):
    """
    加载指定帧的深度图（支持GPU）
    
    Args:
        data_root: 数据根目录路径
        frame_idx: 帧索引
        device: torch device (cpu/cuda)，如果为None则返回numpy数组
    
    Returns:
        depth_map: 深度图 (H, W) (numpy array 或 torch tensor)
    """
    depth_path = os.path.join(data_root, f'{frame_idx}_depth.npz')
    
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"找不到深度图文件: {depth_path}")
    
    depth_map = np.load(depth_path)['arr_0']
    
    # 转换为tensor（如果需要）
    if device is not None and TORCH_AVAILABLE:
        depth_map = torch.from_numpy(depth_map).float().to(device)
    
    return depth_map


def load_all_depth_maps(data_root, num_frames, device=None):
    """
    加载所有帧的深度图（支持GPU）
    
    Args:
        data_root: 数据根目录路径
        num_frames: 总帧数
        device: torch device (cpu/cuda)，如果为None则返回numpy数组列表
    
    Returns:
        depth_maps: 深度图列表，每个元素是 (H, W) 的深度图
    """
    depth_maps = []
    for i in range(num_frames):
        depth_map = load_depth_map(data_root, i, device=device)
        depth_maps.append(depth_map)
    return depth_maps

