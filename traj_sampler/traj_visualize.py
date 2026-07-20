import os
import json
import argparse
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation


def transform_7_to_4x4(pose_7d):
    """
    将7维位姿 [x, y, z, qw, qx, qy, qz] 转换为4x4变换矩阵
    
    关键理解（基于机械臂拍摄静止物体的场景）：
    - [x, y, z] 是相机在世界坐标系中的位置（机械臂末端位置）
    - [qw, qx, qy, qz] 是"相机到世界"的旋转（四元数，w在前）
    - 返回的矩阵应该是"相机到世界"的变换矩阵，用于将相机坐标系的点转换到世界坐标系
    
    构建方式：
    - 四元数直接表示"相机到世界"的旋转 R_cam_to_world
    - "相机到世界"的平移是相机在世界坐标系中的位置 t_cam_in_world
    - 最终矩阵：T_cam_to_world = [R_cam_to_world | t_cam_in_world]
    """
    cam_position_in_world = pose_7d[:3]  # 相机在世界坐标系中的位置
    quaternion = pose_7d[3:]  # [w, x, y, z] 格式，表示"相机到世界"的旋转
    
    # 转换为 [x, y, z, w] 格式供 scipy 使用
    quaternion_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    
    # 创建旋转矩阵（四元数直接表示"相机到世界"的旋转）
    rotation = Rotation.from_quat(quaternion_xyzw)
    R_cam_to_world = rotation.as_matrix()
    
    # 构建4x4变换矩阵（相机到世界）
    transform = np.eye(4)
    transform[:3, :3] = R_cam_to_world
    transform[:3, 3] = cam_position_in_world
    
    return transform


def create_colored_pointcloud(rgb, depth, intrinsic, pose):
    """
    从RGB和深度图创建彩色点云
    
    Args:
        rgb: RGB图像 (H, W, 3)
        depth: 深度图 (H, W)
        intrinsic: 相机内参矩阵 (3, 3)
        pose: 相机到世界坐标系的变换矩阵 (4, 4)
    
    Returns:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3) 范围 [0, 1]
    """
    height, width = depth.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 将像素坐标转换为相机坐标系下的3D点
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # 有效的深度点
    valid_mask = depth > 0
    
    # 计算3D点（相机坐标系）
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    
    # 组合为齐次坐标
    points_cam = np.stack([x, y, z], axis=1)
    points_cam_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    
    # 转换到世界坐标系
    points_world_homo = (pose @ points_cam_homo.T).T
    points = points_world_homo[:, :3]
    
    # 获取对应的颜色（BGR转RGB并归一化到[0,1]）
    colors_bgr = rgb[valid_mask]
    colors = colors_bgr[:, ::-1] / 255.0  # BGR -> RGB, 归一化
    
    return points, colors


def vis_picture(root='traj_demo/imgs', sampled_indices_file=None):
    """
    可视化点云（支持采样帧）
    
    Args:
        root: 数据根目录
        sampled_indices_file: 采样索引JSON文件路径（可选），如果提供则只可视化采样帧
    """
    data_info = np.load(os.path.join(root, 'info.npy'), allow_pickle=True).item()

    cam_pose_list = data_info['cam_pose_list']
    cam_quat_list = data_info['cam_quat_list']
    cam_intrinsic = data_info['cam_intrinsic']

    # 确定要处理的帧索引
    if sampled_indices_file and os.path.exists(sampled_indices_file):
        # 读取采样索引
        with open(sampled_indices_file, 'r', encoding='utf-8') as f:
            sampled_indices = json.load(f)
        print(f"Loading sampled frames from: {sampled_indices_file}")
        print(f"Total sampled frames: {len(sampled_indices)}")
        print(f"Sampled indices: {sampled_indices[:10]}{'...' if len(sampled_indices) > 10 else ''}")
        frame_indices = sampled_indices
        window_title = f"Sampled Point Cloud ({len(sampled_indices)} frames)"
    else:
        # 使用所有帧
        num = len(cam_pose_list)
        frame_indices = list(range(num))
        window_title = f"Full Point Cloud ({num} frames)"
        if sampled_indices_file:
            print(f"Warning: Sampled indices file not found: {sampled_indices_file}")
            print(f"Using all {num} frames instead.")

    points_list = []
    points_col_list = []
    
    print(f"\nProcessing {len(frame_indices)} frames...")
    for idx, i in enumerate(frame_indices):
        if idx % 10 == 0:
            print(f"  Processing frame {i} ({idx+1}/{len(frame_indices)})...")
        
        # 检查索引是否有效
        if i >= len(cam_pose_list):
            print(f"  Warning: Frame index {i} is out of range (max: {len(cam_pose_list)-1}), skipping.")
            continue
        
        pose44 = transform_7_to_4x4(np.concatenate([cam_pose_list[i], cam_quat_list[i]]))
        pose_cam_to_usd = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1],
        ], dtype=np.float32)

        pose = pose44 @ pose_cam_to_usd
        
        rgb_path = os.path.join(root, f'{i}_rgb.jpg')
        depth_path = os.path.join(root, f'{i}_depth.npz')
        
        if not os.path.exists(rgb_path):
            print(f"  Warning: RGB image not found: {rgb_path}, skipping frame {i}.")
            continue
        if not os.path.exists(depth_path):
            print(f"  Warning: Depth map not found: {depth_path}, skipping frame {i}.")
            continue
        
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"  Warning: Failed to load RGB image: {rgb_path}, skipping frame {i}.")
            continue
        
        depth = np.load(depth_path)['arr_0']
        points, points_col = create_colored_pointcloud(rgb, depth, cam_intrinsic, pose)
        points_list.append(points)
        points_col_list.append(points_col)

    if len(points_list) == 0:
        print("Error: No valid frames to visualize!")
        return

    print(f"\nMerging {len(points_list)} point clouds...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(points_list, axis=0).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(points_col_list, axis=0).astype(np.float64))
    
    print(f"Total points: {len(pcd.points)}")
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([pcd, coord_frame],
                                    window_name=window_title,
                                    width=800,
                                    height=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化点云（支持采样帧）')
    parser.add_argument('--data_root', type=str, default='traj_demo/imgs',
                       help='数据根目录（默认: traj_demo/imgs）')
    parser.add_argument('--sampled_indices', type=str, default=None,
                       help='采样索引JSON文件路径（可选），如果提供则只可视化采样帧（默认: None，使用所有帧）')
    
    args = parser.parse_args()
    
    vis_picture(root=args.data_root, sampled_indices_file=args.sampled_indices)


"""
def vis_picture():
    import open3d as o3d

    root = '/home/a/Downloads/4/imgs'
    data_info = np.load(os.path.join(root, 'info.npy'), allow_pickle=True).item()

    cam_pose_list = data_info['cam_pose_list']
    cam_quat_list = data_info['cam_quat_list']
    cam_intrinsic = data_info['cam_intrinsic']

    num = len(cam_pose_list)
    # num = 10

    points_list = []
    points_col_list = []
    for i in range(num):
        cam_pose_list[i]
        pose44 = transform_7_to_4x4(np.concatenate([cam_pose_list[i], cam_quat_list[i]]))
        pose_cam_to_usd = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1],
        ], dtype=np.float32)

        # pose = np.linalg.inv(pose44) @ pose_cam_to_usd
        pose = pose44 @ pose_cam_to_usd
        # pose = pose_cam_to_usd
        rgb = cv2.imread(os.path.join(root, f'{i}_rgb.jpg'))
        depth = np.load(os.path.join(root, f'{i}_depth.npz'))['arr_0']
        points, points_col = create_colored_pointcloud(rgb, depth, cam_intrinsic, pose)
        points_list.append(points)
        points_col_list.append(points_col)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(points_list, axis=0).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(points_col_list, axis=0).astype(np.float64))
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([pcd, coord_frame],
                                    window_name="Colored Point Cloud",
                                    width=800,
                                    height=600)
"""
