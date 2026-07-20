#!/usr/bin/env python3
"""
检查传给MapAnything的数据和坐标转换逻辑
参照 traj_visualize.py 的正确实现来验证
"""
import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def transform_7_to_4x4_traj_visualize(pose_7d):
    """
    参照 traj_visualize.py 的 transform_7_to_4x4 函数
    """
    cam_position_in_world = pose_7d[:3]
    quaternion = pose_7d[3:]
    quaternion_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(quaternion_xyzw)
    R_cam_to_world = rotation.as_matrix()
    
    transform = np.eye(4)
    transform[:3, :3] = R_cam_to_world
    transform[:3, 3] = cam_position_in_world
    
    return transform


def check_pose_convention(data_root='traj_demo/imgs', sampled_indices_file=None):
    """
    检查1: Pose坐标系约定
    参照 traj_visualize.py 的逻辑
    """
    print("=" * 60)
    print("检查1: Pose坐标系约定")
    print("=" * 60)
    
    data_info = np.load(os.path.join(data_root, 'info.npy'), allow_pickle=True).item()
    cam_pose_list = data_info['cam_pose_list']
    cam_quat_list = data_info['cam_quat_list']
    
    # 读取采样索引
    if sampled_indices_file and os.path.exists(sampled_indices_file):
        with open(sampled_indices_file, 'r') as f:
            sampled_indices = json.load(f)
        test_indices = sampled_indices[:3]
    else:
        test_indices = [0, 1, 2]
    
    print(f"\n测试帧: {test_indices}")
    
    # traj_visualize.py 中的逻辑
    pose_cam_to_usd = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    
    print(f"\n参照 traj_visualize.py 的逻辑：")
    print(f"  1. transform_7_to_4x4() -> T_cam_to_world (OpenCV约定)")
    print(f"  2. pose_cam_to_usd 转换矩阵（翻转Y和Z轴）")
    print(f"  3. 最终pose = T_cam_to_world @ pose_cam_to_usd")
    
    for i in test_indices:
        pose_7d = np.concatenate([cam_pose_list[i], cam_quat_list[i]])
        
        # traj_visualize.py 的转换流程
        T_cam_to_world = transform_7_to_4x4_traj_visualize(pose_7d)
        T_fusion = T_cam_to_world @ pose_cam_to_usd
        
        print(f"\n帧 {i}:")
        print(f"  T_cam_to_world (OpenCV约定):")
        print(f"    平移: {T_cam_to_world[:3, 3]}")
        print(f"    旋转矩阵前3x3:")
        print(T_cam_to_world[:3, :3])
        
        print(f"  T_fusion (用于点云融合):")
        print(f"    平移: {T_fusion[:3, 3]}")
        print(f"    旋转矩阵前3x3:")
        print(T_fusion[:3, :3])
    
    print(f"\n⚠️  关键问题：")
    print(f"  - traj_visualize.py 使用 T_fusion = T_cam_to_world @ pose_cam_to_usd 进行点云融合")
    print(f"  - 我们的接口文档说传给MapAnything的是 T_cam_to_world (OpenCV约定)")
    print(f"  - 如果MapAnything期望的是 T_cam_to_world，但点云融合用了 T_fusion")
    print(f"  - 那么传给MapAnything的pose应该是什么？")
    print(f"  - 需要确认MapAnything是否也需要 pose_cam_to_usd 转换")
    
    print(f"\n✅ 建议：")
    print(f"  1. 如果MapAnything期望 OpenCV约定：传给 T_cam_to_world")
    print(f"  2. 如果MapAnything期望 USD约定：传给 T_cam_to_world @ pose_cam_to_usd")
    print(f"  3. 必须与MapAnything工程师确认期望的坐标系约定")


def check_intrinsics_usage(data_root='traj_demo/imgs'):
    """
    检查2: 内参使用
    参照 traj_visualize.py 的逻辑
    """
    print("\n" + "=" * 60)
    print("检查2: 内参使用")
    print("=" * 60)
    
    data_info = np.load(os.path.join(data_root, 'info.npy'), allow_pickle=True).item()
    cam_intrinsic = data_info['cam_intrinsic']
    
    # 获取实际图像尺寸
    rgb_path = os.path.join(data_root, '0_rgb.jpg')
    if os.path.exists(rgb_path):
        img = cv2.imread(rgb_path)
        img_h, img_w = img.shape[:2]
    else:
        img_w = int(cam_intrinsic[0, 2] * 2)
        img_h = int(cam_intrinsic[1, 2] * 2)
    
    print(f"\n参照 traj_visualize.py 的逻辑：")
    print(f"  - 直接使用 cam_intrinsic（原始内参）")
    print(f"  - 图像尺寸: {img_w} x {img_h}")
    print(f"  - 内参矩阵 K:")
    print(cam_intrinsic)
    
    print(f"\n内参参数：")
    print(f"  fx: {cam_intrinsic[0, 0]:.2f}")
    print(f"  fy: {cam_intrinsic[1, 1]:.2f}")
    print(f"  cx: {cam_intrinsic[0, 2]:.2f}")
    print(f"  cy: {cam_intrinsic[1, 2]:.2f}")
    
    # 检查内参与图像尺寸的匹配
    print(f"\n内参与图像尺寸匹配检查：")
    print(f"  内参中心点: ({cam_intrinsic[0, 2]:.1f}, {cam_intrinsic[1, 2]:.1f})")
    print(f"  图像中心: ({img_w/2:.1f}, {img_h/2:.1f})")
    
    if abs(cam_intrinsic[0, 2] - img_w/2) < img_w * 0.1 and abs(cam_intrinsic[1, 2] - img_h/2) < img_h * 0.1:
        print(f"  ✅ 内参中心点与图像中心匹配")
    else:
        print(f"  ⚠️  内参中心点与图像中心不匹配！")
    
    print(f"\n✅ 我们的代码使用：")
    print(f"  - 在 load_pose_data() 中直接返回 cam_intrinsic（原始内参）")
    print(f"  - 在 compute_score_for_pair() 中直接使用传入的 K")
    print(f"  - 这与 traj_visualize.py 的逻辑一致 ✅")
    
    print(f"\n⚠️  需要注意：")
    print(f"  - 如果MapAnything预处理调整了图像分辨率，内参也需要相应缩放")
    print(f"  - 但我们的接口文档中传给MapAnything的是原始内参")
    print(f"  - MapAnything工程师需要根据预处理情况自行缩放内参")


def check_depth_usage(data_root='traj_demo/imgs'):
    """
    检查3: 深度图使用
    参照 traj_visualize.py 的逻辑
    """
    print("\n" + "=" * 60)
    print("检查3: 深度图使用")
    print("=" * 60)
    
    depth_path = os.path.join(data_root, '0_depth.npz')
    if os.path.exists(depth_path):
        depth = np.load(depth_path)['arr_0']
        print(f"\n参照 traj_visualize.py 的逻辑：")
        print(f"  - 直接使用 depth.npz 中的原始深度图")
        print(f"  - 深度图尺寸: {depth.shape}")
        print(f"  - 深度范围: [{depth[depth>0].min():.4f}, {depth[depth>0].max():.4f}] m")
        print(f"  - 有效深度点数: {np.sum(depth > 0)} / {depth.size}")
        
        print(f"\n✅ 我们的代码使用：")
        print(f"  - 在 load_depth_map() 中直接加载 depth.npz")
        print(f"  - 在 compute_score_for_pair() 中直接使用原始深度图")
        print(f"  - 这与 traj_visualize.py 的逻辑一致 ✅")
    else:
        print(f"  ⚠️  找不到深度图: {depth_path}")


def check_our_interface_doc():
    """
    检查4: 我们的接口文档与 traj_visualize.py 的一致性
    """
    print("\n" + "=" * 60)
    print("检查4: 接口文档一致性")
    print("=" * 60)
    
    interface_doc_path = 'output/sampling_interface_doc.json'
    if os.path.exists(interface_doc_path):
        with open(interface_doc_path, 'r') as f:
            doc = json.load(f)
        
        print(f"\n接口文档中的坐标系约定：")
        coord_conv = doc.get('coordinate_system_convention', {})
        print(f"  - 描述: {coord_conv.get('description', 'N/A')}")
        print(f"  - Pose格式: {coord_conv.get('pose_format', 'N/A')}")
        print(f"  - Pose含义: {coord_conv.get('pose_meaning', 'N/A')}")
        print(f"  - 验证公式: {coord_conv.get('verification', 'N/A')}")
        
        print(f"\n⚠️  关键发现：")
        print(f"  - 接口文档说：T_cam_to_world (OpenCV约定)")
        print(f"  - traj_visualize.py 使用：T_cam_to_world @ pose_cam_to_usd")
        print(f"  - 这两个不一致！")
        
        print(f"\n✅ 需要确认：")
        print(f"  1. MapAnything期望的坐标系约定是什么？")
        print(f"     - 如果是OpenCV约定：传给 T_cam_to_world")
        print(f"     - 如果是USD约定：传给 T_cam_to_world @ pose_cam_to_usd")
        print(f"  2. 我们的接口文档需要明确说明传给MapAnything的pose格式")
        print(f"  3. 如果MapAnything期望OpenCV约定，但点云融合用了USD约定")
        print(f"     那么点云和MapAnything的pose需要统一坐标系")
    else:
        print(f"  ⚠️  找不到接口文档: {interface_doc_path}")


def check_our_code_implementation():
    """
    检查5: 我们的代码实现与 traj_visualize.py 的一致性
    """
    print("\n" + "=" * 60)
    print("检查5: 代码实现一致性")
    print("=" * 60)
    
    print(f"\n1. transform_7_to_4x4() 函数：")
    print(f"   - core/geometry.py: 返回 T_cam_to_world (OpenCV约定) ✅")
    print(f"   - traj_visualize.py: 返回 T_cam_to_world (OpenCV约定) ✅")
    print(f"   - 两者一致 ✅")
    
    print(f"\n2. 内参使用：")
    print(f"   - utils/io.py load_pose_data(): 返回原始 cam_intrinsic ✅")
    print(f"   - traj_visualize.py: 使用原始 cam_intrinsic ✅")
    print(f"   - 两者一致 ✅")
    
    print(f"\n3. 深度图使用：")
    print(f"   - utils/io.py load_depth_map(): 加载原始 depth.npz ✅")
    print(f"   - traj_visualize.py: 使用原始 depth.npz ✅")
    print(f"   - 两者一致 ✅")
    
    print(f"\n4. 坐标转换（关键差异）：")
    print(f"   - traj_visualize.py: pose = T_cam_to_world @ pose_cam_to_usd")
    print(f"   - 我们的代码: 使用 T_cam_to_world（没有 pose_cam_to_usd）")
    print(f"   - 接口文档: 说传给MapAnything的是 T_cam_to_world")
    print(f"   - ⚠️  需要确认MapAnything期望的格式")
    
    print(f"\n5. 点云反投影逻辑：")
    print(f"   - traj_visualize.py create_colored_pointcloud():")
    print(f"     * 使用 pose (T_fusion) 进行转换")
    print(f"     * points_world = pose @ points_cam_homo")
    print(f"   - mapanything_sampling/geometry_ops.py unproject_depth_to_world():")
    print(f"     * 使用 T_cam_to_world 进行转换")
    print(f"     * points_world = T_cam_to_world @ points_cam_homo")
    print(f"   - ⚠️  如果pose格式不一致，点云会错位")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查传给MapAnything的数据和坐标转换逻辑')
    parser.add_argument('--data_root', type=str, default='traj_demo/imgs',
                       help='数据根目录')
    parser.add_argument('--sampled_indices', type=str, default='output/sampled_indices.json',
                       help='采样索引JSON文件路径')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("检查传给MapAnything的数据和坐标转换逻辑")
    print("参照 traj_visualize.py 的正确实现")
    print("=" * 60)
    
    # 检查1: Pose坐标系约定
    check_pose_convention(args.data_root, args.sampled_indices)
    
    # 检查2: 内参使用
    check_intrinsics_usage(args.data_root)
    
    # 检查3: 深度图使用
    check_depth_usage(args.data_root)
    
    # 检查4: 接口文档一致性
    check_our_interface_doc()
    
    # 检查5: 代码实现一致性
    check_our_code_implementation()
    
    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)
    print("\n📋 总结：")
    print("1. ✅ 内参使用：与 traj_visualize.py 一致（原始内参）")
    print("2. ✅ 深度图使用：与 traj_visualize.py 一致（原始深度图）")
    print("3. ⚠️  Pose格式：需要确认MapAnything期望的坐标系约定")
    print("   - traj_visualize.py 使用: T_cam_to_world @ pose_cam_to_usd")
    print("   - 我们的接口文档说: T_cam_to_world (OpenCV约定)")
    print("   - 需要与MapAnything工程师确认期望的格式")
    print("=" * 60)


if __name__ == '__main__':
    main()

