#!/usr/bin/env python3
"""
分段关键帧采样算法（基于MapAnything参数）
整合硬性节点约束和双重几何条件（共视率 + SE(3)距离）

使用方法：
    python3 test_segmented_sampling.py --config configs/exp_segmented_sampling.yaml
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.io import load_pose_data, load_all_depth_maps
from utils.visualizer import visualize_trajectory
from mapanything_sampling import build_overlap_matrix, get_segmented_keyframes, build_action_matrix

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def visualize_statistics_panels(M_overlap, M_action, keyframes, num_frames, save_path=None):
    """
    可视化中间三个统计图：共视率分布、SE(3)距离分布、采样路径指标
    
    Args:
        M_overlap: 共视率矩阵
        M_action: SE(3)距离矩阵
        keyframes: 采样的关键帧索引
        num_frames: 总帧数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1：共视率分布
    ax1 = axes[0]
    mask = ~np.eye(num_frames, dtype=bool)
    off_diagonal = M_overlap[mask]
    ax1.hist(off_diagonal, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Covisibility Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Covisibility Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(off_diagonal.mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {off_diagonal.mean():.4f}')
    ax1.legend()
    
    # 子图2：SE(3)距离分布
    ax2 = axes[1]
    mask = ~np.eye(num_frames, dtype=bool)
    off_diagonal_action = M_action[mask]
    ax2.hist(off_diagonal_action, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('SE(3) Distance', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('SE(3) Distance Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(off_diagonal_action.mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {off_diagonal_action.mean():.4f}')
    ax2.legend()
    
    # 子图3：采样路径指标
    ax3 = axes[2]
    if len(keyframes) > 1:
        path_overlaps = []
        path_actions = []
        for i in range(len(keyframes) - 1):
            idx1, idx2 = keyframes[i], keyframes[i + 1]
            path_overlaps.append(M_overlap[idx1, idx2])
            path_actions.append(M_action[idx1, idx2])
        
        x = range(len(path_overlaps))
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(x, path_overlaps, 'o-', color='red', label='Covisibility', linewidth=2)
        line2 = ax3_twin.plot(x, path_actions, 's-', color='blue', label='SE(3) Distance', linewidth=2)
        
        ax3.set_xlabel('Path Step', fontsize=12)
        ax3.set_ylabel('Covisibility Score', fontsize=12, color='red')
        ax3_twin.set_ylabel('SE(3) Distance', fontsize=12, color='blue')
        ax3.set_title('Sampling Path Metrics', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'Not enough keyframes\nfor path metrics', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics panels saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_matrices_and_sampling(M_overlap, M_action, keyframes, nodes, 
                                    num_frames, poses=None, save_path=None):
    """
    可视化共视率矩阵、SE(3)距离矩阵和采样结果
    
    Args:
        M_overlap: 共视率矩阵
        M_action: SE(3)距离矩阵
        keyframes: 采样的关键帧索引
        nodes: 硬性节点
        num_frames: 总帧数
        poses: 位姿列表（用于3D轨迹可视化，如果为None则不显示）
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(20, 10))
    # 如果提供了poses，用3D轨迹图替换时间轴
    if poses is not None:
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    else:
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.3)
    
    # 子图1：共视率矩阵
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(M_overlap, cmap='hot', vmin=0.0, vmax=1.0, aspect='auto', origin='lower')
    ax1.set_title('Covisibility Matrix (M_overlap)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Target Frame Index', fontsize=10)
    ax1.set_ylabel('Reference Frame Index', fontsize=10)
    plt.colorbar(im1, ax=ax1, label='Covisibility Score', fraction=0.046, pad=0.04)
    
    # 在共视率矩阵上标记采样的关键帧
    for kf in keyframes:
        ax1.axhline(y=kf, color='cyan', linewidth=1, alpha=0.5)
        ax1.axvline(x=kf, color='cyan', linewidth=1, alpha=0.5)
    
    # 子图2：SE(3)距离矩阵
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(M_action, cmap='viridis', aspect='auto', origin='lower')
    ax2.set_title('SE(3) Action Matrix (M_action)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Target Frame Index', fontsize=10)
    ax2.set_ylabel('Reference Frame Index', fontsize=10)
    plt.colorbar(im2, ax=ax2, label='SE(3) Distance', fraction=0.046, pad=0.04)
    
    # 在SE(3)距离矩阵上标记采样的关键帧
    for kf in keyframes:
        ax2.axhline(y=kf, color='cyan', linewidth=1, alpha=0.5)
        ax2.axvline(x=kf, color='cyan', linewidth=1, alpha=0.5)
    
    # 子图3：双重约束掩码矩阵
    ax3 = fig.add_subplot(gs[0, 2])
    # 生成双重约束掩码（示例：overlap > 0.25 AND action >= 0.05）
    overlap_mask = M_overlap > 0.25
    action_mask = M_action >= 0.05
    M_filter = (overlap_mask & action_mask).astype(float)
    im3 = ax3.imshow(M_filter, cmap='gray', vmin=0.0, vmax=1.0, aspect='auto', origin='lower')
    ax3.set_title('Dual Constraint Filter Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Target Frame Index', fontsize=10)
    ax3.set_ylabel('Reference Frame Index', fontsize=10)
    plt.colorbar(im3, ax=ax3, label='Valid (1) / Invalid (0)', fraction=0.046, pad=0.04)
    
    # 子图4-6：中间三个统计图（已移到单独的文件）
    # 这里不再显示，由 visualize_statistics_panels 单独生成
    
    # 子图7：3D轨迹可视化或时间轴可视化（占据底部一行）
    if poses is not None:
        # 使用3D轨迹图替换时间轴
        ax7 = fig.add_subplot(gs[1, :], projection='3d')
        
        # 提取所有相机位置
        from utils.visualizer import _to_numpy
        positions_list = []
        for pose in poses:
            pos = pose[:3, 3]
            positions_list.append(_to_numpy(pos))
        positions = np.array(positions_list)
        
        # 绘制完整轨迹
        ax7.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', alpha=0.3, linewidth=1, label='Full trajectory')
        
        # 绘制所有帧的位置点
        ax7.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=10, alpha=0.5, label='All frames')
        
        # 绘制选中的帧
        selected_positions = positions[keyframes]
        ax7.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2],
                   c='red', s=50, marker='o', label='Selected frames')
        
        # 绘制选中帧之间的连线
        if len(keyframes) > 1:
            selected_positions_ordered = positions[keyframes]
            ax7.plot(selected_positions_ordered[:, 0], 
                   selected_positions_ordered[:, 1], 
                   selected_positions_ordered[:, 2],
                   'r-', linewidth=2, alpha=0.7, label='Sampled path')
        
        # 标记硬性节点（用不同颜色）
        if nodes:
            node_positions = positions[nodes]
            ax7.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                       c='purple', s=100, marker='s', 
                       label=f'Hard nodes ({len(nodes)})', edgecolors='black', linewidths=2)
        
        ax7.set_xlabel('X (m)', fontsize=12)
        ax7.set_ylabel('Y (m)', fontsize=12)
        ax7.set_zlabel('Z (m)', fontsize=12)
        ax7.set_title(f'3D Trajectory with Sampled Keyframes ({len(keyframes)} frames from {num_frames} total)', 
                     fontsize=14, fontweight='bold')
        ax7.legend(loc='upper right', fontsize=10)
        ax7.grid(True)
    else:
        # 如果没有提供poses，使用时间轴可视化（向后兼容）
        ax7 = fig.add_subplot(gs[2, :])
        
        # 绘制时间轴
        ax7.plot(range(num_frames), [0] * num_frames, 'k-', linewidth=0.5, alpha=0.3)
        
        # 标记所有帧
        ax7.scatter(range(num_frames), [0] * num_frames, s=5, c='gray', alpha=0.2, label='All frames')
        
        # 标记硬性节点
        ax7.scatter(nodes, [0] * len(nodes), s=300, c='blue', marker='s', 
                   zorder=6, label=f'Hard nodes ({len(nodes)}): {nodes}', edgecolors='black', linewidths=2)
        
        # 标记采样的关键帧
        ax7.scatter(keyframes, [0] * len(keyframes), s=150, c='red', marker='o', 
                   zorder=5, label=f'Sampled keyframes ({len(keyframes)})', edgecolors='black', linewidths=1)
        
        # 标记硬性节点（用不同颜色）
        for node in nodes:
            if node in keyframes:
                ax7.scatter([node], [0], s=300, c='purple', marker='s', 
                           zorder=7, edgecolors='black', linewidths=2)
        
        ax7.set_xlabel('Frame Index', fontsize=12)
        ax7.set_title(f'Segmented Keyframe Sampling Timeline ({len(keyframes)} frames from {num_frames} total)', 
                     fontsize=14, fontweight='bold')
        ax7.legend(loc='upper right', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(-2, num_frames + 2)
        ax7.set_ylim(-0.1, 0.1)
        ax7.set_yticks([])
    
    plt.suptitle('Segmented Keyframe Sampling Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='分段关键帧采样算法（基于MapAnything参数）')
    parser.add_argument('--config', type=str, default='configs/exp_segmented_sampling.yaml',
                       help='配置文件路径（YAML格式）')
    parser.add_argument('--data_root', type=str, default=None,
                       help='数据根目录（覆盖配置文件中的设置）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径（覆盖配置文件中的设置）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 从配置文件读取参数（命令行参数优先）
    data_root = args.data_root if args.data_root is not None else config.get('data_root', 'traj_demo/imgs')
    output_dir = config.get('output_dir', 'output')
    use_gpu = config.get('use_gpu', False)
    device_str = config.get('device', None)
    
    # 共视率计算参数（MapAnything官方参数）
    covis_config = config.get('covisibility', {})
    depth_assoc_error_thres = covis_config.get('depth_assoc_error_thres', 0.1)
    depth_assoc_rel_error_thres = covis_config.get('depth_assoc_rel_error_thres', 0.005)
    depth_assoc_error_temp = covis_config.get('depth_assoc_error_temp', 0.1)
    denominator_mode = covis_config.get('denominator_mode', 'valid_target_depth')
    min_depth = covis_config.get('min_depth', 0.04)
    load_overlap_matrix = covis_config.get('load_overlap_matrix', None)
    save_overlap_matrix = covis_config.get('save_overlap_matrix', None)
    
    # SE(3)距离参数
    se3_config = config.get('se3_action', {})
    w_trans = se3_config.get('w_trans', 1.0)
    w_rot = se3_config.get('w_rot', 1.0)
    
    # 采样参数
    sampling_config = config.get('sampling', {})
    nodes = sampling_config.get('nodes', None)
    overlap_threshold = sampling_config.get('overlap_threshold', 0.25)
    D_target = sampling_config.get('D_target', 0.1)
    max_path_length = sampling_config.get('max_path_length', 16)
    max_jump_distance = sampling_config.get('max_jump_distance', 10)
    
    # 可视化参数
    vis_config = config.get('visualization', {})
    output_matrix = args.output if args.output is not None else vis_config.get('output_matrix', 'output/segmented_sampling.png')
    output_statistics = vis_config.get('output_statistics', output_matrix.replace('.png', '_statistics_panels.png'))
    
    # 打印配置信息
    print("=" * 60)
    print("分段关键帧采样配置（基于MapAnything参数）")
    print("=" * 60)
    print(f"\n配置文件: {args.config}")
    print(f"数据路径: {data_root}")
    print(f"\n共视率计算参数（MapAnything GT）:")
    print(f"  - C_abs (depth_assoc_error_thres): {depth_assoc_error_thres}")
    print(f"  - C_rel (depth_assoc_rel_error_thres): {depth_assoc_rel_error_thres}")
    print(f"  - C_temp (depth_assoc_error_temp): {depth_assoc_error_temp}")
    print(f"  - 动态阈值公式: threshold = {depth_assoc_error_thres} + {depth_assoc_rel_error_thres} * D_expected - log(0.5) * {depth_assoc_error_temp}")
    print(f"  - 在5.0m处的总阈值: {depth_assoc_error_thres + depth_assoc_rel_error_thres * 5.0 - math.log(0.5) * depth_assoc_error_temp:.4f}m")
    print(f"\nSE(3)距离参数:")
    print(f"  - w_trans: {w_trans}")
    print(f"  - w_rot: {w_rot}")
    print(f"\n采样参数:")
    print(f"  - overlap_threshold: {overlap_threshold}")
    print(f"  - D_target: {D_target}")
    print(f"  - max_path_length: {max_path_length}")
    print(f"  - max_jump_distance: {max_jump_distance}")
    print(f"  - nodes: {nodes if nodes is not None else 'auto-generate'}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_matrix), exist_ok=True)
    
    # 确定设备
    if device_str:
        device = torch.device(device_str) if TORCH_AVAILABLE else None
    elif use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device('cuda')
    elif TORCH_AVAILABLE:
        device = torch.device('cpu')
    else:
        device = None
    
    if device:
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using NumPy (PyTorch not available)")
    
    # 加载位姿数据
    print(f"\nLoading pose data from: {data_root}")
    poses, intrinsics, img_h, img_w, num_frames = load_pose_data(
        data_root, device=device, use_gpu=(device is not None and device.type == 'cuda')
    )
    print(f"Loaded {num_frames} frames")
    print(f"Image size: {img_w}x{img_h}")
    
    # 加载或计算共视率矩阵
    if load_overlap_matrix and os.path.exists(load_overlap_matrix):
        print(f"\nLoading overlap matrix from: {load_overlap_matrix}")
        M_overlap = np.load(load_overlap_matrix)
        print(f"Loaded matrix shape: {M_overlap.shape}")
    else:
        # 加载所有深度图
        print(f"\nLoading depth maps...")
        depth_maps = load_all_depth_maps(data_root, num_frames, device=device)
        print(f"Loaded {len(depth_maps)} depth maps")
        
        # 计算共视率矩阵（使用MapAnything官方参数）
        print(f"\nComputing overlap matrix with MapAnything parameters...")
        M_overlap = build_overlap_matrix(
            poses_all=poses,
            depths_all=depth_maps,
            K_all=intrinsics,
            img_h=img_h,
            img_w=img_w,
            depth_assoc_error_thres=depth_assoc_error_thres,
            depth_assoc_rel_error_thres=depth_assoc_rel_error_thres,
            depth_assoc_error_temp=depth_assoc_error_temp,
            denominator_mode=denominator_mode,
            min_depth=min_depth,
            device=device,
            precompute_world_pts=True
        )
        
        if save_overlap_matrix:
            np.save(save_overlap_matrix, M_overlap)
            print(f"Saved overlap matrix to: {save_overlap_matrix}")
    
    # 计算SE(3)距离矩阵
    print(f"\nComputing SE(3) action matrix...")
    M_action = build_action_matrix(
        T_all=poses,
        w_trans=w_trans,
        w_rot=w_rot,
        device=device
    )
    print(f"Action matrix shape: {M_action.shape}")
    print(f"Action matrix stats: min={M_action.min():.4f}, max={M_action.max():.4f}, mean={M_action.mean():.4f}")
    
    # 确定硬性节点（如果提供，则使用；否则自动生成）
    # 注意：硬性节点必须是50的倍数，避免因噪音导致的帧
    # 自动生成会根据实际帧数生成所有50的倍数节点
    if nodes is not None:
        nodes = sorted(set(nodes))
        print(f"\nUsing provided hard nodes: {nodes}")
        print(f"  (Will be corrected to multiples of 50 if needed)")
    else:
        nodes = None  # 让get_segmented_keyframes自动生成
        print(f"\nAuto-generating hard nodes based on {num_frames} frames...")
        print(f"  (Will generate all multiples of 50: 0, 50, 100, 150, ...)")
    
    # 执行分段采样
    print(f"\nPerforming segmented keyframe sampling...")
    print(f"  - Overlap threshold: {overlap_threshold}")
    print(f"  - Action threshold (D_target): {D_target}")
    
    keyframes = get_segmented_keyframes(
        M_overlap=M_overlap,
        M_action=M_action,
        nodes=nodes,
        D_target=D_target,
        overlap_threshold=overlap_threshold,
        max_path_length=max_path_length,
        max_jump_distance=max_jump_distance,
        device=device
    )
    
    # 获取实际使用的硬性节点（重新计算以匹配函数内部逻辑）
    if nodes is None:
        # 自动生成：必须是50的倍数
        nodes = [0]
        step = 50
        current = step
        while current < num_frames:
            nodes.append(current)
            current += step
        last_node = (num_frames - 1) // step * step
        if num_frames % step == 0:
            if last_node - 1 >= 0 and last_node - 1 not in nodes:
                nodes.append(last_node - 1)
        else:
            if last_node not in nodes and last_node >= 0:
                nodes.append(last_node)
        if nodes and nodes[-1] >= num_frames:
            nodes[-1] = num_frames - 1
    else:
        # 修正提供的nodes
        nodes = sorted(set(nodes))
        corrected_nodes = [nodes[0]]
        for node in nodes[1:-1] if len(nodes) > 1 else []:
            corrected = (node // 50) * 50
            if corrected not in corrected_nodes:
                corrected_nodes.append(corrected)
        if len(nodes) > 1:
            last_node = nodes[-1]
            corrected_last = (last_node // 50) * 50
            if corrected_last not in corrected_nodes:
                corrected_nodes.append(corrected_last)
        nodes = corrected_nodes
        last_ideal = (num_frames - 1) // 50 * 50
        if last_ideal not in nodes:
            nodes.append(last_ideal)
        if nodes[-1] >= num_frames:
            if last_ideal < num_frames:
                nodes[-1] = last_ideal
            else:
                nodes[-1] = num_frames - 1
    
    print(f"\nHard nodes used: {nodes}")
    print(f"\nSampled {len(keyframes)} keyframes:")
    print(f"  Indices: {keyframes}")
    
    # 可视化
    print(f"\nVisualizing results...")
    
    # 1. 生成矩阵分析图（包含3D轨迹图作为底部子图）
    print(f"  - Generating matrix analysis with 3D trajectory: {output_matrix}")
    visualize_matrices_and_sampling(
        M_overlap=M_overlap,
        M_action=M_action,
        keyframes=keyframes,
        nodes=nodes,
        num_frames=num_frames,
        poses=poses,  # 传入poses以显示3D轨迹
        save_path=output_matrix
    )
    
    # 2. 生成中间三个统计图（单独文件）
    print(f"  - Generating statistics panels: {output_statistics}")
    visualize_statistics_panels(
        M_overlap=M_overlap,
        M_action=M_action,
        keyframes=keyframes,
        num_frames=num_frames,
        save_path=output_statistics
    )
    
    # 3. 保存采样结果索引文件（供MapAnything工程师使用）
    import json
    output_indices = vis_config.get('output_indices', os.path.join(output_dir, 'sampled_indices.json'))
    print(f"\nSaving sampled indices to: {output_indices}")
    
    # 转换为列表（如果是numpy数组或torch tensor）
    keyframes_list = [int(idx) for idx in keyframes]
    
    # 保存为JSON格式
    with open(output_indices, 'w', encoding='utf-8') as f:
        json.dump(keyframes_list, f, indent=2, ensure_ascii=False)
    
    print(f"  - Saved {len(keyframes_list)} sampled frame indices")
    print(f"  - Frame indices: {keyframes_list[:10]}{'...' if len(keyframes_list) > 10 else ''}")
    
    # 4. 保存接口文档（说明坐标系约定和文件路径约定）
    output_interface_doc = os.path.join(output_dir, 'sampling_interface_doc.json')
    interface_doc = {
        "sampling_engineer_output": {
            "sampled_indices_file": os.path.basename(output_indices),
            "sampled_indices_path": output_indices,
            "total_sampled_frames": len(keyframes_list),
            "total_original_frames": num_frames,
            "sampling_rate": f"{len(keyframes_list)/num_frames*100:.2f}%"
        },
        "coordinate_system_convention": {
            "description": "OpenCV convention: T_cam_to_world (Camera to World)",
            "pose_format": "4x4 transformation matrix",
            "pose_meaning": "Transforms points from camera coordinate system to world coordinate system",
            "verification": "P_world = T_cam_to_world @ P_cam (where P_cam is in camera coordinates)",
            "important_note": "The pose matrices provided are T_cam_to_world (OpenCV convention). If MapAnything expects a different convention (e.g., USD convention with pose_cam_to_usd transformation), the MapAnything engineer should apply the necessary coordinate transformation. The point cloud visualization script (traj_visualize.py) uses T_cam_to_world @ pose_cam_to_usd for visualization, but the raw data provided follows OpenCV convention."
        },
        "file_path_convention": {
            "data_root": data_root,
            "rgb_image_pattern": "{data_root}/{frame_index}_rgb.jpg",
            "depth_map_pattern": "{data_root}/{frame_index}_depth.npz",
            "pose_info_file": "{data_root}/info.npy",
            "example": {
                "frame_index": 0,
                "rgb_path": f"{data_root}/0_rgb.jpg",
                "depth_path": f"{data_root}/0_depth.npz",
                "pose_info": f"{data_root}/info.npy"
            }
        },
        "data_format": {
            "pose_matrix_shape": [4, 4],
            "pose_matrix_dtype": "float32",
            "intrinsics_shape": [3, 3],
            "intrinsics_dtype": "float32",
            "image_size": {
                "width": int(img_w),
                "height": int(img_h)
            }
        },
        "notes": [
            "All frame indices in sampled_indices.json are zero-indexed",
            "Pose matrices are stored in info.npy as cam_pose_list (position) and cam_quat_list (quaternion)",
            "To convert 7D pose to 4x4 matrix: use transform_7_to_4x4([x, y, z, qw, qx, qy, qz])",
            "The resulting 4x4 matrix is T_cam_to_world (OpenCV convention)",
            "IMPORTANT: The intrinsics and depth maps are provided in their original resolution (320x240).",
            "If MapAnything preprocesses images/depth to a different resolution, the intrinsics must be scaled accordingly.",
            "The intrinsics matrix K is: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] where fx=fy=429.47, cx=160, cy=120"
        ]
    }
    
    with open(output_interface_doc, 'w', encoding='utf-8') as f:
        json.dump(interface_doc, f, indent=2, ensure_ascii=False)
    
    print(f"  - Saved interface documentation to: {output_interface_doc}")
    
    print("\nDone! Segmented sampling test completed successfully.")
    print(f"\nGenerated outputs:")
    print(f"  - Sampled indices (for MapAnything): {output_indices}")
    print(f"  - Interface documentation: {output_interface_doc}")
    print(f"  - Matrix Analysis (with 3D trajectory): {output_matrix}")
    print(f"  - Statistics Panels: {output_statistics}")


if __name__ == '__main__':
    main()
