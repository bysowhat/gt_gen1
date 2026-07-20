#!/usr/bin/env python3
"""
共视率检测测试脚本

使用模块化共视率检测代码计算并可视化帧之间的共视率矩阵。
基于深度重投影验证方法（MapAnything方法）。

核心模块：
- core/geometry_ops.py: GPU几何运算模块
- core/covisibility_gpu.py: 共视性校验核心
- core/sampling_pipeline.py: 并行矩阵构建
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.io import load_pose_data, load_all_depth_maps
from mapanything_sampling import build_overlap_matrix, compute_score_for_pair

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def visualize_covisibility_matrix(covisibility_matrix, save_path=None, highlight_threshold=0.25):
    """
    可视化共视率矩阵
    将共视率超过threshold的部分高亮显示，低共视率区域降低分辨率
    """
    num_frames = covisibility_matrix.shape[0]
    
    # 创建高亮矩阵：共视率超过阈值的部分保持原值，其他部分降低分辨率
    highlight_mask = covisibility_matrix > highlight_threshold
    
    # 对低共视率区域进行下采样（降低分辨率）
    # 创建一个混合矩阵：高共视率区域保持高分辨率，低共视率区域降低分辨率
    display_matrix = covisibility_matrix.copy()
    
    # 对低共视率区域进行平滑/下采样处理
    if num_frames > 50:
        # 对于大矩阵，对低共视率区域进行块平均（降低分辨率）
        block_size = max(2, num_frames // 50)  # 自适应块大小
        low_covis_mask = ~highlight_mask
        
        # 创建低分辨率版本（只对低共视率区域）
        low_res_matrix = covisibility_matrix.copy()
        for i in range(0, num_frames, block_size):
            for j in range(0, num_frames, block_size):
                i_end = min(i + block_size, num_frames)
                j_end = min(j + block_size, num_frames)
                block = covisibility_matrix[i:i_end, j:j_end]
                block_mask = low_covis_mask[i:i_end, j:j_end]
                # 只对低共视率区域进行平均
                if np.any(block_mask):
                    avg_value = np.mean(block[block_mask]) if np.any(block_mask) else 0
                    low_res_matrix[i:i_end, j:j_end] = np.where(
                        block_mask, avg_value, block
                    )
        
        # 混合：高共视率区域用原值，低共视率区域用低分辨率值
        display_matrix = np.where(highlight_mask, covisibility_matrix, low_res_matrix)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # 主图：共视率矩阵（高亮显示）
    ax_main = fig.add_subplot(gs[0, 0])
    
    # 使用两个图层：底层显示全部（低分辨率），顶层高亮显示高共视率区域（高分辨率）
    # 底层：全部数据（低分辨率，灰度显示）
    im1 = ax_main.imshow(
        display_matrix,
        cmap='gray',
        vmin=0.0,
        vmax=1.0,
        aspect='auto',
        origin='lower',
        interpolation='bilinear',  # 使用双线性插值使低分辨率区域更平滑
        alpha=0.4
    )
    
    # 顶层：高亮高共视率区域（高分辨率，彩色显示）
    highlight_matrix = np.where(highlight_mask, covisibility_matrix, np.nan)
    im2 = ax_main.imshow(
        highlight_matrix,
        cmap='hot',  # 使用hot colormap使高亮更明显
        vmin=highlight_threshold,
        vmax=1.0,
        aspect='auto',
        origin='lower',
        interpolation='nearest',  # 高分辨率区域使用最近邻保持清晰
        alpha=1.0
    )
    
    # 添加阈值等高线
    contour = ax_main.contour(
        covisibility_matrix,
        levels=[highlight_threshold],
        colors='red',
        linewidths=2,
        linestyles='--',
        alpha=0.7
    )
    ax_main.clabel(contour, inline=True, fontsize=10, fmt=f'{highlight_threshold:.2f}')
    
    cbar = plt.colorbar(im2, ax=ax_main, label='Covisibility Score (Highlighted > {:.0%})'.format(highlight_threshold), 
                       fraction=0.046, pad=0.04)
    ax_main.set_title(f'New Modular Covisibility Matrix ({num_frames}x{num_frames}) - Highlighted > {highlight_threshold:.0%}', 
                     fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Target Frame Index', fontsize=12)
    ax_main.set_ylabel('Reference Frame Index', fontsize=12)
    ax_main.plot([0, num_frames-1], [0, num_frames-1], 'r--', linewidth=1, alpha=0.5, label='Diagonal')
    ax_main.legend(loc='upper right')
    
    # 子图1：相邻帧的共视率分布
    ax_dist = fig.add_subplot(gs[0, 1])
    distances = []
    covis_scores = []
    for offset in range(1, min(20, num_frames)):
        for i in range(num_frames - offset):
            distances.append(offset)
            covis_scores.append(covisibility_matrix[i, i + offset])
    
    if len(covis_scores) > 0:
        ax_dist.scatter(distances, covis_scores, alpha=0.3, s=10)
        ax_dist.set_xlabel('Frame Distance', fontsize=10)
        ax_dist.set_ylabel('Covisibility Score', fontsize=10)
        ax_dist.set_title('Covisibility vs Frame Distance', fontsize=11, fontweight='bold')
        ax_dist.grid(True, alpha=0.3)
        unique_distances = np.unique(distances)
        mean_scores = [np.mean([covis_scores[i] for i, d in enumerate(distances) if d == dist]) 
                      for dist in unique_distances]
        ax_dist.plot(unique_distances, mean_scores, 'r-', linewidth=2, label='Mean')
        ax_dist.legend()
    
    # 子图2：共视率分布直方图
    ax_hist = fig.add_subplot(gs[1, 0])
    mask = ~np.eye(num_frames, dtype=bool)
    off_diagonal = covisibility_matrix[mask]
    ax_hist.hist(off_diagonal, bins=50, edgecolor='black', alpha=0.7)
    ax_hist.set_xlabel('Covisibility Score', fontsize=10)
    ax_hist.set_ylabel('Frequency', fontsize=10)
    ax_hist.set_title('Off-Diagonal Covisibility Distribution', fontsize=11, fontweight='bold')
    ax_hist.grid(True, alpha=0.3)
    ax_hist.axvline(off_diagonal.mean(), color='r', linestyle='--', linewidth=2, 
                    label=f'Mean: {off_diagonal.mean():.4f}')
    ax_hist.legend()
    
    # 子图3：统计信息
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    
    mask = ~np.eye(num_frames, dtype=bool)
    off_diagonal = covisibility_matrix[mask]
    
    # 计算高亮区域的统计
    highlight_count = np.sum(highlight_mask)
    highlight_ratio = highlight_count / (num_frames * num_frames) * 100
    
    stats_text = (
        f"Matrix Statistics:\n\n"
        f"Shape: {covisibility_matrix.shape}\n"
        f"Min: {covisibility_matrix.min():.4f}\n"
        f"Max: {covisibility_matrix.max():.4f}\n"
        f"Mean: {covisibility_matrix.mean():.4f}\n"
        f"Std: {covisibility_matrix.std():.4f}\n\n"
        f"Highlighted (> {highlight_threshold:.0%}):\n"
        f"  Count: {highlight_count}\n"
        f"  Ratio: {highlight_ratio:.2f}%\n\n"
        f"Off-Diagonal:\n"
        f"  Mean: {off_diagonal.mean():.4f}\n"
        f"  Std: {off_diagonal.std():.4f}\n"
        f"  Median: {np.median(off_diagonal):.4f}\n"
        f"  > {highlight_threshold:.0%}: {np.sum(off_diagonal > highlight_threshold)} ({100*np.sum(off_diagonal > highlight_threshold)/len(off_diagonal):.2f}%)\n"
        f"  > 0.5: {np.sum(off_diagonal > 0.5)} ({100*np.sum(off_diagonal > 0.5)/len(off_diagonal):.2f}%)"
    )
    
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                  family='monospace')
    
    plt.suptitle('New Modular Depth Reprojection Covisibility Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用新的模块化共视率检测代码检查数据')
    parser.add_argument('--data_root', type=str, default='traj_demo/imgs',
                       help='数据根目录')
    parser.add_argument('--output', type=str, default='output/new_covisibility_matrix.png',
                       help='输出图像路径')
    parser.add_argument('--save_matrix', type=str, default='output/new_covisibility_matrix.npy',
                       help='保存矩阵的路径（可选）')
    parser.add_argument('--use_gpu', action='store_true',
                       help='使用GPU加速')
    parser.add_argument('--device', type=str, default=None,
                       help='指定设备 (cpu/cuda/cuda:0等)')
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='分块大小（用于内存优化）')
    parser.add_argument('--precompute', action='store_true',
                       help='预计算世界坐标点（优化性能）')
    parser.add_argument('--depth_assoc_error_thres', type=float, default=0.01,
                       help='深度关联绝对误差阈值（米），默认0.01')
    parser.add_argument('--depth_assoc_rel_error_thres', type=float, default=0.0,
                       help='深度关联相对误差阈值（相对于期望深度），默认0.0（固定阈值）。设置>0启用动态阈值，例如0.02表示2%%相对误差')
    parser.add_argument('--depth_assoc_error_temp', type=float, default=0.0,
                       help='深度关联误差温度参数，默认0.0。设置>0启用soft threshold')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.save_matrix:
        os.makedirs(os.path.dirname(args.save_matrix), exist_ok=True)
    
    # 确定设备
    if args.device:
        device = torch.device(args.device) if TORCH_AVAILABLE else None
    elif args.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
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
    print(f"\nLoading pose data from: {args.data_root}")
    poses, intrinsics, img_h, img_w, num_frames = load_pose_data(
        args.data_root, device=device, use_gpu=(device is not None and device.type == 'cuda')
    )
    print(f"Loaded {num_frames} frames")
    print(f"Image size: {img_w}x{img_h}")
    
    # 加载所有深度图
    print(f"\nLoading depth maps...")
    depth_maps = load_all_depth_maps(args.data_root, num_frames, device=device)
    print(f"Loaded {len(depth_maps)} depth maps")
    
    # 准备参数
    # 统一K的格式（所有帧使用相同的内参）
    if isinstance(intrinsics, list):
        K_all = intrinsics
    else:
        K_all = intrinsics  # 单个内参矩阵，会在build_overlap_matrix中处理
    
    # 使用新的模块化代码计算共视率矩阵
    print(f"\nComputing covisibility matrix using new modular code...")
    print(f"  - Using build_overlap_matrix() from sampling_pipeline.py")
    print(f"  - Precompute world points: {args.precompute}")
    if args.chunk_size:
        print(f"  - Chunk size: {args.chunk_size}")
    
    # 打印阈值配置信息
    if args.depth_assoc_rel_error_thres > 0.0 or args.depth_assoc_error_temp > 0.0:
        print(f"  - Using DYNAMIC threshold:")
        print(f"    * Absolute error threshold: {args.depth_assoc_error_thres:.4f}m")
        print(f"    * Relative error threshold: {args.depth_assoc_rel_error_thres:.4f} ({args.depth_assoc_rel_error_thres*100:.2f}%)")
        print(f"    * Temperature parameter: {args.depth_assoc_error_temp:.4f}")
        print(f"    * Example: For depth=1.0m, threshold = {args.depth_assoc_error_thres + args.depth_assoc_rel_error_thres * 1.0 - 0.693 * args.depth_assoc_error_temp:.4f}m")
    else:
        print(f"  - Using FIXED threshold: {args.depth_assoc_error_thres:.4f}m")
    
    covisibility_matrix = build_overlap_matrix(
        poses_all=poses,
        depths_all=depth_maps,
        K_all=intrinsics,
        img_h=img_h,
        img_w=img_w,
        depth_assoc_error_thres=args.depth_assoc_error_thres,
        depth_assoc_rel_error_thres=args.depth_assoc_rel_error_thres,
        depth_assoc_error_temp=args.depth_assoc_error_temp,
        denominator_mode="valid_target_depth",
        min_depth=0.04,
        device=device,
        precompute_world_pts=args.precompute,
        chunk_size=args.chunk_size
    )
    
    # 保存矩阵
    if args.save_matrix:
        np.save(args.save_matrix, covisibility_matrix)
        print(f"\nMatrix saved to: {args.save_matrix}")
        print(f"Matrix shape: {covisibility_matrix.shape}")
        print(f"Matrix dtype: {covisibility_matrix.dtype}")
    
    # 打印统计信息
    print(f"\nCovisibility Matrix Statistics:")
    print(f"  Shape: {covisibility_matrix.shape}")
    print(f"  Min: {covisibility_matrix.min():.4f}")
    print(f"  Max: {covisibility_matrix.max():.4f}")
    print(f"  Mean: {covisibility_matrix.mean():.4f}")
    print(f"  Std: {covisibility_matrix.std():.4f}")
    print(f"  Diagonal (self-overlap): {np.diag(covisibility_matrix).mean():.4f}")
    
    # 计算非对角元素的统计
    mask = ~np.eye(num_frames, dtype=bool)
    off_diagonal = covisibility_matrix[mask]
    print(f"  Off-diagonal mean: {off_diagonal.mean():.4f}")
    print(f"  Off-diagonal std: {off_diagonal.std():.4f}")
    
    # 可视化
    print(f"\nVisualizing matrix...")
    visualize_covisibility_matrix(covisibility_matrix, save_path=args.output, highlight_threshold=0.25)
    
    print("\nDone! New modular code test completed successfully.")


if __name__ == '__main__':
    main()

