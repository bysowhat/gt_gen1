#!/usr/bin/env python3
"""
主程序：生成采样对
"""
import os
import argparse
import yaml
import numpy as np
from utils.io import load_pose_data
from utils.visualizer import visualize_trajectory
from samplers.time_sampler import TimeSampler
from samplers.spatial_sampler import SpatialSampler
from samplers.adaptive_sampler import AdaptiveSampler
from samplers.action_proxy_sampler import SE3ActionSampler
from samplers.hybrid_sampler import HybridSampler


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_sampler(config, poses, intrinsics, img_h, img_w, device=None, data_root=None):
    """根据配置创建采样器（支持GPU）"""
    sampler_type = config['sampler']['type']
    sampler_params = config['sampler'].get('params', {}).copy()  # 创建副本以避免修改原始配置
    
    # 添加device参数到sampler_params
    if device is not None:
        sampler_params['device'] = device
    
    # 如果使用深度重投影方法，需要传递data_root
    if sampler_params.get('use_depth_reprojection', False) and data_root is not None:
        sampler_params['data_root'] = data_root
    
    if sampler_type == 'time':
        return TimeSampler(poses, intrinsics, img_h, img_w, **sampler_params)
    elif sampler_type == 'spatial':
        return SpatialSampler(poses, intrinsics, img_h, img_w, **sampler_params)
    elif sampler_type == 'adaptive':
        return AdaptiveSampler(poses, intrinsics, img_h, img_w, **sampler_params)
    elif sampler_type == 'action_proxy' or sampler_type == 'se3_action':
        return SE3ActionSampler(poses, intrinsics, img_h, img_w, **sampler_params)
    elif sampler_type == 'hybrid':
        return HybridSampler(poses, intrinsics, img_h, img_w, **sampler_params)
    else:
        raise ValueError(f"未知的采样器类型: {sampler_type}")


def generate_obs_pred_sequences(selected_indices, obs_len, pred_len):
    """
    生成观测-预测序列对，用于训练DP网络
    
    Args:
        selected_indices: 采样后的帧索引列表，例如 [0, 5, 10, 15, 20, 25, 30, ...]
        obs_len: 观测窗口长度（观测帧数）
        pred_len: 预测窗口长度（预测帧数）
    
    Returns:
        observations: 观测序列数组 (N, obs_len)，每行是一个观测序列
        predictions: 预测序列数组 (N, pred_len)，每行是对应的预测序列
    """
    observations = []
    predictions = []
    
    # 滑动窗口生成训练样本
    # 对于每个可能的起始位置，如果观测窗口和预测窗口都在范围内，就生成一个样本
    for i in range(len(selected_indices) - obs_len - pred_len + 1):
        # 观测序列：从位置i开始的obs_len个帧
        obs_seq = selected_indices[i:i+obs_len]
        # 预测序列：从位置i+obs_len开始的pred_len个帧
        pred_seq = selected_indices[i+obs_len:i+obs_len+pred_len]
        
        observations.append(obs_seq)
        predictions.append(pred_seq)
    
    return np.array(observations), np.array(predictions)


def save_sequences(selected_indices, obs_len, pred_len, output_dir, exp_name):
    """
    保存观测-预测序列对到文件（单个npy文件，字典格式）
    
    Args:
        selected_indices: 采样后的帧索引列表
        obs_len: 观测窗口长度
        pred_len: 预测窗口长度
        output_dir: 输出目录
        exp_name: 实验名称
    """
    # 生成序列对
    observations, predictions = generate_obs_pred_sequences(
        selected_indices, obs_len, pred_len
    )
    
    if len(observations) == 0:
        print(f"Warning: Cannot generate sequences (obs_len={obs_len}, pred_len={pred_len}, sampled_frames={len(selected_indices)})")
        return
    
    # 保存为单个字典格式的npy文件
    data = {
        'observations': observations,      # (N, obs_len) 观测序列
        'predictions': predictions,         # (N, pred_len) 预测序列
        'selected_indices': np.array(selected_indices),  # 采样后的帧索引列表
        'obs_len': obs_len,                # 观测窗口长度
        'pred_len': pred_len,              # 预测窗口长度
        'num_samples': len(observations)   # 训练样本数量
    }
    
    output_path = os.path.join(output_dir, f'{exp_name}_data.npy')
    np.save(output_path, data, allow_pickle=True)
    
    print(f"Data saved to: {output_path}")
    print(f"  - Observations shape: {observations.shape}")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Selected indices: {len(selected_indices)} frames")
    print(f"  - Generated {len(observations)} training samples")


def main():
    parser = argparse.ArgumentParser(description='生成轨迹采样对')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--data_root', type=str, default='traj_demo/imgs',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化结果')
    parser.add_argument('--use_gpu', action='store_true',
                       help='是否使用GPU加速（需要安装torch）')
    parser.add_argument('--device', type=str, default=None,
                       help='指定设备 (cpu/cuda/cuda:0等)，如果指定则覆盖--use_gpu')
    
    args = parser.parse_args()
    
    # 确定设备
    try:
        import torch
        if args.device:
            device = torch.device(args.device)
        elif args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = None
    except ImportError:
        device = None
        if args.use_gpu or args.device:
            print("Warning: PyTorch not installed, GPU acceleration disabled")
    
    # 加载配置
    config = load_config(args.config)
    exp_name = config.get('experiment_name', 'default')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from: {args.data_root}")
    use_gpu = (device is not None) if device is not None else False
    poses, intrinsics, img_h, img_w, num_frames = load_pose_data(
        args.data_root, device=device, use_gpu=use_gpu
    )
    print(f"Loaded {num_frames} frames")
    print(f"Image size: {img_w}x{img_h}")
    if device is not None:
        print(f"Using device: {device}")
    
    # 创建采样器
    sampler = get_sampler(config, poses, intrinsics, img_h, img_w, device=device, data_root=args.data_root)
    print(f"Using sampler: {config['sampler']['type']}")
    sampler_params = config['sampler'].get('params', {})
    if sampler_params.get('use_depth_reprojection', False):
        print("Using depth reprojection covisibility method (MapAnything)")
    
    # 执行采样
    print("Sampling...")
    selected_indices = sampler.sample()
    print(f"Selected {len(selected_indices)} frames (sampling rate: {len(selected_indices)/num_frames*100:.2f}%)")
    
    # 获取观测和预测窗口长度（从配置文件）
    obs_len = config.get('obs_len', 5)  # 默认观测5帧
    pred_len = config.get('pred_len', 5)  # 默认预测5帧
    
    print(f"Observation window length: {obs_len} frames")
    print(f"Prediction window length: {pred_len} frames")
    
    # 生成并保存观测-预测序列对（包含选中的索引）
    save_sequences(selected_indices, obs_len, pred_len, args.output_dir, exp_name)
    
    # 可视化
    if args.visualize:
        vis_path = os.path.join(args.output_dir, f'{exp_name}_trajectory.png')
        visualize_trajectory(poses, selected_indices, save_path=vis_path)


if __name__ == '__main__':
    main()

