"""
可视化采样结果（支持GPU）
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# 配置matplotlib使用英文字体（避免中文字体警告）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def _to_numpy(x):
    """
    将输入转换为numpy数组（支持GPU tensor）
    
    Args:
        x: numpy array 或 torch tensor
    
    Returns:
        numpy array
    """
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def visualize_trajectory(poses, selected_indices=None, save_path=None):
    """
    可视化轨迹和采样结果（支持GPU）
    
    Args:
        poses: 位姿列表（可以是numpy array或torch tensor）
        selected_indices: 选中的帧索引列表（可选）
        save_path: 保存路径（可选）
    """
    # 提取所有相机位置（支持GPU tensor）
    positions_list = []
    for pose in poses:
        pos = pose[:3, 3]
        positions_list.append(_to_numpy(pos))
    positions = np.array(positions_list)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制完整轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', alpha=0.3, linewidth=1, label='Full trajectory')
    
    # 绘制所有帧的位置点
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='blue', s=10, alpha=0.5, label='All frames')
    
    # 绘制选中的帧
    if selected_indices is not None:
        selected_positions = positions[selected_indices]
        ax.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2],
                   c='red', s=50, marker='o', label='Selected frames')
        
        # 绘制选中帧之间的连线
        if len(selected_indices) > 1:
            selected_positions_ordered = positions[selected_indices]
            ax.plot(selected_positions_ordered[:, 0], 
                   selected_positions_ordered[:, 1], 
                   selected_positions_ordered[:, 2],
                   'r-', linewidth=2, alpha=0.7, label='Sampled path')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trajectory Sampling Visualization')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_statistics(poses, selected_indices, save_path=None):
    """
    可视化采样统计信息（支持GPU）
    
    Args:
        poses: 位姿列表（可以是numpy array或torch tensor）
        selected_indices: 选中的帧索引列表
        save_path: 保存路径（可选）
    """
    num_total = len(poses)
    num_selected = len(selected_indices)
    
    # 计算距离和角度统计
    if len(selected_indices) > 1:
        distances = []
        angles = []
        for i in range(len(selected_indices) - 1):
            idx1 = selected_indices[i]
            idx2 = selected_indices[i + 1]
            # 计算距离
            pos1 = _to_numpy(poses[idx1][:3, 3])
            pos2 = _to_numpy(poses[idx2][:3, 3])
            dist = np.linalg.norm(pos2 - pos1)
            distances.append(dist)
            # 计算角度
            R1 = _to_numpy(poses[idx1][:3, :3])
            R2 = _to_numpy(poses[idx2][:3, :3])
            R_rel = R2 @ R1.T
            trace = np.trace(R_rel)
            angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
    else:
        distances = [0]
        angles = [0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 统计信息文本
    stats_text = f"""
    Total Frames: {num_total}
    Selected Frames: {num_selected}
    Sampling Rate: {num_selected/num_total*100:.2f}%
    Avg Distance: {np.mean(distances):.3f} m
    Avg Angle: {np.mean(angles):.2f} deg
    """
    axes[0, 0].text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Sampling Statistics')
    
    # 距离分布
    axes[0, 1].hist(distances, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Distance (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distance Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 角度分布
    axes[1, 0].hist(angles, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Rotation Angle (deg)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Angle Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 采样间隔分布
    if len(selected_indices) > 1:
        intervals = np.diff(selected_indices)
        axes[1, 1].hist(intervals, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Frame Interval')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Sampling Interval Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

