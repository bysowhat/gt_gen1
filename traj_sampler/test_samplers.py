#!/usr/bin/env python3
"""
快速测试所有采样器
"""
import sys
sys.path.insert(0, '.')

from utils.io import load_pose_data
from samplers.time_sampler import TimeSampler
from samplers.spatial_sampler import SpatialSampler
from samplers.adaptive_sampler import AdaptiveSampler
from samplers.action_proxy_sampler import SE3ActionSampler


def test_all_samplers():
    """测试所有采样器"""
    print("=" * 60)
    print("加载数据...")
    poses, intrinsics, img_h, img_w, num_frames = load_pose_data('traj_demo/imgs')
    print(f"加载了 {num_frames} 帧数据，图像尺寸: {img_w}x{img_h}")
    print("=" * 60)
    
    # 测试固定步长采样
    print("\n1. 测试固定步长采样 (stride=5)")
    sampler = TimeSampler(poses, intrinsics, img_h, img_w, stride=5)
    indices = sampler.sample()
    print(f"   选中 {len(indices)} 帧 (采样率: {len(indices)/num_frames*100:.2f}%)")
    print(f"   索引范围: {indices[0]} - {indices[-1]}")
    
    # 测试距离采样
    print("\n2. 测试距离采样 (distance_threshold=0.1m)")
    sampler = SpatialSampler(poses, intrinsics, img_h, img_w, distance_threshold=0.1)
    indices = sampler.sample()
    print(f"   选中 {len(indices)} 帧 (采样率: {len(indices)/num_frames*100:.2f}%)")
    print(f"   索引范围: {indices[0]} - {indices[-1]}")
    
    # 测试角度采样
    print("\n3. 测试角度采样 (angle_threshold=10deg)")
    sampler = SpatialSampler(poses, intrinsics, img_h, img_w, angle_threshold=10.0)
    indices = sampler.sample()
    print(f"   选中 {len(indices)} 帧 (采样率: {len(indices)/num_frames*100:.2f}%)")
    print(f"   索引范围: {indices[0]} - {indices[-1]}")
    
    # 测试自适应采样
    print("\n4. 测试自适应采样 (min_overlap=0.3, max_overlap=0.8)")
    sampler = AdaptiveSampler(poses, intrinsics, img_h, img_w, 
                               min_overlap=0.3, max_overlap=0.8)
    indices = sampler.sample()
    print(f"   选中 {len(indices)} 帧 (采样率: {len(indices)/num_frames*100:.2f}%)")
    print(f"   索引范围: {indices[0]} - {indices[-1]}")
    
    # 测试SE(3)动作幅度采样（累积模式）
    print("\n5. 测试SE(3)动作幅度采样 - 累积模式 (threshold=0.1, rot_weight=1.0)")
    sampler = SE3ActionSampler(poses, intrinsics, img_h, img_w,
                                action_threshold=0.1, rot_weight=1.0, use_accumulated=True)
    indices = sampler.sample()
    print(f"   选中 {len(indices)} 帧 (采样率: {len(indices)/num_frames*100:.2f}%)")
    print(f"   索引范围: {indices[0]} - {indices[-1]}")
    # 显示一些动作幅度统计
    if len(indices) > 1:
        actions = []
        for i in range(len(indices) - 1):
            _, _, action_mag = sampler.compute_action_magnitude(indices[i], indices[i+1])
            actions.append(action_mag)
        print(f"   平均动作幅度: {sum(actions)/len(actions):.4f}")
    
    # 测试SE(3)动作幅度采样（单步模式）
    print("\n6. 测试SE(3)动作幅度采样 - 单步模式 (threshold=0.05, rot_weight=1.0)")
    sampler = SE3ActionSampler(poses, intrinsics, img_h, img_w,
                                action_threshold=0.05, rot_weight=1.0, use_accumulated=False)
    indices = sampler.sample()
    print(f"   选中 {len(indices)} 帧 (采样率: {len(indices)/num_frames*100:.2f}%)")
    print(f"   索引范围: {indices[0]} - {indices[-1]}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")


if __name__ == '__main__':
    test_all_samplers()

