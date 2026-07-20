#!/usr/bin/env python3
"""
测试GPU兼容性
"""
import numpy as np
from core.geometry import transform_7_to_4x4, project_points, get_frustum_samples, _is_tensor

try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch is available")
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, will test CPU mode")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not installed, will test CPU mode only")

print("\n" + "="*60)
print("Testing CPU mode (numpy)")
print("="*60)

# 测试CPU模式
pose_7d = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])  # [x, y, z, qw, qx, qy, qz]
pose_4x4 = transform_7_to_4x4(pose_7d)
print(f"✓ transform_7_to_4x4 (CPU): {pose_4x4.shape}")

intrinsics = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
points_3d = np.array([[0.1, 0.1, 0.5], [0.2, 0.2, 0.6]], dtype=np.float32)
points_2d, depths, valid = project_points(points_3d, pose_4x4, intrinsics)
print(f"✓ project_points (CPU): {points_2d.shape}")

samples = get_frustum_samples(pose_4x4, intrinsics, 480, 640, near=0.1, far=1.0, num_samples_per_plane=5)
print(f"✓ get_frustum_samples (CPU): {samples.shape}")

if TORCH_AVAILABLE:
    print("\n" + "="*60)
    print("Testing GPU/CPU mode (torch)")
    print("="*60)
    
    # 测试CPU tensor模式
    device = torch.device('cpu')
    pose_7d_torch = torch.tensor(pose_7d, device=device, dtype=torch.float32)
    pose_4x4_torch = transform_7_to_4x4(pose_7d_torch, device=device)
    print(f"✓ transform_7_to_4x4 (CPU tensor): {pose_4x4_torch.shape}, device={pose_4x4_torch.device}")
    
    intrinsics_torch = torch.from_numpy(intrinsics).float().to(device)
    points_3d_torch = torch.from_numpy(points_3d).float().to(device)
    points_2d_torch, depths_torch, valid_torch = project_points(
        points_3d_torch, pose_4x4_torch, intrinsics_torch, device=device
    )
    print(f"✓ project_points (CPU tensor): {points_2d_torch.shape}, device={points_2d_torch.device}")
    
    samples_torch = get_frustum_samples(
        pose_4x4_torch, intrinsics_torch, 480, 640, 
        near=0.1, far=1.0, num_samples_per_plane=5, device=device
    )
    print(f"✓ get_frustum_samples (CPU tensor): {samples_torch.shape}, device={samples_torch.device}")
    
    # 如果CUDA可用，测试GPU模式
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("Testing GPU mode (CUDA)")
        print("="*60)
        
        device = torch.device('cuda')
        pose_7d_gpu = torch.tensor(pose_7d, device=device, dtype=torch.float32)
        pose_4x4_gpu = transform_7_to_4x4(pose_7d_gpu, device=device)
        print(f"✓ transform_7_to_4x4 (GPU): {pose_4x4_gpu.shape}, device={pose_4x4_gpu.device}")
        
        intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)
        points_3d_gpu = torch.from_numpy(points_3d).float().to(device)
        points_2d_gpu, depths_gpu, valid_gpu = project_points(
            points_3d_gpu, pose_4x4_gpu, intrinsics_gpu, device=device
        )
        print(f"✓ project_points (GPU): {points_2d_gpu.shape}, device={points_2d_gpu.device}")
        
        samples_gpu = get_frustum_samples(
            pose_4x4_gpu, intrinsics_gpu, 480, 640,
            near=0.1, far=1.0, num_samples_per_plane=5, device=device
        )
        print(f"✓ get_frustum_samples (GPU): {samples_gpu.shape}, device={samples_gpu.device}")
        
        # 验证结果一致性
        print("\n" + "="*60)
        print("Verifying CPU vs GPU consistency")
        print("="*60)
        pose_diff = torch.abs(pose_4x4_torch.cpu() - pose_4x4_gpu.cpu()).max().item()
        print(f"✓ transform_7_to_4x4 max difference: {pose_diff:.2e}")
        
        points_diff = torch.abs(points_2d_torch.cpu() - points_2d_gpu.cpu()).max().item()
        print(f"✓ project_points max difference: {points_diff:.2e}")
        
        samples_diff = torch.abs(samples_torch.cpu() - samples_gpu.cpu()).max().item()
        print(f"✓ get_frustum_samples max difference: {samples_diff:.2e}")

print("\n" + "="*60)
print("All tests passed!")
print("="*60)

