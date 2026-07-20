# 共视率检测使用文档

基于深度重投影验证的共视率检测（MapAnything方法）

## 快速开始

### 基本使用（固定阈值）

使用新的模块化代码检查 traj_demo 数据（默认使用固定阈值 0.01m）：

```bash
python3 test_new_covisibility.py --data_root traj_demo/imgs --use_gpu --precompute
```

### 使用动态阈值（推荐）

如果深度估计误差随距离增大，建议使用动态阈值：

```bash
# 使用动态阈值（2%相对误差）
python3 test_new_covisibility.py \
    --data_root traj_demo/imgs \
    --depth_assoc_error_thres 0.01 \
    --depth_assoc_rel_error_thres 0.02 \
    --use_gpu --precompute
```

这样，阈值会随深度动态变化：
- 深度 0.5m → 阈值 = 0.01 + 0.02 × 0.5 = 0.02m
- 深度 1.0m → 阈值 = 0.01 + 0.02 × 1.0 = 0.03m
- 深度 2.0m → 阈值 = 0.01 + 0.02 × 2.0 = 0.05m

### 完整参数

```bash
python3 test_new_covisibility.py \
    --data_root traj_demo/imgs \
    --output output/new_covisibility_matrix.png \
    --save_matrix output/new_covisibility_matrix.npy \
    --use_gpu \
    --precompute \
    --chunk_size 50 \
    --depth_assoc_error_thres 0.01 \
    --depth_assoc_rel_error_thres 0.02
```

### 参数说明

- `--data_root`: 数据根目录（默认：`traj_demo/imgs`）
- `--output`: 输出可视化图像路径（默认：`output/new_covisibility_matrix.png`）
- `--save_matrix`: 保存矩阵的路径（默认：`output/new_covisibility_matrix.npy`）
- `--use_gpu`: 使用GPU加速（推荐）
- `--precompute`: 预计算世界坐标点（优化性能，推荐）
- `--chunk_size`: 分块大小，用于内存优化（默认：None，一次性计算所有）
- `--device`: 指定设备（cpu/cuda/cuda:0等）

#### 深度阈值参数（重要）

- `--depth_assoc_error_thres`: 深度关联绝对误差阈值（米），默认 `0.01`
- `--depth_assoc_rel_error_thres`: 深度关联相对误差阈值，默认 `0.0`（固定阈值）
  - 设置为 `> 0` 时启用**动态阈值**
  - 例如：`0.02` 表示 2% 的相对误差
  - 动态阈值公式：`threshold = depth_assoc_error_thres + depth_assoc_rel_error_thres * D_expected`
- `--depth_assoc_error_temp`: 深度关联误差温度参数，默认 `0.0`

**使用动态阈值的示例**：
```bash
# 使用动态阈值（2%相对误差）
python3 test_new_covisibility.py \
    --data_root traj_demo/imgs \
    --depth_assoc_error_thres 0.01 \
    --depth_assoc_rel_error_thres 0.02 \
    --use_gpu --precompute
```

这样，对于不同深度值，阈值会动态变化：
- 深度 0.5m → 阈值 = 0.01 + 0.02 × 0.5 = 0.02m
- 深度 1.0m → 阈值 = 0.01 + 0.02 × 1.0 = 0.03m
- 深度 2.0m → 阈值 = 0.01 + 0.02 × 2.0 = 0.05m

## 代码架构

新的模块化代码分为三个阶段：

### 阶段 I：GPU 几何运算模块 (`core/geometry_ops.py`)

提供所有几何原语：
- `unproject_depth_to_world()` - 深度图反投影到世界坐标
- `inverse_transform_4x4()` - 4x4变换矩阵求逆
- `project_world_to_pixel()` - 世界点投影到像素平面
- `sample_depths_at_reproj()` - 深度采样

### 阶段 II：共视性校验核心 (`core/covisibility_gpu.py`)

核心共视性检查：
- `check_depth_consistency()` - 深度一致性检查
- `compute_score_for_pair()` - 计算两帧共视率（总控函数）

### 阶段 III：并行矩阵构建 (`core/sampling_pipeline.py`)

批量计算和采样：
- `build_overlap_matrix()` - GPU并行构建共视率矩阵
- `get_adaptive_indices()` - 根据矩阵生成采样索引

## 在代码中使用

### 方式1：在 HybridSampler 中使用（已集成）

`HybridSampler` 已经迁移到使用新的模块化代码：

```python
from samplers.hybrid_sampler import HybridSampler
from utils.io import load_pose_data

# 加载数据
poses, intrinsics, img_h, img_w, num_frames = load_pose_data(data_root, device=device)

# 创建采样器（使用深度重投影方法，固定阈值）
sampler = HybridSampler(
    poses, intrinsics, img_h, img_w,
    use_depth_reprojection=True,  # 启用深度重投影方法
    data_root=data_root,           # 深度图数据路径
    depth_threshold=0.01,          # 绝对误差阈值（米）
    depth_assoc_rel_error_thres=0.0,  # 相对误差阈值（0.0=固定阈值）
    device=device
)

# 创建采样器（使用动态阈值，推荐）
sampler = HybridSampler(
    poses, intrinsics, img_h, img_w,
    use_depth_reprojection=True,
    data_root=data_root,
    depth_threshold=0.01,          # 绝对误差阈值（米）
    depth_assoc_rel_error_thres=0.02,  # 2%相对误差（启用动态阈值）
    device=device
)

# 执行采样
selected_indices = sampler.sample()
```

### 方式2：使用总控函数（推荐）

```python
from core.sampling_pipeline import build_overlap_matrix
from utils.io import load_pose_data, load_all_depth_maps

# 加载数据
poses, intrinsics, img_h, img_w, num_frames = load_pose_data(data_root, device=device)
depth_maps = load_all_depth_maps(data_root, num_frames, device=device)

# 计算共视率矩阵
covisibility_matrix = build_overlap_matrix(
    poses_all=poses,
    depths_all=depth_maps,
    K_all=intrinsics,
    img_h=img_h,
    img_w=img_w,
    device=device,
    precompute_world_pts=True  # 预计算优化
)
```

### 方式2：使用单帧对计算

```python
from core.covisibility_gpu import compute_score_for_pair

# 计算两帧之间的共视率（使用固定阈值）
overlap_score = compute_score_for_pair(
    depth_ref=depth_maps[i],
    pose_ref=poses[i],
    depth_target=depth_maps[j],
    pose_target=poses[j],
    K=intrinsics,
    img_h=img_h,
    img_w=img_w,
    depth_assoc_error_thres=0.01,        # 绝对误差阈值
    depth_assoc_rel_error_thres=0.0,    # 相对误差阈值（0.0=固定阈值）
    device=device
)

# 使用动态阈值（推荐，如果深度估计误差随距离增大）
overlap_score = compute_score_for_pair(
    depth_ref=depth_maps[i],
    pose_ref=poses[i],
    depth_target=depth_maps[j],
    pose_target=poses[j],
    K=intrinsics,
    img_h=img_h,
    img_w=img_w,
    depth_assoc_error_thres=0.01,        # 绝对误差阈值
    depth_assoc_rel_error_thres=0.02,    # 2%相对误差（启用动态阈值）
    device=device
)
```

### 方式3：使用底层几何函数

```python
from core.geometry_ops import unproject_depth_to_world, project_world_to_pixel

# 反投影深度图
P_world, M_valid = unproject_depth_to_world(
    K, pose_ref, depth_ref, device=device
)

# 投影到目标帧
P_2d, D_expected = project_world_to_pixel(
    K, pose_target, P_world, device=device
)
```

## 性能优化建议

1. **使用GPU加速**：添加 `--use_gpu` 参数
2. **预计算世界坐标点**：添加 `--precompute` 参数（可提升约30%性能）
3. **分块处理**：如果内存不足，使用 `--chunk_size 50` 参数

## 输出结果

运行后会生成：
- `output/new_covisibility_matrix.npy` - 共视率矩阵数据（153×153）
- `output/new_covisibility_matrix.png` - 可视化图像（包含4个子图）

## 与原实现的对比

- **原实现** (`core/depth_covisibility.py`): 仍然可用，向后兼容
- **新实现** (`core/geometry_ops.py`, `core/covisibility_gpu.py`, `core/sampling_pipeline.py`): 
  - ✅ 更模块化
  - ✅ 更好的GPU并行支持
  - ✅ 更清晰的代码结构
  - ✅ 完全符合MapAnything逻辑

## 测试结果

使用 traj_demo 数据（153帧）：
- ✅ 成功计算 153×153 = 23,409 个帧对的共视率
- ✅ GPU加速：约 7-8 帧对/秒
- ✅ 对角线元素：1.0（正确）
- ✅ 非对角元素统计：平均值 0.0072，符合预期

