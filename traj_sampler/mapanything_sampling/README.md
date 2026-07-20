# MapAnything风格的共视检测和分段采样核心模块

这个模块包含基于MapAnything论文实现的共视检测和分段采样算法的核心代码。

## 模块结构

```
mapanything_sampling/
├── __init__.py              # 模块导出
├── geometry_ops.py          # GPU几何运算模块（3D点生成、投影、深度采样）
├── covisibility_gpu.py      # 共视性校验核心（深度一致性检查、共视率计算）
├── action_metrics.py        # SE(3)运动度量模块（SE(3)距离计算）
├── sampling_pipeline.py     # 分段采样策略（整合硬性节点约束和双重几何条件）
└── README.md                # 本文件
```

## 核心功能

### 1. 几何运算模块 (`geometry_ops.py`)

提供GPU加速的几何原语：

- `unproject_depth_to_world()`: 将深度图反投影到3D世界坐标系
- `project_world_to_pixel()`: 将3D世界点投影到2D像素坐标系
- `sample_depths_at_reproj()`: 在重投影位置采样深度值
- `inverse_transform_4x4()`: 计算4x4变换矩阵的逆

### 2. 共视性校验核心 (`covisibility_gpu.py`)

实现MapAnything的深度重投影验证方法：

- `check_depth_consistency()`: 深度一致性检查（动态阈值）
- `compute_score_for_pair()`: 计算两帧之间的共视率分数
- `in_image_mask()`: 检查2D点是否在图像边界内

### 3. SE(3)运动度量模块 (`action_metrics.py`)

计算帧之间的SE(3)距离：

- `compute_se3_distance()`: 计算两个位姿之间的SE(3)距离
- `build_action_matrix()`: GPU并行构建所有帧对的SE(3)距离矩阵

### 4. 分段采样策略 (`sampling_pipeline.py`)

整合硬性节点约束和双重几何条件：

- `build_overlap_matrix()`: 构建共视率矩阵（GPU优化）
- `get_segmented_keyframes()`: 分段关键帧采样
- `get_adaptive_indices()`: 根据共视率矩阵生成自适应采样索引

## 使用方法

### 基本导入

```python
# 方式1：从mapanything_sampling模块导入（推荐）
from mapanything_sampling import (
    build_overlap_matrix,
    compute_score_for_pair,
    build_action_matrix,
    get_segmented_keyframes
)

# 方式2：从core模块导入（向后兼容）
from core import (
    build_overlap_matrix,
    compute_score_for_pair,
    build_action_matrix,
    get_segmented_keyframes
)
```

### 计算共视率矩阵

```python
from mapanything_sampling import build_overlap_matrix
from utils.io import load_pose_data, load_all_depth_maps

# 加载数据
poses, intrinsics, img_h, img_w, num_frames = load_pose_data(data_root, device=device)
depth_maps = load_all_depth_maps(data_root, num_frames, device=device)

# 计算共视率矩阵
M_overlap = build_overlap_matrix(
    poses_all=poses,
    depths_all=depth_maps,
    K_all=intrinsics,
    img_h=img_h,
    img_w=img_w,
    depth_assoc_error_thres=0.01,
    depth_assoc_rel_error_thres=0.02,  # 动态阈值
    device=device,
    precompute_world_pts=True
)
```

### 计算SE(3)距离矩阵

```python
from mapanything_sampling import build_action_matrix

# 计算SE(3)距离矩阵
M_action = build_action_matrix(
    T_all=poses,
    w_trans=1.0,
    w_rot=1.0,
    device=device
)
```

### 分段关键帧采样

```python
from mapanything_sampling import get_segmented_keyframes

# 方式1：自动生成硬性节点（推荐）
# 硬性节点会根据实际帧数自动生成所有50的倍数节点
# 自动生成规则：
#   - 总是包含第一帧（0）
#   - 生成所有50的倍数：50, 100, 150, 200, ...
#   - 如果轨迹长度不是50的倍数，选择最接近的50的倍数作为最后一个节点
# 示例（硬性节点生成规则）：
#   - 49帧  -> [0]  (没有50的倍数可选，但采样结果会包含[0, 48])
#   - 50帧  -> [0, 49]  (49是50-1，避免可能的噪音帧50)
#   - 51帧  -> [0, 50]  (50是最接近51的50的倍数)
#   - 100帧 -> [0, 50, 99]  (99是100-1，避免可能的噪音帧100)
#   - 101帧 -> [0, 50, 100]  (100是最接近101的50的倍数)
#   - 150帧 -> [0, 50, 100, 149]  (149是150-1，避免可能的噪音帧150)
#   - 151帧 -> [0, 50, 100, 150]  (150是最接近151的50的倍数)
#   - 153帧 -> [0, 50, 100, 150]  (150是最接近153的50的倍数)
#   - 200帧 -> [0, 50, 100, 150, 199]  (199是200-1，避免可能的噪音帧200)
#   - 201帧 -> [0, 50, 100, 150, 200]  (200是最接近201的50的倍数)
#
# 重要：无论硬性节点如何，采样结果总是包含第一帧(0)和最后一帧(num_frames-1)
# 例如：49帧的采样结果至少包含[0, 48]，即使硬性节点只有[0]
keyframes = get_segmented_keyframes(
    M_overlap=M_overlap,
    M_action=M_action,
    nodes=None,  # None表示自动生成，会根据M_overlap.shape[0]自动计算
    D_target=0.1,  # 最小运动成本
    overlap_threshold=0.25,  # 最小共视率
    device=device
)

# 方式2：手动指定硬性节点
# 如果手动指定，代码会自动修正为50的倍数
# 例如：指定[0, 51, 102, 152]会被修正为[0, 50, 100, 150]
nodes = [0, 50, 100, 150]  # 必须是50的倍数
keyframes = get_segmented_keyframes(
    M_overlap=M_overlap,
    M_action=M_action,
    nodes=nodes,
    D_target=0.1,
    overlap_threshold=0.25,
    device=device
)
```

## 依赖关系

- `core.geometry`: 几何变换辅助函数（`_is_tensor`, `_to_tensor`等）
- `utils.io`: 数据加载函数（`load_pose_data`, `load_all_depth_maps`）

## 向后兼容性

为了保持向后兼容性，`core/__init__.py` 中重新导出了所有 `mapanything_sampling` 的功能，因此以下两种导入方式都可以工作：

```python
# 旧方式（仍然支持）
from core.sampling_pipeline import build_overlap_matrix
from core.covisibility_gpu import compute_score_for_pair

# 新方式（推荐）
from mapanything_sampling import build_overlap_matrix, compute_score_for_pair
```

## 相关文档

- `USAGE_NEW_COVISIBILITY.md`: 共视检测使用文档
- `SEGMENTED_SAMPLING_README.md`: 分段采样使用文档
- `HARD_NODES_FIX.md`: 硬性节点修复说明

