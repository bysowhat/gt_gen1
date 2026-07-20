# 分段关键帧采样算法

## 概述

实现了整合硬性节点约束和双重几何条件的分段关键帧采样算法。该算法使用：
1. **共视率矩阵** (`M_overlap`)：基于深度重投影验证（MapAnything方法）
2. **SE(3)距离矩阵** (`M_action`)：评估帧之间的运动成本

## 核心模块

### 1. `core/action_metrics.py` - SE(3)运动度量模块

#### `compute_se3_distance(T_i, T_j, w_trans, w_rot)`
计算两个位姿之间的SE(3)距离。

**公式**：
```
SE(3)距离 = w_trans * ||t|| + w_rot * |theta|
```
其中：
- `t` 是相对变换的平移向量
- `theta` 是旋转角度（通过轴角表示计算）
- `w_trans` 和 `w_rot` 是权重，用于平衡平移和旋转

#### `build_action_matrix(T_all, w_trans, w_rot, device)`
构建所有帧对的SE(3)距离矩阵（GPU并行优化）。

**输入**：
- `T_all`: 所有帧的位姿列表，每个元素是 (4, 4) 矩阵
- `w_trans`: 平移权重（默认1.0）
- `w_rot`: 旋转权重（默认1.0）
- `device`: torch device

**输出**：
- `M_action`: (N, N) 矩阵，`M_action[i, j]` 表示从帧i到帧j的SE(3)距离

**GPU优化**：
- 使用PyTorch的广播机制，并行计算所有帧对的SE(3)距离
- 对于153帧，一次性计算 153×153 = 23,409 个距离值

### 2. `core/sampling_pipeline.py` - 分段采样策略

#### `get_segmented_keyframes(M_overlap, M_action, nodes, D_target, ...)`
执行分段关键帧采样，整合硬性节点约束和双重几何条件。

**算法流程**：

1. **分段任务**：
   - 按硬性节点（例如 `[0, 50, 100, 152]`）将轨迹分段
   - 每个段独立处理：`F_0 → F_50`, `F_50 → F_100`, `F_100 → F_152`

2. **双重约束掩码**：
   - 生成 `M_filter` 矩阵：
     ```
     M_filter[i, j] = 1 if (M_overlap[i, j] > overlap_threshold) 
                          AND (M_action[i, j] >= D_target)
     ```
   - 在GPU上并行生成，高效过滤有效帧对

3. **图搜索**：
   - 在每个段内，从起始节点到结束节点搜索有效路径
   - 使用贪心策略：优先选择满足双重约束且共视率最高的帧
   - 如果找不到满足约束的帧，放宽约束（只要求共视率）

**参数**：
- `M_overlap`: (N, N) 共视率矩阵
- `M_action`: (N, N) SE(3)距离矩阵
- `nodes`: 硬性节点列表（必须包含的帧索引）
- `D_target`: 最小运动成本阈值（默认0.1）
- `overlap_threshold`: 最小共视率阈值（默认0.25）
- `max_path_length`: 每个段的最大路径长度（默认16）

**输出**：
- `keyframes`: 最终采样的关键帧索引列表

## 使用方法

### 基本使用

```python
from core.sampling_pipeline import build_overlap_matrix, get_segmented_keyframes
from core.action_metrics import build_action_matrix
from utils.io import load_pose_data, load_all_depth_maps

# 加载数据
poses, intrinsics, img_h, img_w, num_frames = load_pose_data(data_root, device=device)
depth_maps = load_all_depth_maps(data_root, num_frames, device=device)

# 计算共视率矩阵（使用MapAnything方法）
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

# 计算SE(3)距离矩阵
M_action = build_action_matrix(
    T_all=poses,
    w_trans=1.0,
    w_rot=1.0,
    device=device
)

# 方式1：自动生成硬性节点（推荐）
# 硬性节点会根据实际帧数自动生成所有50的倍数
# 例如：153帧 -> [0, 50, 100, 150]
#      201帧 -> [0, 50, 100, 150, 200]
nodes = None  # None表示自动生成

# 方式2：手动指定硬性节点（会自动修正为50的倍数）
# nodes = [0, 50, 100, 150]

# 执行分段采样
keyframes = get_segmented_keyframes(
    M_overlap=M_overlap,
    M_action=M_action,
    nodes=nodes,
    D_target=0.1,  # 最小运动成本
    overlap_threshold=0.25,  # 最小共视率
    device=device
)
```

### 命令行测试

```bash
# 使用已计算的共视率矩阵（自动生成硬性节点）
python3 test_segmented_sampling.py \
    --data_root traj_demo/imgs \
    --use_gpu \
    --load_overlap_matrix output/test_optimized.npy \
    --D_target 0.05 \
    --overlap_threshold 0.25

# 手动指定硬性节点（会自动修正为50的倍数）
python3 test_segmented_sampling.py \
    --data_root traj_demo/imgs \
    --use_gpu \
    --load_overlap_matrix output/test_optimized.npy \
    --D_target 0.05 \
    --overlap_threshold 0.25 \
    --nodes 0 50 100 150

# 从头计算（包括共视率矩阵）
python3 test_segmented_sampling.py \
    --data_root traj_demo/imgs \
    --use_gpu \
    --D_target 0.1 \
    --overlap_threshold 0.25 \
    --save_overlap_matrix output/overlap_matrix.npy
```

## 参数调优

### SE(3)距离权重

- `w_trans`: 平移权重（默认1.0）
  - 增大：更重视平移距离
  - 减小：更重视旋转角度

- `w_rot`: 旋转权重（默认1.0）
  - 增大：更重视旋转角度
  - 减小：更重视平移距离

**建议**：根据应用场景调整。如果相机运动主要是平移，可以增大 `w_trans`；如果主要是旋转，可以增大 `w_rot`。

### 双重约束阈值

- `D_target`: 最小运动成本阈值
  - 增大：要求更大的运动幅度（更稀疏的采样）
  - 减小：允许更小的运动幅度（更密集的采样）

- `overlap_threshold`: 最小共视率阈值
  - 增大：要求更高的共视率（更保守的采样）
  - 减小：允许更低的共视率（更激进的采样）

**建议**：
- 对于训练数据：`D_target=0.05-0.1`, `overlap_threshold=0.25-0.3`
- 对于测试数据：`D_target=0.1-0.2`, `overlap_threshold=0.3-0.4`

## 性能

- **SE(3)距离矩阵计算**：GPU并行，153帧约需 < 1秒
- **分段采样**：CPU图搜索，153帧约需 < 1秒
- **总时间**：包括共视率矩阵计算，约需 2-5分钟（取决于GPU）

## 测试结果示例

对于153帧的数据：
- 硬性节点：`[0, 51, 102, 152]`
- 双重约束：`overlap_threshold=0.25`, `D_target=0.05`
- 采样结果：41个关键帧
- 有效帧对：656/23409 (2.8%)

## 注意事项

1. **硬性节点**：必须包含首尾帧（0 和 num_frames-1）
2. **共视率矩阵**：建议使用动态阈值（`depth_assoc_rel_error_thres > 0`）
3. **GPU加速**：强烈推荐使用GPU，可显著提升计算速度
4. **内存优化**：对于大矩阵，可以使用 `chunk_size` 参数分块处理

