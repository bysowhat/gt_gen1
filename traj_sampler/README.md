# 轨迹关键帧采样代码库

本代码主要用于从原始相机轨迹中进行稀疏采样。核心算法整合了**共视率检测**和**SE(3)运动度量**，通过这两个约束确保采样帧既具有足够的视野重叠，又有合理的运动变化。

## 核心功能

### 主要特性

1. **基于深度重投影的共视检测**（MapAnything方法）
   - 使用深度图进行精确的共视率计算
   - 支持GPU加速，高效处理大规模数据
   - 动态阈值适应不同深度范围

2. **SE(3)运动度量**
   - 同时考虑平移和旋转的运动成本
   - GPU并行计算所有帧对的SE(3)距离

3. **分段关键帧采样**
   - 整合硬性节点约束和双重几何条件
   - 自动生成或手动指定关键节点
   - 确保采样帧满足共视率和运动成本双重约束

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**：
- `numpy`, `scipy`, `opencv-python`, `matplotlib`, `pyyaml`
- `torch>=1.9.0` (可选，用于GPU加速)

### 2. 准备数据

确保数据目录结构如下：
```
traj_demo/imgs/
├── 0_rgb.jpg          # RGB图像
├── 0_depth.npz        # 深度图
├── 1_rgb.jpg
├── 1_depth.npz
├── ...
└── info.npy            # 位姿信息（包含cam_pose_list和cam_quat_list）
```

### 3. 运行分段采样（推荐）

```bash
# 使用默认配置
python3 test_segmented_sampling.py --config configs/exp_segmented_sampling.yaml

# 指定数据路径
python3 test_segmented_sampling.py --config configs/exp_segmented_sampling.yaml --data_root traj_demo/imgs

# 指定输出路径
python3 test_segmented_sampling.py --config configs/exp_segmented_sampling.yaml --output output/my_result.png
```

### 4. 输出结果

运行后会生成以下文件：

- **`output/sampled_indices.json`**: 采样的关键帧索引列表（供MapAnything工程师使用）
- **`output/sampling_interface_doc.json`**: 接口文档（说明坐标系约定和文件路径约定）
- **`output/segmented_sampling.png`**: 矩阵分析图（包含共视率矩阵、SE(3)距离矩阵、3D轨迹可视化）
- **`output/segmented_sampling_stats.png`**: 统计图（共视率分布、SE(3)距离分布、采样路径指标）

## 目录结构

```
YUV/
├── mapanything_sampling/      # 核心算法模块（推荐使用）
│   ├── geometry_ops.py        # GPU几何运算（3D点生成、投影、深度采样）
│   ├── covisibility_gpu.py    # 共视性校验核心（深度一致性检查、共视率计算）
│   ├── action_metrics.py      # SE(3)运动度量（SE(3)距离计算）
│   ├── sampling_pipeline.py   # 分段采样策略（整合硬性节点约束和双重几何条件）
│   └── README.md              # 模块详细文档
│
├── test_segmented_sampling.py # 分段采样主程序（推荐使用）
│
├── configs/                    # 配置文件
│   └── exp_segmented_sampling.yaml  # 分段采样配置（推荐）
│
├── core/                       # 核心工具库
│   ├── geometry.py             # 几何变换辅助函数
│   └── covisibility.py         # 旧的共视检测（已废弃，保留用于兼容）
│
├── utils/                      # 工具函数
│   ├── io.py                   # 数据加载函数
│   └── visualizer.py           # 可视化工具
│
├── samplers/                   # 旧的采样器实现（已废弃，保留用于兼容）
│   └── ...
│
└── generate_pairs.py           # 旧的采样对生成脚本（已废弃，保留用于兼容）
```

## 核心模块说明

### mapanything_sampling/ - 核心算法模块

#### 1. geometry_ops.py - GPU几何运算模块

提供GPU加速的几何原语：
- `unproject_depth_to_world()`: 将深度图反投影到3D世界坐标系
- `project_world_to_pixel()`: 将3D世界点投影到2D像素坐标系
- `sample_depths_at_reproj()`: 在重投影位置采样深度值
- `inverse_transform_4x4()`: 计算4x4变换矩阵的逆

#### 2. covisibility_gpu.py - 共视性校验核心

实现MapAnything的深度重投影验证方法：
- `check_depth_consistency()`: 深度一致性检查（动态阈值）
- `compute_score_for_pair()`: 计算两帧之间的共视率分数
- `in_image_mask()`: 检查2D点是否在图像边界内

**共视率计算原理**：
- 将参考帧的深度图反投影到3D世界坐标
- 将3D点投影到目标帧的像素坐标系
- 检查重投影深度与目标帧深度的一致性
- 使用动态阈值：`threshold = C_abs + C_rel * D_expected - log(0.5) * C_temp`

#### 3. action_metrics.py - SE(3)运动度量模块

计算帧之间的SE(3)距离：
- `compute_se3_distance()`: 计算两个位姿之间的SE(3)距离
- `build_action_matrix()`: GPU并行构建所有帧对的SE(3)距离矩阵

**SE(3)距离公式**：
```
SE(3)距离 = w_trans * ||t|| + w_rot * |theta|
```
其中：
- `t` 是相对变换的平移向量
- `theta` 是旋转角度（通过轴角表示计算）
- `w_trans` 和 `w_rot` 是权重，用于平衡平移和旋转

#### 4. sampling_pipeline.py - 分段采样策略

整合硬性节点约束和双重几何条件：
- `build_overlap_matrix()`: 构建共视率矩阵（GPU优化）
- `get_segmented_keyframes()`: 分段关键帧采样
- `get_adaptive_indices()`: 根据共视率矩阵生成自适应采样索引

**分段采样算法流程**：

1. **分段任务**：按硬性节点（例如 `[0, 50, 100, 152]`）将轨迹分段
2. **双重约束掩码**：生成过滤矩阵，要求同时满足：
   - 共视率 > `overlap_threshold`
   - SE(3)距离 >= `D_target`
3. **图搜索**：在每个段内，从起始节点到结束节点搜索有效路径

## 配置说明

### 分段采样配置（configs/exp_segmented_sampling.yaml）

```yaml
# 数据路径
data_root: traj_demo/imgs
output_dir: output

# 设备配置
use_gpu: true
device: null  # null表示自动选择，或指定 "cuda", "cpu", "cuda:0" 等

# 共视率计算参数（MapAnything官方参数）
covisibility:
  depth_assoc_error_thres: 0.1      # C_abs: 绝对误差阈值（米）
  depth_assoc_rel_error_thres: 0.005 # C_rel: 相对误差阈值
  depth_assoc_error_temp: 0.1        # C_temp: 误差温度参数
  denominator_mode: "valid_target_depth"  # 分母模式
  min_depth: 0.04  # 最小有效深度（米）
  load_overlap_matrix: null  # 加载已计算的共视率矩阵（.npy文件路径）
  save_overlap_matrix: null  # 保存共视率矩阵（.npy文件路径）

# SE(3)距离计算参数
se3_action:
  w_trans: 1.0  # 平移权重
  w_rot: 1.0    # 旋转权重

# 分段采样参数
sampling:
  nodes: null  # 硬性节点（null表示自动生成，或指定列表如 [0, 50, 100, 150]）
  overlap_threshold: 0.25  # 最小共视率阈值
  D_target: 0.1            # 最小运动成本阈值（SE(3)距离）
  max_path_length: 16      # 最大路径长度（软限制）
  max_jump_distance: 10    # 最大跳跃距离（帧数）

# 可视化参数
visualization:
  output_matrix: "output/segmented_sampling.png"
  output_statistics: "output/segmented_sampling_stats.png"
  output_indices: "output/sampled_indices.json"
```

### 参数调优建议

#### 共视率参数
- `depth_assoc_error_thres` (C_abs): 绝对误差阈值，默认0.1米
- `depth_assoc_rel_error_thres` (C_rel): 相对误差阈值，默认0.005
- `depth_assoc_error_temp` (C_temp): 误差温度参数，默认0.1

**动态阈值公式**：
```
threshold = C_abs + C_rel * D_expected - log(0.5) * C_temp
```

#### SE(3)距离权重
- `w_trans`: 平移权重（默认1.0）
  - 增大：更重视平移距离
  - 减小：更重视旋转角度
- `w_rot`: 旋转权重（默认1.0）
  - 增大：更重视旋转角度
  - 减小：更重视平移距离

**建议**：根据应用场景调整。如果相机运动主要是平移，可以增大 `w_trans`；如果主要是旋转，可以增大 `w_rot`。

#### 双重约束阈值
- `D_target`: 最小运动成本阈值
  - 增大：要求更大的运动幅度（更稀疏的采样）
  - 减小：允许更小的运动幅度（更密集的采样）
- `overlap_threshold`: 最小共视率阈值
  - 增大：要求更高的共视率（更保守的采样）
  - 减小：允许更低的共视率（更激进的采样）

**建议**：
- 对于训练数据：`D_target=0.05-0.1`, `overlap_threshold=0.25-0.3`
- 对于测试数据：`D_target=0.1-0.2`, `overlap_threshold=0.3-0.4`

#### 硬性节点
- `nodes: null`: 自动生成（推荐）
  - 自动生成规则：总是包含第一帧(0)，生成所有50的倍数节点
  - 例如：153帧 -> `[0, 50, 100, 150]`
- `nodes: [0, 50, 100, 150]`: 手动指定
  - 如果手动指定，代码会自动修正为50的倍数
  - 例如：指定`[0, 51, 102, 152]`会被修正为`[0, 50, 100, 150]`

## 使用示例

### 示例1：基本使用（自动生成硬性节点）

```bash
python3 test_segmented_sampling.py --config configs/exp_segmented_sampling.yaml
```

### 示例2：使用已计算的共视率矩阵（加速）

```yaml
# 在配置文件中设置
covisibility:
  load_overlap_matrix: "output/overlap_matrix.npy"  # 加载已计算的矩阵
  save_overlap_matrix: "output/overlap_matrix.npy"  # 保存矩阵供下次使用
```

### 示例3：手动指定硬性节点

```yaml
sampling:
  nodes: [0, 50, 100, 150]  # 手动指定硬性节点
```

### 示例4：调整采样密度

```yaml
sampling:
  D_target: 0.05           # 减小阈值，更密集的采样
  overlap_threshold: 0.2   # 减小阈值，更激进的采样
```

## 编程接口

### Python API

```python
from mapanything_sampling import (
    build_overlap_matrix,
    build_action_matrix,
    get_segmented_keyframes
)
from utils.io import load_pose_data, load_all_depth_maps

# 加载数据
poses, intrinsics, img_h, img_w, num_frames = load_pose_data(
    data_root, device=device
)
depth_maps = load_all_depth_maps(data_root, num_frames, device=device)

# 计算共视率矩阵
M_overlap = build_overlap_matrix(
    poses_all=poses,
    depths_all=depth_maps,
    K_all=intrinsics,
    img_h=img_h,
    img_w=img_w,
    depth_assoc_error_thres=0.1,
    depth_assoc_rel_error_thres=0.005,
    depth_assoc_error_temp=0.1,
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

# 执行分段采样
keyframes = get_segmented_keyframes(
    M_overlap=M_overlap,
    M_action=M_action,
    nodes=None,  # 自动生成硬性节点
    D_target=0.1,
    overlap_threshold=0.25,
    max_path_length=16,
    max_jump_distance=10,
    device=device
)

print(f"Sampled {len(keyframes)} keyframes: {keyframes}")
```

## 输出文件说明

### sampled_indices.json

采样的关键帧索引列表，供MapAnything工程师使用：

```json
[
  0,
  5,
  12,
  18,
  ...
]
```

### sampling_interface_doc.json

接口文档，说明坐标系约定和文件路径约定：

```json
{
  "sampling_engineer_output": {
    "sampled_indices_file": "sampled_indices.json",
    "total_sampled_frames": 25,
    "total_original_frames": 153,
    "sampling_rate": "16.34%"
  },
  "coordinate_system_convention": {
    "description": "OpenCV convention: T_cam_to_world (Camera to World)",
    "pose_format": "4x4 transformation matrix"
  },
  "file_path_convention": {
    "rgb_image_pattern": "{data_root}/{frame_index}_rgb.jpg",
    "depth_map_pattern": "{data_root}/{frame_index}_depth.npz",
    "pose_info_file": "{data_root}/info.npy"
  }
}
```

## 性能优化

### GPU加速

- **共视率矩阵计算**：GPU并行，153帧约需 10-30秒（取决于GPU）
- **SE(3)距离矩阵计算**：GPU并行，153帧约需 < 1秒
- **分段采样**：CPU实现，153帧约需 < 1秒

### 矩阵缓存

首次运行后，可以保存共视率矩阵供后续使用：

```yaml
covisibility:
  save_overlap_matrix: "output/overlap_matrix.npy"
```

下次运行时加载：

```yaml
covisibility:
  load_overlap_matrix: "output/overlap_matrix.npy"
```

这样可以跳过耗时的共视率矩阵计算，直接进行采样。

## 旧版采样器（已废弃）

本代码库还保留了旧的采样器实现（`samplers/`目录），用于向后兼容。这些采样器使用基于Pose的共视检测方法，不如新的基于深度重投影的方法精确。

### 旧版采样器类型

1. **固定时间步长** (`TimeSampler`)
2. **空间采样** (`SpatialSampler`)
3. **自适应采样** (`AdaptiveSampler`)
4. **SE(3)动作幅度采样** (`SE3ActionSampler`)
5. **共视-运动混合采样** (`HybridSampler`)

**注意**：推荐使用新的分段采样算法（`test_segmented_sampling.py`），它基于MapAnything方法，更加精确和高效。

## 相关文档

- `mapanything_sampling/README.md`: 核心模块详细文档
- `USAGE_NEW_COVISIBILITY.md`: 共视检测使用文档
- `SEGMENTED_SAMPLING_README.md`: 分段采样详细文档
- `CODE_REORGANIZATION.md`: 代码重组说明

## 常见问题

### Q: 如何选择使用CPU还是GPU？

A: 在配置文件中设置：
```yaml
use_gpu: true  # 启用GPU（如果有CUDA）
device: null    # 自动选择，或手动指定 "cuda", "cpu", "cuda:0"
```

### Q: 共视率矩阵计算很慢怎么办？

A: 
1. 使用GPU加速（设置 `use_gpu: true`）
2. 首次计算后保存矩阵，后续直接加载（设置 `save_overlap_matrix` 和 `load_overlap_matrix`）

### Q: 采样结果太少/太多怎么办？

A: 调整双重约束阈值：
- 采样太少：减小 `D_target` 或 `overlap_threshold`
- 采样太多：增大 `D_target` 或 `overlap_threshold`

### Q: 硬性节点如何设置？

A: 
- 推荐使用自动生成（`nodes: null`）
- 如果需要手动指定，确保节点是50的倍数（代码会自动修正）
