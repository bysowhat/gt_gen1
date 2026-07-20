# 采样结果接口说明

本文档说明采样代码输出的文件格式，以及如何将这些文件用于MapAnything推理。

## 输出文件

### 1. `sampled_indices.json` - 采样帧索引列表

**这是什么？**
- 一个JSON数组，包含所有需要用于训练的帧编号
- 例如：`[0, 7, 13, 20, 27, ...]` 表示选中了第0、7、13、20、27...帧

**格式示例**:
```json
[0, 7, 13, 20, 27, 34, 40, 45, 49, 50]
```

**重要说明**:
- 所有索引从0开始（第1帧是0，第2帧是1，以此类推）
- 这是采样算法的最终结果，MapAnything工程师需要根据这些索引加载对应的数据

### 2. `sampling_interface_doc.json` - 接口文档

**这是什么？**
- 一个JSON文件，包含所有接口约定的详细信息
- 包括：采样统计、坐标系说明、文件路径、数据格式等

**包含内容**:
- 总共有多少帧，采样了多少帧
- 坐标系约定（OpenCV格式）
- 文件路径规则
- 数据格式说明

## 关键约定（必须遵守）

### 坐标系约定

**简单说**：我们提供的Pose矩阵是OpenCV格式的，表示"从相机坐标系到世界坐标系"的变换。

**为什么重要？**
- 如果坐标系格式不对，3D点云会错位，MapAnything验证会失败
- 必须使用OpenCV约定：`T_cam_to_world`

**技术细节**（供参考）:
- Pose矩阵是4x4的矩阵
- 验证公式：`P_world = T_cam_to_world @ P_cam`

### 文件路径规则

**数据根目录**：`traj_demo/imgs`（可以在配置文件中修改）

**文件命名规则**：
- RGB图像：`{数据根目录}/{帧编号}_rgb.jpg`
- 深度图：`{数据根目录}/{帧编号}_depth.npz`
- Pose信息：`{数据根目录}/info.npy`（所有帧的Pose都在这里）

**实际例子**：
- 第0帧：
  - RGB：`traj_demo/imgs/0_rgb.jpg`
  - 深度：`traj_demo/imgs/0_depth.npz`
- 第50帧：
  - RGB：`traj_demo/imgs/50_rgb.jpg`
  - 深度：`traj_demo/imgs/50_depth.npz`

## 数据格式

### Pose矩阵（位姿）

**存储位置**：`info.npy`文件中
- `cam_pose_list`：位置信息 `[x, y, z]`
- `cam_quat_list`：旋转信息（四元数）`[qw, qx, qy, qz]`

**如何转换**：
- 使用函数 `transform_7_to_4x4([x, y, z, qw, qx, qy, qz])`
- 得到4x4矩阵，格式是OpenCV约定的 `T_cam_to_world`

### 相机内参

**存储位置**：`info.npy`文件中的`cam_intrinsic`
- 是一个3x3的矩阵
- 对应图像尺寸：320x240像素

### 图像和深度图

- **RGB图像**：320x240像素，JPG格式
- **深度图**：240x320像素，NPZ格式（numpy压缩文件）

## 工作流程

### 步骤1：运行采样代码

```bash
python3 test_segmented_sampling.py --config configs/exp_segmented_sampling.yaml
```

**输出文件**：
- `output/sampled_indices.json` - 采样帧索引
- `output/sampling_interface_doc.json` - 接口文档

### 步骤2：MapAnything工程师使用

**读取采样结果**：
```python
import json

# 读取采样索引
with open('output/sampled_indices.json', 'r') as f:
    sampled_indices = json.load(f)
    # 结果：[0, 7, 13, 20, ...]

# 读取接口文档
with open('output/sampling_interface_doc.json', 'r') as f:
    interface_doc = json.load(f)
    # 包含所有接口约定信息
```

**加载数据**：
```python
import numpy as np
import cv2

# 数据根目录
data_root = interface_doc['file_path_convention']['data_root']

# 遍历每个采样帧
for frame_idx in sampled_indices:
    # 1. 加载RGB图像
    rgb_path = f"{data_root}/{frame_idx}_rgb.jpg"
    rgb = cv2.imread(rgb_path)
    
    # 2. 加载深度图
    depth_path = f"{data_root}/{frame_idx}_depth.npz"
    depth = np.load(depth_path)['arr_0']
    
    # 3. 加载Pose（从info.npy中读取）
    data_info = np.load(f"{data_root}/info.npy", allow_pickle=True).item()
    
    # 获取位置和旋转
    position = data_info['cam_pose_list'][frame_idx]  # [x, y, z]
    quaternion = data_info['cam_quat_list'][frame_idx]  # [qw, qx, qy, qz]
    
    # 转换为4x4矩阵（OpenCV格式）
    pose_7d = np.concatenate([position, quaternion])
    T_cam_to_world = transform_7_to_4x4(pose_7d)
    
    # 4. 加载内参
    K = data_info['cam_intrinsic']  # 3x3矩阵，对应320x240分辨率
    
    # 5. 转换为MapAnything需要的格式
    # ... (根据MapAnything的具体要求)
```

## 注意事项

### ⚠️ 必须注意的事项

1. **索引从0开始**
   - 第1帧的索引是0，第2帧的索引是1
   - 所有索引都是整数

2. **坐标系必须一致**
   - 必须使用OpenCV约定：`T_cam_to_world`
   - 如果MapAnything需要其他格式，需要转换

3. **文件路径要正确**
   - 确保`data_root`路径正确
   - 文件命名必须符合规则：`{帧编号}_rgb.jpg` 和 `{帧编号}_depth.npz`

4. **数据格式要正确**
   - Pose矩阵必须是4x4，float32类型
   - 内参矩阵必须是3x3，float32类型

### 📋 检查清单

使用前请确认：
- [ ] `sampled_indices.json`格式正确（JSON数组）
- [ ] 所有索引都是整数，从0开始
- [ ] `sampling_interface_doc.json`包含完整信息
- [ ] 坐标系约定明确（OpenCV: T_cam_to_world）
- [ ] 文件路径规则清晰
- [ ] 数据格式说明完整

## 常见问题

**Q: 如果MapAnything预处理了图像分辨率怎么办？**
A: 如果图像被缩放了，内参矩阵也需要相应缩放。缩放比例 = 新尺寸 / 原始尺寸（320x240）

**Q: 坐标系格式不对怎么办？**
A: 我们提供的是OpenCV格式（T_cam_to_world）。如果MapAnything需要其他格式，需要应用相应的坐标转换。

**Q: 如何验证数据是否正确？**
A: 可以检查：
- 索引是否在有效范围内（0到总帧数-1）
- 文件路径是否存在
- Pose矩阵是否正确（4x4，float32）
- 坐标系是否正确（OpenCV约定）
