# MapAnything接口检查报告

本文档检查采样代码输出的数据格式，确保与MapAnything的要求一致。

## 检查结果

### ✅ 1. 相机内参 - 一致

**我们提供的数据**：
- 直接使用原始内参（`cam_intrinsic`）
- 图像尺寸：320 x 240像素
- 内参矩阵：`[[429.47, 0, 160], [0, 429.47, 120], [0, 0, 1]]`

**检查结果**：✅ 与参考实现（`traj_visualize.py`）一致

**注意事项**：
- 如果MapAnything预处理时调整了图像分辨率，需要相应缩放内参矩阵
- 缩放公式：`新内参 = 原始内参 × 缩放比例`

---

### ✅ 2. 深度图 - 一致

**我们提供的数据**：
- 直接使用原始深度图（`depth.npz`）
- 深度图尺寸：240 x 320像素
- 深度范围：约 0.01 ~ 1.02 米

**检查结果**：✅ 与参考实现（`traj_visualize.py`）一致

**注意事项**：
- 如果MapAnything预处理时调整了深度图分辨率，需要相应缩放

---

### ⚠️ 3. Pose坐标系 - 需要确认

**我们提供的数据**：
- Pose格式：OpenCV约定，`T_cam_to_world`（相机到世界）
- 通过 `transform_7_to_4x4()` 函数转换得到
- 4x4矩阵，表示从相机坐标系到世界坐标系的变换

**参考实现（`traj_visualize.py`）**：
- 点云可视化时使用了额外的坐标转换：`T_cam_to_world @ pose_cam_to_usd`
- 这个转换是**可视化专用的**，用于Open3D显示
- **不影响我们传给MapAnything的数据格式**

**关键差异**：
- 我们提供：`T_cam_to_world`（OpenCV约定）
- 可视化使用：`T_cam_to_world @ pose_cam_to_usd`（USD约定，仅用于显示）

**需要确认**：
- ⚠️ MapAnything期望的坐标系格式是什么？
  - 如果是OpenCV约定：直接使用我们提供的数据 ✅
  - 如果是USD约定：需要应用 `pose_cam_to_usd` 转换

---

## 数据格式总结

### 我们提供的数据

| 数据类型 | 格式 | 尺寸/分辨率 | 说明 |
|---------|------|------------|------|
| RGB图像 | JPG | 320x240 | 原始分辨率 |
| 深度图 | NPZ | 240x320 | 原始分辨率 |
| 内参矩阵 | 3x3 | - | 对应320x240分辨率 |
| Pose矩阵 | 4x4 | - | OpenCV约定（T_cam_to_world） |

### 坐标系说明

**我们使用的坐标系**：OpenCV约定
- Pose矩阵：`T_cam_to_world`
- 含义：将点从相机坐标系转换到世界坐标系
- 验证公式：`P_world = T_cam_to_world @ P_cam`

**可视化使用的坐标系**：USD约定（仅用于显示）
- 额外应用了 `pose_cam_to_usd` 转换
- 这是Open3D显示需要的，不影响数据本身

---

## 重要提醒

### 1. 坐标系约定

**必须确认**：MapAnything期望的坐标系格式
- ✅ 如果是OpenCV约定：直接使用我们提供的数据
- ⚠️ 如果是USD约定：需要转换

**如何转换**（如果需要）：
```python
# USD坐标转换矩阵
pose_cam_to_usd = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

# 转换
T_usd = T_cam_to_world @ pose_cam_to_usd
```

### 2. 分辨率预处理

**如果MapAnything预处理了分辨率**：
- 图像/深度图被缩放 → 内参矩阵也必须相应缩放
- 缩放比例 = 新尺寸 / 原始尺寸

**缩放示例**：
```python
# 假设图像从320x240缩放到640x480
scale_w = 640 / 320  # 2.0
scale_h = 480 / 240  # 2.0

# 缩放内参
K_scaled = np.array([
    [fx * scale_w, 0,           cx * scale_w],
    [0,           fy * scale_h, cy * scale_h],
    [0,           0,           1]
])
```

---

## 验证清单

### 采样代码端（已完成）✅

- [x] 提供 `sampled_indices.json` - 采样帧索引列表
- [x] 提供 `sampling_interface_doc.json` - 接口文档
- [x] 明确说明坐标系约定：OpenCV格式（T_cam_to_world）
- [x] 明确说明内参格式：原始分辨率（320x240）
- [x] 明确说明深度图格式：原始分辨率（240x320）

### MapAnything端（需要确认）⚠️

- [ ] **确认坐标系约定**：OpenCV还是USD？
- [ ] **确认分辨率处理**：是否预处理了图像/深度图？
- [ ] **确认内参缩放**：如果预处理了，内参缩放是否正确？
- [ ] **验证点云对齐**：最终的点云是否对齐正常？

---

## 使用示例

### 读取采样结果

```python
import json
import numpy as np
import cv2

# 1. 读取采样索引
with open('output/sampled_indices.json', 'r') as f:
    sampled_indices = json.load(f)
    # 结果：[0, 7, 13, 20, ...]

# 2. 读取接口文档
with open('output/sampling_interface_doc.json', 'r') as f:
    interface_doc = json.load(f)

# 3. 获取数据根目录
data_root = interface_doc['file_path_convention']['data_root']

# 4. 加载数据
for frame_idx in sampled_indices:
    # 加载RGB
    rgb = cv2.imread(f"{data_root}/{frame_idx}_rgb.jpg")
    
    # 加载深度图
    depth = np.load(f"{data_root}/{frame_idx}_depth.npz")['arr_0']
    
    # 加载Pose
    data_info = np.load(f"{data_root}/info.npy", allow_pickle=True).item()
    position = data_info['cam_pose_list'][frame_idx]
    quaternion = data_info['cam_quat_list'][frame_idx]
    
    # 转换为4x4矩阵（OpenCV格式）
    pose_7d = np.concatenate([position, quaternion])
    T_cam_to_world = transform_7_to_4x4(pose_7d)
    
    # 加载内参
    K = data_info['cam_intrinsic']  # 对应320x240分辨率
    
    # 如果MapAnything需要USD格式，进行转换
    # pose_cam_to_usd = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    # T_usd = T_cam_to_world @ pose_cam_to_usd
```

---

## 总结

### ✅ 已确认一致
1. **内参格式**：原始分辨率（320x240），与参考实现一致
2. **深度图格式**：原始分辨率（240x320），与参考实现一致

### ⚠️ 需要确认
1. **坐标系约定**：我们提供OpenCV格式，需要确认MapAnything期望的格式
2. **分辨率预处理**：如果MapAnything预处理了分辨率，需要确认内参缩放逻辑

### 📋 关键行动项
- 与MapAnything工程师确认坐标系约定（OpenCV vs USD）
- 如果预处理了分辨率，确认内参缩放是否正确
- 验证最终的点云对齐是否正常
