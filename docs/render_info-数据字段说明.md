# `render_info.npy` 数据字段说明

本文档说明 `render/render_trajectory.py` 渲染整条采样轨迹时，写入每侧目录下
`render_info.npy` 的全部字段：各字段是什么、形状、单位/坐标系约定，以及生成它的代码位置。

---

## 1. 文件位置与整体结构

每渲染一条轨迹，输出目录布局如下：

```
<out>/<part_stem>/seam{sid}_{hand}{index}_traj{j}/
    left/   {k}_rgb.jpg   {k}_depth.exr  ...  render_info.npy   ← 本文档主题（左目）
    right/  {k}_rgb.jpg   {k}_depth.exr  ...  render_info.npy   ← 右目（结构相同，side="right"）
    _traj_meta.npy        ← 轨迹级 meta（见 §5）
    _DONE_...             ← 完成哨兵（断点续跑用）
```

- `k` = 关键帧行号（`0..L-1`，L 为下采样后关键帧总数）。
- `--observe-only` 时只渲染 `observe==1` 的帧，帧号 `k` 可能不连续，映射记录在 `frame_indices`。
- **`render_info.npy` 每侧一个，含该侧【所有帧】的信息**（不是每帧一个文件）。

### 读取方式

```python
import numpy as np
info = np.load("left/render_info.npy", allow_pickle=True).item()  # 注意 .item()
print(info.keys())
```

`render_info.npy` 存的是一个 **Python dict**（`allow_pickle=True` 序列化），
所以读回后要 `.item()` 取出字典。

生成代码：`render_job()` 中
```python
np.save(left_dir / "render_info.npy",
        build_side_render_info("left", job, records["left"]), allow_pickle=True)
```

---

## 2. 坐标系与约定（先读这段，再看字段）

| 约定项 | 取值 | 含义 |
|---|---|---|
| `cam_pose_arm_convention` | `"usd"` | `cam_pos_list/cam_quat_list` 是相机在 **arm(base) 系** 下的位姿，且已翻到 **USD 光学约定**（+X 右、+Y 下、+Z 朝前，即 ROS→USD 乘了 `diag([1,-1,-1,1])`）。 |
| `cam_pose_w_convention` | `"ros"` | `cam_pose_w_*` 是相机在 **世界系** 下的位姿，采用 **ROS 光学约定**（Isaac `quat_w_ros`）。 |
| `extrinsic_convention` | `"ros"` | 相机外参（挂载在 Link6 上的偏移）采用 ROS 约定。 |
| `extrinsic_ref_link` | `"Link6"` | 相机外参参考坐标系为机械臂 Link6。 |
| `depth_type` | `"distance_to_image_plane"` | 深度语义：到**像平面**的垂直距离（非到光心的欧氏距离/射线长）。 |

- **四元数一律 `wxyz` 顺序**（字段名带 `_quat` 或 `_wxyz`）。
- **base 系**：机械臂根 Link 坐标系（`root_link`），是训练/推理里相机位姿的落脚系。
- **世界系**：Isaac 仿真世界系，含 env 偏移与离地抬升，一般训练不直接用，仅供追溯/调试。
- 相机 arm 系位姿由世界系相机位姿与世界系 base 位姿反解得到：
  `T_arm = inv(T_base) @ T_cam @ diag([1,-1,-1,1])`（见 `_cam_pose_arm`）。

---

## 3. 逐帧字段（沿 axis0 按帧堆叠，F = 该侧渲染的帧数）

由 `frame_record_for_side()`（单帧单侧）产出，再由 `build_side_render_info()` 用
`np.stack(..., axis=0)` 沿帧维堆叠。**同一 `render_info` 内所有逐帧数组第 0 维都是 F，
且按渲染顺序（帧号升序）一一对应。**

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `frame_indices` | `(F,)` | int64 | 每帧对应的**关键帧行号 `k`**（即文件名 `{k}_rgb.jpg` 的 `k`）。observe-only 时不连续，这是帧↔文件↔轨迹行的映射键。 |
| `n_frames` | 标量 | int | 该侧渲染帧数 F（= `len(frame_indices)`）。 |
| `cam_pos_list` | `(F,3)` | float64 | 相机在 **arm(base) 系** 的平移（米），USD 光学约定。 |
| `cam_quat_list` | `(F,4)` | float64 | 相机在 **arm(base) 系** 的旋转四元数 `wxyz`，USD 光学约定。与 `cam_pos_list` 配对即相机外参（相对机械臂 base）。 |
| `cam_intrinsic` | `(F,3,3)` | float32 | 每帧相机内参矩阵 K（`[[fx,0,cx],[0,fy,cy],[0,0,1]]`，像素单位）。同侧各帧通常一致，逐帧存以防万一。 |
| `jointstates` | `(F,6)` | float32 | **本关键帧的 6 个关节角**（弧度），顺序见轨迹级 `joint_names`。即驱动机械臂到该帧姿态的关节值。 |
| `cam_2d` | `(F,)` | int64 | 本帧是否为 observe（观测）关键帧：`1`=是，`0`=否。来自采样数组第 7 列。（旧名 `observe`）|
| `camera_3d` | `(F,)` | int64 | 本帧是否为 goal（焊接到位/目标）关键帧：`1`=是，`0`=否。来自采样数组第 8 列。（旧名 `goal`）|
| `cam_pose_w_pos` | `(F,3)` | float64 | 相机在 **世界系** 的平移（米），ROS 约定。含 env 偏移+离地抬升，调试用。 |
| `cam_pose_w_quat_wxyz` | `(F,4)` | float64 | 相机在 **世界系** 的旋转 `wxyz`，ROS 约定。 |
| `base_pose_w_pos` | `(F,3)` | float64 | 机械臂 base（root_link）在**世界系**的平移。用于把世界系相机位姿还原到 base 系。 |
| `base_pose_w_quat_wxyz` | `(F,4)` | float64 | 机械臂 base 在**世界系**的旋转 `wxyz`。 |
| `z_lift` | `(F,)` | float64 | 本帧整组（工件+机械臂）为离地所做的抬升量（米）。世界系 Z 里已包含它，抵消它即得贴地前坐标。 |

> 说明：因每帧是不同 env 渲染，`base_pose_w_*`/`z_lift` 逐帧存的是该帧所在 env 的值；
> 而相机 arm 系位姿 `cam_pos_list/cam_quat_list` 已消去 env 偏移与抬升，是训练直接可用的干净外参。

---

## 4. 轨迹级常量字段（整条轨迹恒定，不随帧变化）

由 `build_side_render_info()` 直接写入，**同一 `render_info` 内为单值/单数组**。

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `side` | str | `"left"` 或 `"right"`，标明本文件属于哪一目。 |
| `traj_key` | dict | 轨迹唯一标识：`{"seam_id", "hand", "index", "traj_j"}`。焊缝号 / 手别(forehand\|backhand) / init pose 候选序号 / 该 pose 下第几条采样轨迹。 |
| `sampled_len` | int | 采样轨迹关键帧总数 L（= 下采样后 `(L,8)` 的 L）。注意 `n_frames`(F) ≤ L（observe-only 时 F<L）。 |
| `joint_names` | list[str] | 6 个关节名，顺序与 `jointstates` 列一一对应（取自 cuRobo robot yml 的 `cspace.joint_names`）。 |
| `workpiece_pose7` | `(7,)` float64 | 工件在 **base 系** 的位姿 `[x,y,z, qw,qx,qy,qz]`。整条轨迹工件固定不动。 |
| `seam_line_base` | `(N,3)` float64 或 `None` | 当前焊缝折线在 **base 系** 的采样点（N 个 3D 点）。取焊缝线失败时为 `None`。 |
| `cam_pose_arm_convention` | str | `"usd"`，见 §2。 |
| `cam_pose_w_convention` | str | `"ros"`，见 §2。 |
| `left_extrinsic_pos` | `(3,)` | 左目相机相对 Link6 的外参平移（取自 `configs/default.yaml`）。 |
| `left_extrinsic_quat_wxyz` | `(4,)` | 左目相机相对 Link6 的外参旋转 `wxyz`。 |
| `right_extrinsic_pos` | `(3,)` | 右目相机相对 Link6 的外参平移。 |
| `right_extrinsic_quat_wxyz` | `(4,)` | 右目相机相对 Link6 的外参旋转 `wxyz`。 |
| `extrinsic_convention` | str | `"ros"`，外参约定，见 §2。 |
| `extrinsic_ref_link` | str | `"Link6"`，外参参考 link。 |
| `depth_type` | str | `"distance_to_image_plane"`，深度语义，见 §2。 |

> `left_extrinsic_*` / `right_extrinsic_*` 是配置文件里的**标定外参（相对 Link6）**；
> 而逐帧 `cam_pos_list/cam_quat_list` 是**渲染实际用到的、相对 base 的相机位姿**——
> 二者一个是相机→Link6，一个是相机→base，通过 Link6 的 FK 联系起来，用途不同勿混。

---

## 5. 对照：同目录 `_traj_meta.npy`（轨迹级，非本文件但常一起用）

`_traj_meta.npy` 是**轨迹级** meta（左右目共用一份，存在轨迹根目录），字段：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `sampled` | `(L,8)` float | 原始采样数组 `[q1..q6, observe, goal]`。 |
| `observe` / `goal` | `(L,)` int64 | 整条轨迹的 observe/goal 标记（全 L 帧，非仅渲染帧）。 |
| `positions` | `(L,6)` float | 每关键帧的 6 关节角。 |
| `workpiece_pose7` | `(7,)` | 工件在 base 系位姿（同 render_info）。 |
| `seam_id`/`hand`/`index`/`traj_j` | 标量 | 轨迹标识（同 `traj_key`）。 |
| `joint_names` | list[str] | 关节名顺序。 |
| `n_obstacles` | int | 本轨迹场景中的障碍物个数。 |
| `status` | str | 采样/规划状态（如 success 等，来自 trajectories entry）。 |
| `goal_index` | | 目标索引（来自 trajectories entry）。 |
| `variant` | | 变体标签（来自 trajectories entry）。 |
| `observe_only` | bool | 本次渲染是否只渲了 observe 帧。 |
| `n_rendered` | int | 实际渲染的帧数（= 每侧 F）。 |
| `frame_indices` | `(F,)` int64 | 实际渲染帧号，与 render_info 的 `frame_indices` 一致。 |

---

## 6. 配套的图像文件

| 文件 | 格式 | 说明 |
|---|---|---|
| `{k}_rgb.jpg` | 图片（OpenCV BGR 写盘） | 该帧 RGB，`k` = 关键帧行号。`store_rgb` 内部把 RGB→BGR 后落盘。 |
| `{k}_depth.exr` | EXR（16-bit half float + ZIP） | 该帧深度图，语义为 `distance_to_image_plane`（到像平面的垂直距离，米）。`store_depth` squeeze 成 2D 后落盘。 |

`{k}` 与 `render_info` 里 `frame_indices` 的元素一一对应：
`frame_indices[i]` 就是 `render_info` 内第 `i` 帧（各逐帧数组第 `i` 行）对应的磁盘文件 `{k}_rgb.jpg`。

---

## 7. 典型用法示例

```python
import numpy as np, cv2, os
d = "left"
info = np.load(os.path.join(d, "render_info.npy"), allow_pickle=True).item()

F = info["n_frames"]
for i in range(F):
    k = int(info["frame_indices"][i])         # 磁盘帧号
    rgb   = cv2.imread(os.path.join(d, f"{k}_rgb.jpg"))            # (H,W,3) BGR
    depth = cv2.imread(os.path.join(d, f"{k}_depth.exr"),
                       cv2.IMREAD_UNCHANGED)                        # (H,W) float, 米

    K       = info["cam_intrinsic"][i]         # (3,3) 内参
    cam_pos = info["cam_pos_list"][i]          # (3,) base 系相机平移（USD 光学）
    cam_q   = info["cam_quat_list"][i]         # (4,) base 系相机旋转 wxyz
    q6      = info["jointstates"][i]           # (6,) 本帧关节角
    is_obs  = int(info["cam_2d"][i])           # 是否观测帧（旧名 observe）
    is_goal = int(info["camera_3d"][i])        # 是否目标帧（旧名 goal）

# 轨迹级常量
seam = info["seam_line_base"]                  # (N,3) 或 None，base 系焊缝折线
wp7  = info["workpiece_pose7"]                 # (7,) base 系工件位姿
```

---

## 8. 代码出处速查

| 内容 | 位置（`render/render_trajectory.py`） |
|---|---|
| 单帧单侧记录 | `frame_record_for_side()` |
| 堆叠帧 + 补轨迹级常量 | `build_side_render_info()` |
| 相机世界系→base 系(USD 光学)反解 | `_cam_pose_arm()` |
| 写 `render_info.npy` / `_traj_meta.npy` | `render_job()` 末尾 |
| 相机内外参来源（单一真源） | `_load_cameras_from_config()` ← `configs/default.yaml` |
| RGB/深度落盘 | `render/depth_io.py` 的 `store_rgb` / `store_depth` |
