# render/ — 并行环境快速渲染

参考 `va_simulation/va_sim23_multi.py` 的并行环境思路：一次性铺 N 个 env
（warehouse 背景 + 机械臂 + 工件），用左右目 `TiledCamera` 批量读图，对工件相对
机械臂的每个位姿渲染 **左右目 RGB + 深度**。机械臂统一置于 cfg 里的固定初始关节角
`retract_config`。

## 依赖
- conda 环境 `env_isaaclab`（已装 isaacsim + isaaclab）。
- `pxr` / converters 只能在 `AppLauncher.app` 启动后用，代码已按此组织。

## 渲染内容
- 环境：`full_warehouse.usd`（默认从 Omniverse S3 拉，可用 `--warehouse-usd` 改本地）。
- 机械臂：`configs/default.yaml → robot.cfg_path` 的 cuRobo yml 里的 `urdf_path`，
  运行时 URDF→USD（缓存于 `render/_assets_cache/ur12e/`），固定底座、关重力。
- 工件：`--obj` 指定的 watertight `.obj`，运行时 OBJ→USD（缓存于
  `render/_assets_cache/workpiece/`），纯视觉静态体。
- 位姿：`--seam-npy` 里的 `workpiece_pose7`（T_base←workpiece，米，wxyz），可多个；
  每个位姿一个并行 env。机械臂与工件按给定相对位姿摆在地面上，**不强制工件在地板之上**。
- 相机：左右目 `TiledCamera`，挂在 Link6 上（外参/内参为用户标定值，ros 约定，2208×1242）。
  深度用 `distance_to_image_plane`（针孔 z 深度，单位 m）。

## 用法
```bash
conda run -n env_isaaclab python render/render_seam.py \
    --obj '/media/a/upan/tempt/2/柱_1JdzFk001Mz34qC38vE3On_part_watertight.obj' \
    --seam-npy '/media/a/upan/tempt/2/柱_1JdzFk001Mz34qC38vE3On/seam_40.npy' \
    --out /tmp/render_out --headless
```
常用参数：`--num-envs` / `--max-envs`（并行数）、`--spacing`（环境间距 m）、
`--settle-steps`（读图前 step 帧数）、`--force-convert`（强制重转 USD）、
`--robot-cfg`（覆盖机器人 yml）。

## 输出结构
```
<out>/<obj_stem>/<seam_stem>/pose_{p}/
    left_rgb.png    left_depth.exr     # 左目
    right_rgb.png   right_depth.exr    # 右目
    meta.npy        # 关节角/内参/外参/工件位姿等元信息（dict）
```
深度 EXR 为 16-bit half float + ZIP 压缩（读取见 `depth_io`，需 OpenEXR 支持）。

## 文件
- `render_seam.py`：主入口（AppLauncher 样板、建并行场景、渲染、保存、分批）。
- `asset_convert.py`：URDF/OBJ→USD（带缓存）+ 定位 Link6 子路径。
- `depth_io.py`：RGB png / 深度 EXR 保存助手（搬自 va_simulation）。
