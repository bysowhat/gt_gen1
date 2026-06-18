# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个为视觉引导机器人焊缝检测任务生成**地面真值（GT）数据**的流水线。运行于 NVIDIA Isaac Lab / Isaac Sim 仿真环境，输入焊缝工件数据，输出：
- **Stage 1 (pose)**：每条焊缝的最优相机观测位姿（6-DOF）
- **Stage 2 (traj)**：访问各观测点的最优 STOMP 运动轨迹

## 运行命令

**前提**：必须在已安装 Isaac Lab 的 Python 环境中运行（无标准 requirements.txt，依赖 `omni.isaac.lab`、PyTorch、NumPy）。

```bash
# 多 GPU 并行（推荐，8 GPU × 2 任务）
python main_controller_sp.py

# 单 GPU 顺序
python main_controller.py

# 多 GPU + NVIDIA MPS（需要 sudo）
python main_controller_sp_new.py

# 直接调用单阶段（供调试，通常由 controller 作为子进程调用）
python optimize_pose.py --headless --input_folder <in> --output_folder <out> \
       --sub_folder_pose pose --sub_folder_traj traj --device cuda
python optimize_traj.py --headless --output_folder <out> \
       --sub_folder_pose pose --sub_folder_traj traj --device cuda
```

**工具脚本**：
```bash
python open_data.py                  # 统计 input/ vs completed/ 中的 pkl 数量
python dirs/dir_save.py             # 快照当前输入文件夹列表
python dirs/dir_load.py             # 清理不在 A_folders.txt 中的文件夹（rmtree 默认注释）
```

> **注意**：所有数据路径（`root_dir`、`input_root`、`output_root`）在各 controller 文件中硬编码，修改时需直接编辑源文件。

## 架构概览

### 两阶段流水线

```
main_controller_sp.py (GTGenerator)
   扫描 input_root → 校验文件夹（需有 .pkl + .usd）
   调度子进程跨 NUM_GPUS × PER_GPU_JOBS 个槽位并行执行
   │
   ├─► subprocess: optimize_pose.py → OptimizerPose（ES 采样优化）
   │      输出: output/<folder>/pose/<seam>.pkl + path.json
   │
   └─► subprocess: optimize_traj.py → StompTraj（STOMP 轨迹优化）
          输出: output/<folder>/traj/<seam>.pkl
          成功后: shutil.move(input_folder → completed/)
```

### 核心模块职责

| 文件 | 职责 |
|------|------|
| `optimizer_pose.py` | 进化策略 6-DOF 相机位姿求解器（主要计算逻辑） |
| `stomp_traj.py` | STOMP 轨迹优化器 |
| `scene_pose.py` / `scene_traj.py` | Isaac Lab 仿真场景（机器人、工件、碰撞传感器、RayCaster） |
| `config_pose.py` / `config_traj.py` | 超参数配置（代价权重、关节限位、迭代次数等） |
| `ik_cam_pose.py` / `ik_cam_traj.py` | UR12e 机械臂解析 IK/FK（含 DH 参数和相机-法兰变换） |
| `robot/a1_cfg.py` | Isaac Lab `ArticulationCfg`，加载 `robot/a1_ur12e.usd` |

### 关键设计决策

- **子进程隔离**：每个（文件夹, 阶段）启动独立 Isaac Sim 进程，因为 `AppLauncher` 是进程全局单例，同时保证 GPU 显存干净回收。
- **并行 ES 优化**：`OptimizerPose` 用 `num_batches × (num_randoms_new + num_randoms_old)` 个并行 Isaac Lab 环境同时评估代价样本；`StompTraj` 同理。
- **幂等性**：`optimize_pose.py` 跳过已存在 `pose/` 的文件夹；`optimize_traj.py` 跳过已存在 `traj/` 的文件夹，若无 `pose/` 则删除整个输出文件夹，重跑安全。
- **水平/垂直工件**：`config_pose.py` 和 `config_traj.py` 各维护两套关节限位，由 seam 数据中的 `horizontal` 标志选择。
- **随机种子固定**：`optimizer_pose.py` 和 `stomp_traj.py` 均硬编码 `seed=3`。

### 代价函数构成

**Pose 阶段**（权重在 `ConfigurationPose` 中）：碰撞(25)、视觉旋转(1)、视觉位置(1)、工件内部(19)、视野空间(21)、遮挡(22)、方向(23)

**Traj 阶段**（权重在 `ConfigurationTraj` 中）：控制(0.1, 加速度)、碰撞(20)、视觉(1)、可见性(1)、方向(0.25)

## 已知问题

- `optimize_traj.py` 约 132 行在 `process()` 内部引用裸变量 `config`（而非 `self.config`），若不在 `__main__` 上下文中调用可能导致 `NameError`。
- `ScenePoseCfg.piece` 在类定义时读取 `Configuration().usd_path`，`ScenePose.__init__` 会动态覆盖——重构时注意初始化顺序。
