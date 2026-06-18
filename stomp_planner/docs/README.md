# GT Overall — 焊缝观测位姿与观测轨迹 GT 生成流水线

本目录是 `gt_overall` 项目的详细设计文档。项目运行于 **NVIDIA Isaac Lab / Isaac Sim** 仿真环境，
为「视觉引导机器人焊缝检测」任务批量生成**地面真值（Ground Truth, GT）数据**。

## 项目要解决的问题

给定一个待焊工件（USD 网格 + 若干条焊缝的几何描述），机器人需要用手眼相机把每条焊缝**完整观测**一遍。
由于相机视野（FOV）有限、存在遮挡、机械臂有关节限位与碰撞约束，一条焊缝通常**不能一次拍完**，
需要规划出**若干个观测点（viewpoint）**，依次移动到每个观测点拍摄，拼接出整条焊缝。

流水线分两个阶段：

| 阶段 | 入口脚本 | 求解器 | 产出 |
|------|----------|--------|------|
| **Stage 1 — Pose（观测点）** | `run_gt_pose.sh` | 进化策略 ES（`optimizer_pose.py`） | 每条焊缝的一组 6-DOF 最优相机观测位姿 |
| **Stage 2 — Traj（观测轨迹）** | `run_gt_traj.sh` | STOMP（`stomp_traj.py`） | 依次访问各观测点的多条最优运动轨迹 |

两个阶段是**严格串行**的：Pose 阶段先确定「在哪些点位拍」，Traj 阶段再规划「怎么动过去拍」。
Traj 阶段直接消费 Pose 阶段的输出（`output/<folder>/pose/*.pkl`）。

```
   工件数据 (.usd + 每条焊缝一个 .pkl)
            │
   ┌────────▼─────────┐   run_gt_pose.sh
   │  Stage 1: Pose   │   ES 6-DOF 位姿优化（optimizer_pose.py + scene_pose.py）
   │  生成观测点       │
   └────────┬─────────┘
            │  output/<folder>/pose/<seam>.pkl   （每条焊缝一组观测位姿 + 关节角）
            │  output/<folder>/path.json         （usd_path / pc_path 记录）
   ┌────────▼─────────┐   run_gt_traj.sh
   │  Stage 2: Traj   │   STOMP 轨迹优化（stomp_traj.py + scene_traj.py）
   │  生成观测轨迹     │
   └────────┬─────────┘
            │  output/<folder>/traj/<seam>.pkl   （每条焊缝多条完整轨迹）
            ▼
        GT 数据集
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [01_整体流程与调度.md](01_整体流程与调度.md) | 三个 shell 入口、controller 调度器、子进程隔离、超时/黑名单、断点续跑与幂等性 |
| [02_Pose阶段_观测点生成.md](02_Pose阶段_观测点生成.md) | `optimize_pose.py` / `optimizer_pose.py` / `scene_pose.py` 的 ES 算法、状态机、代价函数 |
| [03_Traj阶段_轨迹生成.md](03_Traj阶段_轨迹生成.md) | `optimize_traj.py` / `stomp_traj.py` / `scene_traj.py` 的 STOMP 算法、观测点排序、代价函数 |
| [04_数据格式.md](04_数据格式.md) | 输入 seam pkl、Pose 输出、Traj 输出、path.json 的字段与张量形状 |
| [05_运动学与数学库.md](05_运动学与数学库.md) | UR12e 解析 IK/FK、DH 参数、相机-法兰变换、四元数/旋转数学库、机器人 USD 配置 |
| [06_配置参数速查.md](06_配置参数速查.md) | `config_pose.py` / `config_traj.py` 全部超参数及其含义、调参建议 |

## 代码结构总览

```
gt_overall/
├── run_gt_pose.sh / run_gt_traj.sh / run_gt.sh / run_gt_split.sh   # shell 入口
│
├── main_controller_sp_pose.py      # Pose 阶段多 GPU 调度器（PoseGenerator）
├── main_controller_sp_traj.py      # Traj 阶段多 GPU 调度器（TrajGenerator）
├── main_controller_sp.py           # Pose+Traj 合并调度器（GTGenerator）
├── main_controller.py              # 单 GPU 顺序版
├── main_controller_sp_new.py       # 多 GPU + NVIDIA MPS 版
│
├── optimize_pose.py                # Pose 子进程入口（OptimizePose）：读数据→建场景→逐 seam 优化→存盘
├── optimizer_pose.py               # ★ ES 6-DOF 位姿求解器（核心计算，约 1450 行）
├── scene_pose.py                   # Pose 仿真场景：碰撞、FOV、遮挡、方向代价
├── config_pose.py                  # Pose 超参数
├── ik_cam_pose.py                  # Pose 阶段 UR12e 解析 IK/FK
├── opt_math_pose.py                # Pose 阶段四元数/旋转矩阵数学库
├── opt_utils_pose.py               # Pose 阶段向量构造/插值工具
│
├── optimize_traj.py                # Traj 子进程入口（OptimizeTraj）：读 pose 结果→排序→STOMP→存盘
├── stomp_traj.py                   # ★ STOMP 轨迹求解器
├── stomp_utils_traj.py             # STOMP 控制代价矩阵、有限差分、滤波矩阵
├── scene_traj.py                   # Traj 仿真场景：碰撞、视野代价
├── config_traj.py                  # Traj 超参数
├── ik_cam_traj.py                  # Traj 阶段 UR12e 解析 IK/FK（与 ik_cam_pose 同，仅 import 不同）
├── gt_math_traj.py                 # Traj 阶段数学库（与 opt_math_pose 子集相同）
│
├── robot/
│   ├── a1_cfg.py                   # Isaac Lab ArticulationCfg
│   └── a1_ur12e.usd / a1_ur12e_s.usd  # 机器人 USD（traj 用 a1_ur12e，pose 用 a1_ur12e_s）
│
├── open_data.py                    # 调试：查看 pkl 结构
├── dirs/                           # 输入文件夹快照/清理工具
├── utils/                          # 统计/拷贝工具
├── logger/                         # 各 slot 运行日志
└── skip_strategy.md                # 跳过/完成幂等性策略说明
```

> **重要约定**：所有数据根路径（`INPUT_ROOT`、`OUTPUT_ROOT` 等）在 `run_*.sh` 中硬编码；
> 子进程的 Isaac Sim 解释器路径 `/workspace/isaaclab/_isaac_sim/python.sh` 在 controller 中硬编码。
> 随机种子在 `optimizer_pose.py` 与 `stomp_traj.py` 中固定为 `seed=3`，保证结果可复现。
