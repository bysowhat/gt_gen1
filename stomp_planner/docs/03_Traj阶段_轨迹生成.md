# 03 — Traj 阶段：观测轨迹生成

Traj 阶段读取 Pose 阶段为每条焊缝确定的若干**观测点（目标关节角）**，规划机械臂从起始位姿依次访问所有观测点的
**关节空间运动轨迹**。求解器是 **STOMP**（Stochastic Trajectory Optimization for Motion Planning），
同样在 Isaac Lab 中并行采样、并行评估。每条焊缝产出 `num_batch(24)` 条 GT 轨迹（同一任务的多样解）。

涉及文件：`optimize_traj.py`（入口）、`stomp_traj.py`（STOMP 核心）、`stomp_utils_traj.py`（数学工具）、
`scene_traj.py`（仿真+代价）、`config_traj.py`（超参）、`ik_cam_traj.py`（IK/FK）。

---

## 1. 入口 `optimize_traj.py`

`OptimizeTraj.run()`：
1. 若 `output/<folder>/pose/` 不存在 → `shutil.rmtree(output/<folder>)` 清理残留并返回（前置缺失）；
2. 读 `path.json` 拿 `usd_path`；
3. 载入 `pose/*.pkl`（每条 seam 是一个 `output_single` 的 list），应用 `blacklist_traj.txt`；
4. 创建 `SceneTraj`（只建一次）；
5. `process()` 逐 seam。

### 1.1 观测点排序（最短路径）`process()`

对一条 seam 的每个数据项：
```python
fixed_pts = data["joint"]   # (N, M, 6) —— N 个观测点位姿快照组，M 个观测点，6 关节
```
- 枚举 M 个观测点的**全部 M! 种访问顺序**：`fixed_pts → (N, M!, M, 6)`；
- 在每种顺序前面拼上机械臂**起始关节角** `scene.initial_joint_pos`：`→ (N, M!, M+1, 6)`；
- 计算每种顺序在关节空间的总路径长度 `Σ ‖相邻点差‖`，对每个 N 取最短顺序 `→ (N, M, 6)`；
- 把 N 条复制/截断到 `num_batch(24)` 条：`fixed_pts = repeat(...)[:num_batch]`。

即：先用「关节空间最短遍历」决定**访问观测点的顺序**，再为这一定序逐段做 STOMP。

### 1.2 分段 STOMP 拼接

```python
scene.reset(robot_pose, seam_median)
stomp = StompTraj(config, scene, device)
for j in range(M):                       # 相邻观测点对 (起点→点1→点2→…)
    path = fixed_pts[:, j:j+2]           # (B, 2, 6) 段两端点
    trajectory, _ = stomp.solve(path, has_vision)
    # 第 0 段保留全部时间步；后续段去掉首点（与上一段末点重合）后拼接
    idx_3d_part[-1] = 1                  # 标记该段末步是一个「观测点」
trajectory = cat(所有段, dim=-1)         # (B, D=6, T_total)
idx_3d     = cat(所有段标记, dim=-1)     # (T_total,)  哪些时间步是观测点
```
> 注：每段单独调用 `stomp.solve` 优化一段；`InitializeTrajectory` 内部其实也支持「多 transfer 段一次性优化」，
> 但 `optimize_traj.py` 当前是**逐段**调用（每段 2 个固定点）。

输出 `output_single`：`trajectory (B, 6, T_total)`、`idx_3d`、`robot_pose`、原始 seam 几何等，
写入 `traj/<seam>.pkl`。详见 [04_数据格式.md](04_数据格式.md)。

> **已知 bug**（CLAUDE.md 亦提及）：`process()` 内 `Stomp(config, ...)` 引用的是**裸全局 `config`**
> 而非 `self.config`。仅在 `__main__` 上下文（全局有 `config`）下能正常工作。

---

## 2. STOMP 核心 `StompTraj`

### 2.1 并行维度（`config_traj.py`）

- `num_batch = 24`：每条焊缝生成 24 条 GT 轨迹；
- `num_rollouts_new = 30`、`num_rollouts_old = 20`：每轮新采 30 条噪声轨迹 + 复用 20 条历史好轨迹（+1 条当前最优 = `num_rollouts_all = 51`）；
- `num_dimensions = 6`、`num_timesteps_init/next = 51`、`delta_t = 0.1`、`num_iterations = 20`；
- `num_envs = num_batch × num_rollouts_new × 2 = 1440` 个并行仿真环境。

### 2.2 顶层流程 `solve(fixed_pts, has_vision)`

```python
resetVariables(fixed_pts)        # 按段数动态分配张量、生成控制代价矩阵 R 与滤波矩阵
InitializeTrajectory(fixed_pts)  # 线性插值初始化每段，clamp 关节限位，算初始代价
for _ in range(num_iterations):  # 20 轮
    generateNoisyRollouts()      # 采样/复用噪声轨迹
    computeNoisyRolloutsCosts()  # 评估每条 rollout 各关节各时间步代价
    computeProbabilities()       # 代价 → 指数权重（逐 (维度 d, 时间步 t)）
    updateParameters()           # 加权更新最优轨迹，post-filter 平滑，仅在代价下降时接受
return parameters_optimized, parameters_total_cost
```

### 2.3 噪声采样 `generateNoisyRollouts` / `filterNoisyRollouts`

- 噪声从 `N(0, R⁻¹)` 采样（`R` 是控制代价矩阵，其逆作为协方差 → 自动产生**平滑**的噪声）；
- 每段（init 段 + 各 next 段）分别采样后拼接（next 段去掉首点避免重复）；
- `filterNoisyRollouts`：按 `noise_scale` 缩放，可选 `pre_filter` 平滑（默认关），叠加到当前最优后 **clamp 关节限位**；
- 复用模式：从历史 rollout 中按总代价取最优 `num_rollouts_old` 条重用，`[-1]` 槽放当前最优（精英保留）。

`noise_scale` / `filter_scale` 自适应：当 `碰撞代价 + 视野代价` 高于阈值时用 `_high`，否则 `_low`
（碰撞严重时加大噪声幅度跳出，接近可行时减小幅度精修）。

### 2.4 代价 `computeStateCost`

```python
collision_cost = scene.computeCollisionCost(rollouts) * collision_cost_weight(20)   # (B,R,D,T)
vision_cost[:, :, :, 10:41:10] = scene.computeVisionCost(...) * vision_cost_weight(1)
```
- **碰撞代价**：把 rollout 每个时间步的关节角写入仿真、step 物理、读接触传感器 `collided()`，
  逐 (batch, rollout, timestep) 得 0/1，再广播到所有关节维度。分批喂入 `num_envs` 个环境并行算。
- **视野代价**：只在时间步 `[10,20,30,40]`（采样关键步，省算力）计算：
  - 默认（`has_pc=False`）：`dir_cost` = 相机 z 轴与「相机→焊缝中点 `seam_median`」方向的夹角（度），
    乘以随时间步衰减的权重 `0.5^t`（越早的时间步越重要，保证一上来就朝向目标），× `dir_cost_weight(0.25)`；
  - 若有点云（`has_pc=True`）：用相机 FOV + raycast 统计可见点云比例，惩罚不可见点（`visible_cost`）。

### 2.5 控制代价

`computeParametersControlCosts_1`：对轨迹求**加速度**（7 点中心差分模板，`stomp_utils_traj.py`），
代价 = 0.5·Σ加速度²，× `control_cost_weight(0.1)`。
控制代价矩阵 `R = Aᵀ A`（A 为加速度有限差分矩阵），其逆 `R⁻¹` 既用于代价也用作噪声协方差。
（另有 `_2` 版本对速度计代价，当前用 `_1` 加速度版。）

### 2.6 概率与更新

经典 STOMP：逐 `(batch, 维度 d, 时间步 t)` 在 `num_rollouts_all` 条 rollout 上做
`prob ∝ exp(-h·(cost-min)/(max-min))`（`h=2`），归一化；
`updateParameters` 用 `Σ prob·noise` 得到更新噪声，经 `filter_matrix_R` **平滑**（`post_filter=True`）后叠加到最优轨迹，
首末点强制不动（`[...,0]=0, [...,-1]=0`），仅当总代价下降时接受。

### 2.7 滤波/平滑矩阵 `generateSmoothingMatrix_1`

STOMP 的更新平滑矩阵由 `R⁻¹` 拟合而来。本项目对**多段轨迹**（中间含不可移动的观测点）做了扩展：
用五次多项式（`fit_quintic`）拟合单点的平滑曲线列，沿各段镜像拼接，并用 `damping` 衰减跨段耦合，
保证平滑不会把中间「必须精确到达」的观测点抹掉。这是相对标准 STOMP 的主要定制。

---

## 3. `scene_traj.py` 场景构成

与 `scene_pose.py` 类似，但：
- 机器人 USD 用 `robot/a1_ur12e.usd`（pose 用简化版 `_s`）；
- 接触传感器 Link1–5 **不带 filter**（直接 net_force 判碰撞）；
- 视野代价模型不同：
  - 3D 视锥顶点 `p1..p8`（注释掉）；
  - 实际用 **2D 相机视锥**（5 面体 `p10..p14`，`compute_normal_1`）做点云可见性，或用 `dir_cost`（朝向中点）；
- `reset(robot_pose, seam_median)`：摆机器人、记录 `seam_median`（视野代价目标）；
- `computeVisionCost` 用 `UR12e_t.forward_cam_pose`（FK）从关节角算出相机位姿，再判朝向/可见性；
- `selectPC`（若用点云）：在焊缝附近裁剪点云、体素下采样、按端点/正面区域过滤出 `pc_used`。
- `getJoints`：traj 场景里基于 IK + 仿真碰撞挑关节解（与 pose 的 `getJoints` 思路一致，含 `field` 圆锥代理）。

`num_envs = 1440`，`sim dt = 1/100`。

---

## 4. 输出

每条 seam → `traj/<seam>.pkl`，是 `output_single` 的 list。核心字段：

| 字段 | 形状 | 含义 |
|------|------|------|
| `trajectory` | `(num_batch, 6, T_total)` | 24 条关节空间轨迹 |
| `idx_3d` | `(T_total,)` | 0/1 标记，1 表示该时间步是一个观测点（应拍照） |
| `robot_pose` | `(7,)` | 机器人底座位姿（工件系，含 base 变换） |
| `piece_pose_original` / `robot_pose_original` | `(7,)` | 原始世界系位姿 |
| `seam_line` / `seam_limits` | — | 原始焊缝几何（便于可视化/校验） |

详见 [04_数据格式.md](04_数据格式.md)。
