# 02 — Pose 阶段：观测点生成

Pose 阶段为每条焊缝求解一组 **6-DOF 相机观测位姿**，使得这些位姿合起来能「完整、清晰、无碰撞」地拍下整条焊缝，
且**观测点数量尽量少**。求解器是一个高度并行的**进化策略（Evolution Strategy, ES）**，
在 Isaac Lab 中用数百个并行环境同时评估候选位姿的代价。

涉及文件：`optimize_pose.py`（入口）、`optimizer_pose.py`（ES 核心）、`scene_pose.py`（仿真+代价）、
`config_pose.py`（超参）、`ik_cam_pose.py`（IK/FK）。

---

## 1. 入口 `optimize_pose.py`

`OptimizePose.run()`：
1. 扫描 input 文件夹：`.usd` → `config.usd_path`，`.ply` → `config.pc_path`，每个 `.pkl` 载入为一条 seam 数据；
2. 应用黑名单（`blacklist_pose.txt`）与 reference 过滤；
3. 写 `output/<folder>/path.json`（记录 usd_path/pc_path，供 Traj 阶段读取）；
4. 创建 `ScenePose`（**只建一次场景**，所有 seam 复用）；
5. `process()` 逐 seam 优化。

`process()` 对每条 seam：
```python
robot_pose   # (M, 7)  该 seam 有 M 个候选机器人底座位姿（机器人可停在不同位置观测）
piece_pose   # (M, 7)  对应工件位姿
seam_line    # (N, 3)  焊缝中心线采样点
seam_tangent # (N, 3)  每点切向
seam_limits  # (N, 2, 3) 每点两条「坡口边」方向向量（焊缝是两面夹角的凹槽）
horizontal   # (M,)   该底座位姿下工件是水平还是垂直（选关节限位）
seam_median = seam_line[N//2]   # 焊缝中点
```
对 M 个 (robot_pose, piece_pose) 逐个：
```python
robot_pose_rel = scene.reset(robot_pose[i], horizontal[i], piece_pose[i])  # 把机器人/工件摆好
optimizer = OptimizerPose(cfg, scene)
optimizer.resetSeamData(seam_line, seam_tangent, seam_limits)
cam_pose, joint, start_pts, end_pts = optimizer.solve()   # ★ 核心
```
每个成功解保存为一个 `output_single` dict（字段见 [04_数据格式.md](04_数据格式.md)），
所有解组成 list 写入 `pose/<seam>.pkl`。

> 注：`robot_pose` 在 `scene.reset` 内被变换到**工件坐标系**（piece 作为世界原点），返回相对位姿存盘。

---

## 2. 焊缝几何模型

每个焊缝采样点 `i` 由三组量描述：
- `seam_line[i]`：点的 3D 坐标；
- `seam_tangent[i]`：焊缝走向（切向）；
- `seam_limits[i]` = `(limit0, limit1)`：构成坡口的**两个侧面方向向量**（从焊缝指向两侧母材表面）。

由此可以推导一个**理想观测方向**：`z_dir = -(limit0 + limit1)/2`（坡口角平分线的反方向，即相机应从凹槽正上方往里看）。
切向作为相机 y 轴，叉乘得 x 轴，构成相机姿态的初值。

---

## 3. ES 核心 `OptimizerPose`

### 3.1 并行维度

- `num_batches = 8`：8 个**独立搜索实例**（batch）。每个 batch 各自维护一套「当前正在优化的相机位姿 + 已观测进度」的状态机，互不干扰，相当于 8 次独立尝试。
- `num_randoms_new = 100`：每次迭代每个 batch 采样 100 个噪声扰动 rollout（`num_randoms_old=0` 表示不复用旧 rollout）。
- `num_envs = num_batches × num_randoms_new = 800`：仿真里用 800 个并行环境一次性评估所有候选。

### 3.2 顶层流程 `solve()`

```python
defineVariables()          # 清零所有状态张量
initializePose()           # 构造 8 类初始相机位姿（见 3.3）
while i <= max_iterations(300):
    generateNoise()        # 采样高斯噪声 → 候选位姿
    computeNoisyPoseCosts() # 仿真评估每个候选的代价
    computeProbabilities()  # 代价 → softmax 权重
    updateParameters()      # 加权更新「当前最优位姿」
    resetVariables()        # ★ 观测进度状态机（推进/重置焊缝段）
    if self.stop: break     # 已有 num_poses(4) 个 batch 走完整条焊缝
selectOutputs()            # 选出观测点最少的方案
if refine: refineOutputs() # 把每个观测点位姿精修到局部最优
# 把「反向焊缝」batch 的 start/end 索引映射回原始焊缝方向
```

### 3.3 初始位姿 `initializePose()` — 8 个变体

从焊缝**两端点**出发各构造一组相机姿态，并做对称扩展，得到 **8 个初始位姿变体**：

1. 起点、终点各算一个基础朝向 `rot0`（z=坡口平分线反向、y=切向、x=y×z），经 SVD 正交化为 SO(3)；
2. 绕相机 X 轴 **±45°** 倾斜 → 4 个；
3. 再绕相机 Z 轴 **+180°**（相机翻转，y 轴反向）→ 8 个；
4. 位置 = 端点 + 相机系下 `[0,0,-0.5]` 旋转到世界（相机沿自身 −z 后退 0.5m，看向焊缝）。

同时为每个变体记录其对应的**焊缝方向**：偶数索引(0,2,4,6) 用原始 `seam_line`，
奇数索引(1,3,5,7) 用 `seam_line.flip(0)`（反向）。
这样 8 个 batch 同时从「焊缝正向/反向、相机两种倾斜、相机两种翻转」探索，提高找到可行解的概率。
（`length_multiple = num_batches // 8 = 1`，即正好 8 个 batch 一一对应。）

### 3.4 观测进度状态机 `resetVariables()`

这是 Pose 阶段最特别的部分：它不是「优化一个固定目标」，而是**边优化边推进焊缝覆盖**。

每个 batch 维护一个当前焊缝段 `[start_pts, end_pts)`，目标是为这一段找到一个零代价（完全可见、无碰撞、方向合格）的相机位姿。

- **段内成功**：当 `pos_cost + rot_cost <= 0`（`end_update`），说明当前位姿能看清这一段 →
  保存该位姿，并把 `end_pts` 沿 `pts_list` 向后推进一格（扩大可见段）。
- **段内失败**：迭代超过 `num_iterations(100)` 或连续 `num_next(30)` 次更新都没改善（`start_update`）→
  说明这一段一个位姿看不全 → 把 `start_pts` 重置到当前 `end_pts` 附近（开启**新观测点**），并通过 `retrievePoses` 尝试复用对称方向已找到的位姿。
- **整条完成**：当 `end_pts >= num_points`（`mask_finish`）→ 该 batch 标记 `finish`。
- **终止**：当 `finish` 的 batch 数 ≥ `num_poses(4)` → `self.stop = True`。

`pts_list` 定义了焊缝段的推进刻度：`arange(0, N+1, num_steps=5)`，并在首尾插入 `1` 和 `N-1`，
使端点附近更密（端点处观测约束不同）。

每个 batch 把成功保存的位姿序列记录在 `cam_pose_list[i]`、`joints_list[i]`、`start_pts_list[i]`、`end_pts_list[i]` 中。
一个 batch 走完整条焊缝所需的位姿个数 = 它的观测点数。

### 3.5 候选生成 `generateNoise()` / `computePoses()`

- 从多元高斯 `N(0, diag(sigma²))`（`sigma=0.1`，6 维：Δ位置 xyz + Δ旋转 so3）采样 `num_randoms_new` 个噪声；
- **自适应步长**：普通迭代缩放 `0.1 × clamp(当前代价, 2, 60)`（代价大时探索范围大）；精修阶段固定 `0.2`；
- `computePoses` 把噪声叠加到当前最优位姿：位置增量经当前姿态旋转后相加，旋转增量经 Rodrigues（`matrix_from_so3`）转旋转矩阵右乘；
- 末位 `[-1]` 槽固定为「上一轮最优」（精英保留）。

### 3.6 概率与更新 `computeProbabilities()` / `updateParameters()`

经典 STOMP/ES 风格的指数加权：
```python
# 对 pos 和 rot 各自独立处理
取每个 batch 代价最小的 num_select(2) 个 rollout
exponents = -h * (cost - min) / (max - min)        # h = exponentiated_cost_sensitivity = 5
prob = importance_weight * exp(exponents)，归一化
noise_optimized = Σ prob_k · noise_k               # 凸组合
```
`updateParameters` 用凸组合的噪声生成新位姿，`computeOptimizedCost` 只在**总代价下降**时接受更新
（精修模式下还要求 `pos+rot<=0` 可行且连续代价 `_o` 改善）。未被接受的 batch `next_count += 1`。

### 3.7 精修 `refineOutputs()`

`selectOutputs()` 先选出所有 `finish` 且观测点数等于**最小值**的 batch（即最优方案）。
然后 `refineOutputs` 对选中的「A 个方案 × B 个观测点」共 A×B 个位姿，开启 `original_costs=True` 模式，
再跑 `max_iterations_refine(40)` 轮 ES，把每个观测位姿精修到**局部最优的连续方向代价**（在保持可行的前提下让相机正对焊缝）。

精修时每隔 `span(=1)` 轮存一张快照，最终把多张快照拼起来 → 每个观测点产出**多个候选 GT 位姿**
（输出形状 `(num_snapshots × A, B, 7)`），为下游提供姿态多样性。

最后 `solve()` 把奇数（反向焊缝）batch 的 `[s_inv, e_inv)` 索引映射回原始焊缝：`[N - e_inv, N - s_inv)`。

---

## 4. 代价函数（在 `scene_pose.py` 中计算）

`OptimizerPose.computeCosts(cam_poses)` 组合两类代价：

```python
collision_costs, joints = scene.computeCollisionCost(cam_poses, joints_optimized)
vision_pos, vision_rot, vision_pos_o, vision_rot_o = scene.computeVisionCost_1(...)

pos_costs = collision_cost_weight(25) * collision + vision_pos_cost_weight(1) * vision_pos
rot_costs = collision_cost_weight(25) * collision + vision_rot_cost_weight(1) * vision_rot
```
其中 `vision_pos = insides·w_in + block·w_block + orientation·w_ori`，`vision_rot = insides·w_in`。
（pos 代价管「相机放哪 + 朝向」，rot 代价只管「能否进 FOV」，二者分别选 rollout，再合起来判可行。）

各权重（`config_pose.py`，`dist_limit=0.5`）：

| 代价项 | 权重 | 物理含义 |
|--------|------|----------|
| **collision 碰撞** | 25 | 机械臂/底座/相机体与工件或自身碰撞（仿真接触力） |
| **insides 视野内** | 19/0.5 = 38 | 焊缝点是否落在相机视锥 FOV 内（越界惩罚） |
| **block 遮挡** | 22/0.5 = 44 | 相机到焊缝点的视线是否被工件挡住（raycast） |
| **orientation 方向** | 23/30 | 观测方向是否合理（切向夹角、坡口平面内角、与坡口面法向夹角） |
| **space 视野空间** | 21/0.5 = 42 | （`has_space_cost=False`，当前代价路径未启用，见 `computeVisionCost_0`） |

### 4.1 碰撞代价 `computeCollisionCost` → `getJoints`

1. 把相机位姿变换到机器人底座系；
2. **解析 IK**（`UR12e_t.solve_fairino_ec`）求出 8 个关节解；NaN 置 -20，越限位的解打 -500 分；
3. 以「与上一最优关节角 `joints_optimized` 距离最近」为奖励，取最优 `K=2` 个关节解；
4. 把这些关节角写入仿真，并在相机位置放一个**圆锥 `field`**（半径 0.335/2、高 0.4，代表相机/传感器外壳实体），
   step 物理后用接触传感器 `collided()` 判断是否碰撞；
5. 关节越限位**或**发生碰撞 → 碰撞代价记 1。返回最优关节角与代价。

> `field` 圆锥的作用：相机本体是有体积的，单纯用关节链碰撞检测不到「相机壳撞到工件」，
> 因此在相机位姿处摆一个圆锥代理，专门检测传感器头的碰撞。

### 4.2 视野内代价 `visionInsides`

相机 FOV 建模为一个**六面体视锥**（顶点 `p1..p8`，由 `scl/scl_z` 缩放，z 从 0.4 延伸到 0.4×(1+scl_z)）。
对焊缝点，把 6 个面的法向用相机姿态旋转后，算点到各面的有向距离 `dist = -(向量·法向)`；
全部 `<=0` 即在视锥内，代价 = 各面正越界距离之和（在外面多远）。

### 4.3 遮挡代价 `visionBlock`

从相机位置向焊缝点发射射线（外加在射线周围 `block_radius=0.04`、`num_block_pts=6` 的一圈点增强鲁棒性），
用 warp `raycast_mesh` 打工件网格。若命中点早于到达焊缝点（被挡），
代价 = `1 - 命中距离/2`（挡得越近代价越大），对所有射线求和。

### 4.4 方向代价 `visionOrientation`

观测方向是否「正」由三部分加权（`tgt 0.3 + plane 0.2 + normal 0.5`）：
- **tangent**：视线与焊缝切向的夹角应落在 [30°, 60°]（不能顺着焊缝看，也不能完全垂直）；
- **plane**：视线在坡口平面内的投影方向与角平分线的偏差；
- **normal**：视线与两个坡口面法向的夹角应接近 45°±15°（保证两个坡口面都能看清）。

焊缝**端点**（`block_mask` 为 True，即首/末点）只用 tangent 代价；中间点用 plane+normal。
`_o`（original）变体是去掉 `clamp(min=0)` 的**连续**代价，供精修阶段用作可微目标。

---

## 5. `scene_pose.py` 场景构成

`ScenePoseCfg`（继承 `InteractiveSceneCfg`）：

| 实体 | 说明 |
|------|------|
| `robot` | A1_CFG（UR12e 机械臂），`__init__` 内动态把 USD 改为 `robot/a1_ur12e_s.usd` |
| `piece` | 工件 RigidObject，**kinematic**（不受力），网格来自 `usd_path` |
| `field` | 圆锥（相机本体碰撞代理），density 5000，平时藏在 z=1e6，检测时摆到相机处 |
| `contact_*` | piece、两个底座、Link1–5 的接触传感器；Link 之间配 `filter_prim_paths_expr` 检测**自碰撞** |

- `num_envs = 800`，`env_spacing = 16`（环境间隔够大，避免相邻环境互相干扰 raycast）；
- 启动时把 piece USD 的所有 mesh 合并成一个 **warp mesh**（用于遮挡 raycast）；8 顶点 mesh 用硬编码立方体面索引；
- `reset(robot_pose, horizontal, piece_pose)`：把 robot pose 变换进工件系，机器人底座再叠加 `[0,0,0.26]` + 绕 z 90° 的 `base_transform`，并按 `horizontal` 标志切换水平/垂直关节限位。

> **CLAUDE.md 提到的初始化顺序坑**：`ScenePoseCfg.piece.spawn.usd_path = Configuration().usd_path`
> 是在**类定义时**求值的——那时 `Configuration().usd_path` 是空字符串（占位）。
> 真正的工件路径在 `ScenePose.__init__` 第 166 行 `scene_cfg.piece.spawn.usd_path = self.cfg.usd_path` 动态覆盖。
> 重构时若调整类定义/实例化顺序，务必保留这个运行时覆盖，否则会加载空 USD。

---

## 6. 输出

每条 seam → `pose/<seam>.pkl`，内容是 `output_single` 的 list（每个候选机器人位姿一个）。
关键字段：`cam_pose`（相机位姿快照集，形状 `(num_snapshots×A, B, 7)`）、`joint`（对应关节角）、
`start_pts`/`end_pts`（每个观测点覆盖的焊缝段索引）、`seam_median`、`horizontal`、以及原始 seam 几何。
详见 [04_数据格式.md](04_数据格式.md)。Traj 阶段会用 `joint` 字段作为各观测点的目标关节角。
