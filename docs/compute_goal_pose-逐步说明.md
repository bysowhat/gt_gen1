# compute_goal_pose 逐步说明（面向零背景读者）

> 目的：把 `Scene.compute_goal_pose()` 从入口到最底层子函数【一步一步】讲清楚，
> 让完全不了解这套代码的人也能读懂「它在算什么、怎么算、每个子函数负责哪一块、
> 为什么会失败、怎么调试」。
>
> 代码位置：
> - 入口 `gt_gen/scene.py::Scene.compute_goal_pose`（约 753 行）
> - per-seam 主体 `scripts/compute_goal_poses2.py::compute_goal_poses`
> - 进化策略优化器 `stomp_planner/optimizer_pose.py::OptimizerPose`
> - 去 Isaac 的场景/碰撞/视觉 `stomp_planner/scene_pose2.py::ScenePose2`
> - 解析 IK `stomp_planner/ik_cam_pose.py::UR12e_t`
> - 超参 `stomp_planner/config_pose.py::ConfigurationPose`

---

## 0. 一句话概括

给定一条焊缝（3D 空间里的一条线）和当前 3D 世界（工件 mesh + 障碍物），
`compute_goal_pose` 用**进化策略（ES）**搜索出一组**相机观测位姿**（goal pose），
使这组位姿**合起来能把整条焊缝看完**，且每个位姿都满足：

1. 相机视野（FOV）里真的框住了焊缝点（不是拍到旁边）；
2. 焊缝点没被工件/障碍挡住（视线不被遮）；
3. 观测方向相对焊缝两个坡口面的角度合理（接近 45°，别掠射）；
4. 机械臂能摆到这个相机位姿（IK 有解、关节不越界）；
5. 机械臂本体不撞工件/不自撞，手臂也不挡住自己镜头的视野锥。

输出是一串位姿 + 对应的机械臂关节角，写进 `self.goal_poses[seam_id]`。
下游 `compute_pose_and_plan_path` 再拿这串位姿去「边走边看」规划轨迹。

---

## 1. 术语速查（先扫一眼，后面反复用）

| 术语 | 含义 |
|------|------|
| **焊缝 seam** | 工件上要检测的一条焊接缝，代码里离散成 N 个采样点的折线 `seam_line (N,3)` |
| **观测位姿 / goal pose** | 相机（连同机械臂末端）应该摆到的 6-DOF 位姿，7 维表示 `[x,y,z, qw,qx,qy,qz]` |
| **forehand / backhand（正手/反手）** | 工件相对机器人两种摆放朝向，各自有一套关节限位与候选初始位姿 |
| **init pose（初始位姿）** | 工件相对机器人「怎么摆」的一个确定摆放，由 `plan_init_pose` 求出候选，`set_init_pose` 选定 |
| **ES 进化策略** | 一种不需要梯度的优化：撒一堆随机扰动 → 评估代价 → 按代价加权融合出更优解 → 迭代 |
| **piece 系（工件 mesh 系）** | 以工件网格自身坐标为原点的坐标系。焊缝、障碍几何都在这个系里 |
| **base_link 系（机器人基座系）** | 以机器人固定底座为原点。cuRobo 的正向运动学根就是它 |
| **FOV 视锥** | 相机能看到的六面体视野（近大远小的方台），`visionInsides` 判点是否在其中 |
| **field 锥** | 相机镜头正前方一个小圆锥「视野禁区」，手臂近端 Link1/2/3 若探进来就挡住自己视线 |
| **K / B / DOF** | 输出张量三个维度：K=收敛后候选快照数，B=覆盖整缝所需位姿个数，DOF=6 关节 |

---

## 2. 调用链全景图

```
Scene.compute_goal_pose()                         ← gt_gen/scene.py 入口
  ├─ 读参数（configs/default.yaml 的 compute_goal_pose 段）
  ├─ 造焊缝数据 seam_line/seam_tangent/seam_limits  ← _seam_data_arrays（工件系）
  ├─ 算工件↔机器人相对摆放 robot_pose7 / piece_pose7（从当前 init pose）
  ├─ ScenePose2(cfg, ...)                          ← 建碰撞世界 + 视觉几何 + IK
  ├─ scene2.reset(robot_pose, horizontal, piece)   ← 摆工件进碰撞世界、选关节限位
  ├─ [可选] _inject_obstacles_into_scenepose2      ← 把障碍并进碰撞世界
  ├─ Optimizer = OptimizerPose(cfg, scene2)
  ├─ optimizer.resetSeamData(...)
  └─ optimizer.solve()  ★核心★                     ← stomp_planner/optimizer_pose.py
        ├─ defineVariables()      初始化一堆状态张量
        ├─ initializePose()       造 8 个初始朝向变体 + 算初始代价
        ├─ while i<=max_iterations:   ES 主循环
        │     ├─ generateNoise()          撒高斯扰动生成 num_randoms 个候选
        │     ├─ computeNoisyPoseCosts()  批量评估所有候选代价
        │     │     └─ computeCosts()
        │     │           ├─ scene.computeCollisionCost()  → getJoints()
        │     │           │     ├─ UR12e_t.solve_fairino_ec()   解析 IK（8 解）
        │     │           │     └─ _collided_batch()            碰撞+自碰+field锥
        │     │           └─ scene.computeVisionCost_1()
        │     │                 ├─ visionInsides()      在不在 FOV 里
        │     │                 ├─ visionBlock()        被不被遮挡（warp raycast）
        │     │                 └─ visionOrientation()  观测角好不好
        │     ├─ computeProbabilities()   代价→概率（softmax 式）
        │     ├─ updateParameters()       概率加权融合出更优位姿
        │     └─ resetVariables()  ★覆盖推进★ 看完一段就推进焊缝窗口
        ├─ selectOutputs()        从 8 个批里挑「视角数最少」的若干批
        └─ refineOutputs()        对选中的继续精修，堆叠成 K 维快照
```

---

## 3. 输入 / 输出数据契约

### 3.1 前提条件

调用前必须先：
```python
scene.plan_init_pose()             # 求出候选初始位姿
scene.set_init_pose(hand, index)   # 选定「工件相对机器人怎么摆」
```
否则 `compute_goal_pose` 第一行就 `raise RuntimeError`（`scene.py:794`）。

### 3.2 输入（方法内部自动组装，无需手动传）

| 数据 | 形状 | 坐标系 | 来源 |
|------|------|--------|------|
| `seam_line` | (N,3) | 工件 mesh 系 | `_seam_data_arrays()`，两端点插值 N 段（默认 20） |
| `seam_tangent` | (N,3) | 工件 mesh 系 | 焊缝切线（单位向量） |
| `seam_limits` | (N,2,3) | 工件 mesh 系 | 焊缝两个坡口面方向 d1/d2 |
| `robot_pose7` | (7,) | 工件 mesh 系 | = `pose7(inv(T_workpiece_in_base))`，机器人基座在工件系里的位姿 |
| `piece_pose7` | (7,) | — | 恒为 identity `[0,0,0, 1,0,0,0]`（工件放在原点） |

> **关键坐标系约定**：优化在「工件放在原点、机器人基座在 `robot_pose7`」的框架里做。
> `ScenePose2.reset` 会把这套换算成 `robot_base_inv_pose`（piece→base_link 变换），
> 再把工件 mesh 摆进 cuRobo 的 base_link 系碰撞世界。

### 3.3 输出

返回一个 dict（同时写入 `self.goal_poses[seam_id]`）：

| 键 | 形状 | 含义 |
|----|------|------|
| `cam_pose` | (K,B,7) | 相机观测位姿（**工件 piece 系**，wxyz） |
| `joints` | (K,B,6) | 每个位姿对应的机械臂 6 关节角 |
| `start_pts` / `end_pts` | (K,B) | 每个位姿覆盖的焊缝点区间 `[start,end)` |
| `robot_pose_rel` | (7,) | reset 返回的 robot_pose（工件系） |

无解时返回 `None`（并把 `None` 写进 `goal_poses`）。

**(K,B,DOF) 三维语义**（务必理解，下游全靠它）：
- **DOF=6**：UR12e 六轴关节角。
- **B**：覆盖整条焊缝所需的观测位姿个数。B=1 表示一个视角看完整条缝；B>1 表示要
  **按顺序**访问的多视角序列。
- **K**：候选解的「快照 × 批」数量，**不是** K 个独立最优解，而是收敛后近似等价的一批解。
  用的时候第一维通常取 `0` 即可（见 §8）。

---

## 4. 逐层单步走

### 4.1 入口 `Scene.compute_goal_pose`（scene.py:753）

单步顺序：

1. **检查前提**（794）：没设 init pose 直接报错。
2. **读参数**（799-805）：从 `configs/default.yaml` 的 `compute_goal_pose` 段读
   `include_obstacles / horizontal / device`；显式传参 > yaml > 内置默认。
   - `horizontal`：0=水平工件 / 1=垂直工件 / 其它=all（全范围），决定用哪套关节限位。
3. **加载优化器模块**（807-810）：`_load_compute_goal_poses2()` 惰性 import
   `scripts/compute_goal_poses2.py`，拿到 `ConfigurationPose / OptimizerPose / ScenePose2`
   三个类。这里有个技巧：注入一个 `scene_pose` 轻量 stub，顶掉 `optimizer_pose.py` 顶层
   那句只用于类型注解的 `from scene_pose import ScenePose`，避免连带 import isaaclab。
4. **组装 ES 配置**（813-826）：`ConfigurationPose()` 默认值 + 把 yaml 段里同名字段
   `setattr` 覆盖上去，最后重算派生量 `num_envs = num_batches*(num_randoms_new+num_randoms_old)`。
   > ⚠ ES 采样规模（`num_batches` 默认 8 / `num_randoms_new` 默认 100）调太小会触发优化器内部
   > 假设崩溃，默认 8×100 稳定。
5. **造焊缝数据**（829-832）：`_seam_data_arrays()` 把当前焊缝变成
   `seam_line/seam_tangent/seam_limits`（工件系），转成 GPU 张量。
6. **算相对摆放**（835-840）：
   ```python
   T = self.cur_init_pose.T_workpiece_in_base          # 工件在 base 系的 4x4
   robot_pose7 = mat44_to_pose7(inv(T))                # 机器人基座在工件系
   piece_pose7 = [0,0,0, 1,0,0,0]                      # 工件放原点
   ```
7. **建场景**（842-843）：`ScenePose2(cfg, obj_path=工件obj, robot_cfg_path=...)`
   → 见 §4.3。
8. **reset**（847）：`scene2.reset(robot_pose, horizontal, piece_pose)`，把工件摆进碰撞
   世界、选关节限位 → 见 §4.4。
9. **注入障碍**（848-849）：若 `include_obstacles` 且本焊缝有障碍，
   `_inject_obstacles_into_scenepose2` 把障碍实体和工件一起 `update_world` 进碰撞世界
   （避障；open_cylinder 纯视觉的跳过）。
10. **建优化器 + 喂焊缝**（851-852）。
11. **求解**（853）：`optimizer.solve()` → 见 §4.5 起，**核心**。
12. **打包结果**（855-868）：把 `cam_pose/joints/start_pts/end_pts/robot_pose_rel`
    detach clone 成 dict；`cam_pose is None` 则记 None。

### 4.2 焊缝数据 `_seam_data_arrays`（scene.py:733）

把当前焊缝（`self.seam` 里的两端点 + 坡口方向）变成优化器要的三件套：
- `seam_line (N,3)`：两端点线性插值 N 个点（`_interp_line`，N=`plan_init_pose.n_seg`，默认 20）。
- `seam_tangent (N,3)`：焊缝方向单位向量（直缝→每点相同）。
- `seam_limits (N,2,3)`：两个坡口面方向 d1、d2（`bisector = unit(d1+d2)`）。

都在**工件 mesh 系**，与障碍几何、weld_json 同框。

### 4.3 场景构造 `ScenePose2.__init__`（scene_pose2.py:83）

一次性搭好三样东西：

1. **纯 torch 配置**（90-135）：代价权重、关节限位占位、FOV 视锥六面体的 8 个角点
   `p1..p8` 及其 6 个面法向 `self.normal`（`compute_normal`）。这套六面体就是相机 FOV。
2. **工件 mesh**（138 `_load_piece`）：trimesh 读 `*_part.obj`，顶点/面存下来；再建一个
   warp mesh（`_build_wp_mesh`）供遮挡 raycast 用。
3. **cuRobo 碰撞世界**（140 `_init_curobo`）：
   - 加载机器人 cfg（含固定底座 `xiaoyu_base_link`，所以 cuRobo 的 FK 根**就是** base_link）；
   - 建 `RobotWorld`（整臂碰撞球 + mesh 碰撞世界 + 自碰撞）；
   - 选出 Link1/2/3 的碰撞球掩码 `_field_sphere_mask`（field 锥只测这几个近端球）。

### 4.4 `ScenePose2.reset`（scene_pose2.py:217）★坐标系关键★

1. **换算到工件局部系**（221-224）：若给了 `piece_pose`，把 `robot_pose` 用
   `inv(piece_pose)` 复合到工件系。这里 `piece_pose=identity`，所以 `robot_pose` 不变。
2. **算 piece→base_link 变换**（231-233）：
   ```python
   robot_base = robot_pose.clone()
   robot_base_inv_pose = inv(robot_base)      # piece→base_link
   ```
   > 🔴 **历史 bug（已修）**：这里曾从原版 Isaac `scene_pose.py` 逐字拷来一段
   > `base_transform = [0,0,0.26, 0.7071068,0,0,0.7071068]`（0.26m + 绕 z 90°），
   > 用来叠机器人底座。但去 Isaac 版的 cuRobo FK 根**已经是 base_link**（底座在模型里），
   > `robot_pose` 本就是 base_link 位姿，再叠一次 = 把底座重复计一遍，令工件/相机/IK 帧
   > 整体错转 90°，导致工件碰撞≈100%、IK 无解≈58%、几乎所有焊缝失败。**现已去掉。**
   > 详见 memory `scenepose2-base-transform-bug` 与 `docs`。
   > 原版 Isaac `scene_pose.py` 里 root≠base_link，那里 base_transform 有效，**勿动**。
3. **选关节限位**（236-245）：按 `horizontal` 选水平/垂直/all 三套之一。
4. **摆工件进碰撞世界**（247-250）：把工件 mesh 按 `robot_base_inv_pose` 位姿
   `update_world` 到 cuRobo 的 base_link 系碰撞世界。

### 4.5 求解顶层 `OptimizerPose.solve`（optimizer_pose.py:241）

```python
defineVariables()          # 初始化所有状态张量（batch/候选/覆盖计数等）
initializePose()           # 造 8 个初始朝向 + 算初始代价    → §4.6
i = 1
while i <= max_iterations:                                   # 默认 300
    generateNoise()          # 撒扰动                        → §4.7
    computeNoisyPoseCosts()  # 批量评估代价                  → §4.8
    computeProbabilities()   # 代价→概率                     → §4.9
    updateParameters()       # 加权融合出更优位姿            → §4.10
    resetVariables()         # 覆盖推进：看完一段推进窗口    → §4.11
    if self.stop: break      # 够 num_poses(默认4) 个批 finish 就停
selectOutputs()            # 挑视角数最少的批                → §4.12
if cam_pose_select is None: return None×4
if refine: refineOutputs() # 精修 + K维快照                  → §4.13
# 反缝索引回映射（奇数批用翻转焊缝）                          → §4.14
return cam_pose_select, joints_select, start_pts_select, end_pts_select
```

**并行结构**：`num_batches`（默认 8）个「批」同时独立优化，每个批是一条正在生长的
观测序列。每个批每轮撒 `num_randoms_new`（默认 100）个候选一起评估。

### 4.6 初始位姿 `initializePose`（optimizer_pose.py:543）

从焊缝首尾两端各构造相机初始朝向，共 **8 个变体**：
- 基准朝向：镜头 -z 指向焊缝坡口中分线反向，y 沿焊缝切线（543-573）。
- 绕相机 X 轴 ±45°（576-578）→ 2 种；
- 首/尾两端（569）→ 2 种；
- 绕 Z 轴翻 180°（585）→ 2 种；
- 合计 2×2×2 = 8，相机沿视线后退 0.5m（592-593）。

`num_batches=8` 时 `length_multiple=1`，即每个批恰好一个初始变体。
最后 `computeCosts` 算一遍初始代价存进 `pos/rot_optimized_cost`。

> 奇偶批用不同焊缝方向：偶数批用原焊缝，奇数批用 `seam_line.flip(0)`（翻转）。
> 这是为了从两个方向生长覆盖窗口，最后在 §4.14 把奇数批的索引映射回原焊缝。

### 4.7 撒扰动 `generateNoise`（optimizer_pose.py:614）

1. 从多元高斯（协方差 `diag(sigma²)`，sigma 默认 0.1）采 `num_randoms_new` 个
   6 维扰动（位置 3 + 旋转 3），乘一个随代价自适应的 `scale`（代价大→扰动大，探索更广）。
2. 最后一个 slot（`[:,-1]`）固定为「零扰动=上一轮最优」，保证精英不丢。
3. `computePoses`（676）把扰动施加到当前最优位姿上：位置加 `quat_apply(quat, delta_pos)`，
   旋转右乘 `matrix_from_so3(delta_rot)`，得到 `num_randoms_all` 个候选位姿。

### 4.8 代价评估 `computeNoisyPoseCosts` → `computeCosts`（optimizer_pose.py:703/715）

对每个候选位姿算两个总代价：
```
pos_costs = collision_cost_weight * collision + vision_pos_cost_weight * vision_pos
rot_costs = collision_cost_weight * collision + vision_rot_cost_weight * vision_rot
```
权重：碰撞 25、视觉 pos/rot 各 1（`config_pose.py:34-36`）。碰撞项权重远大于视觉，
意味着**碰撞/IK 无解几乎一票否决**。

代价由两大块组成：

#### (A) 碰撞代价 `computeCollisionCost` → `getJoints`（scene_pose2.py:255/265）

对每个相机候选位姿：
1. **piece→base_link**（273-274）：把相机位姿从工件系变到机器人基座系。
2. **解析 IK**（277-280）：`UR12e_t.solve_fairino_ec` 给出**8 组**关节解（§4.15）。
3. **过滤 + 选解**（281-291）：
   - `nan→-20`，标记越界解 `outbound`；
   - 用「与参考构型 `joints_opt` 的距离」当奖励，越界的减 500；
   - 取 top-K（K=2）个最优解，越界仍无解的记 `cost=1`（IK 无解/越界）。
4. **碰撞判定** `_collided_batch`（298, 308）：对选出的关节解，
   - `get_collision_constraint`：整臂碰撞球 vs 工件/障碍 mesh（穿透>0 即撞）；
   - `get_self_collision`：自碰撞；
   - **field 锥**（321-331）：Link1/2/3 的碰撞球是否落进镜头正前方那个小圆锥
     （半径 0.335/2、高 0.4、半角≈22.7°）——若落进去，手臂就挡住了自己的视线。
   任一命中 → `cost=1`。
5. K 个候选解取「代价最小」的那个（301-305），返回 `cost∈{0,1}` 和其关节角。

> **碰撞代价是 0/1 的硬门槛**：只有 IK 有解 + 不撞 + 不挡视线，这个位姿才有资格被接受。
> 诊断脚本 `/tmp/diag_seam34.py` 就是拆开这里统计 IK 无解率 / 工件碰撞率 / 自碰率 / field 锥命中率。

#### (B) 视觉代价 `computeVisionCost_1`（scene_pose2.py:336）

对当前焊缝窗口 `[start_pts, end_pts)` 里的每个焊缝点，算三项（都在工件系）：

1. **`visionInsides`（395）**：焊缝点在不在相机 FOV 六面体内。做法是把点代入六个面
   的法向内积，全在内侧（各 dist≤0）代价为 0，否则按超出量累加。→ 「有没有拍到」。
2. **`visionBlock`（420）**：从相机向焊缝点发一束射线（主射线 + `num_block_pts` 条
   在小圆盘上偏移的射线），用 warp `mesh_query_ray` 看是否在到达焊缝点**前**先撞上工件/障碍。
   撞了=被遮挡，代价>0。→ 「视线通不通」。
3. **`visionOrientation`（477）**：观测方向相对焊缝两坡口面的角度。理想是接近 45°带
   （`45°±15°`），掠射（太平）惩罚大。分三个子项 tgt/pl/nm 加权。→ 「角度好不好」。

三项加权求和：
```
vision_pos_cost = insides*w1 + block*w2 + orientation*w3   # 位置类（能不能看到 + 角度）
vision_rot_cost = insides*w1                               # 只看 FOV 内含
```
外加一个 **gate 版本** `vision_pos_cost_gate`：把 orientation 目标带放宽
`orient_gate_relax` 度（默认 0，等价旧版）。gate 专门用于「接受门槛」——判定一个焊缝点
到底算不算「真正看到了」，把「角度好不好」降级为偏好而非硬卡。

### 4.9 概率 `computeProbabilities`（optimizer_pose.py:762）

对每个批的候选代价取 top-`num_select`（默认 2）个最小的，用
`prob ∝ exp(-h·(cost-min)/(max-min))`（h=5）归一化成概率。代价越低概率越高，
是 STOMP/CMA 式的 softmax 加权。

### 4.10 更新 `updateParameters`（optimizer_pose.py:803）

用概率对选中的扰动做**凸组合**得到 `noise_optimized`，施加到当前最优位姿上得到候选
`cam_pose_optimized`，再 `computeOptimizedCost`（830）判定：只有新代价更低才接受更新
（`update_idx`）。未被接受的批 `next_count += 1`（连续失败会触发窗口重置，见 §4.11）。

### 4.11 覆盖推进 `resetVariables`（optimizer_pose.py:436）★最难的一块★

这是「一个视角看完一段焊缝 → 推进到下一段」的状态机。核心变量：
- `start_pts / end_pts`：当前批正在尝试覆盖的焊缝点区间 `[start,end)`。
- `pts_list`：覆盖检查点序列 `arange(0,N+1,num_steps)` 再插入 `1` 和 `N-1`
  （例：N=20,num_steps=5 → `[0,1,5,10,15,19,20]`）。窗口就沿这个序列生长。
- `finish`：该批是否已覆盖完整条焊缝。

单步逻辑：
1. **能不能接受当前视角**（442）：`end_update = (pos_optimized_gate + rot_optimized_cost) <= 0`
   ——用 **gate 代价**判定（碰撞=0 且 FOV 内 且 视线通 且 角度未超软上限）。达标就
   把当前位姿存进 `cam_pose_saved`。
2. **推进 end 点**（480-483）：达标则把 `end_pts` 沿 `pts_list` 前移一格
   （窗口向后扩，让这个视角尝试多看一段）。
3. **看不动就重置 start**（449, 487-505）：若连续 `num_iterations`(100) 轮或
   `num_next`(30) 次更新失败仍没达标，就把 `start_pts` 推到当前 `end`，
   **开一个新视角**从这里继续（存下上一个视角进 `cam_pose_list`）。
4. **强制首尾**（`forced`，452-467）：可选，单独拍第一段和倒数第二段（默认关）。
5. **整缝完成**（472-480）：`end_pts >= num_points` 且达标 → `finish=True`，
   第一次完成时存下最终区间。
6. **停止**（539-540）：已 `finish` 的批数 ≥ `num_poses`(默认 4) → `self.stop=True`，跳出主循环。

**直觉**：每个批像一个人拿相机，从焊缝一端开始，尽量在一个位置多看一段；看不动了就
挪到新位置接着看；直到把整条缝看完。最终每个批产出的「视角序列」就是 B 个位姿。
不同批因为初始朝向/焊缝方向不同，会得到不同长度（不同 B）的序列。

### 4.12 挑最优批 `selectOutputs`（optimizer_pose.py:292）

- 没有任何批 finish → `cam_pose_select=None`（→ 上层返回 None，**无解**）。
- 否则在所有 finish 的批里，找出**视角数最少**（`cam_pose` 堆叠后第 0 维最小=用最少位姿
  看完整条缝）的那些批，stack 成 `(A0, B, 7)`（A0=并列最少的批数）。

### 4.13 精修 `refineOutputs`（optimizer_pose.py:342）

对选出的 `A0×B` 个位姿再跑 `max_iterations_refine`（默认 40）轮 ES 精修
（这次 `original_costs=True`，用「未放宽」的全量朝向代价，把角度往 45° 收敛）。
每 `span`（默认 1）步存一次快照，共 `num_snapshots` 个。结尾：
```python
cam_pose_list.reverse()                        # 先反转→dim0 前段是最收敛的迭代
cam_pose_select = cat(cam_pose_list, dim=0)    # (num_snapshots*A0, B, 7)
```
于是输出第一维 **K = num_snapshots × A0**（例：40×4=160）。这就是 §3.3 说的 K：
一批收敛后近似等价的快照，不是 K 个独立解。

### 4.14 反缝索引回映射（optimizer_pose.py:270-288）

奇数批（初始用翻转焊缝）存的 `start/end_pts` 是「翻转缝」上的下标，调用方拿的是原缝，
所以把 `[s,e)` 映射回 `[N-e, N-s)`。这样 `start_pts/end_pts` 无论来自哪个批都对齐原焊缝。

### 4.15 解析 IK `UR12e_t.solve_fairino_ec`（ik_cam_pose.py:59）

给一个相机 4×4 位姿，先右乘手眼外参逆 `cam_inv`（相机→Link6，从
`configs/default.yaml` 的 `sensor.camera` 读，**单一真源**），得到法兰位姿 T，
再用 UR/Fairino 六轴解析逆解公式一次算出**全部 8 组**关节解（2×2×2 分支：j1 两解 ×
j5 两解 × j2 两解）。这些解回到 `getJoints` 里过滤越界、选最接近参考构型的。

> ⚠ IK 用的相机外参必须和 raycast 传感相机是同一台。历史上这里写死过旧机械臂相机
> （差 0.185m/91°），换相机后必须改，现已改为读 config。见 memory `camera-single-source-config`。

---

## 5. 关键超参一览（config_pose.py）

| 参数 | 默认 | 作用 |
|------|------|------|
| `num_batches` | 8 | 并行优化的批数（=8 个初始朝向） |
| `num_randoms_new` | 100 | 每批每轮撒的候选数 |
| `num_envs` | 800 | = num_batches×num_randoms，IK/碰撞的并行规模 |
| `max_iterations` | 300 | ES 主循环上限 |
| `max_iterations_refine` | 40 | 精修轮数（影响 K） |
| `num_steps` | 5 | 覆盖窗口沿 pts_list 的步长 |
| `num_iterations` | 100 | 单视角卡住多少轮后重置 start |
| `num_next` | 30 | 单视角卡住多少次更新失败后重置 start |
| `num_poses` | 4 | 有几个批 finish 就停 |
| `collision_cost_weight` | 25 | 碰撞权重（远大于视觉→硬门槛） |
| `orient_gate_relax` | 0 | 接受门槛的朝向放宽角度（0=旧版行为） |
| `refine` | True | 是否做精修 |
| `scl / scl_z` | 7/11, 9.5/11 | FOV 视锥尺寸缩放 |

---

## 6. 什么情况会返回 None（无解）

`selectOutputs` 里**没有任何批 finish** 就无解。常见根因（按诊断经验）：

1. **坐标系错**（历史头号杀手）：`reset` 里重复叠 base_transform → 工件碰撞≈100%、
   IK 目标被转到够不着 → 全灭。**已修**（§4.4）。
2. **IK 无解率高**：相机位姿够不到（机器人到工件太远/角度刁钻），或关节限位太紧
   （`horizontal` 选错）。
3. **工件/障碍碰撞率高**：init pose 摆放让工件正好挡在机械臂工作空间里。
4. **视锥/遮挡**：焊缝在深腔里，任何能拍到的位姿都被工件自遮挡。
5. **ES 规模太小**：`num_batches/num_randoms_new` 调太小，优化器内部假设崩溃。

---

## 7. 怎么单步调试（实操）

### 7.1 环境与运行

- 需要 GPU + curobo + warp 环境（`env_isaaclab`）。
- **显存**：`compute_goal_pose` 会吃不少显存；本地 11.6G 卡跑「边走边看」会 OOM。
  建议在大卡上，或用 `Scene.save/load` 拆进程（算 goal → save → 另进程 load 可视化）。
  见 memory `plan-explore-path-gpu-oom` / `scene-save-load-cross-process-viz`。
- **不能同进程**先 `compute_goal_pose` 再启 isaacsim（warp 会污染），必须拆两次运行。

单条焊缝最小复现（对应 `scripts/demo_scene.py::demo_main`）：
```python
scene = Scene(cfg="configs/default.yaml", workpiece_obj=..., weld_json=...)
scene._set_cur_seam(34)
scene.add_obstacle_type2()
scene.plan_init_pose()
scene.set_init_pose("forehand", 0)
res = scene.compute_goal_pose()      # ← 在这里下断点/加 monkeypatch
```

### 7.2 下断点的关键位置

| 想看什么 | 断点位置 |
|----------|----------|
| 坐标系对不对 | `scene_pose2.py:231-250` reset，打印 `robot_base_inv_pose` |
| IK 有没有解 | `scene_pose2.py:280` `joints_all_ik`、289 `cost` |
| 三种碰撞谁在杀 | `scene_pose2.py:308` `_collided_batch`（d_world/d_self/cone_hit）|
| 视觉三项谁在杀 | `scene_pose2.py:351/361/366` insides/block/orientation |
| 覆盖推进卡在哪 | `optimizer_pose.py:442` end_update、472 mask_finish、535 num_finish |
| 为什么无解 | `optimizer_pose.py:296` `self.finish.any()` |

### 7.3 现成的诊断脚本

`/tmp/diag_seam34.py`：不改仓库文件，用 monkeypatch 给 `getJoints/_collided_batch/
computeVisionCost_1/selectOutputs` 打补丁，跑完打印一份「三分量代价 + 碰撞细分
（IK无解/工件碰撞/自碰/视锥）+ 覆盖进度（finish/end_pts）」报告。**排查无解首选它**。
用法：`python /tmp/diag_seam34.py`（先确保 `scene1.pkl` 是修复后代码重新生成的）。

### 7.4 可视化验证

- `Open3DSceneVisualizer.show_goal_pose_collision(hand, index, compare=True)`：
  按 ScenePose2 判碰口径分色画整臂碰撞球，`compare=True` 开双窗对照
  （robot_base_inv_pose 摆放 vs T_workpiece_in_base 真值），当初就是靠它定位 base_transform bug。
- `show_scene_isaacsim(goal_arm_index=[0,1])` / `show_trajectory_isaacsim`：
  把机械臂摆到 goal 关节角渲染。**注意**：这些方法渲染帧是对的，但画的是
  `goal_poses` 里的数据——若 pkl 是 base_transform 修复**之前**生成的，会显示错转 90°
  的臂，必须用修复后代码重新生成 pkl 再看。

---

## 8. 下游怎么用输出（承上启下）

`compute_pose_and_plan_path`（scene.py:871）对每个候选 init pose：
1. `set_init_pose` → `compute_goal_pose` 得 `joints (K,B,6)`；
2. **只取 variant 0**（K 维第 0 个，收敛后快照通常等价）；
3. 造一张三态体素图 `vm`，按位姿顺序 `0..B-1` 逐个 `plan_explore_path`：
   - pose[0] 从初始关节角（retract）起步；
   - pose[i] 从 pose[i-1] **实际到达的末关节角**起步，且全程共享同一张 vm
     → 后一个 pose 继承前一个「边走边看」观测到的世界；
4. B 个 pose 全 reached 才算这条序列成功。

所以 `compute_goal_pose` 的产物（一串有序观测位姿 + 关节角）是「边走边看」轨迹规划的
**目标序列**。理解了 (K,B,DOF) 就理解了两阶段如何衔接。

---

## 附：一分钟速记

> `compute_goal_pose` = 在「工件放原点、机器人基座在 robot_pose」的框架里，用 8 个批 ×
> 每批 100 个高斯扰动的**进化策略**，搜索能覆盖整条焊缝的相机位姿序列。每个候选位姿要过
> **碰撞门槛**（解析 IK 有解 + 不撞工件/障碍/不自撞 + 手臂不挡视野锥）和**视觉代价**
> （在 FOV 内 + 视线不被遮 + 观测角接近 45°）。看完一段推进焊缝窗口，看完整条缝算一个批
> finish；挑视角数最少的批精修，堆叠成 (K,B,6) 输出。无解时返回 None——头号历史根因是
> reset 里重复叠底座 base_transform（已修）。
