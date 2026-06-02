# Phase 1 在线主动观测 — 完整实现细节文档

> 本文把 Phase 1（焊缝在线主动观测的探索阶段）从问题定义到每一行算法逻辑、每一个超参、
> 每一处坑与修复，全部写清楚。目标：任何人读完这一篇就能完全理解代码在做什么、为什么这么做。
>
> 代码位置：`/home/a/Projects/Github/gt_gen_hanfeng/phase1/`
> 运行环境：`/home/a/miniforge3/envs/env_isaaclab/bin/python`（已装 isaacsim/isaaclab/cuRobo/open3d/libigl/trimesh+embreex）

---

## 0. 问题定义

- **机械臂**：固定底座 6 自由度 UR12e（臂展约 1.9 m），cuRobo 配置文件
  `/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml`。
- **工件**：`/media/a/新加卷/hanfeng/segment_sub_output/<工件>_part/` 下，每个工件一个 `.obj` 网格 + 多个 `seam_*.pkl`。
- **焊缝**：一条连续焊缝约 20 个采样点（长 10–72 mm）。
- **观测约束**：相机到焊缝点的距离须落在 **[0.30, 0.60] m**，且要在焊缝开口的"楔形"角内、视线无遮挡、目标在视锥内。
- **难点**：机械臂初始姿态看不到焊缝；工件上有大量遮挡；要**增量探索**找到能合规观测焊缝的相机位姿（以及到达它的无碰撞路径）。

**两阶段流水线**：
- **Phase 1（本文）**：探索 → 找到合格观测位姿 → 触发 "phase switch"。
- **Phase 2**（后续）：在合格观测位姿附近精细沿焊缝扫描。

**Phase 1 的输出**：一条规划路线（每步相机位姿 + 关节角），最终到达一个能合格观测焊缝的位姿。

---

## 1. 基础约定

### 1.1 初始关节角（重要区分）
`config.INITIAL_JOINT_ANGLES`（= ur12e.yml 的 `retract_config`）：
```
[1.5707824, -2.0071660, 1.3613485, -0.9599629, -1.5707706, 0.0]
```
- **这是 Phase 1 的真正起点**（远离工件的 "home" 位姿）。
- pkl 里的 `joint_angles` 是**焊接到位姿**（EE 末端贴在焊缝中点 P_weld 上，距离≈0），**不是初始姿态**，Phase 1 不用它当起点（只可作目标参考）。
- 历史坑：早期 viz 误用 `pkl.joint_angles` 当起点，导致"从贴着焊缝开始、必须先移开"的反直觉行为。

### 1.2 坐标系
- **world 系**：全局，mesh/焊缝/相机都在 world 里。
- **base 系**：机械臂底座；`robot_pose_world`（4×4，存在 pkl 里）做 world↔base 变换。cuRobo 在 base 系算，`ArmKin` 负责换算。
- **相机系（OpenCV 约定）**：`z` = 光轴（向前看），`x` = 右，`y` = 下。
  所以相机 4×4 位姿 `T` 里：`T[:3,2]` = 光轴方向，`T[:3,3]` = 相机位置，`up = -T[:3,1]`。

### 1.3 pkl 数据结构（`_viz_helpers.load_seam_data`）
| 字段 | 形状 | 含义 |
|------|------|------|
| `seam_line` | (N,3) | 焊缝采样点（world） |
| `seam_tangent` | (N,3) | 每点切线方向 |
| `seam_limits` | (N,2,3) | 每点两侧的"面方向"`boundary_dirs`（单位向量，从焊缝指向开口外，张成观测楔形） |
| `p_weld` | (3,) | 焊缝中点 = `seam_line[middle]` |
| `p_weld_idx` | int | 中点索引 |
| `robot_pose` | (4,4) | 底座在 world 的位姿 |

---

## 2. 模块总览（里程碑 → 文件）

| 里程碑 | 文件 | 职责 |
|--------|------|------|
| M1.1 | `depth_source.py` | 工件网格 → 模拟深度图（trimesh+embreex 光线投射） |
| M1.2 | `mapping.py` | PyTorch 三态体素栅格在线建图（free/occupied/unknown） |
| M1.3 | `observation_check.py` | 合格观测检查（4 条硬约束） |
| M1.4 | `arm_kin.py` | cuRobo 封装：FK / IK / 整臂碰撞 / 边碰撞 / 轨迹代价 |
| M1.5 | `graph.py` | 持久化全局图 + Dijkstra（GBPlanner2 风格） |
| M1.6 | `gain.py` | 增益函数（体积增益 + 目标偏置） |
| M1.7 | `exploration.py` | 单 cycle 探索（采样→IK→碰撞→评分→选最优→加图→切换检测） |
| M1.8 | `exploration.py` | 多 cycle 主循环 + 卡住回退（Global step） |
| M1.8.5 | `geodesic.py` + `exploration.py` | 绕障（测地距离场）、目标导向候选、合格位姿求解器、边碰撞、滚转采样等一系列修复 |
| 工具 | `tests/batch_eval.py`, `tests/visualize_route.py` | 批量评估成功率 + 保存/可视化规划路线 |
| 配置 | `config.py` | 所有超参集中 |

---

## 3. M1.1 — 模拟深度图 `depth_source.py`

### `CameraIntrinsics`
- `from_fov(fov_h_deg, fov_v_deg, height, width)` 由视场角构造内参（fx, fy, cx, cy）。
- 默认相机：FOV 86°×57°（RealSense D455），分辨率可调（批量评估用 240×320 省时）。

### `RaycastSim`
- 用 `trimesh` 加载工件 `.obj`，`embreex`（Intel Embree BVH）加速光线求交。
- `render(camera_pose: 4×4) -> (H,W) 深度图（米，0=未命中）`：
  1. 每像素方向（相机系）`dir = normalize([(u-cx)/fx, (v-cy)/fy, 1])`。
  2. 转 world：`dirs_world = dirs_cam @ R.T`。
  3. `mesh.ray.intersects_first` 只取命中三角形 id（比 `intersects_location` 快 4–5 倍）。
  4. 自己用 ray-plane（命中三角形法线 + 顶点）算精确距离。
- `render_batch` 是简单循环版。

---

## 4. M1.2 — 体素栅格建图 `mapping.py`

### `VoxelMap`
- 稠密 3D `uint8` 张量（GPU），状态码：`0=unknown, 1=free, 2=occupied`。
- 构造：`VoxelMap(bounds=(2,3), resolution, max_range, device)`，`bounds = [[xmin,ymin,zmin],[xmax,ymax,zmax]]`。
- 关键方法：
  - `world_to_index(pts)`：world 坐标 → 体素 idx（越界整行 = -1）。
  - `index_to_world(idx)`：idx → 体素中心 world 坐标。
  - `query(pts)`：单点状态。
  - `raycast(origins, dirs, max_range)`：批量射线 → 命中距离 + 沿途 unknown 计数（增益用）。
  - `integrate(depth, intrinsics, camera_pose)`：把一帧深度写进地图。

### `integrate` 逻辑（每条射线分块处理）
对每条射线沿光轴等距采样（步长 = `res/2`，保证不漏穿薄物）：
- `sample_d < depth - res/2` → 标 **free**（命中点之前是空的）。
- `|sample_d - depth| ≤ res/2` → 标 **occupied**（命中体素）。
- `sample_d > depth + res/2` → 不写（保留 unknown）。
- `depth == 0`（未命中）→ 整条射线到 max_range 全标 free。
- **occupied 写在 free 之后**，让 occupied 优先。

### ⚠️ 坑与修复：occupied 保护（M1.8.5）
**问题**：射线 free-carving 会把工件的 occupied 体素**侵蚀成 free**——某个角度的射线擦过工件、命中更远处，于是把中途的工件体素错标 free。结果碰撞几何"少了肉"，导致规划路径从那里穿过工件。
**修复**：integrate 时**已 occupied 的体素不被 free 覆盖**（静态工件的占据是"黏性"的）：
```python
cur = self._state[f_idx...]; keep = cur != STATE_OCCUPIED; f_idx = f_idx[keep]
```

---

## 5. M1.3 — 合格观测检查 `observation_check.py`

### `Viewpoint(pos, dir, up)` / `ObservationResult`
`is_valid_observation(viewpoint, p_seam, tangent, voxel_map, intrinsics, seam_limits, d_min, d_max, los_unknown_as_block)`
依次检查 **4 条硬约束**（任一失败即不合格，短路返回）：

1. **距离** `check_distance`：`d_min ≤ ‖cam_pos - p_seam‖ ≤ d_max`（默认 [0.30, 0.60]）。
2. **视锥** `check_in_frustum`：把 p_seam 投到相机系，必须在水平/竖直 FOV 半角内（86°→±43°，57°→±28.5°）且在前方。
3. **视线** `check_line_of_sight`：沿 cam→p_seam 射线在体素图里逐点查，遇 occupied = 遮挡。
   - `los_unknown_as_block=True`（严格）：unknown 也算遮挡（未探明的不信任）。
   - `=False`（宽松）：unknown 当可通（求解器内部用，见 §11）。
4. **楔形** `check_seam_wedge`：相机必须落在焊缝两侧 `boundary_dirs (d1,d2)` 张成的开口楔形内。

> 历史：原本有第 5 条"入射角(incidence)"约束，**应用户要求删除**。

### 楔形判定细节
- `d1, d2` = 两条 boundary_dir（可投影到 ⊥tangent 平面）。
- **中线（bisector）** `bis = normalize(d1 + d2)` —— 楔形的中央方向，"最正"地看焊缝的方向。
- 开口角 `gap = arccos(d1·d2)`，半角 `half = gap/2`，`cos_half = cos(half - margin)`。
- 令 `rel = normalize(cam_pos - p_seam)`（投到 ⊥tangent），**`cos_to_bis = rel·bis ≥ cos_half` 即在楔形内**。
- gap 可以是 90°（L 形直角缝）、45°（锐角凹槽）等任意角；半角自适应。

### `wedge_bisector(seam_limits, tangent)`（M1.8 抽出复用）
单独返回中线单位方向（从焊缝指向开口外），供评分的"中线对齐"项 + 目标候选采样 + 测地种子复用。退化（limits 非法/反向）返回 None。

### ⚠️ 历史坑
- `seam_limits` 一度被误解成"法向量"；实为 `boundary_dirs`（面方向，指向开口外）。
- 非 watertight 网格用 `trimesh.contains` 不可靠 → 改用 **libigl `fast_winding_number`**（winding>0.5 判内部）。
- 薄壳网格射线 LOS 会漏判 → 可视化里用 mesh 光线投射当真值。

---

## 6. M1.4 — cuRobo 封装 `arm_kin.py`

`ArmKin(robot_yml_path, robot_pose_world)`，6 DoF。
| 方法 | 作用 |
|------|------|
| `fk(theta)` | 正运动学，返回 `ee_pos_world / ee_quat_world`（含 base→world 换算） |
| `ik(pos_world, quat_wxyz)` | 逆运动学（批量），返回 `success` 掩码 + `theta` |
| `update_world(voxel_map)` | 把体素图的 occupied 喂给 cuRobo 当碰撞障碍 |
| `collides(theta)` | 整臂（含自碰撞）碰撞检测，批量 → bool |
| `edge_free(theta_a, theta_b, step_rad)` | 两构型间关节空间**直线插值逐点查碰撞**，全过才 free |
| `trajectory_cost(theta_a, theta_b)` | 关节空间 L2 距离 |

### ⚠️ cuRobo 集成坑（已解决）
- `warp.torch` 在 warp 1.13 被移除 → monkey-patch 补回。
- `use_cuda_graph=True` 锁 batch size → 改 False。
- 空 `WorldConfig` 不清旧 cuboid → 用一个远处占位 cuboid 兜底。
- `goal_position.view(-1)` 非连续张量报错 → 加 `.contiguous()`。
- 关节限位（实测）：见 ur12e.yml，部分关节范围超 ±π。

---

## 7. M1.5 — 全局图 + Dijkstra `graph.py`

GBPlanner2 风格的**持久化无向图**（跨 cycle 累积，不重置）。

### `GraphNode`
`node_id, theta(6), ee_pos_world(3), ee_dir_world(3), cycle_added, last_gain, metadata`。

### `ExplorationGraph`
- 底层 `networkx.Graph` + `sklearn.BallTree`（6D 关节空间近邻）。
- `add_node / add_edge(weight)`（边权 = 关节空间距离；调用方负责验证 free）。
- `nearest_node(theta) / k_nearest / nodes_within(theta, radius)`。
- `dijkstra(src,dst) / dijkstra_length / dijkstra_to_many(src, dsts)`：最短路径（边权加权）。
- `frontier_nodes(gain_threshold)`：`last_gain ≥ 阈值` 的节点，按 gain 降序。
- `update_node_gain / stats`。

---

## 8. M1.6 — 增益函数 `gain.py`

`gain = w_vol · volumetric_gain + w_target · target_bias`

### `volumetric_gain(cam_pos, cam_dir, cam_up, voxel_map, cfg)`
- 在相机视锥内撒 `n_rays_h × n_rays_v`（默认 8×8=64）根稀疏射线。
- 每条射线在体素图里 raycast，累加沿途 **unknown 体素数**。
- 物理意义：朝未知空间看 = 信息增益大 → 驱动"探索/建图"。

### `target_bias(cam_pos, cam_dir, p_target)`
- `= cos(夹角(cam_dir, p_target - cam_pos))`，范围 [-1,+1]。
- **只奖励"朝向"目标，不奖励"距离"**（这是后面 M1.8 要补 approach 项的根本原因）。

### `GainConfig`：`w_vol, w_target, n_rays_h/v, max_range, fov_h/v_deg`。

---

## 9. M1.7 — 单 cycle 探索 `exploration.py: run_one_cycle`

一个 cycle 的 7 步：

### Step 0：更新碰撞世界
`arm_kin.update_world(voxel_map)`。

### Step 1：采样候选末端位姿
候选 = **探索候选 + 目标导向候选**（比例由 `frac_goal_candidates`=0.4 控制，M1.8.5 加）：

**探索候选** `sample_candidate_ee_poses(ee_pos_now, p_target, cfg, n_box)`：
- 位置：以当前 EE 为心、`sample_box_half`(=0.30) 的笛卡尔立方体内均匀采样。
- 朝向：光轴 z 指向 p_target，叠加 **yaw/pitch 抖动**（各 ±12°，必须 < 视锥半角否则目标晃出视野）。
- **roll 抖动**：绕光轴随机旋 ±`roll_jitter_deg`(=180°)。roll 不改变"看哪"，但**决定 IK 可行性**。
- 构造 OpenCV 系基 `R=[x,y,z]`，输出 (n,4,4)。

**目标导向候选** `sample_observation_candidates(p_target, cfg, n_goal, seam_limits, tangent)`：
- 直接在 **shell∩wedge** 里采样"理想观测位姿"：在楔形内随机方向 × `[d_min,d_max]` 随机距离放相机，精确瞄准 p_target。
- 每个位置**系统扫 `goal_roll_sweep`(=12) 个 roll**（roll 是 IK 关键 DOF，远侧位姿只有窄段 roll 可解，随机单 roll 命中率仅 ~0.5%，扫 12 个 ~5%）。
- 够不到时 IK 滤掉（靠探索候选绕行接近）；够得到时直接锁定合格观测。

### Step 2：IK 过滤 `filter_by_ik`
对每个候选位姿跑 cuRobo IK，保留有解的，返回 `(poses_pass, thetas_pass)`。

### Step 3：整臂碰撞过滤 `filter_by_collision`
对 IK 解 `collides()`，保留不撞的。

### Step 4–5：评分 + 选最优 `score_candidates`
（完整公式见 §10）。

### 锁定检测 + 边碰撞检查（M1.8.5）
- **锁定**：遍历碰撞通过候选，对**每一个**做 `detect_phase_switch`，第一个"能合格观测**且** `edge_free(theta_now→它)`"的直接选中（phase switch）。
  关键：合格观测不一定是分数最高那个，必须逐个验，否则漏掉。
- **否则**：按分数降序，取**第一个 `edge_free`** 的候选（保证 theta_now→它的关节空间直线运动不穿工件）。
- 若没有任何边无碰撞的候选 → 本 cycle `success=False`（触发回退）。

### Step 6：加节点 + 边
新位姿加入图；与最近节点之间**仅当 `edge_free` 成立才连边**（保证 Dijkstra 路径可执行）。

### Step 7：phase switch 检测 `detect_phase_switch`
遍历所有焊缝点，返回当前位姿能合格观测的点索引列表。

### `CycleResult`
`success, next_theta, next_ee_pose_world, best_score/gain/traj_cost, best_node_id, n_candidates_total/after_ik/after_collision, phase_switch_triggered, p_seam_observed`。

---

## 10. 评分公式（`score_candidates` 完整版）

对每个候选：
```
score = gain                                    # M1.6: w_vol·vol + w_target·target_bias
      + approach                                # M1.8/8.5: 拉向观测壳 (测地或欧氏)
      + bisector                                # M1.8: 对准楔形中线
      - lambda_cost · trajectory_cost           # 关节空间运动代价
```

- **gain**：`compute_gain`（§8），驱动探索/朝向。
- **approach**（混合，M1.8.5）：
  - 若有测地场 `geo_field`：取候选体素的测地距离 `gd`；网格内可达用 `min(gd, geo_field_clamp)`，**网格外/不可达退回欧氏** `|‖cam-P‖ - d_mid|`（边界处两者≈相等，平滑衔接）。
  - `approach = -w_approach · 距离`（`w_approach`=12000）。
  - 物理意义：把相机往"合格观测区（shell∩wedge）"拉；测地版能绕障。
- **bisector**（M1.8）：`+ w_bisector · (rel_unit · 中线)`（`w_bisector`=3000），让最终观测正对楔形中央。无 seam_limits 则跳过。
- **trajectory_cost**：`lambda_cost`(=50) × 关节空间 L2 距离，惩罚大幅移动。
- 注：`gains` 只含 M1.6 gain（不含 approach/bisector），方便 CycleResult 报告与 frontier 复用。

---

## 11. M1.8.5 — 绕障：测地距离场 `geodesic.py`

### 为什么需要
原 approach 用**欧氏距离**，无视障碍。当合格观测区在**工件另一面**时，绕行会让欧氏距离变大 → approach 把机械臂往近侧表面顶，绕不过去。

### `workspace_bounds(robot_pose, arm_reach, pad)`
网格范围 = base 为心、半边长 = `arm_reach`(1.9) + `pad`(0.8) 的立方盒。
**测地场活在网格里，所以网格必须罩住整个工作空间**（不能像早期那样只罩 mesh±0.5，否则远处起点查不到场值、失去梯度）。

### `build_shell_wedge_goal_mask(...)`
合格观测区的体素"种子"：到 p_target 距离 ∈[d_min,d_max] **且**在楔形内（用 seam_limits 解析判定）**且**可通行（非 occupied）。返回 (nx,ny,nz) bool。

### `geodesic_field(voxel_map, goal_mask)`
**GPU 波前 BFS（并行 Bellman-Ford）**：
- 种子（shell∩wedge 可通行格）距离 = 0；occupied 格 = inf（挡住）；free + unknown 都可通行（未知区乐观当通，边走边修）。
- 26 邻接，边权 = `res · √(dx²+dy²+dz²)`。
- 反复松弛（pad 边界为 inf 后按偏移切片取邻居最小值 + 边权），直到收敛。
- 输出每格"穿自由空间绕开障碍到合格观测区的真实路程"（米）；不可达 = inf。
- 实测：18 万格 ~27 ms，51 万格（工作空间网格）~250 ms。

### `sample_field_at(field, voxel_map, pts)`
查任意 world 点所在格的测地距离（越界返回 oob_value）。

### 物理意义
- **直线可达的缝**（如 seam_24）：种子在近侧正前方，梯度直接朝里，测地≈欧氏，行为不退化。
- **需绕障的缝**（如 seam_73）：种子在远侧楔形，场绕开工件，梯度从第 1 个 cycle 就引导相机朝绕路走。

---

## 12. M1.8 — 多 cycle 主循环 `run_phase1`

### `Phase1State`（跨 cycle 状态）
`cycle_idx, theta_now, voxel_map, graph, arm_kin, seam_*, p_target, p_observed(set),
stuck_count, last_global_cycle, best_dist_to_shell, solver_calls,
cycle_history, global_cycle_indices, ee_pos_history, ee_pose_history(4×4), theta_history(6)`。
> 后三个 history 记录**完整规划路径**（位置 + 位姿 + 关节角），供可视化复现。

### `Phase1Result`
`success, final_state, phase_switch_at, p_observed_first, total_time_s, fail_reason`。

### 主循环（每 cycle）
```
while cycle_idx < max_cycles:
  1) 在 theta_now 处拍 depth → voxel_map.integrate()              # 建图
  2) detect_phase_switch(当前位姿) → 若观测到任何点: 成功返回      # 切换检测
  3) 决策:
     if is_local_stuck(state):                                    # 卡住
         3a) 若 use_pose_solver 且 solver_calls<max: 调用求解器(§13)
             找到 → 移到该位姿(不立即判成功, 下一 cycle 拍图严格确认), continue
         3b) 否则 do_global_step()(§12.3); 没 frontier → 失败返回
     else:                                                        # 正常
         cyc = run_one_cycle(...)                                 # 单步探索
         if not cyc.success: stuck_count++
         else:
             theta_now = cyc.next_theta                           # 贪心执行最佳候选
             记录 ee_pose/theta history
             更新进度/卡住计数(§12.1)
  cycle_idx++
return 失败(max_cycles_reached)
```

### 12.1 进度与卡住判据（`dist_to_shell` + 关键修复）
- `dist_to_shell(ee, p_target, cfg)` = EE 到观测壳 [d_min,d_max] 的距离（壳内=0）。
- **进壳后不再触发 global**（已在正确区域，继续本地精修扫图清 LOS）；
  否则进度指标 `dist_to_shell` 进壳即饱和=0、被误判卡住 → global 把自己弹走 → 在壳边反复横跳。
- 未进壳时：`d_shell < best - progress_eps` 才算有进展，否则 `stuck_count++`。

### 12.2 `is_local_stuck`
`stuck_count ≥ patience(3)` **且** 距上次 global step 至少做过 1 次 local。

### 12.3 Global step `do_global_step`（GBPlanner2 风格回退）
1. `reevaluate_node_gains`：地图变了，重算图中所有节点的 gain。
2. `pick_frontier_node`：在 `frontier_nodes`（gain≥阈值）里选 `score = gain·exp(-global_lambda_cost·path_cost)` 最高的（path_cost = Dijkstra 关节路程）。
3. `dijkstra(当前节点 → frontier)`，沿路径逐节点移动 + 每点拍图建图。
4. 路径都是已验证 `edge_free` 的边 → 可执行。没 frontier/没路径 → 返回 False。

---

## 13. M1.8.5 — 合格位姿求解器 `solve_observation_pose`

针对**可行域极小的刁钻焊缝**（深凹角 + 远侧 + 臂展边缘，如 seam_73，合格位姿仅约 0.017%）。
探索循环每 cycle 只撒 ~百候选，命中针尖几乎不可能 → 卡住时调用本求解器：
- 一次性**大批量**（`solver_budget`=20000，分 chunk=1000）在 shell∩wedge 内精确瞄准 + roll 扫描。
- 批量 IK → 碰撞 → 合格检查（**宽松 LOS**：unknown 不算挡，找"几何上可观测的可达位姿"）。
- 返回最"居中"（贴中线）的合格 `(theta, pose, observed)`；某 chunk 一旦有命中就停（不耗尽预算）。
- 主循环拿到后**移过去拍一帧**（unknown→free），下一 cycle 严格判定确认。
- `solver_max_calls`(=3) 限制总调用次数控制开销。每格清 `empty_cache` 防 OOM。

---

## 14. `config.py` 全部超参速查

| 组 | 参数 | 默认 | 含义 |
|----|------|------|------|
| 起点 | `INITIAL_JOINT_ANGLES` | retract_config | Phase1 起点关节角 |
| 采样 | `sample_box_half` | 0.30 | EE 笛卡尔采样盒半边长(米)；太小绕不动 |
| | `n_samples` | 100 | 每 cycle 候选数 |
| | `yaw/pitch_jitter_deg` | 12 | 朝向抖动，须 < 视锥半角 |
| | `roll_jitter_deg` | 180 | 绕光轴滚转，管 IK 可行性 |
| | `frac_goal_candidates` | 0.4 | 目标导向候选占比 |
| | `goal_roll_sweep` | 12 | 每个目标位置扫几个 roll |
| 增益 | `w_vol` | 1.0 | 体积增益权重 |
| | `w_target` | 2000 | 目标偏置权重 |
| | `gain_n_rays_h/v` | 8/8 | 增益射线数 |
| | `gain_max_range` | 2.0 | 增益射线最远(米) |
| approach | `w_approach` | 12000 | 拉向观测壳权重 |
| | `w_bisector` | 3000 | 对准楔形中线权重 |
| 测地 | `use_geodesic_approach` | True | 用测地场绕障 |
| | `geo_field_clamp` | 5.0 | 测地距离上限/不可达填充(米) |
| | `geo_wedge_margin_deg` | 0 | 种子楔形收紧余量 |
| 评分 | `lambda_cost` | 50 | 关节运动代价权重 |
| | `g_high/g_low` | 100/5 | (保留)阈值 |
| 观测 | `obs_d_min/max` | 0.30/0.60 | 观测距离区间(米) |
| | `obs_los_unknown_as_block` | True | 严格 LOS：unknown 当障碍 |
| 体素 | `voxel_res` | 0.02 | (默认分辨率；批量用 0.05) |
| | `voxel_max_range` | 2.0 | raycast 最远 |
| | `arm_reach` | 1.9 | 臂展(米)，定网格大小 |
| | `workspace_pad` | 0.8 | 臂展外余量(米) |
| 相机 | `cam_fov_h/v_deg` | 86/57 | 视场角 |
| | `cam_height/width` | 360/640 | 分辨率 |
| cuRobo | `edge_step_rad` | 0.05 | 边碰撞插值步长(rad) |
| 求解器 | `use_pose_solver` | True | 卡住时启用 |
| | `solver_budget` | 20000 | 每次预算 |
| | `solver_max_calls` | 3 | 最多调用次数 |
| 主循环 | `max_cycles` | 100 | 安全终止 |
| | `patience` | 3 | 连续几次没进展算卡住 |
| | `progress_eps` | 0.01 | 进展最小缩短(米) |
| Global | `global_frontier_gain_threshold` | None(=g_low·2) | frontier 阈值 |
| | `global_lambda_cost` | 0.5 | global 路径代价权重 |

---

## 15. 工具脚本

### `tests/batch_eval.py` — 批量评估 + 保存路线
- 跑 `segment_sub_output` 下所有工件所有 `seam_*.pkl`（`--limit N` 限制）。
- **每完成 1 条立即更新** `tmp/batch_eval/summary.json` + `.csv`，并写 `routes/<工件>__<seam>.npz`。
- **可续跑**：已有 `.npz` 默认跳过（`--redo` 强制重跑）。
- 用**地面真值占据图**建图（工件 mesh 用 libigl 填 occupied，其余 free）。
- 命令：
  ```
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python phase1/tests/batch_eval.py --redo > tmp/batch_eval/run.log 2>&1 &
  ```

### 路线 `.npz` 字段（完整规划路径）
`part, seam, obj, success, switch_at, fail_reason, observed, cycles, solver_calls,
ee_path(N,3), ee_poses(N,4,4), theta_path(N,6), graph_pts(M,3), final_theta,
cam_fov, obs_d, seam_line, seam_tangent, seam_limits, p_weld, p_weld_idx, robot_pose`。

### `tests/visualize_route.py` — 可视化单条路线
- 无参 = 列出所有路线 + 成功率；带 `.npz` 路径 = 开 Open3D 窗口。
- 画：灰工件 / 品红焊缝(绿起点红终点) / 橙 P_weld / 黑起点球 / 红折线+蓝点 EE 路线 /
  每路点相机坐标系 / 终点视锥(成功黄/失败紫) / 青线楔形中线 + 灰线两边界 / 紫探索图节点。
- 用 `O3DVisualizer` 的 `add_3d_label` 在每路点标**步数**（`0 起点`/`1`/…/`N 观测`或`N 终止`）。

### 其它可视化
`tests/visualize_single_cycle.py`（M1.7 单 cycle）、`tests/visualize_multi_cycle.py`（M1.8 多 cycle 含中线对齐）、`tests/visualize_gain.py`（增益）。

---

## 16. 测试现状

- 单测：`test_single_cycle.py`(13) + `test_multi_cycle.py`(12) + `test_observation_check.py` + `test_mapping.py`(20) 等，合计 **50+ passed**。
- 端到端（地面真值图，多 seed）：
  - **seam_24 / seam_54 / seam_64**：稳定成功，1–5 cycle 进壳并触发观测。
  - **seam_73**：唯一难例（见 §17）。

---

## 17. 已知问题与限制

### 17.1 seam_73：针尖可行域
- 合格观测区在工件**远侧朝下的深凹角**里，又接近臂展极限（合格区离 base 1.44–1.66 m）。
- 6000 个理想观测候选 → IK 仅 ~14 → 碰撞 ~13 → 完全合格 **~1**（约 0.017%）。
- 几何上可观测（确认存在合格位姿），但可行域是"针尖"，随机/目标采样每 cycle 命中近乎不可能。
- 求解器（budget 4 万）在**真值图 + 宽松 LOS** 能找到（~35 s）；但端到端增量图 + 严格 LOS 仍难闭环。
- 决策方向（用户已选）：**加专门的合格位姿求解器**（§13，已实现，仍在打磨端到端闭环）。

### 17.2 ⚠️ 求解器远跳穿工件（当前未解决）
- 本地步已加 `edge_free` 检查，相邻路点直线运动不穿工件。
- **但求解器找到远侧位姿后是直接 teleport**（`theta_now = sol_theta`），这一大跳**没有规划无碰撞路径**，直线插值会穿过工件（可视化里表现为某一步线段切过 mesh，如 seam_2 加求解器后的 step8→9，关节跳幅 ~5.9 rad、EE 跳 0.57 m）。
- 根因：大幅重构型的"跳"需要**真正的运动规划**（cuRobo motion_gen）沿无碰撞轨迹移动，而非关节空间直线插值。
- 同理：Global step 沿 Dijkstra 走的是**已验证 edge_free 的图边**，那些是安全的；只有"求解器跳"和（理论上）任何未验证的大跳不安全。

### 17.3 可视化是"路点直线"近似
- 保存的是**路点**（每 cycle 一个相机位姿）。相邻路点间画的是直线；本地步因 edge_free 检查过、直线插值无碰撞，所以直线≈真实；但求解器跳那段直线不代表真实可执行轨迹（需 motion 规划）。

### 17.4 在线 vs 已知 CAD 的张力
- "在线探索未知空间"的设定 与 "工件 CAD 已知、碰撞应对真值工件" 之间存在张力。
- 当前批量评估用**地面真值占据图**（工件填 occupied，其余 free，全已知）做碰撞 + LOS，规避了"未知区碰撞不安全"的问题，但也使 LOS 永远通畅（不体现严格 LOS 的遮挡探索）。真·在线（起点全 unknown）需要把 unknown 当障碍做保守碰撞，是后续工作。

---

## 18. 关键设计决定与坑（编年）

| 决定/坑 | 原因/修复 |
|---------|-----------|
| 弃 nvblox 改自写 PyTorch 体素图 | nvblox CUDA 工具链编译困难 |
| trimesh+embreex `intersects_first` | 比 `intersects_location` 快 4–5 倍 |
| libigl `fast_winding_number` | 工件非 watertight，`trimesh.contains` 不可靠 |
| 删除 incidence 约束 | 用户要求，4 条约束 |
| 楔形用 中线+半角（非法向） | seam_limits 是 boundary_dirs，开口角可任意 |
| 起点用 retract_config 非 pkl.joint_angles | pkl 是焊接到位姿不是初始 |
| 加 approach 项 | target_bias 只管朝向不管距离，EE 会飘向开阔空间 |
| 进度/卡住用"到壳距离"非绝对分数 | approach 让绝对分变负，绝对阈值误判 |
| sample_box_half 0.15→0.30 | 0.15 太小，远距离绕不动卡在 ~1.1 m |
| 加 roll 抖动 + 目标候选 roll 扫描 | roll 决定 IK 可行性，固定 roll 漏掉可达位姿 |
| yaw/pitch 抖动 30°→12° | 30°>视锥半角 28.5°，目标被晃出视野 |
| 进壳后不触发 global | 否则进度饱和误判卡住、反复横跳清不了 LOS |
| 锁定检测验全部候选(非 argmax) | 合格位姿不一定分最高，只验 argmax 会漏 |
| 测地距离场绕障 | 欧氏距离无视障碍，绕行被当倒退 |
| 网格 = 工作空间(base±(reach+pad)) | 测地场活在网格里，网格须罩住相机会经过的全空间 |
| 合格位姿求解器 | 针尖可行域焊缝靠探索循环命中不到 |
| 本地步 edge_free + 图边 edge_free | 路点本身不撞≠路径不撞，直线插值会穿工件 |
| occupied 保护(integrate 不擦) | 射线 free-carving 侵蚀工件占据，碰撞图"少肉"致穿模 |

---

## 19. 待办（下一步）

- **M1.8.5 收尾**：求解器远跳改为 cuRobo motion 规划（或把求解器位姿连进图后 Dijkstra 到达），消除穿工件的大跳。
- **批量成功率**：跑全部 234 条焊缝，统计成功率，分析失败类型（多为 §17.1 类针尖缝）。
- **M1.9**：把深度源从 RaycastSim 切到 Isaac Sim 真渲染。
- **M1.10**：端到端集成 + 视频；Phase1 输出观测位姿 → cuRobo motion_gen 规划整条无碰撞执行轨迹。
