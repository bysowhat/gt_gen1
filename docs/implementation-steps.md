# GT 生成实现步骤（逐步追踪） 

实现 [`gt-generation-curobo-implementation.md`](./gt-generation-curobo-implementation.md)
与 [`privileged-nbv.md`](./privileged-nbv.md) 的工程落地清单。

## 工作方式

- **自底向上**：先做能独立验证的原语，再串主循环。
- **一次一步**：每步完成并**人工确认**后，才做下一步。
- 可视化（体素图 / raycast / 候选 / B）穿插在各步验证中，不单列。

---

## Step 0 — 决策锁定 + 环境 + 代码骨架  ✅ 已完成

**8 项决策（已锁定，见 `configs/default.yaml`）：**

- [x] **决策 A**：**整臂本体**（whole_arm；机器人 cfg 含 9 个 link 碰撞球）
- [x] **决策 B**：`ur12e.yml`（6 轴 `xiaoyu_arm_joint1..6`，`base_link`→`xiaoyu_tip_link`，含 retract_config）
- [x] **决策 C**：watertight `.obj` mesh；焊缝/目标来自 `*_part.pkl` 的 `hanfeng_*` 字段
- [x] **决策 D**：自己对 mesh raycast，深度上限 3.0 m
- [x] **决策 E**：左目针孔相机 960×540（rgb+depth）；内参 fx≈fy≈477.4, cx=480, cy=270；
      装在 **Link6**，pos+quat(**wxyz**) 见 `configs/default.yaml`；参考 `kejian_benchmark/gt_benchmark_pose_fast`
- [x] **决策 F**：`env_isaaclab`（curobo 0.0.post1.dev24 / torch 2.7+cu128 / CUDA ✓）
- [x] **决策 G**：用本项目**自定义** GT 格式（不沿用 traj.pkl），schema 在 Step 11 定
- [x] **决策 H**：ROI = 整臂可达工作空间实测包围盒 + 余量（见下「ROI 标定」），分辨率 4 cm

### ROI 标定（决策 H 落实）

**实测**（20000 个随机关节角，整臂碰撞球的 base 系包围盒）：

| 轴 | 整臂可达范围(含球半径) | 跨度 |
|-----|------------------------|--------|
| x   | −1.61 → 1.84 m         | 3.45 m |
| y   | −1.80 → 1.85 m         | 3.65 m |
| z   | −0.98 → 2.29 m         | 3.28 m |

- 距 base 最大半径 ≈ **2.30 m**（末端最远可达 2.33 m，z 最高 2.32 → 焊缝 z=1.78 在可达范围内 ✓）。
- 关节限位受限（非整圈）：j1∈[−0.78,3.92]、j2∈[−2.60,0.36]、j3∈[−0.08,2.54]、j4∈[−3.30,−0.25]……
  所以可达区不是完整球，而是约 **3.5×3.6×3.3 m** 的盒子。

**确定的 ROI**（实测包围盒 + ~0.15–0.2 m 余量取整）：

```yaml
roi:
  center: [0.1, 0.0, 0.65]     # base 系，米
  dims:   [3.8, 3.9, 3.6]      # 米
  voxel_size_m: 0.04
```

- 覆盖检查：x 0.1±1.9=[−1.8,2.0] ⊇[−1.61,1.84] ✓；y ±1.95 ⊇[−1.80,1.85] ✓；z 0.65±1.8=[−1.15,2.45] ⊇[−0.98,2.29] ✓。
- 体素数 ≈ 95×98×90 ≈ **0.84M**（voxel=0.04；uint8 <1 MB），sync 的 ESDF 距离变换 <1 s。
  （0.02 m 会到 ~6.7M、sync 偏慢；焊接精度要求高时可再调细。）

**单一 ROI 来源**：`roi.center/dims/voxel_size_m` 是唯一事实来源，`init_curobo` 的 cuRobo voxel
世界与 `build_roi_voxmap` 都从它读 → 两网格**同框**（`build_roi_voxmap` 默认再外扩 1 体素，
包住 cuRobo 每轴 `1+floor(dim/voxel)` 的"+1"边界层）。已验证同框下 voxmap↔cuRobo 占据 100% 吻合、差 0。

**产出**：`configs/default.yaml`、`gt_gen/`（各步函数桩 + `config.py` + `compat.py`）、
`scripts/verify_step0.py`。

**验证**：`conda run -n env_isaaclab python scripts/verify_step0.py` → `STEP0_OK`
（环境 + 配置加载 + MotionGen 初始化/warmup 全通过）。

> ⚠️ **环境坑（已解决）**：warp 1.13.0 把 torch 互操作移到顶层（`wp.device_from_torch`），
> 而 curobo dev 检出仍调用旧的 `wp.torch.device_from_torch`（仅 world_mesh.py 1 处）。
> 用 `gt_gen/compat.py` 的 shim 修复（须在 `import curobo` 前 `import gt_gen.compat`），
> 不改动 curobo/warp 安装。warmup 还需世界至少有一个障碍物（占位 cuboid）。

---

## Step 1 — cuRobo 基础封装（IK + 规划，空世界）  ✅ 已完成

- [x] 封装 `init_curobo / solve_ik / plan_to_pose / plan_to_config / fk`（`gt_gen/curobo_iface.py`）
- **依赖**：Step 0
- **验证**：`scripts/verify_step1.py` → `STEP1_OK`（FK、IK 误差~0、plan_to_pose、plan_to_config 全过）。

> ⚠️ **坑（已解决）**：MotionGen 内置 IK（`mg.solve_ik`/cuda-graph）对**远离 retract 的目标收敛差**
> （位置误差 8–20cm），导致 `plan_single` 报 `IK_FAIL`。解决：handle 内建**专用多种子 IKSolver**
> （`num_seeds=50`、`use_cuda_graph=False`、无世界，世界碰撞交给规划/扫掠检查），
> `plan_to_pose` 改为 **IK→`plan_to_config`（关节空间规划）**。混用 `solve`/`solve_batch`
> 会触发 cuda-graph "changing goal type" 报错，故 IK 统一用 `solve_batch`。
> VOXEL 世界初始为全自由（ESDF=-max），Step 5 再灌障碍。

---

## Step 2 — 三态体素地图数据结构  ✅ 已完成

- [x] ROI 网格 `FREE/OCCUPIED/UNKNOWN`；初始全 UNKNOWN；坐标互转；增删查
      （`gt_gen/voxmap.py`：`ThreeStateVoxelMap` + `build_roi_voxmap`）
- **依赖**：Step 0
- **验证**：`conda run -n env_isaaclab python scripts/verify_step2.py [--viz]` → `STEP2_OK`
      （建图/形状、2000 点坐标 round-trip、越界 get→UNKNOWN/set 跳过、单个+批量 set-get、
      `non_free_mask` 与 `counts` 自洽、`build_roi_voxmap` 形状与中心；`--viz` 出 `/tmp/voxmap.png`）。

> **约定**：坐标系 = 机械臂 `base_link`（与 cuRobo 规划世界一致，Step 5 同步最省事）；
> `origin`=体素 [0,0,0] 最小角，`voxel_to_world` 返回体素**中心**；另暴露 `.center`
> 供 Step 5 映射到 cuRobo 的 center-based VoxelGrid。越界 `get`→UNKNOWN（仍属非 FREE，保守）。
> 所有查询/设置方法支持单个 `(3,)` 或批量 `(N,3)`。

---

## Step 3 — 传感器模拟（raycast 真值场景）  ✅ 已完成

- [x] 给定相机位姿 + 内参 + 真值场景，raycast 出穿过/命中体素
      （`gt_gen/sensor.py`：`load_camera_model` / `build_kinematics` / `link6_pose` /
      `camera_pose_from_config` / `load_truth_scene` / `raycast_observe`）
- **依赖**：Step 0
- **验证**：`conda run -n env_isaaclab python scripts/verify_step3.py [--viz]` → `STEP3_OK`
      （合成墙场景：大墙全命中 occ z≈D / free z<D / 都在 max_depth；max_depth 截断无命中；
      小墙部分命中；`camera_pose_from_config` 的 R 正交·det=1·相机-Link6 偏移=外参模长）。

> **设计**：`raycast_observe` 是**纯几何**（相机位姿 4x4 + 内参 + base 系 trimesh），输出
> base 系点 `(free_points, occ_points)`——free=各射线 near→min(命中距,max_depth) 前一格采样，
> occ=命中距≤max_depth 的命中点；**体素化交给 Step 4**（故 Step 3 只依赖 Step 0、可独立快测）。
> 射线方向 OpenCV 光学帧 `d=normalize([(u-cx)/fx,(v-cy)/fy,1])`→`R` 转 base；
> 相机位姿 `T_base_cam = T_base_Link6 @ T(extrinsic)`，FK 用轻量 `CudaRobotModel`（不起 MotionGen）。

---

## Step 4 — 观测更新 `observe_and_update`  ✅ 已完成

- [x] 把 Step 3 结果合并进三态图（穿过→FREE，命中→OCCUPIED）
      （`gt_gen/mapping.py`：`observe_and_update` + `commit_observation`）
- **依赖**：Step 2、3
- **验证**：`conda run -n env_isaaclab python scripts/verify_step4.py [--viz]` → `STEP4_OK`
      （单次观测：occ 贴墙 z≈0.97 / free 在墙前 / 墙后仍 UNKNOWN；
      再观测墙撤走：OCCUPIED 不降级、墙后 UNKNOWN→FREE、UNKNOWN 单调减少）。

> **合并策略（OCCUPIED 粘滞，保守避障）**：occ 点无条件置 OCCUPIED（允许 UNKNOWN/FREE→OCC）；
> free 点只把【当前非 OCCUPIED】体素置 FREE（绝不降级已知障碍）；单次观测内先 free 后 occ
> 保证命中体素最终为 OCCUPIED。离散化下边界体素会被不同射线判定冲突，粘滞保证「宁可多障碍、绝不少障碍」。
> `free_step` 默认取 `voxmap.voxel_size`（沿射线约每体素一采样）。

---

## Step 5 — voxmap → cuRobo 碰撞世界同步（未知=障碍）  ✅ 已完成

- [x] `sync_collision_world`：把「OCCUPIED ∪ UNKNOWN」灌进 cuRobo voxel 碰撞世界
      （`gt_gen/collision_sync.py`）
- **依赖**：Step 1、2
- **验证**：`conda run -n env_isaaclab python scripts/verify_step5.py` → `STEP5_OK`
      （全 UNKNOWN 同步 → retract/q2 均判碰撞；标 FREE retract 区 → retract 无碰撞、q2 仍碰撞；
      再标 FREE q2 区 → q2 无碰撞）。

> **实现**：cuRobo VOXEL 世界用 **ESDF**（带符号距离场，约定**占据为正、自由为负**，
> 见 `WorldVoxelCollision.get_sphere_distance(compute_esdf)`；初始全自由 = `-max_esdf_distance`）。
> `sync_collision_world`：① 取 cuRobo voxel 中心(base 系，`create_xyzr_tensor(transform_to_origin)`)；
> ② 在 voxmap 查三态，**非 FREE = 占据**；③ `scipy.ndimage.distance_transform_edt` 从占据掩码算
> 带符号 ESDF（`d_in - d_out`，比二值填充有梯度，利于规划）；④ `update_voxel_data` 写回。
> 按**中心查表**而非按下标对齐，故 voxmap 与 cuRobo 网格分辨率/范围无需逐一相等（cuRobo 每轴
> `1+floor(dim/voxel)`，voxmap 用 `round`，差一不影响）。全占据/全自由有快捷分支。

---

## Step 6 — 整臂扫掠体积 + `motion_stays_in_free`  ✅ 已完成

- [x] 算 `qi→qi+1` 整臂扫掠体积，判断是否 ⊆ FREE
      （`gt_gen/swept.py`：`swept_volume` + `motion_stays_in_free`）
- **依赖**：Step 2（+ 决策 A 的碰撞球配置）
- **验证**：`conda run -n env_isaaclab python scripts/verify_step6.py` → `STEP6_OK`
      （全 UNKNOWN→运动被拒；扫掠体积标 FREE→通过；扫掠区放 1 个 OCCUPIED→又被拒；零位移自由→通过）。

> **做法**：q_from→q_to 关节空间线性插值，子步数按"任一碰撞球单步位移 ≤ 分辨率"定（不漏体素，
> 分辨率默认取 `voxmap.voxel_size`）；逐步批量 FK 整臂碰撞球（决策 A=whole_arm），每球**保守**覆盖
> 其 AABB 与球相交的所有体素（中心到球心 ≤ r + 半个体素对角线）→ 并集 = 扫掠体积。
> 判定：扫掠体积内**只要有一个非 FREE**（OCCUPIED 或 UNKNOWN，含越界按 UNKNOWN）→ 该段不通过。
> 这是保守探索的核心闸门：只执行"整条臂扫掠体积已确认全自由"的运动。

---

## Step 7 — `reach_pt` + 阻塞段 `B` 计算  ✅ 已完成

- [x] `plan_on_truth` 求 P\*；向前扫求 `reach_pt`；取前方一小段 UNKNOWN 为 `B`
      （`curobo_iface.plan_on_truth` + `reach_b.compute_reach_pt` / `compute_blocking_B`）
- **依赖**：Step 1、2、6
- **验证**：`conda run -n env_isaaclab python scripts/verify_step7.py` → `STEP7_OK`
      （真值上 P* 规划到**由 pkl 几何算的 `goal_pose`**(`seam_ee_pose`，放开 roll 的 metric，非 pkl 关节角)，
      P*=78 点：全 UNKNOWN→reach_idx=0、B=1025；前半 FREE→reach_idx=39 停在未知前沿、B=454；
      整条 FREE→reach_idx=77 到终点、B=0；B 全为 UNKNOWN 体素）。

> **要点**：`plan_on_truth(handle, start, goal, pose_cost_metric=)` 须用【真值 world】(含真实障碍 mesh)的
> handle——P* 在已知全部障碍下规划，挡住探索的只会是 UNKNOWN。goal 可为关节角或**末端位姿**
> (pos,quat_wxyz)；位姿走 `plan_to_pose`(可放开 roll)。`compute_reach_pt` 用 `motion_stays_in_free`
> 沿 P* 向前扫，首个非 FREE 段前停；`compute_blocking_B` 取 reach_pt 前方 k(=`params.nbv.k_lookahead`) 段
> 扫掠体积里**仍 UNKNOWN** 的体素（不含 FREE/OCCUPIED）= "卡住下一步、只因没看过"的格子（滑动窗口）。

---

## Step 8 — 候选视点生成  ✅ 已完成

- [x] `cluster_centroids(B)` → `standoff_poses_looking_at` → 眼在手 IK → 保守可达性过滤
      （`gt_gen/candidates.py`：`cluster_centroids` / `standoff_poses_looking_at` /
      `look_at_pose` / `generate_candidates` → `Candidate` 列表）
- **依赖**：Step 1、2、7
- **验证**：`conda run -n env_isaaclab python scripts/verify_step8.py [--viz]` → `VERIFY_STEP8_OK`
      （冷启动 seam_22：初始圆柱 FREE 罩住 retract、cur_cfg=retract、沿 P* reach_idx=8/56、B=172。
      4 个 verify_* 各验一函数：① cluster_centroids 聚 1 簇心且在 B 包围盒内、无空簇；
      ② standoff_poses_looking_at 生成 10 个位姿，眼在 FREE/光轴对准 T/视线通/站距合档/旋转正交；
      ③ look_at_pose 平移=p/光轴对准 T/正交/X水平Y朝下/T 投影到画幅中心/退化仍正交；
      ④ generate_candidates 端到端产 5 个候选，逐个朝 B(≤20°)、IK 误差≤4.6mm、整臂扫掠⊆FREE 可达、
      自碰撞 OK、眼在 FREE。`--viz` 每个函数分步/逐窗用 open3d 展示几何。聚类参数取自 default.yaml.cluster）。

> **眼在手关键**：cuRobo IK 解的是 `ee_link`(xiaoyu_tip_link)，而相机挂在 `Link6`。要让
> **相机**到达某位姿 `T_base_cam`，须先换算成对应 **ee_link** 位姿再 IK：
> `T_base_ee = T_base_cam @ T_cam_ee`，其中 `T_cam_ee` 是 cam→ee 的固定刚体变换（与构型无关，
> 用任一构型 FK 一次性求出）。候选三关过滤：①IK 有解（关节限位+自碰撞，handle.ik）；
> ②真实 FK 光轴对准 T（兜 cuRobo 朝向阈值偏松，默认 ≤20°）；③`motion_stays_in_free`
> 当前构型→候选整臂扫掠 ⊆ FREE（保守，走不到的丢）。
> **初始 FREE 基础**：候选的站位/视线/可达都建立在初始引导 FREE 空间之上——冷启动用
> **圆柱体法** `set_initial_free_cylinder`（参数取自 `default.yaml` 的 `init_free.cyl_*`）罩住
> retract 整臂；这正是反复确认过的「眼在手可达视点常很有限」难点：候选基本是在初始 FREE
> 块内小幅重定向去看 B。**坑/接 stuck**：若无任何可达候选 → 主循环转「就近揭示」兜底（§6）。

---

## Step 9 — 特权 NBV 打分与选择  ✅ 已完成

- [x] `raycast_reveal`（假设性）→ `gain = reveal ∩ B` → `score` → argmax；含 v1 直线加权 fallback
      （`gt_gen/nbv.py`：`raycast_reveal` / `score_candidate` / `best_next_view_using_oracle` /
      `best_next_view_line_weighted` → `NBVResult`）
- **依赖**：Step 3、7、8
- **验证**：`conda run -n env_isaaclab python scripts/verify_step9.py [--viz]` → `VERIFY_STEP9_OK`
      （冷启动 seam_22，|B|=172、5 候选：① raycast_reveal 各候选 reveal 非空在界内、reveal∩B gain
      =[164,164,170,170,170]；② score_candidate λ=0 时 score=gain、argmax 选 gain=170；
      ③ best_next_view_using_oracle 端到端 status=ok、选中 cfg 揭开 B 170、自碰撞OK、可达；
      ④ 直线加权 fallback 跑通。`--viz` 显示选中视点 + 揭开的 B(绿)/未揭开(橙) + FOV 视锥）。

> **要点**：④的 raycast 是【假设性】的——从候选相机位姿对真值 mesh 投射、算"将确定"的体素，
> `gain = |reveal ∩ B|`，**不改地图**（真正改图是主循环 ⑥）；真值 mesh 的遮挡天然处理（被挡的 B
> 不算揭开）。`score = gain − λ·path_cost`（λ=`params.nbv.lambda_cost`，默认 0 → 纯比揭开量）。
> **P\* 注入**：`best_next_view_using_oracle(..., p_star=)` 支持外部传入真值最优路——主循环每轮算
> 一次 P* 注入即可，既省一次规划又避免 cuRobo 多种子的随机抖动（不传则函数内部自行规划）。

---

## Step 10 — 主循环编排（①~⑦）+ 冷启动 + 卡住处理

- [ ] 准备阶段 + ①~⑦ 串成 `generate_gt`；含 `look_around` 冷启动、`handle_stuck`
- **依赖**：Step 1–9
- **验证**：简单场景里机械臂从 retract 推进、绕障、到达目标；全程只走确认自由区。

---

## Step 11 — GT 导出

- [ ] 轨迹（关节角 + 时间戳 + 起点/目标/成功标志）按项目格式序列化
- **依赖**：Step 10
- **验证**：输出结构符合既有 `pkl`/约定，可被下游读取。

---

## Step 12 — 批量产 GT + 鲁棒性

- [ ] 多场景循环 / 批量并行；不可行场景标记；失败日志；参数化配置
- **依赖**：Step 10、11
- **验证**：跑一批场景产出数据集；不可行场景被正确跳过并记录。

---

## 依赖关系一览

```
Step0 ─┬─ Step1(cuRobo) ─┬─────────────────────────┐
       ├─ Step2(体素图) ─┼─ Step4(观测) ─ ...        │
       └─ Step3(raycast)─┘                          │
Step2+1 → Step5(碰撞同步)                            │
Step2+A → Step6(扫掠) → Step7(reach_pt/B) → Step8(候选) → Step9(NBV打分)
        所有 → Step10(主循环) → Step11(导出) → Step12(批量)
```

## 进度

- [x] **Step 0** — 完成（`STEP0_OK`）
- [x] **Step 1** — 完成（`STEP1_OK`）
- [x] **Step 2** — 完成（`STEP2_OK`）
- [x] **Step 3** — 完成（`STEP3_OK`）
- [x] **Step 4** — 完成（`STEP4_OK`）
- [x] **Step 5** — 完成（`STEP5_OK`）
- [x] **Step 6** — 完成（`STEP6_OK`）
- [x] **Step 7** — 完成（`STEP7_OK`）
- [x] **Step 8** — 完成（`STEP8_OK`）
- [x] **Step 9** — 完成（`VERIFY_STEP9_OK`）
- [ ] **Step 10** — 待开始（下一步）

每完成一步在对应小节打勾并在此记录。
