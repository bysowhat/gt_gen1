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
- [x] **决策 H**：ROI = 可达范围 + 1.2 m，分辨率 2 cm

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

## Step 4 — 观测更新 `observe_and_update`

- [ ] 把 Step 3 结果合并进三态图（穿过→FREE，命中→OCCUPIED）
- **依赖**：Step 2、3
- **验证**：一次观测后地图正确更新；多次可累积。

---

## Step 5 — voxmap → cuRobo 碰撞世界同步（未知=障碍）

- [ ] `sync_collision_world`：把「OCCUPIED ∪ UNKNOWN」灌进 cuRobo voxel 碰撞世界
- **依赖**：Step 1、2
- **验证**：伸进 UNKNOWN 的构型判碰撞；该区观测变 FREE 后同构型变无碰撞。

---

## Step 6 — 整臂扫掠体积 + `motion_stays_in_free`

- [ ] 算 `qi→qi+1` 整臂扫掠体积，判断是否 ⊆ FREE
- **依赖**：Step 2（+ 决策 A 的碰撞球配置）
- **验证**：全在自由区的运动通过；伸进未知的被拒。

---

## Step 7 — `reach_pt` + 阻塞段 `B` 计算

- [ ] `plan_on_truth` 求 P\*；向前扫求 `reach_pt`；取前方一小段 UNKNOWN 为 `B`
- **依赖**：Step 1、2、6
- **验证**：`reach_pt` 停在未知前沿；`B` 是下一段未知体素（可视化对照）。

---

## Step 8 — 候选视点生成

- [ ] `cluster_centroids(B)` → `standoff_poses_looking_at` → IK → 保守可达性过滤
- **依赖**：Step 1、2、7
- **验证**：候选朝向 B、站位在 FREE 区、从当前位姿可达。

---

## Step 9 — 特权 NBV 打分与选择

- [ ] `raycast_reveal`（假设性）→ `gain = reveal ∩ B` → `score` → argmax；含 v1 直线加权 fallback
- **依赖**：Step 3、7、8
- **验证**：选出的视点揭开 B 最多；fallback 路径也能跑。

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
- [ ] **Step 4** — 待开始（下一步）

每完成一步在对应小节打勾并在此记录。
