# GT 生成实现步骤（逐步追踪）

实现 [`gt-generation-curobo-implementation.md`](./gt-generation-curobo-implementation.md)
与 [`privileged-nbv.md`](./privileged-nbv.md) 的工程落地清单。

## 工作方式

- **自底向上**：先做能独立验证的原语，再串主循环。
- **一次一步**：每步完成并**人工确认**后，才做下一步。
- 可视化（体素图 / raycast / 候选 / B）穿插在各步验证中，不单列。

---

## Step 0 — 决策锁定 + 环境 + 代码骨架

**先敲定 8 项决策（否则后续返工）：**

- [ ] **决策 A**：自由约束作用于**整臂本体**还是**仅末端/工具**
- [ ] **决策 B**：机器人型号；有无 URDF + cuRobo robot 配置 `.yml`
- [ ] **决策 C**：真值场景格式（mesh / Isaac Sim / 已体素化）及如何提供
- [ ] **决策 D**：传感器/渲染（Isaac Sim 渲染深度 / 自己对 mesh raycast）
- [ ] **决策 E**：相机内参、FOV、量程、手眼外参
- [ ] **决策 F**：cuRobo 版本（核对「按版本核对」的 API）
- [ ] **决策 G**：GT 输出格式（对齐项目现有 `pkl` / `joint_angles`）
- [ ] **决策 H**：ROI 包围盒范围 + 体素分辨率

**产出**：仓库结构、配置文件、模块空函数桩。
**验证**：cuRobo 可 import；机器人配置可加载；`motion_gen.warmup()` 跑通。

---

## Step 1 — cuRobo 基础封装（IK + 规划，空世界）

- [ ] 封装 `init_curobo / solve_ik / plan_to_pose / plan_to_config`
- **依赖**：Step 0
- **验证**：空世界里从 `retract_config` 规划到目标位姿拿到平滑轨迹；IK 能解位姿。

---

## Step 2 — 三态体素地图数据结构

- [ ] ROI 网格 `FREE/OCCUPIED/UNKNOWN`；初始全 UNKNOWN；坐标互转；增删查
- **依赖**：Step 0
- **验证**：建图、set/get、坐标转换正确；能简单可视化。

---

## Step 3 — 传感器模拟（raycast 真值场景）

- [ ] 给定相机位姿 + 内参 + 真值场景，raycast 出穿过/命中体素
- **依赖**：Step 0
- **验证**：简单场景里 raycast 结果（FREE / 命中 OCCUPIED）符合预期。

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

当前：**Step 0（待决策）**。每完成一步在上面打勾并记录。
