# plan_joint_single 规划提速 · 方案

## 背景

`scripts/demo_scene.py` 的调用链：

```
demo_scene.py
  → Scene.compute_pose_and_plan_path        (gt_gen/scene.py:1040)
    → Scene.plan_explore_path               (gt_gen/scene.py:1115)
      → generate_gt                         (gt_gen/main_loop.py:717)
        → si.plan_joint_single  ×多次        (gt_gen/stomp_iface.py:118)
          → stomp_planning_api.plan_to_joint_single
```

`generate_gt` 主循环里，**步①（直达 goal）每轮调一次 `plan_joint_single`，步②（规划 P*）按需再调一次**，`max_rounds=6`（`configs/default.yaml`），所以一次 `generate_gt` 大概 6~12 次 `plan_joint_single`。当前 `plan_joint_single` 执行很慢，本文分析成本结构并给出分层优化方案。

---

## 成本结构：慢在哪里

每次 `si.plan_joint_single` → `api.plan_to_joint_single` 都会**从零构建一个 `StompPlanner`**（`stomp_planning_api.py:512`），再跑 STOMP 求解。两块成本：

### 1. 重建 planner（纯重复开销）

`StompPlanner.__init__` 里最贵的是：

- `CollisionOracle(...)` → `RobotWorld(cfg)`（`plan_path_stomp_obstacle.py:136`）：每次都重新加载机器人配置、建碰撞检查器 CUDA buffer、做首次 kernel 预热。
- **机器人本体一轮都没变，只有碰撞 world 变了**（voxmap 逐轮长大）。这块是可以省掉的重复开销。

### 2. STOMP 迭代计算量

`plan_joint` 里新建 `ObstacleStomp` 并跑 `solve`（`plan_path_stomp.py:182`），恒跑满 `num_iterations`：

| 旋钮 | 值 | 来源 |
|------|----|----|
| `num_iterations` | 120 | `configs/default.yaml` planner.stomp |
| `num_batch` | 8 | 同上 |
| `num_rollouts_all` | 51 (=30 new + 20 old + 1) | `plan_path_stomp.py:115` |
| `num_timesteps` | 151 | `configs/default.yaml` planner.stomp |

每次调用 ≈ `120 × 8 × 51 × 151` ≈ **740 万次碰撞距离评估**（每次 = cuRobo FK + 世界碰撞 + 自碰撞）。这是实打实的 STOMP 迭代计算量，线性正比于上面几个旋钮。

### 3. 附带：每轮 empty_cache

`main_loop.py:800` 每轮开头 `torch.cuda.empty_cache()`——会把已缓存显存全部还给驱动，下一次规划又得重新分配，和"复用 planner"的思路冲突。这是之前为治 `plan_explore_path` 显存 OOM 加的（大卡/拆进程是另一条治法）。

---

## 优化计划（分层，按"收益/风险"排序）

### 第 0 步：先加拆分计时（必做，几行）

现在 `stomp_iface.py:133` 的 `[计时]` 把"建 planner + solve"算在一起，看不出占比。先把两段分开打印，确认到底是"重建"贵还是"迭代"贵，避免盲目调参。

### 第 1 层：复用 planner，只换 world（结构优化，不改精度，收益最大）

cuRobo `RobotWorld` 支持 `update_world(world_config)`（`curobo/wrap/model/robot_world.py:192`，内部 `load_collision_model`），可只换碰撞世界、不重建机器人和检查器。做法：

- 按 `checker_type` 缓存两个 `StompPlanner`：一个 PRIMITIVE（步①的 voxmap cuboid 世界）、一个 MESH（步②的 `world_plan`）。步②的 `world_plan` 跨轮不变，甚至可只建一次。
- 每轮只 `oracle.rw.update_world(new_world)`，跳过 robot cfg 加载 + 检查器分配 + kernel 预热。
- 配套：把 `main_loop.py:800` 的 `empty_cache()` 改为不是每轮都调（否则复用意义被抵消），显存风险需一起评估。

**⚠️ 风险点**：`CollisionOracle` 建时按当时 world 的盒/面数定了 `n_cuboids/n_meshes` 上限（`plan_path_stomp_obstacle.py:128-129`）。voxmap 长大后 cuboid 数可能超上限，需要建 planner 时给一个足够大的 cap。要实测确认 `load_collision_model` 是"重分配"还是"截断"。

### 第 2 层：调 STOMP 计算量（换精度换速度，需要拍板）

计算量线性正比于这几个旋钮（改 `configs/default.yaml`）：

- `num_timesteps: 151 → 81` 左右（几乎减半计算，轨迹分辨率下降）
- `num_iterations: 120 → 60~80`（收敛深度下降，可能掉成功率）
- `num_rollouts_new/old`（30/20）→ 适度减小

风险：绕障成功率下降，需要小规模跑良率对比。

### 第 3 层：算法级（收益高但改动大）

- **早停**：`solve` 现在恒跑满 `num_iterations`（`plan_path_stomp.py:185`）。加"已找到无碰撞 + 在限位候选且代价稳定就提前退出"，实际常常提前收敛，省一大截迭代。
- **热启动**：用上一轮轨迹做初值，替代线性插值初始化，减少所需迭代数。改动最大。

---

## 建议顺序

先做 **第 0 步 + 第 1 层**（不动精度、收益最大、风险可控），实测提速后，再决定要不要碰第 2/3 层。
