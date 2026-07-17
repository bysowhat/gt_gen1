# obstacle_type1_demo_main 流程提速方案

> 背景：`compute_goal_pose` 与 `si.plan_joint_single` 两个内核已单独优化过，本方案**不动这两个内核**，
> 只针对 `scripts/demo_scene.py::obstacle_type1_demo_main` 的**流程编排 / cuRobo 句柄复用 / 主循环每轮开销**提速。

## 〇、实测数据（2026-07-16，PROFILEMAIN 逐行计时）

工件：`钢柱_1dlcPQ...`（长 15.375m），seam 89，RTX 3060 12G。总计 **118.9s**：

| 行 | 内容 | 耗时 | 占比 | 说明 |
|---|---|---:|---:|---|
| L254 | `plan_init_pose_fast(verbose=False)` | 20.5s | 17.2% | 「纯几何、很快」预期外的重 |
| L260 | `compute_pose_and_plan_path("forehand")` #1（无障碍）| **79.1s** | **66.5%** | init#0 一次成功即返回；内含 compute_goal_pose 6.14s + 建 2 个 MotionGen + generate_gt×2 |
| L263 | `add_obstacle_type1(...)` | 0.24s | 0.2% | 轻 |
| L271 | `compute_pose_and_plan_path("forehand")` #2（加障碍后）| 18.8s | 15.8% | **全部耗在 compute_goal_pose 无解×2（8.97+8.91s），未产出轨迹** |
| L272 | `save` | 0.01s | — | 同进程只此一次落盘 |

**关键结论**：
1. 中间的 save/load 已被注释（②已由手工完成），当前单进程直连。
2. 大头是 **#1 的 79s**，但逐行 profiler 看不到内部拆分——需进一步埋点区分「建 MotionGen」vs「generate_gt 主循环」（已加 `PROFILEMAIN` 分段计时，见 §四）。
3. **#2 的 18.8s 是纯浪费**：加障碍后 goal pose 无解，却花两次 ES（各 ~8.9s、num_batches=8 主循环 301 轮）才发现。
4. `plan_init_pose_fast` 20.5s 出乎意料，需拆 `_fast_build_ctx`（建 15m 大 mesh 点集）vs `_fast_solve_weld`（4896 候选逐步过滤）。

---

## 一、这条流程实际在干什么


`obstacle_type1_demo_main` 单进程运行（无 isaacsim，可视化全部注释）：

1. `plan_init_pose_fast` —— 纯几何，建 `_fastctx`（按工件缓存），较轻
2. `save scene1 → load scene1`
3. `compute_pose_and_plan_path("forehand")` ← **重头戏**
4. `save scene1 → load scene1`
5. `add_obstacle_type1(...)` —— 纯几何，轻
6. `save scene2 → load scene2`
7. `compute_pose_and_plan_path("forehand")` ← **又一遍重头戏**
8. `save scene2`

每次 `compute_pose_and_plan_path`（`gt_gen/scene.py:1114`）内部：
最多 2 个 init pose，每个 → `compute_goal_pose`（ScenePose2 + ES）→ 循环内再 `save` 一次
→ `_build_explore_world` → 每个观测位姿 `plan_explore_path` → `generate_gt` 主循环。

## 二、优化点（按预计收益排序）

### 1. 复用 cuRobo / MotionGen 句柄（预计最大头）

`_build_explore_world`（`gt_gen/scene.py:1295`）每次都**新建两个 MotionGen**
（`h_truth` MESH + `h_expl` VOXEL），各自 `warmup` + 建独立 `IKSolver`
（`gt_gen/curobo_iface.py:173-191`）。这是秒级开销，却在**每个 init pose × 两次
compute_pose_and_plan_path** 都重来，累计可能 4–8 次全量构造。

- **`h_expl`（VOXEL 全自由世界）完全不依赖场景内容**（`ci.init_curobo(self.cfg)` 不传 world）。
  整个 demo 只需建 **1 次**，全程复用。
- **`h_truth`（MESH）** 依赖工件摆放 + 障碍，但 MotionGen 支持 `update_world(WorldConfig)`——
  比重建 MotionGen + warmup 便宜得多。建 1 次，之后每个 init pose / 加障碍后只 `update_world`。
- 需要一个 Scene 级句柄缓存（类似现有 `_k2ctx` / `_fastctx` 的模式），
  让 `_build_explore_world` 走"存在就复用 / 更新"分支。

### 2. 去掉同进程里多余的 save/load（①的前置条件，本身也省时间）

本函数全程单进程、无 isaacsim，中间 5–6 次 `save+load` 是纯磁盘往返：
`load`（`gt_gen/scene.py:933`）会重跑 `__init__` → `load_config` + 重新解析**整个 weld_json 的所有焊缝**，
`save` 要 pickle 整个 scene（含轨迹 / mesh）。`compute_pose_and_plan_path` 内部
`gt_gen/scene.py:1146` 还有一次**循环内 save**。

- 同进程直接持有 scene 对象即可，只在最后确实要落盘时 `save` 一次。
- **这是 ①的前提**：`load` 后句柄被丢成 `None`，只要还在 load，句柄就无法复用。

### 3. `generate_gt` 主循环每轮固定开销（`gt_gen/main_loop.py:799` 起）

主循环最多 `max_rounds=200` 轮，每轮都有：

- `torch.cuda.empty_cache()`（`gt_gen/main_loop.py:800`）——带 GPU 同步开销，逐轮累积；
  评估去掉或降频（如每 N 轮一次）。
- `si.world_from_voxmap_auto(cfg, voxmap)`（`gt_gen/main_loop.py:810`）——
  **每轮把 voxmap 全量转 mesh 重建 world/checker**。voxmap 是增量长大的，
  可评估增量更新 world 而非每轮全量重建。
- `_build_explore_world` 开头的 `gc.collect()` + `empty_cache()`（`gt_gen/scene.py:1314-1318`）——
  句柄复用后（①）这段 OOM 防御可放宽。

### 4. 加障碍后的第二次 `compute_pose_and_plan_path` 是否要重算 goal pose（需算法确认）

Type1 障碍加在**扫掠空间**里。第二次调用会把 `compute_goal_pose`（ScenePose2 + ES，num_envs≈800）
整段重跑。若障碍不影响观测位姿的可达性 / 视线，第一次的 `goal_poses`（joints 变体）可直接复用，
第二次**只重跑 `generate_gt`**（碰撞世界变了、路径必须重规划，但目标关节角不必重解）。
这一条需先确认障碍语义，收益可能很大（省掉一整轮 ES）。

### 5. ScenePose2 的复用（次要）

`compute_goal_pose`（`gt_gen/scene.py:1085`）每个 init pose 都 `new ScenePose2`（含 curobo robot world）
+ `new Optimizer`。若 ScenePose2 的 robot world 可只 `reset` 不重建，能省每 init pose 的建世界开销。
compute_goal_pose 已优化，此条列作备选，看是否已覆盖。

## 三、建议执行顺序（依实测修订）

1. **先跑分段计时（§四）**：确认 #1 的 79s 里「建 2 个 MotionGen」占多少、`generate_gt` 主循环占多少——
   决定先做 ①（句柄复用）还是 ③（主循环）。
2. **#2 的 18.8s 纯浪费**（收益明确、可先摘）：加障碍后 goal pose 无解还跑两次 ES。
   选项：(a) 复用 #1 的 `goal_poses`（需确认障碍是否遮挡视线/可达——若 plate 挡住则不能复用）；
   (b) 无解时把 init-pose 尝试上限降到 1，省掉第二次 ES；(c) 加障碍后先快速可行性判定再决定是否重跑 ES。
3. **①（句柄复用）**：`h_expl`（VOXEL，与场景无关）全程建 1 次；`h_truth`（MESH）建 1 次后 `update_world`。
   预计砍掉 #1/#2 里重复的 MotionGen warmup。
4. **`plan_init_pose_fast` 20.5s**：按 §四拆分后，若 `_fast_build_ctx` 占大头则缓存/降采样大 mesh 点集；
   若 `_fast_solve_weld` 占大头则优化 4896 候选的过滤向量化。
5. **③（主循环每轮开销）**：`empty_cache` 降频、`world_from_voxmap_auto` 增量更新。

## 四、已加的分段计时（PROFILEMAIN=1）

除 `scripts/demo_scene.py` 的逐行 profiler 外，`gt_gen/scene.py` 已加 `PROFILEMAIN` 连动的分段打点，
把上面的大头进一步拆开：

- `[PROFILEMAIN][plan_init_pose_fast] _fast_build_ctx=… _fast_solve_weld=…` —— 拆 L254 的 20.5s。
- `[PROFILEMAIN][compute_pose_and_plan_path] _build_explore_world 总=…（init#N）` —— 建世界总耗时。
- `[PROFILEMAIN][_build_explore_world] h_truth(MESH)=… h_expl(VOXEL)=…` —— 拆两个 MotionGen 构建。
- `[PROFILEMAIN][generate_gt] pose#N 主循环=… status=… rounds=…` —— 每个观测位姿的主循环耗时。

`gt_gen/main_loop.py::generate_gt` 内部再按【步骤累计】计时，循环结束打印一行汇总，
定位主循环里哪一步是瓶颈（每步总耗时 / 占比 / 每轮均摊）：

```
[PROFILEMAIN][generate_gt] 主循环分步耗时（N 轮，合计 X.XXXs）：
    step2_pstar   …s  …%  (每轮 …ms)   # 步② 规划 P*（plan_joint_single / plan_pose_single / plan_to_pose_all）
    nbv           …s  …%  (每轮 …ms)   # 步③④ best_next_view_using_oracle（含假设性 raycast 打分）
    step1_direct  …s  …%  (每轮 …ms)   # 步① 直达 goal 的规划
    step1_world   …s  …%  (每轮 …ms)   # 步① world_from_voxmap_auto（voxmap→mesh，每轮全量重建）
    move          …s  …%  (每轮 …ms)   # 步⑤ _move_to（实际移动 + 沿途拍）
    observe       …s  …%  (每轮 …ms)   # 准备阶段 + 步⑥ _observe（raycast 实拍）
    sync_world    …s  …%  (每轮 …ms)   # 步0 sync_collision_world（voxmap 非 FREE → h_expl）
    empty_cache   …s  …%  (每轮 …ms)   # 步0 torch.cuda.empty_cache
    stuck         …s  …%  (每轮 …ms)   # no_reachable_candidate 兜底 handle_stuck
```

下一步：`PROFILEMAIN=1` 再跑一次，据这四行把 #1 的 79s 定量拆开，再拍板先做 ① 还是 ③。


## 五、已落地的 nbv 优化（2026-07-16，实测）

按 §四埋点跑下来，`generate_gt` 主循环 80%+ 是 **nbv（best_next_view_using_oracle）**，单次 nbv 调用 ~32s。
再在 nbv 内部细分，确认瓶颈**不是**串行 IK（只占 ~15s→批量后 2.5s），而是两处 Python/小批量同步热点。
已实施四刀（均保持算法语义不变，纯 numpy 部分对拍逐位一致）：

| 刀 | 改动 | 文件 | 效果 |
|---|---|---|---|
| ① 批量 IK | `generate_candidates` 逐位姿 288 次 `solve_ik` → 分块 `solve_batch`（chunk×num_seeds 控显存） | `curobo_iface.solve_ik_batch_configs` / `candidates.py` | candgen 的 ik 段 ~15s → 2.5s |
| ② 减 return_seeds | 每位姿只用 top-2，`return_seeds` 200 → 8（`num_seeds` 仍 200，搜索不变） | `candidates.py` / `default.yaml`(nbv.ik_return_seeds/ik_batch) | 省 GPU→CPU 拷回 |
| A 打分去 Python + 批量 FK | `_count_intersection` set 遍历 → numpy `isin`；两次焊枪代价 FK → 全候选一次批量（`_gun_costs_batched`） | `nbv.py` | score 的 cost 段 ~73ms/候选 → 0.5ms |
| B 扫掠体素化向量化 | `voxelize_spheres` 逐球 meshgrid+set → 统一偏移网格广播（判据不变） | `swept.py` | `motion_stays_in_free` 106ms → 61ms/次，filt 13.4s → 7.8s |

**总效果**：`obstacle_type1_demo_main` 总时长 **214s → 104.5s（≈2.05×）**；单次 nbv ~32s → ~22s；功能不退化（`序列成功`）。

**剩余大头（未做，可选）**：
- `raycast_reveal`（carve_observe）**204ms/候选 × ~56 = 11.5s**，是打分循环现在的 #1。每候选对 217k 体素做一遍 CPU 视锥筛 + warp kernel + GPU 同步。批量化需把多相机做成 2D warp kernel（voxels×cameras），改动到 `sensor.py`，较大。
- `_build_explore_world` 每次重建 2 个 MotionGen（h_truth MESH ~19s、h_expl VOXEL ~5s），是 nbv 之外的另一 ~24s 大头；h_expl 与场景无关可全程建 1 次、h_truth 建 1 次后 `update_world`。
- `plan_init_pose_fast` ~22s（`_fast_build_ctx` 12.7s + `_fast_solve_weld` 8.3s）。


