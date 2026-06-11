# Step 10 实现计划 — 主循环编排（goal 后退 standoff + 纯三态 voxel 探索世界） 

> 状态：**待批准**（计划文档，未开始编码）。

## Context（为什么这么改）

旧 Step 10（已被回退到 `c01faf7 step9`）为了让焊枪能 mm 级贴到焊缝表面，给探索世界
`h_expl` 加了「UNKNOWN 体素 + 工件 mesh」的 **voxel+mesh 混合**架构，用来消除体素 ~4cm
膨胀造成的「碰到 voxel 但其实没碰到工件 mesh」的假碰撞。

本次换一条更简单的路线：**把 `goal_pose` 沿焊缝两面角平分方向（bisector）后退 10cm**，
让末端目标落在离工件表面足够远（10cm > 体素膨胀带）的自由空间里。这样：

- 纯三态 voxel 探索世界（非 FREE = 障碍）就够用，**不再需要 voxel+mesh 混合策略**；
- GT 轨迹**终点就停在这个 standoff 位姿**（离焊缝 10cm），不再向表面靠近（已与用户确认）；
- 10cm 这个值作为单一来源写进 `configs/default.yaml`。

**voxel 仍是之前的三态方案**：`UNKNOWN/FREE/OCCUPIED`（`gt_gen/voxmap.py`）。主循环每轮
通过观测**更新占用**（FREE/OCCUPIED 随 raycast 增长），再 `sync_collision_world` 把最新
「非 FREE」推进 cuRobo voxel 碰撞世界 —— 探索世界的障碍完全来自三态图，不含任何 mesh。

---

## 架构：4 个「世界」分工（详解）

总览：`h_truth`（真值，知道全部障碍）只用来「决定往哪看 / 选下一步」；`h_expl`（探索，
只知道已观测自由区）才是「机械臂真正能走的世界」，GT 全部来自它。`truth_scene` 是 raycast
的几何源，`voxmap` 是横跨两者的「已观测记忆」。

```
                 选方向(特权)                       记忆
   h_truth(MESH) ───────────►  NBV/P*  ───┐    ┌── voxmap(三态) ──┐
        ▲                                 │    │  (读: 算B/可达/打分)│
        │ raycast 几何                     ▼    ▼                   │ 写: 观测更新占用
   truth_scene(trimesh) ──► observe/reveal ──► 更新FREE/OCCUPIED ──┘
                                                  │ sync(非FREE→ESDF)
                                                  ▼
                                          h_expl(VOXEL) ──► 规划①/⑤ ──► GT
```

---

### ① `h_truth` —— MESH 真值世界（含工件 mesh）

**做什么用**：扮演「全知特权专家」。在已知全部障碍（工件 mesh）的前提下算最优路、生成并
校验候选视点、给候选打分。它**不参与 GT 的实际执行**，只指导探索方向。

**怎么建**（参考 `verify_step8.build_scene` L56–63）：

```python
# 工件@机械臂基座位姿 = inv(T_world_robot) ∘ T_world_piece（cuRobo Pose, wxyz）
mp = _p(robot_pose).inverse().multiply(_p(piece_pose)).get_pose_vector()[0].cpu().numpy().tolist()
world  = WorldConfig(mesh=[Mesh(name="workpiece", file_path=obj, pose=mp)])
h_truth = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH)
```
- 输入：`cfg`（Config）、工件 `obj` 路径、`mp`（工件 7 维位姿 `[x,y,z,qw,qx,qy,qz]`）。
- 输出：`CuroboHandle`（含 `mg` MotionGen、`ik` IKSolver、`ta`、`config`、`voxel` 元信息）。

**调用的函数 / 输入输出**：

| 函数（`gt_gen/...`） | 输入 | 输出 | 作用 |
|---|---|---|---|
| `free_pose_metric(h_truth, free_rot=(0,))` | handle | `PoseCostMetric` | 放开焊枪绕接近轴自转（roll 自由），P\*/IK 都用它 |
| `plan_on_truth(h_truth, cur_cfg, goal_pose, max_attempts, pose_cost_metric=metric)` | 起点关节角 `(6,)`、`goal_pose=((x,y,z),(qw,qx,qy,qz))`、metric | `P*` 插值关节序列 `(T,6) np.float`；失败 `None` | 全知最优路（②的 P\*）。挡路的只可能是 UNKNOWN |
| `best_next_view_using_oracle(h_truth, cur_cfg, vm, truth_scene, goal_pose, camera_model=cam, pose_cost_metric=metric, p_star=P)` | 当前构型、voxmap、truth_scene、goal、已算好的 P\* | `NBVResult(status, cfg, cam_pose, target, gain, score, reach_idx, n_B, n_candidates, P_star)` | 一轮 NBV：②reach_pt/B → ③候选 → ④假设性 raycast 打分 → argmax |
| （②③④ 内部）`compute_reach_pt` / `compute_blocking_B` | handle, vm, P\* | `reach_idx:int` / `B:(M,3) 体素下标` | 沿 P\* 扫到自由区尽头、取前方一小段仍 UNKNOWN 的体素 |
| （③内部）`generate_candidates(h_truth, vm, B, cam, cur_cfg)` | handle, vm, B, 相机, 当前构型 | `list[Candidate]`（config/cam_pose/target/ik_pos_err/look_err_deg） | 候选 IK + 看向校验 + 可达过滤；**用 `check_state(h_truth)` 过滤工件碰撞** |
| `check_state(h_truth, cfg)` | 关节角 `(6,)` | `(feasible:bool, constraint:float)` | 单构型是否撞工件/自碰/越限。验证阶段也用它判「0 真值碰撞」 |

> 为什么候选生成挂在 h_truth：候选构型不能扎进真实工件，需要工件 mesh 的碰撞判定；而
> 「能不能经自由区走到」由 `motion_stays_in_free(handle, vm, …)` 查 **voxmap** 决定（见下），
> 与 handle 的世界无关，所以保守性不受影响。

---

### ② `h_expl` —— VOXEL 纯三态探索世界（无 mesh）

**做什么用**：机械臂**真正执行**的世界。只把「已观测自由区（FREE）」当可通行，
「OCCUPIED ∪ UNKNOWN」一律当障碍 → 规划出的轨迹天然只走确认自由区。GT 全部来自这里。

**怎么建**（`init_curobo` 不传 world_model 即默认 VOXEL 全自由，见 `curobo_iface.py` L125–129）：

```python
h_expl = ci.init_curobo(cfg)     # world_model=None → {"voxel": {dims/pose/voxel_size}} + CollisionCheckerType.VOXEL
```
- 输入：`cfg`（ROI 的 `dims/center/voxel_size_m` 从 `cfg.roi` 读，与 voxmap 同框）。
- 输出：`CuroboHandle`，初始为**全自由** voxel 世界（占据特征待 sync 写入）。

**调用的函数 / 输入输出**：

| 函数 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `sync_collision_world(h_expl, vm)`（`collision_sync.py`） | h_expl、三态 vm | `{'occupied':int, 'free':int}` | **每轮先调**：按体素中心查 vm，非 FREE→占据，用 scipy 距离变换算 ESDF（占据+/自由−）写回 cuRobo。副作用：更新 voxel feature |
| `plan_to_pose(h_expl, cur_cfg, goal_pose, max_attempts, pose_cost_metric=metric)` | 当前构型、goal、metric | `MotionGenResult`（`.success` / `.get_interpolated_plan()`）；IK 无解 `None` | **①** 试在确认自由区直接规划到 goal；成功即收尾 |
| `plan_to_config(h_expl, cur_cfg, target_cfg, max_attempts)` | 当前构型、目标构型 `(6,)` | `MotionGenResult` | **⑤** 移动到选中视点（在 `_move_to` 内） |
| `res.get_interpolated_plan().position` | — | `(T,6) torch→np` | 取插值轨迹，`[1:]` 接进 GT（去掉与上段重复的首点） |

> `h_expl` 的占据**完全来自三态 voxmap**：每轮观测把更多 UNKNOWN 变成 FREE/OCCUPIED，
> `sync_collision_world` 把最新「非 FREE」重灌进 ESDF → 已确认走廊随轮次延长。无任何工件 mesh。

---

### ③ `truth_scene` —— trimesh 真值网格（raycast 几何源）

**做什么用**：所有 raycast 的几何来源。两种 raycast 角色不同但同一几何：
- **⑥ 实拍**（真实发生、提交进 voxmap）；
- **④ 假设性**（只为 NBV 打分、不改图，已封装在 `best_next_view_using_oracle` 内部）。

**怎么建**（`sensor.load_truth_scene`）：

```python
truth_scene = load_truth_scene(obj, mesh_pose=mp)   # 与 h_truth 用同一 obj + 同一 mp
```
- 输入：工件 `obj` 路径、`mp`（同 h_truth 的工件位姿）。
- 输出：trimesh 场景（内含 ray 引擎），表达 base 系下的工件表面。

**调用的函数 / 输入输出**：

| 函数 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `observe_and_update(vm, cam_pose, cam, truth_scene, max_depth, pixel_stride)`（`mapping.py`） | voxmap、相机位姿 `4x4`、相机模型、truth_scene、`max_depth=cfg.max_depth_m` | `dict`（本次生效的 FREE/OCCUPIED 体素数） | **⑥**：raycast→穿过标 FREE、命中标 OCCUPIED，提交进 vm。**更新占用的唯一入口** |
| （④内部）`raycast_reveal(vm, cam_pose, cam, truth_scene)`（`nbv.py`） | 同上（候选相机位姿） | `reveal:(N,3) 体素下标` | 假设「去那会确定哪些体素」，与 B 求交得 `gain`；**不改图** |

---

### ④ `voxmap` —— 三态体素图（已观测记忆，横跨上面三者）

**做什么用**：整个系统**唯一的持久记忆**，记录每个体素是 `UNKNOWN(0)/FREE(1)/OCCUPIED(2)`。
所有「该往哪看、能不能走、走廊到哪、是否进展」都查它。

**怎么建**（`voxmap.build_roi_voxmap` + 冷启动 `init_free.set_initial_free_cylinder`）：

```python
vm = build_roi_voxmap(cfg)                      # 按 cfg.roi 建全 UNKNOWN 图
n  = set_initial_free_cylinder(h_truth, vm, config=cfg)   # 罩住 retract 整臂的圆柱预标 FREE（冷启动立足之地）
```
- 输入：`cfg.roi`（center/dims/voxel_size_m）、`cfg.init_free`（cyl_radius_m/cyl_height_m）。
- 输出：`ThreeStateVoxelMap`；初始除圆柱内为 FREE 外全 UNKNOWN。

**被谁读 / 写**：

| 方向 | 函数 | 查/改什么 |
|---|---|---|
| 读 | `compute_reach_pt` / `compute_blocking_B`（`reach_b.py`） | 沿 P\* 的扫掠体素是否 FREE；前方一小段哪些仍 UNKNOWN = B |
| 读 | `motion_stays_in_free(handle, vm, q_from, q_to)`（`swept.py`） | 候选/移动的整臂扫掠体素是否全 ⊆ FREE（保守可达性，验证也用它） |
| 读 | `sync_collision_world(h_expl, vm)` | 取各体素三态 → 非 FREE 当障碍灌进 h_expl |
| 读 | `raycast_reveal` / `score_candidate` | 打分时只数 reveal 里仍 UNKNOWN 且 ∈ B 的 |
| 读 | `vm.counts()[FREE]` | 进展判定（每轮 FREE 是否增长） |
| **写** | `observe_and_update`→`commit_observation` | 穿过的体素置 FREE、命中置 OCCUPIED（sticky）——**更新占用** |
| **写** | `set_initial_free_cylinder` | 冷启动把圆柱内整块置 FREE |

---

## 改动清单

### 1. `configs/default.yaml` — 新增 `goal:` 段 + `params.loop.observe_every_n`（单一来源）

新增顶层 `goal:` 段：

```yaml
goal:                                   # 末端目标相对焊缝的摆放
  standoff_cm: 10.0                     # 沿两面角平分方向(bisector, 背离工件)后退的 standoff 距离(cm)
                                        # 目的：目标离工件表面够远(>体素膨胀带)，纯 voxel 探索世界不会误判碰撞
```

并在既有 `params.loop` 段补一个采样间隔（步⑤ 沿途拍照的 N）：

```yaml
params:
  loop:
    max_rounds: 200                     # 主循环最大轮次（已存在）
    stuck_rounds: 5                     # 连续无进展判卡死（已存在）
    observe_every_n: 10                 # 步⑤ _move_to 移动时，每隔 N 个插值路点 _observe 一次（边走边拍）
```

> `config.params` 已把整个 `params` dict 暴露出来，`_move_to` 直接读
> `params["loop"]["observe_every_n"]`（缺省回退 10），**无需新增 config 访问器**。

### 2. `gt_gen/config.py` — 加访问器

```python
@property
def goal_standoff_m(self) -> float:
    """末端目标沿 bisector 后退的 standoff 距离（米）。见 default.yaml goal.standoff_cm。"""
    return float(self.raw.get("goal", {}).get("standoff_cm", 0.0)) / 100.0
```

### 3. `scripts/plan_seam.py` — 已改（本次会话早先），保持

`seam_ee_pose(d, index=None, offset_m=0.0)` 现返回 `((pos, quat_wxyz), dbg)`，
`dbg = {"seam_mid", "seam_bisector"}`，`pos = mid + offset_m * bis`。无需再动。

### 4. 修复被 `seam_ee_pose` 新返回值打断的调用方（必须，否则 verify 报错）

`seam_ee_pose` 由返回 `(pos,quat)` 改成了 `(pose, dbg)`，以下调用要解包：

- `scripts/verify_step7.py:45`：`goal_pose = seam_ee_pose(d)` → `goal_pose, _ = seam_ee_pose(d)`
- `scripts/verify_step8.py:53`：`goal_pose = seam_ee_pose(d)` → `goal_pose, _ = seam_ee_pose(d)`
  （`verify_step9` 复用 `verify_step8.build_scene`，改这一处即覆盖）

> 注：step7/8/9 的 goal 保持 offset=0（落在焊缝表面）—— 它们测候选/NBV 机件，与 standoff
> 无关，不引入 offset 以免动既有断言。standoff 仅 step10 使用。

### 5. `gt_gen/main_loop.py` — 实现（当前是 stub）

按 `docs/privileged-nbv.md §4.5`「准备阶段 + ①~⑦」实现。

**签名**：
```python
def generate_gt(h_truth, h_expl, voxmap, truth_scene, goal_pose,
                camera_model=None, params=None):
    """完整 ①~⑦ 主循环。返回 (GT, status, info)。
       GT: (T, dof) np.float64；status ∈ {reached, infeasible, stuck, max_rounds}。"""
```
- `camera_model` None → `load_camera_model(h_truth.config)`；`params` None → `h_truth.config.params`。
- 起点 `cur_cfg = list(h_truth.config.retract_config)`，`GT = [cur_cfg]`。

**准备阶段（冷启动）**：vm 已由调用方罩好初始 FREE；先 `_observe(...)` 一次（首次更新占用）；
`metric = free_pose_metric(h_truth, free_rot=(0,))`。

**主循环**（≤ `params.loop.max_rounds`，默认 200）。下面**逐过程**说明：每一步在做什么、
调用哪个函数、输入什么、得到什么、然后怎么用。一轮的骨架：

```
准备：cur_cfg=retract; GT=[cur_cfg]; free_prev=vm.counts()[FREE]; reach_prev=-1; stale=0
      _observe(...)（冷启动补拍一次）; metric=free_pose_metric(h_truth, free_rot=(0,))

for round in range(max_rounds):
    torch.cuda.empty_cache()
    sync_collision_world(h_expl, vm)                              # 步0
    res = plan_to_pose(h_expl, cur_cfg, goal_pose, metric)        # 步①
    if res 成功: GT += plan[1:]; status="reached"; break
    P   = plan_on_truth(h_truth, cur_cfg, goal_pose, metric)      # 步②
    r   = best_next_view_using_oracle(h_truth, cur_cfg, vm,       # 步③④
              truth_scene, goal_pose, camera_model=cam,
              pose_cost_metric=metric, p_star=P)
    分支 r.status →                                              # 步⑤
        scene_infeasible      : status="infeasible"; break
        ok                    : seg=_move_to(h_expl, vm, cur_cfg, r.cfg, ...); cur_cfg=r.cfg
        corridor_confirmed    : （见 5c）按 reach_idx 推进或收尾
        no_reachable_candidate: handle_stuck(...)（见 5d）
    _observe(vm, h_truth, cur_cfg, cam, ...)                      # 步⑥ 终点补拍
    进展判定（步⑦）：free/reach 是否增长 → 更新 stale；stale≥stuck_rounds → status="stuck"; break
else:
    status="max_rounds"
return np.asarray(GT), status, info
```

> **一句话读懂整个循环**：机械臂只敢在「亲眼看过是空的」区域里走。每一轮先问「现在能不能直接
> 走到目标？」（步①），不能就请一位「上帝视角的教练」（h_truth）指一个**最值得去看一眼**的位置
> （步②③④），走过去、边走边拍照把更多未知变成已知（步⑤⑥），再回到步① 重问。已知区域像
> 水面一样一轮轮扩大，直到淹没目标，步① 自然成功。步⑦ 防止「原地打转、毫无新发现」时无限循环。

#### 步 0 —— `sync_collision_world(h_expl, vm)`（`gt_gen/collision_sync.py`）

**这一步在解决什么问题**：`h_expl` 是机械臂真正规划用的世界，但 cuRobo 的 voxel 碰撞世界本身不认识
「三态」——它只有「障碍 / 非障碍」。我们的安全准则是「只走亲眼确认过是空的格子」，也就是说
**UNKNOWN 必须被当成障碍**（没看过 = 不许进），只有 FREE 才放行。这一步就是把三态 voxmap 的最新
状态「翻译」成 cuRobo 能用的障碍场：凡是**非 FREE**（OCCUPIED 或 UNKNOWN）的体素，统统标成障碍。

**为什么必须每轮开头都做一次**：上一轮机械臂走动 + 拍照，让一批 UNKNOWN 翻成了 FREE/OCCUPIED
（voxmap 变了）。但 `h_expl` 还停留在上一轮的障碍场，不重灌的话，步① 看到的还是旧走廊、找不到
刚打通的新路。所以每轮第一件事就是把「最新已知地图」推给 `h_expl`，让后续规划基于最新认知。

- **输入**：`h_expl`（VOXEL handle）、`vm`（三态图，已含上一轮观测的增量）。
- **输出**：`{'occupied':int, 'free':int}`（这次写进去多少占据 / 自由体素，纯诊断用）。
- **底层动作**：按体素中心查 vm 三态 → 非 FREE 记为占据 → scipy 距离变换算 ESDF（占据处为正、
  自由处为负的有符号距离场）→ 写回 `h_expl` 的 voxel feature。cuRobo 规划时用这个 ESDF 判碰撞。

#### 步 ① —— `plan_to_pose(h_expl, cur_cfg, goal_pose, ..., pose_cost_metric=metric)`

**这一步在问什么**：「以现在已经确认是空的这片区域，机械臂能不能**一口气**从当前位置规划到最终
目标（standoff goal）？」这是每轮的乐观尝试——**能收尾就立刻收尾**。注意它跑在 `h_expl` 上，UNKNOWN
已经在步0 被当成障碍，所以**只要规划成功，这条路必然全程在已确认 FREE 里**，天生满足保守安全准则。

**成功意味着什么**：已知自由区已经连通了「当前位置 → 目标」，探索任务完成。把这段插值轨迹接进 GT，
`status="reached"`，**break** 退出循环。这就是正常结束的出口。

**失败意味着什么**：当前已知 FREE 区还没把目标和当前位置连起来——中间还隔着没看过的 UNKNOWN
（被当障碍挡住了）。失败不是错误，而是「还得再探索」的信号，于是落到步②，去问教练「下一步该看哪」。

- **输入**：`h_expl`（步0 刚同步过）、起点 `cur_cfg (6,)`、`goal_pose=((x,y,z),(qw,qx,qy,qz))`、
  `metric`（放开焊枪绕接近轴 roll 自转，让 IK 多一个自由度更容易求解）。
- **输出**：`MotionGenResult`（`.success`；成功时 `.get_interpolated_plan().position` → `(T,6)`）；
  连 IK 都无解时返回 `None`。
- **怎么用**：成功 → `GT += plan[1:]`（去掉与上段重复的首点）、`status="reached"`、break；失败 → 进步②。

#### 步 ② —— `P = plan_on_truth(h_truth, cur_cfg, goal_pose, ..., pose_cost_metric=metric)`

**为什么要一个「作弊」的规划**：步① 失败只告诉我们「现在走不通」，但没告诉我们「该往哪个方向探索
才能走通」。于是请出全知教练 `h_truth`——它知道工件 mesh 的全部真实形状，在「假设所有障碍都已知」的
前提下规划一条**理想最优路 P\***。关键洞察：因为真障碍对 P\* 全都已知、已避开，那么**唯一可能挡住我们
沿 P\* 前进的，就只剩「还没观测的 UNKNOWN」**——而 UNKNOWN 是可以靠「去看一眼」消除的。于是 P\* 就成了
「最该优先把哪一段未知区看清楚」的路标。这是整套特权 NBV 的核心：用真值算出方向，但只用它**指路**，
绝不让它替机械臂走（机械臂仍只在 `h_expl` 的已知 FREE 里走）。

- **输入**：`h_truth`（MESH handle，含工件 mesh）、`cur_cfg`、`goal_pose`、`metric`。
- **输出**：插值关节序列 `P:(T,6)`；若真值上目标本身就不可达（机械臂构型 / 自碰 / 撞工件无解）→ `None`。
- **怎么用**：把 `P` 作为 `p_star` 注入步③。**每轮只算一次 P\*** 再复用，既省一次规划，也避免 cuRobo
  多种子 IK 的随机性让 P\* 每次抖动。`P is None` 会在步③ 里转成 `scene_infeasible`。

#### 步 ③④ —— `r = best_next_view_using_oracle(h_truth, cur_cfg, vm, truth_scene, goal_pose, camera_model=cam, pose_cost_metric=metric, p_star=P)`

**这一步把「P\* 指的方向」变成「具体去哪拍一张」**。它内部串了四件事（`gt_gen/nbv.py`）：

1. **reach_pt —— 我沿 P\* 最远能保守走到哪？** `compute_reach_pt(h_truth, vm, P)` 从 `cur_cfg` 出发沿
   P\* 逐段检查：每一小段的**整臂扫掠体积**是否全部落在 FREE 里（`motion_stays_in_free` 查 vm）。一路放行，
   直到第一段碰到非 FREE 就停，返回那个最远的安全下标 `reach_idx`。直觉：P\* 是理想路，但我们只敢沿它
   走到「已确认安全」的那一点为止，再往前就是未知。
2. **阻塞段 B —— 到底是哪些「没看过的格子」在挡路？** `compute_blocking_B(h_truth, vm, P, reach_idx, k)`
   取 reach_pt **前方 k 段**（`k_lookahead` 默认 6）扫掠体积里**仍是 UNKNOWN** 的体素，汇成 `B:(M,3)`。
   B 的定义很精确：「**卡住下一步、且唯一原因是还没看过**」的格子（不含已知 OCCUPIED——那是真障碍，看了也过不去）。
   把 B 看清楚（变成 FREE 或 OCCUPIED），P\* 的下一段要么打通、要么确认此路不通。
   - **若 B 为空** → 前方该看的都看完了 → 返回 `corridor_confirmed`（见 5c）。
3. **候选 —— 站在哪、怎么转头，能把 B 拍进画面，且我还走得到？** `generate_candidates(h_truth, vm, B, cam, cur_cfg)`：
   把 B 聚类成几个目标块 → 每块周围按 standoff 距离布相机位姿 → 眼在手 IK 反解关节角 →
   过滤掉「看不到 B 的」「IK 失败的」「撞工件的」（`check_state(h_truth)`，要 mesh）「从当前位置走不过去的」
   （`motion_stays_in_free` 查 vm，保守）。剩下的就是**既看得见 B、又确认走得到**的候选 `list[Candidate]`。
   - **若一个候选都不剩** → 返回 `no_reachable_candidate`（见 5d）。
4. **打分 argmax —— 哪个候选最划算？** 对每个候选做一次**假设性** raycast `raycast_reveal(vm, cand.cam_pose, cam, truth_scene)`：
   「如果真把相机挪过去，会确定哪些体素？」（只算、**不改图**）。`gain = |reveal ∩ B|`（能揭开多少阻塞格子），
   `score = gain − λ·关节位移`（揭得多又走得近的更优），取 score 最大的候选返回。

- **输入**：`h_truth`、`cur_cfg`、`vm`、`truth_scene`、`goal_pose`、`cam`、`metric`、`p_star=P`。
- **输出**：`NBVResult(status, cfg, cam_pose, target, gain, score, reach_idx, n_B, n_candidates, P_star)`，
  `status ∈ {ok, scene_infeasible, corridor_confirmed, no_reachable_candidate}`。

#### 步 ⑤ —— 按 `r.status` 决定这一轮怎么走

NBV 的四种结局对应四条出路：

- **`scene_infeasible`** —— 步② 的 P\* 是 `None`，连全知教练在真值上都规划不到目标（目标位姿本身不可达：
  超出工作空间 / 必然自碰 / 必然撞工件）。再探索也没意义 → `status="infeasible"`，**break**。
- **`ok`** —— 选出了最值得去的下一视点 `r.cfg`。执行 `_move_to(h_expl, vm, cur_cfg, r.cfg, ...)` 真正把臂
  移过去：**在 `h_expl` 上规划**（保证整段只走 FREE），轨迹接进 GT，且**沿途每隔 N 个插值路点就拍一次照**
  （`N = params.loop.observe_every_n`，默认 10，边走边观测）；到达后 `cur_cfg = r.cfg`。这是最常见的「正常探索一步」。
- **`corridor_confirmed`** —— 见 5c。
- **`no_reachable_candidate`** —— 见 5d。

##### 5c. `corridor_confirmed` 分支 —— 「前方没未知了，但①还没成」的看似矛盾局面

**它什么意思**：`B` 为空，说明 reach_pt 前方那 k 段已经全部确认（没有未知格子挡路了），没什么可看的。
但能走到这个分支，恰恰是因为**步① 刚刚失败了**（成功的话早就 break 了）。于是出现一个矛盾：
「前方没未知，①却没能规划到目标。」

**为什么会矛盾**：① 想**一口气**规划到**最终目标**，而 B 只检查了 reach_pt **前方 k 段**（k 默认 6，看不到
更远）。所以真实情况多半是：**近处这一段确实通了，但走廊更远处还有未确认段**；或者 cuRobo 这一次
采样没搂到那条可行路（规划器有随机性）。

**怎么处理——别贪心，先把已确认的这段吃下来**：
- `reach_idx = r.reach_idx`（NBV 已带回）；
- 若 `reach_idx == len(P)-1`（整条 P\* 都已落在 FREE 里）→ `_move_to(h_expl, vm, cur_cfg, P[-1], ...)`
  直接走到终点，`status="reached"`，break；
- 否则 `_move_to(..., 目标 = P[reach_idx] 对应构型)`，沿 P\* 往前挪一段，记作一次进展，继续下一轮。
  下一轮 sync 后已确认走廊更长、且观测点前移，① 或 reach_pt 会再往前推。一段段把已知走廊「吃」到目标。

##### 5d. `no_reachable_candidate` 分支 → `handle_stuck(...)` —— 「该看的看不到 / 够不着」

**它什么意思**：B 非空（确实有未知挡路），但步③ 一个可用候选都没生成——要么 B 被真障碍**遮死**了
（没有任何视点能把它拍进画面），要么所有能拍到它的视点机械臂**都走不到**。盯着 B 已经没辙了。

**怎么处理——换个目标，先把自由区摊大**：转「就近揭示」兜底 `handle_stuck`。不再非 B 不可，改为
去揭开**任意** frontier（`_frontier_cells(vm)`：自身 UNKNOWN 且邻接 FREE 的格子 = 探索前沿），挑一个
「揭开未知最多、又走得到」的候选走过去。直觉：先把已知空间整体扩大，常常能**间接**绕开遮挡、
打通原本通往 B 的路。
- 若连就近 frontier 也没有可达候选 → 返回失败信号，本轮**计为「无进展」**，叠加进步⑦ 的 stuck 判定。

#### 步 ⑥ —— 观测更新：把「走过的地方」变成「看过的地方」

这是 voxmap 占用增长的**唯一来源**，也是整个循环能推进的发动机。`_move_to` 在步⑤ 移动时已经**沿途**每
`observe_every_n` 个路点拍一次（`_observe`，N 从 `params.loop.observe_every_n` 读，默认 10），到达后再在终点构型补拍一次：raycast 命中工件表面的格子标 OCCUPIED、
射线穿过的空格子标 FREE，写回 voxmap。每多拍一张，就有一批 UNKNOWN 翻面。增长之后，下一轮步0 同步出
更长的走廊、步② 的 reach_pt 也能沿 P\* 推得更远——循环就是靠这个「看得越多 → 走得越远 → 看得更多」滚动的。

#### 步 ⑦ —— 进展判定：识别「原地打转」并及时止损

**为什么需要**：若某轮探索既没翻开新格子、reach_pt 也没前进，说明这一轮白干了。偶尔一轮没进展正常
（候选选得不好），但**连续多轮**都没进展，基本就是卡死了（目标被真障碍彻底围死、或可达性走到死角），
该停下报告而不是空转 200 轮。

**怎么判**：
- 本轮末取 `free_now = vm.counts()[FREE]`、`reach_now = r.reach_idx`；
- 只要 `free_now > free_prev`（看到了新空间）**或** `reach_now > reach_prev`（沿 P\* 推进了）→ 算有进展，`stale=0`；
  两者都没动 → `stale += 1`；随后更新 `free_prev / reach_prev`；
- `stale >= params.loop.stuck_rounds`（默认 5）连续无进展 → `status="stuck"`，break；
- for 循环自然跑满 `max_rounds`（默认 200）仍没 reached → `status="max_rounds"`。这两者都是「没能完成」的诚实出口。

---

**私有辅助函数**（都在 `gt_gen/main_loop.py` 内）：

| 函数 | 输入 | 输出 | 做什么 |
|---|---|---|---|
| `_observe(vm, handle, q, cam, scene, max_depth, pixel_stride=16)` | voxmap、handle（取 FK）、构型 `q`、相机模型、truth_scene、最大深度 | `dict`（本次生效 FREE/OCCUPIED 数） | `camera_pose_from_config(handle, q, cam)` 算相机 4x4 位姿 → `observe_and_update(vm, cam_pose, cam, scene, max_depth)` 实拍并写回 vm。**更新占用唯一入口** |
| `_move_to(h_expl, vm, cur_cfg, target_cfg, cam, scene, max_depth, every_n=None)` | h_expl、vm、起点/目标构型、相机、scene、深度 | 该段插值轨迹 `(t,6)` 或 `None`（规划失败） | `plan_to_config(h_expl, cur_cfg, target_cfg)` → `get_interpolated_plan().position` → 接进 GT；沿途每 `every_n` 点 `_observe`，终点补拍。边走边看。`every_n=None` 时取 `params.loop.observe_every_n`（默认 10） |
| `handle_stuck(h_truth, h_expl, vm, cur_cfg, cam, scene, ...)` | 两个 handle、vm、当前构型、相机、scene | 是否取得进展（bool）/ 新 cur_cfg | 「就近揭示」：以 `_frontier_cells(vm)`（FREE 邻接 UNKNOWN）为目标，`generate_candidates(h_truth, vm, frontier, cam, cur_cfg)` 选揭开任意未知最多的可达候选 → `_move_to` 过去；无可达候选 → 失败信号 |
| `_frontier_cells(vm)` | voxmap | `(M,3)` 体素下标 | 取所有「自身 UNKNOWN 且 6-邻接里有 FREE」的体素 = 探索前沿，喂给 handle_stuck 当目标 |
| `look_around(handle, vm, cur_cfg, cam, scene, max_depth)` | handle、vm、构型、相机、scene、深度 | 无（就地观测） | 冷启动/兜底的小幅环视。v1 实现为「当前构型补拍一次 + 转 handle_stuck」，保留函数位以备加强（按关节 ±dq 摆动多拍） |

> 各步**世界归属**速记：步 0/①/⑤ 的真实规划在 `h_expl`(VOXEL)；步 ②/③④ 的特权决策在 `h_truth`(MESH)；
> 步 ⑥ 的 raycast 几何来自 `truth_scene`(trimesh)；可达性/进展判定/B 全查 `voxmap`(三态)。

### 6. `scripts/verify_step10.py` — 新建（参考 verify_step8/9）

- `build_scene_step10(args)`：读 seam → `goal_pose, _ = seam_ee_pose(d, offset_m=cfg.goal_standoff_m)`（**带 10cm**）；
  建 `h_truth`(MESH)、`h_expl`(VOXEL)、`truth_scene`、`vm`(+圆柱 FREE)、`cam`。
- 跑 `GT, status, info = generate_gt(h_truth, h_expl, vm, truth_scene, goal_pose)`。
- **断言**：① status==reached；② 每对相邻 GT `motion_stays_in_free(h_truth, vm_final, ·)` ⊆ FREE
  → 0 保守违例（FREE 只增不减，用最终 vm 近似，文档注明）；③ 每点 `check_state(h_truth)` feasible
  → 0 真值碰撞；④ GT 终点末端位姿 ≈ goal（确认停在 standoff）。
- 打印 `VERIFY_STEP10_OK`（GT 路点数、轮数、|B| 轨迹、FREE 增长）。`--viz` 复用 step8 的 open3d 工具画整条 GT。

### 7. 文档

- `docs/implementation-steps.md`：Step 10 打勾、进度表 `[ ]`→`[x]`，记一行「goal 后退
  `goal.standoff_cm`、纯三态 voxel 探索世界、无 voxel+mesh」。
- 本文件 `docs/step10-plan.md` 即独立计划文档。

---

## 不做什么（范围边界）

- **不**实现 GT 序列化/导出（Step 11；GT 不存时间戳——既有用户决策）。
- **不**改 `drop_collision_links`（保持 `[]`，焊枪也参与避障；单一来源）。
- **不**给 `h_expl` 加任何工件 mesh（核心：纯三态 voxel）。
- **不**改 step7/8/9 的几何/断言（仅解包 `seam_ee_pose` 返回值这一兼容性修复）。

---

## 验证（端到端）

```
conda run -n env_isaaclab python scripts/verify_step7.py    → STEP7_OK
conda run -n env_isaaclab python scripts/verify_step8.py    → VERIFY_STEP8_OK
conda run -n env_isaaclab python scripts/verify_step9.py    → VERIFY_STEP9_OK
conda run -n env_isaaclab python scripts/verify_step10.py   → VERIFY_STEP10_OK
```
`verify_step10` 用 seam_22：status=reached、0 保守违例、0 真值碰撞、终点≈standoff goal。
`--viz` 目视：整条 GT 全程在 FREE 内推进、绕过 UNKNOWN、停在离焊缝 10cm 处。

## 关键文件

- 改：`configs/default.yaml`、`gt_gen/config.py`、`gt_gen/main_loop.py`、
  `scripts/verify_step7.py`、`scripts/verify_step8.py`、`docs/implementation-steps.md`
- 新建：`scripts/verify_step10.py`
- 复用（不改）：`gt_gen/{nbv,reach_b,candidates,swept,collision_sync,mapping,sensor,voxmap,init_free,curobo_iface}.py`、
  `scripts/plan_seam.py`（本次已改）
