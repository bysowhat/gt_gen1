# API 重组方案：状态类 `Scene` + 可组合原语

> 目标：把现有散落在 `scripts/` 的编排/几何逻辑收进 `gt_gen/`，提供一个持有「当前状态
> （3D 世界 + 机械臂）」的状态类 `Scene`，配上一组可组合的 API 原语，干净支持用法 1/2/3。
> 本文是**方案**，确认后再改代码。讨论确认点见文末「待确认 / 已确认」。

---

## 1. 设计依据（已与作者确认） 

- **候选初始位姿 = 工件↔机械臂的相对 pose**，由 `scripts/plan_init_pose.py:InitPoseLookupSolver`
  算出。每个候选携带 `(R, t)`（mesh world→base：`p_base = R·p_world + t`）、可达该焊缝的关节角
  `q`、`ee_pos_in_base`、评分。**它定义一整套自洽的世界布局**（工件在 base 系的摆放 → truth_scene；
  若 goal 以工件系给定则随之得 base 系 goal_pose），不是机械臂的起步关节角（起步仍是固定 retract）。
- **goal pose 由用户直接输入**（不再由焊缝几何 bisector/tangent+standoff 计算）。可选两种坐标系：
  `base`（规划用 base 系目标，固定）或 `workpiece`（工件/mesh 世界系，随候选 `(R,t)` 变换到 base）。
  焊缝本身仍用于「候选初始位姿求解」与「类型 2/3 焊缝相对障碍放置」。
- **「全局视野下的规划路径」= 在 MESH 世界全知规划**：在 `h_truth`（工件 + 全部障碍都已知）里直接
  规划 retract→goal，作为「这套布局是否可行」的前置闸门；失败即该候选不可行。
- **落点**：散落逻辑收进 `gt_gen/`，新增一个状态类编排；`scripts/` 改为薄壳 CLI 调用 API。

### 1.1 为什么两个用法的步骤顺序不同（关键）

| 用法 | 障碍类型 | 顺序 | 原因 |
|------|----------|------|------|
| 用法 1 | 类型 2/3（焊缝旁 C 型 / open box） | 先障碍、后候选 | 类型 2/3 是**焊缝相对**几何，随工件摆放走；障碍参数（哪侧/距离）先随机定好，每个候选再投影到 base 系 |
| 用法 2 | 类型 1（关节扫掠走廊） | 先候选、后障碍 | 类型 1 依赖「retract→goal 默认轨迹」的扫掠走廊，而默认轨迹依赖工件摆放+goal，必须 `apply_init_pose` 之后才能算 |

`Scene` 用「状态变更 + 脏标记缓存失效」自然承载这种依赖顺序：`apply_init_pose()` 会重算
工件 in base / truth_scene / 各 cuRobo 世界（goal 为工件系时一并重算 goal_pose）；焊缝相对障碍随之重新投影。

---

## 2. 现状盘点（重组前）

**`gt_gen/` 已有干净原语**（保留，被 `Scene` 复用）：
- `config.py`：`Config` / `load_config`（ROI、retract、init_free、planner backend、障碍参数单一来源）。
- `curobo_iface.py`：`CuroboHandle` + `init_curobo(...)`；`solve_ik` / `plan_to_pose(_all)` /
  `plan_to_config` / `plan_on_truth` / `check_state` / `fk` / `free_pose_metric`。两种世界：
  **h_truth**（MESH=工件+障碍）、**h_expl**（VOXEL 三态）。
- `voxmap.py`（`ThreeStateVoxelMap`）、`init_free.py`（`set_initial_free_*`）、`sensor.py`/`mapping.py`
  （raycast→更新）、`collision_sync.py`、`swept.py`、`reach_b.py`、`candidates.py`（NBV 视点）、`nbv.py`。
- `obstacles.py`：原语 `Box`/`Tube` + 形状构造（plate / open_box / pipe…）+ `build` / `to_world_config`。
- `obstacle_placement.py`：**障碍类型 1**（关节扫掠走廊）全套：`compute_link_sweep` /
  `place_in_corridor` / `search_placement` / `generate_scenes` / `validate_scene`（三条件）/ `build_world`。
- `main_loop.py`：`generate_gt(...)` = 完整 ①~⑦「边走边看」主循环。
- `gt_export.py`：`export_gt(trajectory, meta, out_path)`。

**散落在 `scripts/`（需收进 `gt_gen/`）**：
- 焊缝 I/O：`plan_init_pose.py:load_welds(_weld_angle3.json)` + `find_obj`（goal 改为用户直接输入，
  不再用 `plan_seam.py:seam_ee_pose` 从焊缝几何算）。
- **障碍类型 2**（焊缝旁 C 型 plate）：几何写死在 `place_seam_plate.py`。
- **障碍类型 3**（open box / open cylinder 套焊缝，6 个面距 `dis_cm` 定尺寸）：几何写死在
  `viz_seam_open_box_isaacsim.py`。
- **候选初始位姿**：`plan_init_pose.py:InitPoseLookupSolver`。
- **编排**：`place_obstacles.py` / `place_obstacles_to_gt.py` / `verify_step10.py`（混 CLI + I/O + viz + 逻辑）。

**缺口**：没有统一的「当前状态」对象可编程驱动；编排与关键几何都困在 CLI 脚本里。

---

## 3. 目标模块结构（重组后）

```
gt_gen/
  config.py                # 加 seam_obstacle 段属性（类型2/3 随机参数单一来源）
  curobo_iface.py          # 不变
  voxmap.py sensor.py mapping.py collision_sync.py swept.py reach_b.py
  candidates.py nbv.py main_loop.py obstacles.py init_free.py gt_export.py   # 不变
  obstacle_placement.py    # 类型1（基本不变，少量签名规整）

  # ---- 新增 / 收编 ----
  seam.py                  # load_welds(_weld_angle3.json) / find_obj / 由 weld dict 现算 seam_line/tangent/limits
  init_pose.py             # InitPoseLookupSolver（从 scripts 迁入）+ InitPoseCandidate dataclass
  obstacles_seam.py        # 类型2（C 型 plate）+ 类型3（open box/cylinder）焊缝相对放置
  scene.py                 # ★ Scene 状态类（本方案核心）
  recipes.py               # run_usecase1 / run_usecase2 / run_usecase3 便捷驱动
  scene_io.py              # Scene ↔ npz（兼容 viz_placed_obstacle_isaacsim 字段）

scripts/                   # 全部改薄：解析 CLI → 调 gt_gen API → 存盘/可视化
```

---

## 4. 核心：`Scene` 状态类

一个类持有全部「当前状态」，属性按 **世界状态 / 机械臂状态 / 静态输入** 分组（仍是单一类）。

### 4.1 状态字段

```python
class Scene:
    # —— 静态输入 ——
    cfg            : Config
    workpiece_obj  : str                  # 工件 mesh 路径（_part.obj / _watertight.obj）
    seam           : dict                 # 选中那条焊缝的 weld dict（load_welds：p0/p1/mid/bisector/boundary_dirs）；
                                          #   seam_line/tangent/limits 由它现算（口径同 plan_init_pose.save_seam_pkl）
    seam_id        : str                  # 焊缝标识（weld_json stem + seam_index）

    # —— 世界状态（3D） ——
    init_pose      : InitPoseCandidate | None   # 当前工件↔臂相对 pose (R,t)；None=尚未求解/应用
                                                #   （无 pkl 预焊位姿，规划前必须 compute_init_pose_candidates + apply_init_pose）
    goal_user      : tuple                 # 用户【直接输入】的 goal ((x,y,z),(qw,qx,qy,qz))；不由焊缝计算
    goal_frame     : str                   # "base"|"workpiece"：goal_user 所在坐标系
    goal_pose      : tuple                 # 规划用 base 系 goal。goal_frame=="base"→恒等于 goal_user（固定）；
                                           #   =="workpiece"→apply_init_pose 用候选 (R,t) 变换 goal_user 得到
    goals          : list                  # 用法4 的 goal 序列（由 compute_goal_poses 写入）；run_goal_sequence 依次走
    obstacles      : list[ObstacleSpec]    # 已放障碍（带类型标签 + 焊缝相对/绝对参数）
    truth_scene    : trimesh.Trimesh       # 工件+障碍（base 系）——raycast 几何源
    voxmap         : ThreeStateVoxelMap    # 三态记忆（+ 初始 FREE）。唯一持久探索记忆，generate_gt_explore
                                           #   就地累积；用法4 跨段继承（序列内只 reset 一次）

    # —— 机械臂状态 ——
    cur_cfg        : list[float]           # 当前关节角。【所有用法起步均为 retract】（与初始位姿无关）。
                                           #   generate_gt_explore 成功后推进到该段终点 → 用法4 下一段的起点

    # —— cuRobo 世界（惰性构建，脏标记缓存） ——
    _h_truth       : CuroboHandle          # MESH：工件（真实尺寸）+障碍（真实尺寸）——NBV/候选避障/真值碰撞判定
    _h_plan        : CuroboHandle          # MESH：工件(真实尺寸)+障碍(外扩 buffer)——主循环步② 规划参考路 P* 用；
                                           #   = generate_gt 的 h_truth_plan 形参。无 buffer 时退回 None（步② 用 h_truth）
    _world_plan    : WorldConfig           # ★ 与 _h_plan 同源、同一个「工件+膨胀障碍」MESH 世界的【裸 WorldConfig 形态】。
                                           #   由 build_world(workpiece, prims, buffer_m) 产出；_h_plan 即用它 init_curobo 而来。
                                           #   两形态分别服务两个后端的步②：curobo 后端用 _h_plan(读 handle 内部世界，忽略本字段)；
                                           #   stomp 后端用 _world_plan（plan_pose_single 只吃 WorldConfig，不接收 handle）。
                                           #   无 buffer 时退回真值世界 world。一次 build_world → 两形态都备好。
    _h_expl        : CuroboHandle          # VOXEL 三态——①/⑤ 实际无碰撞规划
    _camera_model  : CameraModel
    _solver        : InitPoseLookupSolver  # 候选求解器句柄（首次 compute_init_pose_candidates 建查表，按工件 key 复用）
    _dirty         : set[str]              # {"worlds","truth_scene","goal"} 缓存失效标记
```

`ObstacleSpec` 记录「类型 + 生成参数 + 焊缝相对/绝对」，使焊缝相对障碍能在 `apply_init_pose` 后
按新布局重新实例化为 base 系 `prims`。

### 4.2 API 方法（= 可组合原语）

**A. 构建 / 输入**
```python
@classmethod
def from_seam(cls, cfg, obj_path, weld_json, goal=None, *, seam_index=0, goal_frame="base") -> Scene
    # 输入与 scripts/plan_init_pose.py --solve 一致：工件 mesh(obj_path) + 焊缝 _weld_angle3.json(weld_json)。
    #   【不再吃已解算好的 seam_*.pkl】——那种 pkl 已把某个初始位姿(robot_pose)焊死，与本 API
    #   「位姿一律由 InitPoseLookupSolver 求解」相悖。
    # weld_json 经 load_welds() 解析为多条焊缝；seam_index 选其中一条（一个 Scene = 一条焊缝）。
    #   选中的 weld dict（p0/p1/mid/bisector/boundary_dirs）即 self.seam；seam_line/tangent/limits 由它现算。
    #   焊缝仅用于：①候选初始位姿求解(InitPoseLookupSolver)；②类型2/3 焊缝相对障碍放置。
    # goal 由用户【直接输入】((x,y,z),(qw,qx,qy,qz))，不由焊缝几何计算；用法4 传 None（由 compute_goal_poses 产）。
    # goal_frame：
    #   "base"      → goal 即规划用 base 系目标，固定不变（apply_init_pose 不动它）；
    #   "workpiece" → goal 在工件/mesh 世界系，apply_init_pose 用候选 (R,t) 变换到 base 系。
    # init_pose 起始为 None（无预焊位姿）：规划前必须先 compute_init_pose_candidates + apply_init_pose。
    # voxmap 暂不建。

def set_goal(self, goal, *, goal_frame=None) -> None    # 直接改 goal（及其 frame）；按 frame 重算 goal_pose
                                                        #   （goal_frame=None 时沿用当前 frame）

def compute_goal_poses(self, n: int, **kw) -> Scene       # ★ 由你后续实现（本方案先留接口）
    # 契约【传入 scene、输出 scene】：读 self 的 工件+焊缝+已 apply 的 init_pose（base 系布局已定），
    #   算出 n 个 base 系 goal，写入 self.goals 后 return self。供 run_goal_sequence 依次走。
    # 需在 apply_init_pose 之后调（依赖 base 系布局）。
```

**B. 候选初始位姿**（用法 1 步 3 / 用法 2 步 2）
```python
def compute_init_pose_candidates(self, n: int, **solver_kw) -> list[InitPoseCandidate]
    # 包 InitPoseLookupSolver.solve_one_weld_lookup；按 combined_score 取前 n

def apply_init_pose(self, cand: InitPoseCandidate) -> None
    # 设工件↔臂相对 pose (R,t)；重算工件 in base、truth_scene；
    # 若 goal_frame=="workpiece" 则用 (R,t) 把 goal_user 变换为 base 系 goal_pose（goal_frame=="base" 则不动）；
    # 焊缝相对障碍随之重新投影；置脏标记 {"worlds","truth_scene","goal"}
```

**C. 障碍物**（用法 1 步 2 / 用法 2 步 3）
> **添加时机规则（重要）**：
> - **类型 2/3（焊缝相对）必须在世界初始化之前添加，添加后不再更改**；可混合添加多个
>   （如 2 个 type2 + 1 个 type3）。它们随 `apply_init_pose` 一并投影到 base，是整个运行/序列里的【静态世界】。
> - **类型 1（关节扫掠走廊，base 绝对）可在「规划到某个 goal 之前」动态添加**（用法2/用法4 各段按需加）。
```python
def add_obstacle_type1(self, rng, *, link=None, otype=None) -> PlacementResult
    # 关节扫掠走廊：内部先算 retract→【当前 goal】默认轨迹 → compute_link_sweep
    #   → search_placement（三条件验证）。需当前 init_pose 已定 + 当前 goal 已定。绝对几何（base 系）。
    #   挂在哪个 goal 由调用方决定：放置前先 set_goal(goals[i])，本方法即按该 goal 的走廊放置。
    #   可分 N 次增量添加，每加一个后重跑 run_goal_sequence 保一致（见 §8⑥⑦）。

def add_obstacle_type2(self, rng=None, *, parallel="b", n_cm=10, length_pct=80,
                       width_cm=30, thickness_cm=3, angle_deg=0) -> PlacementResult
    # 焊缝旁 C 型 plate（焊缝相对）。存 ObstacleSpec(relative=True)。须在世界初始化前加、之后不变。

def add_obstacle_type3(self, rng=None, *, kind="open_box", dis_cm=(10,)*6,
                       wall_cm=2) -> PlacementResult
    # open box / open cylinder 套焊缝（焊缝相对）。须在世界初始化前加、之后不变。
    # type2/type3 可重复调用以叠加多个（混合）；每次 append 一条 ObstacleSpec(relative=True)。

def clear_obstacles(self) -> None
```

**D. 世界 / 体素（多为内部，按需暴露）**
```python
def reset_voxmap(self, init_free="cylinder", **kw) -> None
    # 丢弃当前探索记忆，新建一张全 UNKNOWN 的 ThreeStateVoxelMap，再在 retract 邻域种一小块
    #   "已知 FREE" 作起步引导。动机：相机在末端 Link6，起点视野极小；若 voxmap 全 UNKNOWN，
    #   整臂扫掠必含 UNKNOWN → reach_pt=0 → 第一步就动不了（见 init_free.py 头注）。
    # init_free 选 FREE 块形态（透传 set_initial_free_*；缺参从 cfg 取）：
    #   "space"   → set_initial_free_space   各关节 ±dq 扫掠并集，贴合 blob，体素最省
    #   "cylinder"→ set_initial_free_cylinder base 竖轴圆柱整块，规整、与轨迹无关（默认）
    #   "box"     → set_initial_free_box     base 系轴对齐长方体整块
    # 调用时机：用法1/2 每个候选重试都重来一张（探索记忆不可跨候选带）；用法4 整个序列【只调一次】，
    #   之后各段继承累积同一张 voxmap（故名 reset、可反复调，但用法4 不在段循环里调）。不进 _dirty。

def _ensure_worlds(self) -> None    # 据 _dirty 惰性(重)建 h_truth/h_plan/world_plan/h_expl
    # 仅当 _dirty 含 "worlds" 才动；没动过直接复用缓存。变更方法(apply_init_pose/add_obstacle_*)
    #   只置脏，把"重建 4 个世界(各含 warmup)"推迟到真正规划前一次补齐——避免候选循环里反复重建。
    # build_world(工件,障碍,buffer) 跑一次产出膨胀 WorldConfig，存为 _world_plan，再用它 init_curobo
    #   得 _h_plan（同源两形态，分供 stomp / curobo 后端步②；无 buffer 时退回真值 world，_h_plan=None）。
    # 世界变了不重 init_curobo，走 mg.update_world 把新世界灌进已有 handle，复用 warmup
    #   （沿用 obstacle_placement 在"工件"/"工件+障碍"间切换的成熟做法）。
```

**E. 规划**（用法 1 步 4、用法 3）
```python
def plan_global(self, start_cfg=None) -> GlobalPlanResult
    # 全知 MESH 规划 start(默认 retract)→goal，在 h_plan(工件+膨胀障碍)上算一条参考路 P*。
    # 包 plan_to_pose_all / plan_on_truth（带 metric 放开 roll）。两个角色：
    #   ① 可行性闸门：失败(None)→这套布局/候选不可行，换候选；
    #   ② 产出 P* 轨迹 → 作为 generate_gt_explore 的【初始轨迹 p_star_init】喂入（见下）。
    # 它就是 generate_gt 内部步② 每轮重规划 P* 所用的【同一原语】（只是起点换成当前 cur_cfg）。

def generate_gt_explore(self, p_star_init, start_cfg=None) -> GtResult   # (GT, status, info)
    # 边走边看主循环。【前提】先有 plan_global 成功的结果，把它当 p_star_init 传进来：
    #   第一轮(rnd==0)直接拿它当 P*、不重规划（绕开"同一空间这里 plan 却失败"的抖动）。
    # 之后每一轮内部都会【再次调用同一全知 MESH 规划】(= plan_global，起点=当前 cur_cfg)重算 P*，
    #   随探索进展把 P* 不断更新——故 plan_global 既是入口闸门，也是循环内反复复用的原语。
    # start_cfg=None → 用 scene.cur_cfg 作起点；成功(reached)时把 cur_cfg 推进到该段终点构型，
    #   从而用法 4 的下一段(init→g0→g1→…)无缝从上段终点续起。
    # 包 main_loop.generate_gt(h_truth,h_expl,voxmap,truth_scene,goal_pose,camera_model,params,
    #   h_truth_plan=h_plan, p_star_init=<plan_global 的 P*>, world_plan=world_plan)

def run_goal_sequence(self, goals=None, *, init_free="cylinder") -> list[GtResult]
    # 用法4 的可重跑单元：goals=None → 用 self.goals。reset_voxmap(只一次) → 从 retract 起，对 goals
    #   逐个 set_goal+plan_global+generate_gt_explore（voxmap 跨段继承，cur_cfg 链式推进）。
    #   用【当前完整障碍集】一次性产出整条序列。
    # 段失败（plan_global=None 或 status!=reached）→【终止整序列】，返回已成功的前 k 段 + 失败信号；
    #   自身不重试（不换 init_pose/障碍/goal），重试由上层决定后整条重跑（决定⑬）。
    # 关键：每次新增一个类型1 障碍后【重调本方法】，使每个 goal 的 GT 都对齐到最新的全障碍世界
    #   （解决"后段 type1 令前段不一致"——见 §8）。只加 1 个 type1 时调一次即可、无需重跑。
```

**F. 导出 / 序列化**
```python
def export_gt(self, gt, out_path, meta=None) -> None        # 包 gt_export.export_gt
def concat_gt(self, gts) -> GT                              # 用法4：逐段 GtResult 列表 → 一条连续轨迹
                                                            #   （去相邻段重复接缝点、续接索引/时间），供整条回放/导出
def to_npz(self, path) -> None                              # 障碍+工件+轨迹（兼容现有 viz）
@classmethod
def from_npz(cls, cfg, path) -> Scene                       # 复原（如 place_obstacles_to_gt 入口）
```

**G. 高层用法驱动（`recipes.py`，可选便捷壳）**
```python
def run_usecase1(cfg, obj_path, weld_json, *, seam_index=0, obstacle_type=2|3, n_init=8, **kw) -> GtResult
def run_usecase2(cfg, obj_path, weld_json, *, seam_index=0, n_init=8, **kw) -> GtResult
def run_usecase3(scene_or_state, goal_pose=None, with_obstacle=None) -> GtResult
def run_usecase4(cfg, obj_path, weld_json, init_pose, *, seam_index=0, n_goals=3,
                 static_obstacles=None, type1_specs=None, **kw) -> list[GtResult]
    # 多段路点序列：(前置)加类型2/3 静态障碍 → apply init_pose → compute_goal_poses(n_goals)
    #   → 分 N 次加类型1，每次后重跑 run_goal_sequence（voxmap 跨段继承）；type1_specs=None 时仅跑一次。
    #   返回终态（含全部 type1）每个 goal 的 GtResult 列表。
```

---

## 5. 各用法的组合写法（伪代码）

### 用法 1（类型 2/3：焊缝旁 C 型 / open box，世界初始化前加、之后不变）
```python
scene = Scene.from_seam(cfg, obj_path, weld_json, goal)   # 1 给定工件 obj + 焊缝 json + goal
scene.add_obstacle_type2(rng); scene.add_obstacle_type2(rng)  # 2 可混合叠加多个（焊缝相对、静态）
scene.add_obstacle_type3(rng, kind="open_box")      #   例：2 个 type2 + 1 个 type3
for cand in scene.compute_init_pose_candidates(n):  # 3 候选初始位姿
    scene.apply_init_pose(cand)                     #   重算 goal/truth_scene；type2/3 随焊缝重投影(参数不变)
    gp = scene.plan_global()                        # 4 全局规划=可行性闸门 + 产初始 P*
    if gp is None:                                  #   失败→换候选(回到3)
        continue
    scene.reset_voxmap(init_free="cylinder")
    res = scene.generate_gt_explore(gp.trajectory)  # 5 边走边看(P* 当初始轨迹)，失败→换候选
    if res.status == "reached":
        scene.export_gt(res.gt, out); break
```

### 用法 2（类型 1：关节扫掠走廊）
```python
scene = Scene.from_seam(cfg, obj_path, weld_json)         # 1
cands = scene.compute_init_pose_candidates(n)       # 2 候选初始位姿（先）
for cand in cands:
    scene.apply_init_pose(cand)                     #   定布局 → 默认轨迹可算
    if scene.add_obstacle_type1(rng).ok is False:   # 3 随机障碍（依赖默认轨迹的扫掠走廊）
        continue
    gp = scene.plan_global()                        # 4 全局规划=可行性闸门 + 产初始 P*
    if gp is None:                                  #   加障碍后不可行→清障换候选
        scene.clear_obstacles(); continue
    scene.reset_voxmap(init_free="cylinder")
    res = scene.generate_gt_explore(gp.trajectory)  # 5 边走边看(P* 当初始轨迹)，失败→换候选
    if res.status == "reached":
        scene.export_gt(res.gt, out); break
    scene.clear_obstacles()
```

### 用法 3（给定当前状态 + goal，规划路径；可选加障碍）
```python
scene = Scene.from_seam(cfg, obj_path, weld_json, goal)   # 或 Scene.from_npz(cfg, scene_npz)
# 可选：scene.add_obstacle_type1/2/3(...)
gp = scene.plan_global(start_cfg=scene.cur_cfg)     # 全局规划=可行性闸门 + 产初始 P*
res = scene.generate_gt_explore(gp.trajectory, start_cfg=scene.cur_cfg)   # 边走边看
```

### 用法 4（给定工件+焊缝+初始位姿 → 算 n 个 goal → 逐段路点序列，世界探索状态【跨段继承累积】）
```python
scene = Scene.from_seam(cfg, obj_path, weld_json, goal=None)  # goal 由 compute_goal_poses 产出，from_seam 可不给
scene.add_obstacle_type2(rng); scene.add_obstacle_type3(rng)  # 类型2/3：世界初始化前加、之后不变（静态、可混合多个）
scene.apply_init_pose(init_pose)                    # 初始位姿=工件↔臂【相对 pose】(R,t)，定 base 系布局；type2/3 投影到 base
#   机械臂起步关节角【恒为 retract】（与是哪个初始位姿无关）
goals = scene.compute_goal_poses(n=3)               # ★你实现：算 n 个 base 系 goal 写入 scene.goals（传入scene→输出scene）
#   该 api 契约：传入 scene、返回 scene（goals 已 set 进 self.goals）；这里 goals 即 scene.goals 的别名

# run_goal_sequence = 可重跑单元：reset_voxmap 一次 → 从 retract 起逐 goal 边走边看（voxmap 跨段继承、cur_cfg 链式）
gts = scene.run_goal_sequence()                     # 默认走 scene.goals，用当前障碍集（仅 type2/3）跑出整条序列
# 各段 GT 可逐段导出，或拼接成一条整序列：scene.export_gt(concat(gts), out)
```
其内部展开（即 `run_goal_sequence` 做的事）：
```python
scene.reset_voxmap(init_free="cylinder")            # ★仅此一次：建全 UNKNOWN + 种起步 FREE；之后不再 reset
gts = []
for k, g in enumerate(scene.goals):                 # 依次 init→g0, g0→g1, g1→g2
    scene.set_goal(g, goal_frame="base")            #   切到本段目标（base 系，固定）
    gp = scene.plan_global(start_cfg=scene.cur_cfg) #   本段全局规划=闸门+初始 P*（k=0 起点=retract；k>0=上段终点）
    if gp is None: break
    res = scene.generate_gt_explore(gp.trajectory, start_cfg=scene.cur_cfg)  # 边走边看，就地累积 voxmap
    if res.status != "reached": break
    gts.append(res.gt)                              #   res 成功后 cur_cfg 已推进到 g，作下段起点
```

### 用法 4b（在用法4 基础上，分 N 次添加类型1：每加一个就用全障碍集【重跑整条序列】）
```python
scene = ...（同用法4：from_seam(cfg, obj, json) → 加 type2/3 → apply_init_pose → scene = compute_goal_poses(n)，goals=scene.goals）
gts = scene.run_goal_sequence()                     # 第 0 轮：仅 type2/3 的整条序列（可选，作基线）

for r in range(N):                                  # 要添加类型1 共 N 次（N 个 type1）
    scene.set_goal(scene.goals[target[r]])          # 针对某个 goal 的走廊放置（调用方指定挂哪个 goal）
    scene.add_obstacle_type1(rng)                   # 第 r 个 type1：按【原始逻辑】放置，累积进 scene.obstacles
    gts = scene.run_goal_sequence()                 # ★关键：每加一个 type1 就【对每个 goal 重跑一遍】边走边看，
#                                                   #   使所有 goal 的 GT 都对齐到含 r+1 个 type1 的最新世界
# 终态 gts = 含全部 N 个 type1 的一致序列。N==1 时即上面用法4 的单次 run，无需"重跑"。
```
> **跨段继承 / 一致性三要点**：① **voxmap 在每次 `run_goal_sequence` 开头 `reset` 一次**，序列内各段共享、
> 就地累积（段 k>0 起点必已探明 FREE，无需重种）。② **类型2/3 是前置静态世界**；**类型1** 可分 N 次增量添加，
> 每个按原始走廊逻辑放置。③ **每新增一个 type1 → 重跑整条 `run_goal_sequence`**，让每个 goal 的 GT 都基于
> 当前完整障碍集——这样后加的 type1 不会留下"前段没考虑它"的不一致（N==1 无需重跑）。


---

## 6. 关键设计原则

1. **单一状态对象**：`Scene` 持有世界 + 机械臂状态；焊缝相对障碍以**焊缝局部参数**存储，随
   `apply_init_pose` 重投影到 base，从而无需用户手动维护一致性。
2. **惰性世界 + 脏标记**：`obstacles`/`init_pose` 变更只置脏标记，下次规划前 `_ensure_worlds`
   统一重建；cuRobo handle 用 `mg.update_world` 复用，避免反复 warmup（沿用 `obstacle_placement`
   在「工件」「工件+障碍」间切换的成熟做法）。
3. **原语 + 配方分层**：A~F 是细粒度可组合 API（「供我组合」）；G 的 `run_usecase*` 是把
   候选/障碍/回退循环固化的便捷壳，二者并存。
4. **后端无关**：`planner_backend`（curobo|stomp）已在 `Config` 抽象，`Scene` 透传，规划方法不分叉。
5. **薄壳脚本**：现有 `place_obstacles.py` 等改为「解析 CLI → 调 `Scene`/`recipes` → 存盘/viz」，
   行为与字段保持兼容，旧 npz 仍可被 `viz_placed_obstacle_isaacsim.py` 回放。
6. **探索记忆可跨段继承 + 多次 type1 重跑保一致**：`voxmap` 是 `Scene` 持有的唯一持久探索记忆，
   `run_goal_sequence` 开头 `reset` 一次、序列内各段就地累积（`g1→g2` 站在 `init→g1` 已探明世界上）。
   类型2/3 前置静态；类型1 可分 N 次增量加，每加一个就重跑整条 `run_goal_sequence`，使各 goal 的 GT 始终
   对齐当前完整障碍集（解决后加 type1 的前段不一致问题）。

---

## 7. 迁移映射（哪段逻辑搬到哪）

| 现状位置 | 迁入 | 说明 |
|----------|------|------|
| `plan_init_pose.py:load_welds / find_obj` | `gt_gen/seam.py` | 读 `_weld_angle3.json` → weld dict；由 weld dict 现算 `seam_line/tangent/limits`（不再读 pkl 的 robot_pose） |
| `plan_init_pose.py:InitPoseLookupSolver` | `gt_gen/init_pose.py` | 产 `InitPoseCandidate(R,t,q,goal_pose,scores)`；可视化留在脚本 |
| `place_seam_plate.py` 几何 | `gt_gen/obstacles_seam.py:place_seam_c_plate(...)` | 类型 2，返回 `prims + meta`，焊缝相对 |
| `viz_seam_open_box_isaacsim.py` 几何 | `gt_gen/obstacles_seam.py:place_seam_open_box/cylinder(...)` | 类型 3，6 面距 `dis_cm` 定尺寸 |
| `place_obstacles.py` 编排 | `Scene.add_obstacle_type1` + `recipes.run_usecase2` | 脚本改薄 |
| `place_obstacles_to_gt.py` | `Scene.from_npz` + `generate_gt_explore` | 脚本改薄 |
| `verify_step10.py` | `recipes.run_usecase3` / 测试 | 作为回归用例保留 |

`obstacle_placement.py`、`main_loop.py`、`obstacles.py` 等核心算法**不动**，只被 `Scene` 调用。

---

## 8. 待确认 / 已确认

**已确认**：① 初始位姿 = 工件↔臂相对 pose `(R,t)`（`InitPoseLookupSolver`），**不是机械臂起步关节角**；
机械臂起步关节角【恒为 retract】，与是哪个初始位姿无关，候选 `q` 仅作参考/校验 goal 可达。
② 全局规划 = MESH 世界全知规划。③ 落点 = 收进 `gt_gen` + 新增状态类。
④ goal 由用户直接输入（base/workpiece 两种 frame 都支持）。
⑤ **障碍添加时机**：类型 2/3（焊缝相对）必须在世界初始化前添加、之后不变，可混合多个；类型 1（base 绝对）
可分 N 次增量添加（每个按原始走廊逻辑放置）。
⑥ **多次类型1 的一致性处理**：每新增一个类型1 后，用当前完整障碍集对每个 goal 重跑一遍 `run_goal_sequence`，
使所有 goal 的 GT 都对齐到最新世界（N==1 时单次即可、无需重跑）。原"后段 type1 令前段不一致"遗留即此法解决。
⑦ **类型1 挂哪个 goal 由调用方指定**：放置第 r 个 type1 前先 `set_goal(goals[target[r]])`，`add_obstacle_type1`
按该 goal 的 retract→goal 默认轨迹走廊放置（原始逻辑）。即每个 type1 显式绑定到调用方选定的某个 goal。
⑧ **类名 = `Scene`**。
⑨ **`compute_goal_poses` 契约 = 传入 scene → 输出 scene**：由你实现，读 self 的工件+焊缝+init_pose 算 n 个
base 系 goal，写入 `self.goals` 后 `return self`；`run_goal_sequence` 默认遍历 `self.goals`。
⑩ **类型2/3 随机参数进 yaml**：`configs/default.yaml` 新增 `seam_obstacle` 段（与 `obstacle_placement` 段对称），
`Config` 加属性读出，作单一来源。
⑪ **失败回退 = 原语 + 配方分层**：`Scene` 出细粒度原语（候选/障碍/规划/`run_goal_sequence`），
`recipes.run_usecase*` 出固化循环的便捷壳；二者并存（§6 原则3）。
⑫ **solver 允许缓存**：`Scene` 持 `InitPoseLookupSolver` 句柄，按工件 key 失效（同工件多焊缝只建一次查表）。
⑬ **段失败处理 = 终止整序列、重试交上层**：`run_goal_sequence` 内某段 `plan_global` 失败或边走边看没到时，
立即停下（序列强链式：后段起点=前段终点），返回已成功的前 k 段 + 明确失败信号（哪段/`plan_global` 或 `explore`/状态）；
**自身不做任何重试**（不换 init_pose/障碍/goal）。要不要重试、改什么由上层（`recipes`/脚本）决定后整条重跑——
保证 `run_goal_sequence` 是确定性一趟，4b「每加一个 type1 就重跑」的语义才不被隐藏状态突变污染。
⑭ **序列输出形态 = 逐段列表 + `concat` helper**：`run_goal_sequence` 返回 `list[GtResult]`（每段含 GT/status/info，
便于单独检视/导出/定位失败段）；另给 `concat_gt(gts)` 把逐段 GT 拼成一条连续轨迹（去相邻段重复接缝点、续接索引），
供「要一条完整 GT 回放/导出」的下游。逐段为主、拼接为派生视图，二者都给。

**待确认**：无（以上 ①–⑭ 已全部确认）。

---

## 9. 实施步骤（确认后执行，自底向上、可逐步验证）

1. `gt_gen/seam.py`：迁 `load_welds`/`find_obj`，由 weld dict 现算 `seam_line/tangent/limits`；脚本改引用。
2. `gt_gen/obstacles_seam.py`：迁类型 2/3 几何，函数化（输入焊缝+参数 → `prims+meta`）；配 viz 自检。
   同时 `configs/default.yaml` 加 `seam_obstacle` 段 + `config.py` 加属性（决定⑩）。
3. `gt_gen/init_pose.py`：迁 `InitPoseLookupSolver`，产 `InitPoseCandidate`。
4. `gt_gen/scene.py`：实现 `Scene`（字段含 `goals`/`_solver` 缓存 + A~F 方法 + `run_goal_sequence` + 脏标记/惰性世界）。
5. `gt_gen/recipes.py`：`run_usecase1/2/3/4`（用法4 待 `compute_goal_poses` 由你实现后接入）。
6. `gt_gen/scene_io.py`：npz 互转（对齐现有字段）。
7. `scripts/` 改薄；跑通现有可视化与 `verify_step10` 回归。

每步完成后人工确认再下一步（沿用项目「一次一步」工作方式）。
```
