# 观测场景：把机械臂悬空放进 USD 里观测焊缝（ObserveAnythingScene 设计）

> 目标：`scripts/find_surface_welds.py` 已从 USD 场景 `/media/a/upan/others/warehouse.usdz`
> 找出焊缝、存 `/tmp/welds1.json`。现在要把机械臂**悬空**放到 USD 里的某个位置去**观测**这些焊缝：
> 机械臂有一个 free-init 空间，该空间不能与 USD 场景碰撞；初始位姿采样、焊缝过滤等**沿用**
> `scripts/demo_scene.py` 里 `Scene.plan_init_pose_fast` 的口径，全部读 `configs/default.yaml`。
>
> 本文只谈设计与步骤，**不改代码**。

---

## 0. 一句话结论

- 新建 `ObserveAnythingScene(Scene)`，接收 **USD + welds json**（而非 `.obj + _weld_angle3.json`）。
- **下游可整段复用父类**：`set_init_pose` / `compute_goal_pose` / `compute_pose_and_plan_path`
  / `_build_explore_world` / `plan_explore_path` 一行不改。
- **`plan_init_pose_fast` 需在子类中重写**（override 同名方法），采样口径与过滤链见 §2。
- **焊缝 json schema 对齐**：`find_surface_welds.py` 把 `p0/p1` 改存 `corrected_p0/corrected_p1`
  后（你自己重新生成 json），键名与父类 `load_welds` 完全一致 → `_load_seam` 有望直接复用（§2.3）。
- 需要一个**前置步骤**：把 USD 导出成世界系（米）的合并 `.obj`，当作 `workpiece_obj`（§3）。

---

## 1. 为什么下游能复用：Scene 是「帧相对」的

`Scene` 的每一个下游方法只认一个量：`InitPoseCandidate.T_workpiece_in_base`
（工件 mesh 系 ↔ 机械臂 base_link 系的相对位姿，`p_base = R·p_world + t`）。

- 焊接场景：机械臂 base 固定在原点，把**工件**摆到臂前 → 求 `T_workpiece_in_base`。
- 观测场景：**场景（仓库）在世界里固定**，把**机械臂悬空**放到世界某处看焊缝。

设机械臂 base 在世界的位姿为 `T_base_world`（base→world），则

```
T_workpiece_in_base  ==  T_scene_in_base  ==  inv(T_base_world)
```

即「动机械臂」与「动工件」在数学上是**同一个量**。因此只要 `ObserveAnythingScene`
能产出合法的 `InitPoseCandidate`，下游对它是无感的：

| 下游方法 | 依赖什么 | 观测场景是否满足 |
|---|---|---|
| `set_init_pose` | 候选列表 + `T_workpiece_in_base` | ✅ 由重写的 `plan_init_pose_fast` 产出 |
| `compute_goal_pose` | `self.cur_init_pose.T_workpiece_in_base`、`self.workpiece_obj`（ScenePose2 obj_path）、`self.cfg.robot_cfg_path`、本焊缝 `_seam_data_arrays` | ✅（obj 见 §3） |
| `_build_explore_world` | `workpiece_pose7` + `CuMesh(file_path=self.workpiece_obj)` + `load_truth_scene(self.workpiece_obj)` | ✅（obj 见 §3） |
| `compute_pose_and_plan_path` / `plan_explore_path` | 上面两个 + `self.cur_cfg`(retract 起步) | ✅ |

> 注意 `_seam_frame` / `_seam_data_arrays` 读的是 `self.seam` 的
> `p0_world/p1_world/mid_world/boundary_dirs/bisector_world` 字段。改名后 welds json
> 的键与父类 `load_welds` 一致，`_load_seam` 解析出的这套字段可直接被 `compute_goal_pose` 吃（见 §2.3）。

---

## 2. `plan_init_pose_fast`：子类重写（override 同名方法）

父类 `plan_init_pose_fast` → `plan_init_pose.py:_fast_build_ctx` / `_fast_solve_weld`，其内核假设
「工件是一个可以被**摆平、绕 90° 翻成 8 种朝向**的小工件」，且**通过移动工件**来确定工件-机械臂相对位姿。
观测场景里可动的是**机械臂**、且场景是世界里固定的大仓库，采样方式与两条工件专属过滤都要改。
子类**重写同名的 `plan_init_pose_fast`**，产出结构与父类一致的候选。

### 2.1 采样方式：直接在世界系采样机械臂 base 位姿（不再动工件）

不再 `lay_flat` + 8 朝向枚举，也不再移动工件。改为：**机械臂在场景中竖直放置**（base z 轴对齐世界 up），
直接采样机械臂在 USD 世界里的 **xyz 位置 + yaw 朝向**：

- **位置 xyz**：以**焊缝中点**为原点，在**周边 2.5m 的立方体**采样区内取格点；
  xy 采样间隔 20cm，z 采样间隔 10cm。
- **朝向 yaw**：绕世界竖直轴，`0 → 360°`，采样间隔 20°。

每个采样点得到机械臂 base 在世界的位姿 `T_base_world`（竖直 + yaw），
再由 §1 的关系换算候选：`T_workpiece_in_base = inv(T_base_world)`。
**这就是「移动机械臂来确定机械臂在 USD 中处于哪个位置」**，而非移动工件。

上述参数**新开一段** `configs/default.yaml: observeanything.plan_init_pose_fast` 读取（§2.4）。

### 2.2 过滤链（按你的指定：两保留、一去掉、一改判据）

在 base_link 系下（`p_base = R·p_world + t`，`R,t` 由 `T_base_world` 反解）逐候选过滤：

| 过滤 | 焊接语义 | 观测场景处理 |
|---|---|---|
| **端点在 ee 范围**（`ee_xy_range_m`/`ee_z_range_m`） | 焊缝可达 | **保留**（照搬 `_inrange`） |
| **工件+障碍 vs init_free 无交集** | free 空间别撞场景 | **保留（核心需求）**：`_pts_in_init_free(scene_pts_base, region)` 有交集则丢 |
| ~~底座-工件 XY 投影不相交（`base_overlap_filter`）~~ | 臂别压工件下 | **去掉**（整仓库 footprint 必然盖住 base） |
| 工件最近点 base-x > `workpiece_x_min_m` → **改为：焊缝中点距离 base-x > 0.4** | 目标在臂前方 | **改判据**：用 `seam_mid_base.x > x_min` 取代「工件最近点」 |
| 正面 bisector base-z≥0 | 焊缝朝上（正脸可见） | **保留**：只在能看到焊缝正面的朝向放相机，兼作正/反手分类依据 |
| 轻去重 | 通用 | **保留** |

> 「焊缝中点距离 base-x > 0.4」：把焊缝中点变换到 base 系 `seam_mid_base = R·mid_world + t`，
> 要求 `seam_mid_base.x > x_min`（`x_min` 读配置，默认 0.4）。这比「整场景最近点 base-x」稳健，
> 大场景最近点常落在 base 脚下会全灭。

### 2.3 焊缝 json schema（改名后对齐，`_load_seam` 有望直接复用）

**改动 1（你来做）**：`scripts/find_surface_welds.py` 里把输出键 `p0` / `p1` 改成
`corrected_p0` / `corrected_p1`（`find_welds` 末尾 append 的 dict，约 L542–543），然后**你自己重新生成 json**。

改名后 welds json 的键为：`corrected_p0` / `corrected_p1` / `bisector` / `boundary_dirs` /
`edge_dir` / `length` / `prim_path` …，与父类 `Scene._load_seam` → `plan_init_pose.load_welds`
所读的 `corrected_p0` / `corrected_p1` / `bisector` / `boundary_dirs` **完全一致**。

结论：**`_load_seam` 有望不必覆盖，直接复用父类**（读 `corrected_p0/1`、`bisector`、`boundary_dirs`，
按 `seam_min_length_m` 过滤短缝）。落地时确认一次：父类 `load_welds` 对 `boundary_dirs` 的长度/
结构假设与 `find_surface_welds.py` 输出一致即可；若有差异再最小化覆盖 `_load_seam`。

### 2.4 新增配置段 `observeanything.plan_init_pose_fast`

在 `configs/default.yaml` **新开** `observeanything` 项目，其下 `plan_init_pose_fast` 放本方法参数：

```yaml
observeanything:
  plan_init_pose_fast:
    # 采样：以焊缝中点为原点、周边 2.5m 立方体内采样机械臂 base 的 xyz + yaw
    sample_cube_half_m: 2.5        # 立方体“周边 2.5m” = 半边长（采样区 [-2.5, 2.5]³，边长 5m）
    xy_step_m: 0.20                # xy 采样间隔
    z_step_m: 0.10                 # z  采样间隔
    yaw_min_deg: 0
    yaw_max_deg: 360
    yaw_step_deg: 20               # yaw 采样间隔
    # 可达/过滤（口径同 plan_init_pose_fast）
    ee_xy_range_m: [0.4, 1.0]      # 端点在 ee 范围（保留）
    ee_z_range_m: [-0.1, 0.1]
    seam_center_x_min_m: 0.4       # 焊缝中点 base-x 下限（取代工件最近点判据）
    standoff_cm: 0.0
    workpiece_x_voxel_m: 0.1       # 场景体素点用于 init_free 交集判定
    front_face_filter: true        # 正面 bisector base-z≥0：保留（只在焊缝正脸可见的朝向放相机）
# init_free 仍走顶层 init_free.*_for_init（method_for_init / box_*_for_init），与父类一致
```

> init_free 区域**不放进** `observeanything` 段，仍读顶层 `init_free.*_for_init`
> （`_init_free_region_from_cfg` 现成逻辑，与 `_fresh_explore_voxmap` 一致）。

---

## 3. 前置步骤：USD → 世界系合并 `.obj`

下游把 `self.workpiece_obj` 当 mesh **文件**读：
- `compute_goal_pose`：`ScenePose2(obj_path=self.workpiece_obj)`；
- `_build_explore_world`：`CuMesh(file_path=self.workpiece_obj)` + `load_truth_scene(self.workpiece_obj)`；
- 重写的 `plan_init_pose_fast` 自身也要场景顶点/体素点做 init_free 交集判定。

`.usdz` trimesh/curobo 读不了，故 **`ObserveAnythingScene.__init__` 里先把 USD 导出成一个世界系（米）的
合并 `.obj`**（带缓存，按 usd 路径+mtime 命名），复用 `find_surface_welds.py` 现成函数：
- `load_scene_meshes(usd_path)` → 每个 `UsdGeom.Mesh` 的世界系（米）trimesh；
- `merge_meshes(tms)` → 合并；`trimesh` 导出 `.obj`。

因为焊缝 `corrected_p0/corrected_p1` 与导出的场景 mesh **同在世界系（米）**，所以「工件 mesh 系」在观测场景里
就是「USD 世界系」，焊缝坐标可直接用，帧天然自洽。

> ⚠️ 场景可能很大 → mesh 很大 → curobo ESDF/碰撞世界显存吃紧。对策（择一，文档标记为待验证）：
> - 只导出**目标焊缝邻域**的场景块（按焊缝中心裁剪半径 R 内的 mesh）当 workpiece_obj；
> - 或用父类 `_build_explore_world` 已有的 handle 复用 + 显存回收路径，先按整场景跑通再优化。
> 这与 `[[plan_explore_path 显存 OOM]]` 的教训一致：边走边看整场景大概率要拆进程/大卡。

---

## 4. `ObserveAnythingScene(Scene)` 结构

```
class ObserveAnythingScene(Scene):
    def __init__(self, cfg, usd_path, welds_json, scene_obj_cache=None, ...):
        # 1) USD → 世界系合并 .obj（缓存）→ obj_fp
        # 2) super().__init__(cfg, workpiece_obj=obj_fp, weld_json=welds_json)
        #    self.usd_path = usd_path 备用（可视化/裁剪）

    # _load_seam：改名后 json 键已对齐 → 期望【直接复用父类】，不覆盖（落地确认一次）

    def plan_init_pose_fast(self, include_obstacles=True, verbose=False):
        # 重写（override 同名方法）——见 §2：
        #   采样 = 以焊缝中点为原点、周边 2.5m 立方体内取机械臂 base 的 xyz（xy 20cm / z 10cm）
        #          + yaw 0~360° 步长 20°（臂竖直）→ T_base_world → inv() 得 T_workpiece_in_base
        #   过滤 = 端点在 ee 范围（保留）+ init_free vs 场景无交集（保留）
        #          + 焊缝中点 base-x > 0.4（改判据）；去掉底座-工件 XY 相交
        #   产出 InitPoseCandidate 写进 self.init_pose_candidates[seam_id]（结构同父类）
        # 复用 pim.mat44_to_pose7 / rotmat_to_quat_wxyz / _pts_in_init_free
        # / _init_free_region_from_cfg / _voxelize_mesh_points 等纯几何工具

    # set_init_pose / compute_goal_pose / compute_pose_and_plan_path /
    # _build_explore_world / plan_explore_path —— 全部继承父类，不覆盖
```

### 重写的 `plan_init_pose_fast` 内部（几何，纯 numpy）
以单条焊缝 `p0,p1,mid,bis`（世界系）为例：

1. **采样 base 世界位姿**：`mid` 为原点，立方体 `[-H,H]³`（`H = sample_cube_half_m`）内按
   `xy_step_m` / `z_step_m` 取格点 `pos_world`；yaw 从 `yaw_min~yaw_max` 步长 `yaw_step_deg`。
   base 竖直 → `R_base_world = Rz(yaw)`；`T_base_world = [R_base_world | pos_world]`。
2. **换算候选帧**：`R = R_base_world.T`，`t = -R @ pos_world` → `p_base = R·p_world + t`
   （即 `T_workpiece_in_base = inv(T_base_world)`）。
3. **过滤**：
   - `bis_base = R·bis`；`front_face_filter`（保留，默认 true）要求 `bis_base.z ≥ 0`（只在焊缝正脸可见的朝向放相机）；正/反手按 `bis_base.x` 符号分类；
   - `seam_mid_base = R·mid + t`，要求 `seam_mid_base.x > seam_center_x_min_m`（**改判据**）；
   - `_inrange`（端点 `R·p0+t` / `R·p1+t` 在 `ee_xy_range_m`×`ee_z_range_m`）——照搬；
   - `scene_pts_base = 场景体素点 @ R.T + t`；`_pts_in_init_free(scene_pts_base, region).any()`
     则丢 —— **核心**，`region = _init_free_region_from_cfg(cfg)`（读 `init_free.*_for_init`）；
   - 轻去重（同朝向 + 位置阈值）。
4. 组 `cand` dict（字段与父类 `_fast_solve_weld` 逐位一致）→ `InitPoseCandidate.from_kejian2`，
   写入 `self.init_pose_candidates[seam_id]`（正/反手分桶，结构同父类）。

---

## 5. 待定/需拍板

1. **场景规模 / 显存**：整仓库 mesh vs 焊缝邻域裁剪（§3 待验证）。
2. **`_load_seam` 是否真能零覆盖**：取决于父类 `load_welds` 对 `boundary_dirs` 的结构假设是否与
   `find_surface_welds.py` 输出完全一致，落地时确认一次。

> 已定：① 采样立方体「周边 2.5m」= **半边长**（采样区 [-2.5, 2.5]³）；
> ② 正面过滤 `bis_base.z ≥ 0` **保留**（只在焊缝正脸可见的朝向放相机，兼作正/反手分类）。

---

## 6. 落地清单（后续实现时）

- [ ] 【你来做】`find_surface_welds.py`：输出键 `p0`/`p1` → `corrected_p0`/`corrected_p1`，重新生成 json。
- [ ] `ObserveAnythingScene.__init__`：USD → 合并 `.obj`（缓存）+ `super().__init__`。
- [ ] `_load_seam`：期望直接复用父类；落地确认 `boundary_dirs` 结构一致，否则最小化覆盖。
- [ ] 重写 `plan_init_pose_fast`：世界系采样 base(xyz+yaw) + 三条过滤（端点 ee / init_free 无交集 / 焊缝中点 base-x>0.4），复用 `plan_init_pose.py` 几何工具。
- [ ] `configs/default.yaml`：新增 `observeanything.plan_init_pose_fast` 段（§2.4）。
- [ ] demo 脚本：`plan_init_pose_fast` → `set_init_pose` → `compute_pose_and_plan_path`（继承）。
- [ ] 跨进程可视化沿用 `Scene.save/load` + `Open3DSceneVisualizer`（见 `[[scene_save_load_cross_process_viz]]`）。

---

## 附：关键复用点索引

- 相对位姿语义：`gt_gen/scene.py:InitPoseCandidate` / `set_init_pose` / `compute_goal_pose`
- 快速几何过滤链（照搬思路）：`scripts/plan_init_pose.py:_fast_solve_weld`（L2931+）、
  `_pts_in_init_free`（L2391）、`_init_free_region_from_cfg`（L2374）、
  `_voxelize_mesh_points`（L1990）、`mat44_to_pose7`（L249）、`load_welds`（L265）
- USD → mesh：`scripts/find_surface_welds.py:load_scene_meshes`（L102）/ `merge_meshes`（L333）/
  焊缝 json 输出 append（L540–547，改名 `corrected_p0/1` 处）
- init_free 配置：`configs/default.yaml: init_free`（L114，`*_for_init` 键）
- fast 配置参考：`configs/default.yaml: plan_init_pose_fast`（L185，新段照其口径命名）
