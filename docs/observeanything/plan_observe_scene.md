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
- **焊缝 json schema 对齐**：`find_surface_welds.py` 已把 `p0/p1` 改存 `corrected_p0/corrected_p1`
  （**已完成**，json 已重新生成），键名与父类 `load_welds` 完全一致 → `_load_seam` 有望直接复用（§2.3）。
- 需要一个**前置步骤**：从 USD **只裁剪目标焊缝邻域**（焊缝中心半径 R 内的整块 prim mesh），
  求布尔并集做成**闭合(watertight)**的世界系（米）`.obj`，当作 `workpiece_obj`（§3）。

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

**改动 1（已完成）**：`scripts/find_surface_welds.py` 里输出键 `p0` / `p1` 已改成
`corrected_p0` / `corrected_p1`（`find_welds` 末尾 append 的 dict），并已重新生成 json。

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
  crop:
    crop_radius_m: 3.5             # USD→obj 邻域裁剪半径（焊缝中心球内整块 prim 保留）；
                                   # ≳ sample_cube_half_m + ee_xy_range_m 上限，覆盖采样+可达+init_free
    watertight: false              # 默认拼接(concatenate，快、每壳闭合，碰撞够用)；
                                   # 置 true 才走 make_watertight 布尔并集成单一流形(慢、贴合处可能反破)
# init_free 仍走顶层 init_free.*_for_init（method_for_init / box_*_for_init），与父类一致
```

> init_free 区域**不放进** `observeanything` 段，仍读顶层 `init_free.*_for_init`
> （`_init_free_region_from_cfg` 现成逻辑，与 `_fresh_explore_voxmap` 一致）。

---

## 3. 前置步骤：USD →「焊缝邻域裁剪」世界系 `.obj`（默认拼接，闭合可选）

下游把 `self.workpiece_obj` 当 mesh **文件**读：
- `compute_goal_pose`：`ScenePose2(obj_path=self.workpiece_obj)`；
- `_build_explore_world`：`CuMesh(file_path=self.workpiece_obj)` + `load_truth_scene(self.workpiece_obj)`；
- 重写的 `plan_init_pose_fast` 自身也要场景顶点/体素点做 init_free 交集判定。

`.usdz` trimesh/curobo 读不了，且整仓库 mesh 太大（curobo ESDF/碰撞世界显存吃紧，见
`[[plan_explore_path 显存 OOM]]`）。故 **`ObserveAnythingScene.__init__` 里只导出目标焊缝邻域的场景块**
（带缓存，按 usd 路径 + 焊缝 id/中心 + 半径 R 命名）：

1. **加载整场景 prim mesh**：`load_scene_meshes(usd_path)` → 每个 `UsdGeom.Mesh` 的世界系（米）trimesh
   `[(prim_path, Trimesh), ...]`（`__init__` 只做一次，缓存在内存）。
2. **按焊缝中心半径 R 裁剪（整块保留，不切三角形）**：以焊缝中点 `mid_world` 为球心，
   **只保留** bbox（或任一顶点）落入半径 R 内的**整块 prim mesh**。
   > 关键：**按 prim 整块取舍**，绝不按三角形裁切。切三角形会在切面留下破洞 → 破坏闭合性；
   > 整块保留每块 prim（IFC/仓库构件本就是各自闭合的实体），每壳的闭合性才能保住。
   > R 读配置（§2.4 `crop_radius_m`，默认略大于采样立方体半边长 + ee 可达半径，保证采样区+可达范围内
   > 的场景都在块里）。
3. **默认：直接拼接（concatenate）——每缝毫秒级、不会失败**。把裁到的各块拼成一个 mesh
   （不跨块焊顶点）即当 workpiece_obj。结果 `all_shells_watertight=True`——**这正是 curobo 碰撞/ESDF
   真正需要的**：相互重叠的闭合壳占据空间天然取并、内部面机械臂碰不到，对避障无害。
   > **已核实（读代码）：全链路无「整体单一 watertight」硬性要求，只 ESDF 一步要「每壳闭合」，且它专为多闭合壳求并设计。**
   > ① ScenePose2 MESH 碰撞：`scene_pose2.py:_load_piece`(L175，`trimesh.load(process=False)`，无 watertight 断言)
   >   + `_init_curobo`(L227，`Mesh(vertices/faces)` + `CollisionCheckerType.MESH`) + `_collided_batch`(L374，碰撞球
   >   vs 三角形逐面判碰)——**逐三角形，闭合无关**；② `_build_explore_world` h_truth 同为 MESH(`scene.py:1464`)；
   >   ③ warp/trimesh raycast(`scene_pose2.py:_build_wp_mesh` L188 / `sensor.py:raycast_observe` L139，取最近命中)——
   >   **逐三角形，闭合无关**；④ 唯一有拓扑要求的是 ESDF 纠符号(`scene.py:798-812`)：无符号距离
   >   + `igl.fast_winding_number(V,F,·)` + `wn>0.5` 判内——**广义缠绕数对重叠闭合壳求和**(壳内 wn≈1、
   >   重叠区 wn≈2 均判内)，且 `process=False`(L769) 不跨块焊顶点，各壳独立正好可求和；作者原话
   >   `scene.py:745`「igl sign 用合并后的 (V,F)（多个 watertight 组件缠绕数求和）」。⑤ `grep is_watertight/
   >   fill_holes/assert` 全链路零命中。故**拼接（每壳闭合）即满足全部消费者**；布尔并集成单一流形不被任何一步*需要*。
   > 为什么不默认布尔并集：① **速度**——`boolean.union` 是每缝成本大头（随块数/面数增长，可到秒级），
   > 而拼接只是「筛 + 摞」，毫秒级；② **鲁棒**——「每块 watertight」**不保证**其布尔并集 watertight：
   > 两块**共面贴合**（仓库梁/板对齐极常见）、**仅沿棱/点接触**都会让并集边界出现**非流形边/捏合点**，
   > 反而破掉闭合性（`make_watertight` 靠 `drop_degenerate` 兜底、`check_watertight` 专门统计非流形边，
   > 正因这会真实发生）。拼接则拓扑上**仍无边界边**（各壳顶点独立、互不共享边），代价只是**自相交 +
   > 保留内部几何**——对碰撞无害。
4. **可选：整体布尔并集成单一干净流形**（`crop.watertight: true` 时）。走 `watertight/make_watertight.py`
   口径（`components_as_solids` + `trimesh.boolean.union(engine="manifold")` + `drop_degenerate`
   + `merge_vertices`），去掉内部几何、得到单一 2-流形，更漂亮；但更慢、且贴合处可能反而破掉 watertight，
   故仅在下游确需「整体单一 watertight」时开。可用 `check_watertight.py:check_mesh` 自检、不过就回退拼接。

因为焊缝 `corrected_p0/corrected_p1` 与导出的场景 mesh **同在世界系（米）**，所以「工件 mesh 系」在观测场景里
就是「USD 世界系」，焊缝坐标可直接用，帧天然自洽；裁剪只是缩小 mesh，不改坐标系。

> 说明：裁剪半径 R 让 workpiece_obj 只覆盖「采样立方体 + ee 可达 + init_free」需要判碰的邻域，
> 直接解决整场景显存 OOM；不同焊缝各自裁自己的邻域块（缓存分文件）。

---

## 4. `ObserveAnythingScene(Scene)` 结构

```
class ObserveAnythingScene(Scene):
    def __init__(self, cfg, usd_path, welds_json, scene_obj_cache=None, ...):
        # 0) load_scene_meshes(usd) 一次，缓存整场景 prim mesh 列表在内存（世界系/米）
        # 1) super().__init__(cfg, workpiece_obj=<占位/首缝块>, weld_json=welds_json)
        #    self.usd_path = usd_path、self._scene_prims = [...] 备用
        #   ⚠ 邻域块按【焊缝中心半径 R】裁剪 → 每条缝的 workpiece_obj 不同，必须【按缝重建】

    def _crop_neighborhood_obj(self, seam_id):
        # 以该缝 mid_world 为球心、半径 R 整块保留 prim →
        # 默认 concatenate 拼成一个 mesh（毫秒级，all_shells_watertight）；
        # crop.watertight=true 时才走 make_watertight 口径求布尔并集成单一流形 →
        # export 世界系(米) .obj（缓存按 usd+seam_id/中心+R 命名）→ 返回 obj_fp

    def _set_cur_seam(self, seam_id):
        # 切缝时把 self.workpiece_obj 重指向本缝裁剪块 → 让 compute_goal_pose /
        # _build_explore_world / plan_init_pose_fast 都吃到本缝邻域块；再调 super()._set_cur_seam

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

1. **裁剪半径 R 的取值**：默认 `crop_radius_m: 3.5`（≳ 采样半边长 2.5 + ee 可达 1.0）；
   若某些缝邻域仍过大/过小，落地时按显存与召回调。
2. **每缝是否需要「整体单一 watertight」**：默认**否**——每缝直接拼接（毫秒级），结果
   `all_shells_watertight=True`，curobo 碰撞/ESDF 够用（重叠闭合壳占据空间天然取并，内部面碰不到）。
   **已核实（读代码，file:line 见 §3 step3 blockquote）：全链路无「整体单一 2-流形」硬性要求**——
   MESH 碰撞(ScenePose2 `_collided_batch` / `_build_explore_world` h_truth) 与 raycast 都是**逐三角形、闭合无关**；
   唯一有拓扑要求的 ESDF 纠符号用 `igl.fast_winding_number`（`scene.py:798-812`），**专为多闭合壳缠绕数求和**
   （作者原话 `scene.py:745`），拼接的独立闭合壳正好满足。
   注意「每块 watertight」**不保证**其布尔并集 watertight：共面贴合 / 仅棱·点接触会让并集边界出非流形边，
   反而破掉闭合性。仅当落地确认下游硬性要求「整体单一 2-流形」时（当前**并无**此需求），才置
   `crop.watertight: true` 走 `make_watertight` 布尔并集（更慢、且需 `check_watertight` 兜底、不过就回退拼接）。
3. **`_load_seam` 是否真能零覆盖**：取决于父类 `load_welds` 对 `boundary_dirs` 的结构假设是否与
   `find_surface_welds.py` 输出完全一致，落地时确认一次。

> 已定：① 采样立方体「周边 2.5m」= **半边长**（采样区 [-2.5, 2.5]³）；
> ② 正面过滤 `bis_base.z ≥ 0` **保留**（只在焊缝正脸可见的朝向放相机，兼作正/反手分类）；
> ③ `find_surface_welds.py` 键名 `p0/p1 → corrected_p0/corrected_p1` **已完成**、json 已重生成；
> ④ USD→obj **只裁焊缝邻域**（半径 R 整块保留 prim）；**默认拼接**（每壳闭合、毫秒级、碰撞够用），
>   整体布尔并集成单一流形为**可选**（`crop.watertight`，更慢且贴合处可能反破）。

---

## 6. 落地清单（后续实现时）

- [x] `find_surface_welds.py`：输出键 `p0`/`p1` → `corrected_p0`/`corrected_p1`，已重新生成 json。
- [x] `ObserveAnythingScene.__init__`：`load_scene_meshes(usd)` 缓存整场景 prim；`_crop_neighborhood_obj(seam)`
      按焊缝中心半径 R 整块裁剪（AABB 到球心 ≤R）→ 默认 concatenate（`crop.watertight` 时才布尔并集）+ 导出（缓存）；
      `_set_cur_seam` 重指 obj。（`gt_gen/observe_scene.py`）
- [x] `_load_seam`：直接复用父类（`boundary_dirs` 结构一致，零覆盖）。
- [x] 重写 `plan_init_pose_fast`：世界系采样 base(xyz+yaw) → `inv(T_base_world)` + 过滤（正面 bis_base.z≥0 / 端点 ee / 焊缝中点 base-x>0.4 / init_free 无交集 / 轻去重），复用 `plan_init_pose.py` 几何工具。
- [x] `configs/default.yaml`：新增 `observeanything.plan_init_pose_fast` + `observeanything.crop` 段（§2.4）；`gt_gen/config.py` 加 `obs_fast_*`/`obs_crop_*` 访问器。
- [x] demo 脚本：`scripts/demo_observe_scene.py`（`--stage fast|plan|viz`）——`plan_init_pose_fast` → `set_init_pose` → `compute_pose_and_plan_path`（继承）。
- [x] 跨进程可视化沿用 `Scene.save/load` + `Open3DSceneVisualizer`（见 `[[scene_save_load_cross_process_viz]]`）。

---

## 附：关键复用点索引

- 相对位姿语义：`gt_gen/scene.py:InitPoseCandidate` / `set_init_pose` / `compute_goal_pose`
- 快速几何过滤链（照搬思路）：`scripts/plan_init_pose.py:_fast_solve_weld`（L2931+）、
  `_pts_in_init_free`（L2391）、`_init_free_region_from_cfg`（L2374）、
  `_voxelize_mesh_points`（L1990）、`mat44_to_pose7`（L249）、`load_welds`（L265）
- USD → mesh：`scripts/find_surface_welds.py:load_scene_meshes`（L102，世界系/米 per-prim trimesh）/
  `merge_meshes`（L333）/ 焊缝 json 输出 append（已改名 `corrected_p0/1`）
- 邻域裁剪块做成闭合 obj：`watertight/make_watertight.py:make_watertight`（`components_as_solids` +
  `trimesh.boolean.union(engine="manifold")` + `drop_degenerate`）；自检 `watertight/check_watertight.py:check_mesh`
- init_free 配置：`configs/default.yaml: init_free`（L114，`*_for_init` 键）
- fast 配置参考：`configs/default.yaml: plan_init_pose_fast`（L185，新段照其口径命名）
