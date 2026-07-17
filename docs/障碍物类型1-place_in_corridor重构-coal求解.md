# place_in_corridor 重构方案：B 空间 + coal(hppfcl) 精确求解

> 状态：**已实现**（`gt_gen/obstacle_placement.py`）。coal/hppfcl 精确外推 + 水平偏航求解已上线，
> 最小场景验证通过（plate/open_box/gantry/box_beam/pipe/rect_frame/braced_frame/u_channel 均 clear+overlap）。
> 本文只描述 `gt_gen/obstacle_placement.py::place_in_corridor` 的重写方案，
> 不改上层 `add_obstacle_type1`（`scene.py`）与下游碰撞注入链路——接口 `(prims, anchor_eff, meta)` / `None` 保持不变。

---

## 1. 背景与现状缺陷

现行 `place_in_corridor`（`obstacle_placement.py:540`）流程：

1. 时间窗内对扫掠球心算 `sd = _signed_dist_init_free`，剔除离 goal 太近者，`argmax sd` → 锚球心 `c_k`；
2. 朝向：`+X` 对齐路径切向 + `angle_deg` 自转；
3. **3 轮收缩循环**：以 `c_k` 为心的外接半径 `R_b = _bounding_radius_about(prims, c_k)`，若 `R_b > sd_k`
   就按比例缩 `span/tube_r/thickness`，直到障碍整只塞进 init_free 外余量；
4. 兜底 `_clears_init_free_any` + `_overlaps_sweep`，否则 `None`。

**根本缺陷**：第 3 步用「以 `c_k` 为心的包围球半径 `R_b`」当尺寸判据。

- 对**细长 / 中空**障碍（`box_beam` / `pipe` / `rect_frame` / `gantry` / `braced_frame`…），沿 init_free 外法向的
  真实厚度远小于 `R_b`。用 `R_b` 判据会**过度收缩**，把本可原尺寸放下的障碍缩成一小块，甚至缩到 clip 下界仍判放不下 → 误 `None`。
- 「缩尺寸」违背需求：用户要求**按障碍参数生成 1 个原尺寸障碍 C**，不允许为了塞进去而改变它的尺寸。

---

## 2. 需求（用户原话）

> 更改 place_in_corridor 逻辑：先计算目标 link 的**扫掠空间**，然后把 **init_free 中的刨去**，剩下的空间记为 **B**。
> 按照障碍物参数生成 **1 个障碍物 C**，把 C 放到**和 B 相交、且不和 init_free 相交**的区域。
> 怎么放通过**解析解直接计算**出来，不要试。

补充确认（多轮对话）：

- **扫掠空间是整只球扫过的体积**（并集，含半径），不是球心点集。
  → 球 `i` 属于 B 的判据是 `sd(c_i) + r_i ≥ 0`（球面探出 init_free），而非 `sd(c_i) > 0`。
- **不设「最小探出量 ε」**：候选判据就是 `sd(c_i)+r_i ≥ 0`，候选为空则 `None`。
- **障碍 C 保持参数原尺寸**，不缩。
- **求解自由度（本次最终确认）**：`xyz 平移` + `只在水平方向（绕竖直轴）旋转的 1 个偏航自由度`。
  即障碍保持竖直姿态（pitch/roll 不变），只允许在水平面内转向，用来在「正着放塞不进、斜一点就能出 init_free」时救回，
  且不破坏「正面横挡走廊」的语义。**不做自由 6-DOF**（避免解出「侧刃从缝里溜过」的可行但没用姿态）。
- **用公开碰撞库做精确判据**：`coal`（新名）/ `hppfcl`（hpp-fcl，旧名，本机已装）。

---

## 3. 问题的等价化简（关键）

设 init_free 为凸集（当前配置是 box，也支持 cylinder），障碍 C 是若干凸原语（`Box`/`Tube`）的并集，扫掠是球集合。

两条约束：

- ①「C 不进 init_free」：`C ∩ init_free = ∅`。
- ②「C 与 B 相交」：`C ∩ B ≠ ∅`，其中 `B = sweep − init_free`。

**化简**：若 ① 成立（C 整只在 init_free 外），则 `C ∩ sweep` 的任何点都在 init_free 外，故
`C ∩ sweep ≠ ∅ ⟺ C ∩ B ≠ ∅`。于是两约束等价为：

> **C 整只在 init_free 外（①） 且 C 与扫掠球并集有交（②'）。**

不必显式构造 B——只需「C 出 init_free」+「C 碰到扫掠管」。这正是 `_clears_init_free_any` 与 `_overlaps_sweep`
已经在做的精确判据。重写的核心是**如何解析地把 C 摆到同时满足这两者的位姿**，而不再靠缩尺寸。

---

## 4. 新算法（分阶段）

记：锚球 `(c_k, r_k)`；障碍局部系 `+X` 对齐切向的初始朝向 `Rm0`；竖直轴 `ẑ = (0,0,1)`（base 系重力向上）。

### Phase 0 — 输入（不变）
`per_wp[link]`（T,S,4 扫掠球）、`origin[link]`（T,3）、init_free（box/cyl，来自 cfg）、
障碍参数（`otype/size_scale/angle_deg/thickness`）、`goal_pos` + `goal_clearance_m`。

### Phase 1 — 选锚球（闭式，确定性）
1. 时间窗 `[i0,i1]`（`op["pos_t_window"]`）。
2. 对窗内所有扫掠球心算 `sd = _signed_dist_init_free(centers, cfg)`。
3. **B 判据**：`sd(c_i) + r_i ≥ 0`（球面探出 init_free）。
4. 剔除 `‖c_i − goal‖ < goal_clearance_m`（别贴焊缝）。
5. 若无 B 候选 → **`None`**（该 link 扫掠几乎全埋在 init_free 内，上层换 link / 挪 `pos_t_window`）。
6. **锚 `c_k = argmax sd`**（B 候选中 sd 最大者）：离 init_free 最深 → 需要的外推最小、挡路最实。
   记 `sd_k = sd(c_k)`、`t_k` = 其时间步。

> 锚球的作用：① 定初始位置（body 中心落在 `c_k` → 天然含扫掠球 `(c_k,r_k)` → 初始与扫掠有交）；
> ② 定初始朝向（`t_k` 处切向）；③ 给外推提供起点。

### Phase 2 — 初始朝向
- 切向 `tangent = origin[t_k+1] − origin[t_k−1]`；
- `rpy0 = _rpy_align_x_to(tangent, roll_deg=angle_deg)` → `Rm0`（`+X` 横挡走廊）。

### Phase 3 — 造**原尺寸**障碍（删除收缩循环）
- `shape, local_off = _shape_for(otype, span, tube_r, cfg, th=thickness)`，
  `span = 2·tube_r0·size_scale`、`tube_r0` 为 `t_k` 处走廊管半径（沿用现有量法）。
  **不再乘 `scale`、不再 3 轮收缩**。
- body 中心置于 `c_k`：`anchor_eff0 = c_k + Rm0·local_off`；`prims0 = ob.build(otype, anchor_eff0, rpy0, **shape)`。

### Phase 4 — coal 精确分离 + 水平偏航求解（本次核心）
用 coal/hppfcl 把「过度外推」换成「按真实贯入深度 + 接触法线最小外推」，并加 1 个水平偏航自由度。

定义绕**竖直轴** `ẑ`、过锚点 `c_k`、角 `ψ` 的旋转 `Ryaw(ψ)`。对给定 `ψ`：

1. **摆姿**：`Rm(ψ) = Ryaw(ψ)·Rm0`，body 中心仍在 `c_k`（`anchor = c_k + Rm(ψ)·local_off`），得原语世界位姿。
2. **精确分离量**：对每个障碍 prim 与 init_free 几何调 `coal.distance(...)`（`enable_signed_distance=True`），
   取贯入最深者的 `(depth, n̂)`（`depth = −min signed_dist`，`n̂` = 接触法线，指向「把障碍推出 init_free」方向）。
   → **最小外推向量** `v(ψ) = max(depth, 0)·n̂`（真实厚度，不是包围球 `R_b`）。
3. **外推后**：`c_k' = c_k + v(ψ)`，此时 `C(ψ)` 恰好脱离 init_free（约束①满足）。
4. **保交判据**：外推后 `C(ψ)` 是否仍与扫掠球并集有交（`_overlaps_sweep`，或 coal 对锚球做 distance ≤ 0）。
   - 外推越小 → 越可能仍嵌在扫掠管里（约束②'满足）。
5. **偏航求解**：在有界区间 `ψ ∈ [−ψmax, +ψmax]`（如 ±90°）里，取**使 `‖v(ψ)‖` 最小、且外推后仍保交**的 `ψ*`。
   - 实现：先在候选角（0、以及让障碍最薄面朝 `n̂` 的解析角）粗评，再有界一维精化（几步二分/黄金分割，
     每步只做廉价 coal 距离查询——**不是**旧「放→跑规划器→测→改尺寸→重放」的试探，是局部几何求解）。
   - 若整个区间内**无** `ψ` 能同时「出 init_free」且「保交」 → **`None`**（上层换 link / 类型 / 尺寸）。

> 为什么偏航能救回：`v(ψ)` 的大小 = 障碍沿 `n̂`（init_free 面法向）的贯入厚度。水平转向改变障碍呈给 init_free
> 那一面的截面厚度——把薄面转向 `n̂` → 需要外推的量变小 → 外推后更可能仍咬住扫掠管。竖直姿态不变，`+X` 仍大体横挡走廊，
> 不会退化成「侧刃溜过」。

### Phase 5 — 兜底 + 返回
- 断言 `_clears_init_free_any(prims, cfg)`（应由 coal 分离构造性成立）+ `_overlaps_sweep(prims, sph)` + goal 间距；
- 通过 → `return prims, anchor_eff, meta`（`meta` 增补 `yaw_deg=ψ*`、`push_vec=v(ψ*)`、`push_norm=‖v‖`，
  去掉 `shrink_scale`），否则 `None`。

---

## 5. coal / hppfcl 封装设计

### 5.1 运行环境（已验证）
- 本番 env：`env_isaaclab`（`/home/a/miniforge3/envs/env_isaaclab`），含 `warp 1.13.0` + `curobo` + `numpy/scipy`。
- **`import hppfcl` 可用（hpp-fcl 2.4.4）**；`import coal` 不可用（新名 3.x 未装）。其余 env（isaacsim42/45）两者皆无。
- 签名距离 API 实测：
  ```python
  import hppfcl, numpy as np
  g1 = hppfcl.Box(np.array([dx,dy,dz]))          # 全边长
  g2 = hppfcl.Cylinder(radius, height)           # 全高；halfLength=height/2；轴沿局部 +Z
  T  = hppfcl.Transform3f(); T.setRotation(Rm3x3); T.setTranslation(t3)
  req = hppfcl.DistanceRequest(); req.enable_signed_distance = True
  res = hppfcl.DistanceResult()
  d = hppfcl.distance(g1,T1, g2,T2, req, res)     # d<0 即贯入，|d|=贯入深度
  n = np.asarray(res.normal)                      # 接触/分离法线
  p1,p2 = res.getNearestPoint1(), res.getNearestPoint2()
  ```

### 5.2 新增模块级 helper（`obstacle_placement.py`）
- `_fcl():` 导入回退——`try import coal as _fcl / except import hppfcl as _fcl / except → None`。
  返回 `None` 时 `place_in_corridor` **退回旧包围球 `R_b` 路径**（无库环境仍能跑，只是保守）。
- `_fcl_geom_for_prim(prim) -> (geom, Transform3f)`：`ob.Box→Box(dims)`、`ob.Tube→Cylinder(radius,height)`，
  位姿从 `prim.pose`（xyz + wxyz）转 `Transform3f`。
- `_fcl_geom_init_free(cfg) -> (geom, Transform3f)`：按 `cfg.init_free_method_for_init` 造
  box（`Box(hi−lo)` + 中心平移）或 cylinder（`Cylinder(r,h)` + 轴心/高度中点平移）。
- `_penetration(prims_world, init_free_geom) -> (depth, normal)`：遍历 prim 调 `distance`，返回**最深贯入**的
  `(depth≥0, n̂)`；全分离时 `depth=0`。这是替换 `R_b` 的精确核心。
- （可选）`_yaw_solve(...)`：封装 Phase 4 的有界一维偏航搜索，返回 `(ψ*, v*, prims*, anchor*)` 或 `None`。

### 5.3 不改动的既有件
`_signed_dist_init_free` / `_clears_init_free_any` / `_overlaps_sweep` / `_shape_for` / `_rpy_align_x_to`
/ `compute_link_sweep` 全部保留（Phase 1 选锚、Phase 5 兜底仍用它们）。仅 `place_in_corridor` 主体重写。

---

## 6. 与旧版差异一览

| 维度 | 旧版 | 新版 |
|---|---|---|
| 尺寸 | 3 轮收缩到 `R_b ≤ sd_k` | **原尺寸不缩** |
| 分离判据 | 以 `c_k` 为心的包围球 `R_b`（对细长/中空过度外推） | coal 精确贯入深度 + 接触法线 |
| 摆位自由度 | 仅 argmax 定位（位置唯一，朝向切向定死） | xyz 外推 + **水平偏航 ψ**（1 转动 DOF） |
| 求解 | 缩尺寸循环 | 闭式选锚 → coal 最小外推 → 有界一维偏航精化 |
| `None` 条件 | clip 下界仍 `R_b>sd_k` | 无 B 候选，或任何 ψ 都无法「出 init_free 且保交」 |
| `meta` | `shrink_scale/init_free_push=0` | `yaw_deg/push_vec/push_norm`（删 `shrink_scale`） |

---

## 7. 坐标系一致性（不变）
放置全程在 **base_link 系**（扫掠球来自 FK、init_free box、切向、外推向量、coal 查询都在 base 系）。
`place_in_corridor` 返回 base 系原语；`scene.py::add_obstacle_type1` 再用 `inv(T_workpiece_in_base)`
转回工件 mesh 系存进 `ObstacleSpec`（与 type2/3 一致，本次不改）。

---

## 8. 边界与返回 None
1. link 无碰撞球定义 / `T<3` → `None`（同旧）。
2. 时间窗内无 `sd(c_i)+r_i ≥ 0` 且离 goal 够远的球（B=∅，整段扫掠埋在 init_free 内）→ `None`。
3. 偏航区间内任何 ψ 都无法「原尺寸出 init_free 且仍咬住扫掠管」→ `None`（上层换 link/类型/尺寸重试）。
4. 无 coal/hppfcl 库 → 退回旧包围球 `R_b` 路径（保守，可能偏严）。

---

## 9. 待实现清单（确认后执行）
- [x] 删 `place_in_corridor` 的 3 轮收缩循环（旧逻辑抽到 `_place_shrink_fallback`，仅无库回退用）。
- [x] 加 `_fcl()` / `_fcl_T` / `_fcl_geom_for_prim` / `_fcl_geom_init_free` / `_penetration`
      / `_clears_init_free_exact` / `_push_out_of_init_free` / `_yaw_solve`。
- [x] Phase 1 选锚判据改为 `sd + r ≥ 0`（含半径），保留 goal 剔除与 argmax。
- [x] Phase 4 偏航求解（绕竖直轴、有界一维粗扫+邻域细扫、coal 距离评估、迭代法线外推）。
- [x] `meta` 字段调整（`yaw_deg/push_vec/push_norm`，删 `shrink_scale/init_free_push`；回退路径仍保留旧字段）。
- [x] 兜底断言改用 `_clears_init_free_exact`（fcl 精确，与外推判据一致）。上层 `add_obstacle_type1` / 下游不动。

> 实现要点补充：
> - **外推为迭代式**（`_push_out_of_init_free`，≤12 轮）：多 prim 凹障碍（open_box/gantry/frame）单次法线外推
>   不保证全 prim 出 init_free，故每轮取「当前最深贯入」的法线继续推，直到 fcl 精确判定 depth≤tol。
> - **偏航搜索**：粗扫 step 15°（±yaw_max，默认 ±90°，读 `op.get("yaw_search_max_deg",90)`）择 ‖v‖ 最小者，
>   再在其 ±15° 内 step 3° 细化。每个 ψ 用 `Rz(ψ)·Rm0` 绕世界竖直轴、外推后须 `_overlaps_sweep` 保交。
> - **兜底与外推同源**：断言用 `_clears_init_free_exact`（fcl signed-dist），不用保守包围球，避免「fcl 判已分离
>   但包围球判仍相交」而误否掉可行解。无库回退路径才用包围球 `_clears_init_free_any`。

## 10. 明确不做
- 不做自由 6-DOF 姿态优化（只 xyz + 水平偏航）。
- 不改障碍尺寸（原参数尺寸）。
- 不重新规划轨迹、不做三条件验证、不改坐标系存储约定。
- 不引入新配置段（偏航区间 `ψmax` 若需可配，复用 `obstacle_placement` 段加一项，默认 ±90°）。
