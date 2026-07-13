# 缝周多分辨率 voxmap 方案（细化焊缝周围体素、其余区域保持粗）

> 状态：**设计阶段，未改代码**。本文记录动机、可行性分析、约束、逐模块改法与待定项，供确认后落地。
> 相关：[`gt-generation-curobo-implementation.md`](./gt-generation-curobo-implementation.md)、[`implementation-steps.md`](./implementation-steps.md)、[`privileged-nbv.md`](./privileged-nbv.md)。

---

## 0. 需求与结论

**需求**：能否精细化 `voxel_size_m`——焊缝周围一定范围用小体素（更精），其他区域用大体素（省显存/算力）？

**结论**：能，但不是「改一个数」。同一张网格内变分辨率，`ThreeStateVoxelMap` 和 cuRobo 的 **VOXEL** 碰撞世界都不支持（二者都是单一标量 `voxel_size`）。要做多分辨率，得选一种**分层架构**。

**关键前提（决定工作量）**：只要**只用 STOMP 规划**（`cfg.planner_backend='stomp'`），cuRobo 的 VOXEL 世界就不参与碰撞判定，那条「单一 voxel_size」硬约束随之消失，真·多分辨率从「高工作量」降到「中等」。详见 §2。

---

## 1. 背景：两张网格、两种角色

链路里其实有**两张独立的体素网格**，精度诉求主要落在前者：

| 网格 | voxel_size 来源 | 作用 | 精度影响 |
|---|---|---|---|
| **voxmap**（`ThreeStateVoxelMap`） | `build_roi_voxmap` ← `roi.voxel_size_m`（default.yaml，当前 **0.04**） | ①探索记忆 ②**保守 GT 判据** `motion_stays_in_free` ③观测 `carve_observe` | 判据精度、丢候选多少、观测细节 |
| **cuRobo VOXEL 世界** | `init_curobo` ← 同一个 `roi.voxel_size_m` | cuRobo 后端规划器**找候选路径** | 规划找路的精度 |

**松耦合点**：`sync_collision_world`（`collision_sync.py`）是「按世界中心查表」——
`voxmap.get(voxmap.world_to_voxel(centers))`，注释也写了「voxmap 分辨率/范围与 cuRobo 网格无需相等，只需覆盖」。**即 voxmap 与 cuRobo 的分辨率本就可独立设**。

`motion_stays_in_free`（`swept.py`，GT 正确性闸门）跑在 **voxmap** 上，不碰 cuRobo。所以只要把 voxmap 在缝周细化，就能提升「这条路是否真在观测自由空间里」的判据精度和观测细节。

---

## 2. STOMP-only 让约束消失（核心洞察）

看 `main_loop.py:807-815` 的 backend 分支：

- **STOMP 模式**：每步调 `stomp_iface.world_from_voxmap_auto(cfg, voxmap)`，把 voxmap 的**非 FREE 区**转成
  - **MESH**（marching-cubes，`world_from_voxmap`）或
  - **CUBOID**（greedy 合盒，`world_from_voxmap_cuboid`）
  世界，再喂给 `plan_joint_single/plan_pose_single`。STOMP 的碰撞判定走这张 **mesh/primitive 世界**。
- **cuRobo 模式**：才用 `h_expl` 的 VOXEL ESDF 世界 + `sync_collision_world`。

**关键**：mesh/cuboid 世界**没有单一 voxel_size 约束**——它就是一堆三角面 / 长方体，来自哪个分辨率的体素都无所谓，union 到一起 cuRobo 照单全收。那条「同一网格内不能变分辨率」的硬约束是 cuRobo **VOXEL** checker 独有的；STOMP 不碰它。

> **残留**：`main_loop.py:803` 的 `sync_collision_world(h_expl, voxmap)` 目前**无条件**调用，stomp 下也在往 cuRobo voxel 世界灌数据，但规划不读它——stomp 下 `h_expl` 只剩 FK/运动学和 `explain_endpoints` 用途。见 §4 Row 5。

因此，多分辨率**不会**因细 voxel 撑爆显存（记忆中 `plan_explore_path` 的 OOM 是 VOXEL 世界导致；mesh/cuboid 版省得多），也**避开** cuRobo 的「维度须为 voxel_size 整数倍」等断言。

---

## 3. 前提数据结构：`MultiResVoxelMap`

不是「一张网格支持变分辨率」，而是**两张独立的 `ThreeStateVoxelMap` 拼在一起**，外面套一个壳：

- **coarse（粗网格）**：`vs=0.04`，覆盖**整个 ROI**（含 retract→焊缝的整段接近路径）。
- **fine（细网格）**：`vs=0.01`，只覆盖**焊缝为中心的小盒**（如 0.5 m 立方，待定）。
- **归属规则**：一个世界坐标点，落在 fine 盒内 → 归 fine 网格；否则 → 归 coarse 网格。

### ⚠ 最关键的设计约束（后续各行反复用到）

coarse 网格在**物理上**也盖住了 fine 盒那块区域，但必须让 coarse 在 fine 盒内的格子**「挖空」**——查询和建世界时都当它不存在。

> **为什么必须挖空**：coarse 是 4 cm 粒度，fine 盒那块若还没探索就是一片 4 cm 的 UNKNOWN（=障碍）。若不挖空，这片 4 cm 障碍会**盖在** fine 用 1 cm 雕出来的 FREE 通道上，把细通道整个糊死——细化就白做了。所以 coarse 必须在 fine 盒处留一个洞，那块交给 fine 说了算。

> 先做**两级**（粗+细）即够用；壳留扩展口，以后可扩 N 级。

---

## 4. 逐模块改法

改动面集中在 **3 个文件**（voxmap / stomp_iface / swept+sensor）+ main_loop 一行 gate + config。

### Row 1 — `voxmap.py`：新增 `MultiResVoxelMap`

消费者调 voxmap 的方法分两类，多分辨率下处理方式不同：

**A 类：按「世界坐标点」查/设 —— 能干净分派**
- `get(points)`：按「是否在 fine 盒内」把点分两组，分别丢 `fine.get` / `coarse.get`，再按原顺序拼回。`motion_stays_in_free` 的 `voxmap.get(cells)` 走这条，透明。
- `set_world(points, state)`：按点归属分派到对应网格写入。

**B 类：暴露「单一分辨率」的方法 —— 无唯一答案，须改调用方**
- `world_to_voxel(p)` / `voxel_to_world(idx)`：一个整数下标属于哪张网格？歧义。`swept.voxelize_spheres`、`sensor._roi_centers` 直接用裸下标，**不能靠壳透明分派**，得改成「对两张子网格各自算」（见 Row 3、Row 4）。
- `voxel_size`：标量语义没了。保留一个「代表值」给仍读它的旧代码，但真正在意精度处（swept 的 `pad=0.5·vs·√3`、carve 的 `near=vs`）都要改成**按当前处理的子网格取**。
- `non_free_mask()` / `grid`：不再是单一 array。建世界的函数（Row 2）改为分别取两张子网格的 mask，不再调此合并方法。
- `state_centers(state)`：`coarse.state_centers`（排除 fine 盒内）+ `fine.state_centers`，拼接返回。可视化/nbv 用它，透明。

**小结**：壳让 A 类透明；B 类的三个消费者（swept、carve、world 构建）本质是「对整张网格做一遍」，改成「对两张子网格各做一遍再合并」。

### Row 2 — `stomp_iface.world_from_voxmap_auto`：两张世界 union 成一个 STOMP world

当前 `world_from_voxmap`（mesh 版，152 行）/ `world_from_voxmap_cuboid`（cuboid 版，224 行）读一张 voxmap 的 `non_free_mask` + 单一 `vs`，产一批 mesh 或 cuboid。改成：

1. **fine 子网格** → 照原样跑 → 一批 **1 cm** 的细 mesh / cuboid（焊缝周围精细障碍面）。
2. **coarse 子网格** → 先把 fine 盒内的格子从 mask 剔掉（§3「挖空」），再跑 → 一批 **4 cm** 的粗 mesh / cuboid，fine 盒处留空。
3. **合并**：`WorldConfig(mesh = coarse_meshes + fine_meshes)`（cuboid 版同理 `cuboid=coarse+fine`）。**cuRobo 一个 world 放多个 mesh / 多个 cuboid 合法**——这是 STOMP-only 能做多分辨率的根本。

**膨胀（inflate）注意**：两张网格各自按自己的 vs 膨胀 1 层——粗层膨 4 cm、细层膨 1 cm。语义都是「1 体素余量」，但**物理厚度不同**：缝周余量变 1 cm（更薄、更贴合），正合意图；但要确认别薄到擦碰。

### Row 3 — `swept.motion_stays_in_free`：GT 正确性闸门变精（收益最大）

链路 `motion_stays_in_free → swept_volume → voxelize_spheres(voxmap, spheres)`：整臂碰撞球体素化后 `voxmap.get` 查是否全 FREE。精度由两个量决定，都随 vs 变：
- **FREE 空间量化粒度**（carve 雕出的 FREE 多细）；
- **球体素化保守 pad** = `0.5·vs·√3`（`swept.py:79-81`），vs=0.04 时约 **3.5 cm** 过近似。

改法：`voxelize_spheres` 不能再依赖单一 `voxmap.world_to_voxel`（B 类歧义），拆成**按子网格各跑一遍**：
- 球（含半径）触及 fine 盒 → 在 **fine 网格**算相交格、用 fine 的 vs 和 pad、查 `fine.get`；
- 其余 → 走 coarse；
- **保守取并**：任一子网格判非 FREE → 整段判失败。

缝周 pad 从 ~3.5 cm 降到 `0.5·0.01·√3 ≈ 0.87 cm`，FREE 粒度 4 cm→1 cm ⇒ 大量「其实能安全穿过、只因 4 cm 量化擦到相邻格被误杀」的贴缝位姿被救回。**这就是「良率低是判据太粗」的正解。**

### Row 4 — `sensor.carve_observe`：观测两张网格各雕一遍

当前 `_roi_centers(voxmap)`（244 行）一次性生成整张网格所有体素中心，warp kernel 对这些中心做视锥雕刻。改成：
- `_roi_centers` 产**两套**中心：coarse 全体中心（排除 fine 盒内）+ fine 全体中心；缓存 key 仍 `id(voxmap)`。
- kernel 对两套中心**分别**投影 + 雕刻，`near`、`0.5·vs` 按各自网格 vs 取。
- 返回的 `free_idx / occ_idx` 标明属于哪张网格，调用方 `_observe` 按网格写回对应子网格。

**成本**：fine 盒小（0.5 m）但 1 cm ⇒ 50³≈12.5 万格；coarse 覆盖大但 4 cm ⇒ 格数少。相机视锥筛选（`infr` 掩码）后实际参与 warp 的更少，可控。
**坑**：fine 盒边界处，coarse 中心被排除、fine 接管，要保证视锥筛选/雕刻在边界**连续、不留缝、不重复**——实现时专门测。

### Row 5 — `main_loop.sync_collision_world`：STOMP-only 下可整个跳过

这步（`main_loop.py:803`）把 voxmap 灌进 cuRobo 的 **VOXEL** 世界（h_expl）。STOMP 规划不读它（读 Row 2 那张 mesh/cuboid 世界）。两个选择：
- **(a) 推荐**：`if cfg.planner_backend != 'stomp'` 才调 sync。stomp 下整个跳过——省一步，也**彻底不碰**那个单分辨率的 voxel 世界。前提：确认 stomp 下 `h_expl` 除 FK/运动学外无别的碰撞用途（过一遍 `reach_b`、`candidates`、`explain_endpoints` 确认）。
- (b) 保留但只喂 coarse 网格——仅当某些诊断/回退仍依赖 h_expl 占据时才需要。

---

## 5. 改动面汇总

| 文件 | 改动 |
|---|---|
| `gt_gen/voxmap.py` | 加 `MultiResVoxelMap` 壳（A 类透明分派 + B 类留给下面三处） |
| `gt_gen/stomp_iface.py` | `world_from_voxmap_auto` 改「两网格各建 + 挖空 coarse + union」 |
| `gt_gen/swept.py` | `voxelize_spheres` 从「整张网格一遍」改「两子网格各一遍再取并」 |
| `gt_gen/sensor.py` | `_roi_centers` / `carve_observe` 从「整张一遍」改「两子网格各一遍」 |
| `gt_gen/main_loop.py` | 一行 gate 掉 stomp 下的 `sync_collision_world` |
| `configs/default.yaml` + `config.py` | 加 fine 盒的 center / dims / voxel_size_m 参数 |

---

## 6. 待定项（落地前需确认）

1. **确认走 STOMP-only**（`cfg.planner_backend='stomp'`，cuRobo 后端弃用）？
   若 cuRobo 后端仍要保留，那条路径仍受单分辨率约束，多分辨率只能在 stomp 分支生效。
2. **粗网格覆盖多大**：coarse 仍须罩住 retract→焊缝的整段扫掠（否则保守判据把出界当 UNKNOWN→非 FREE，误杀接近路径）。fine 盒只需罩缝周。
   → 待办：统计 retract→seam 扫掠的 base 系包围盒尺寸，据此定 coarse 大小与 fine 盒大小。
3. **mesh 版还是 cuboid 版**做多分辨率 union（`cfg.stomp_params['voxel_world']`）：两级分辨率下哪种拼接更稳。倾向 **cuboid**（合盒对不同分辨率各自合，union 干净），可议。
4. **fine 盒边界连续性**：Row 4 的边界不留缝/不重复、Row 3 的球跨界取并——实现时重点测。

---

## 7. 附：voxel_size 当前来源（现状）

- `roi.voxel_size_m: 0.04`（`configs/default.yaml:74`）是探索 voxmap + cuRobo VOXEL 世界的**单一真源**。
- `config.voxel_size_m`（`config.py:61-62`）读它；`build_roi_voxmap`（`voxmap.py:150`）与 `init_curobo`（`curobo_iface.py:137`）都用它。
- 勿与另几个 `voxel_size_m`（`plan_init_pose.*`=0.01、`plan_init_pose_kejian2.*`=0.01）混淆——那些是**工件 ESDF** 用途，与探索三态图无关。
