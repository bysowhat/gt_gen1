# 缝周多分辨率 voxmap 方案（细化焊缝周围体素、其余区域保持粗）

> 状态：**设计已定稿，待落地代码**。§6 四项待定已拍板（STOMP-only 删 cuRobo 规划支路 / coarse=整 ROI + fine=焊缝 AABB 外扩 10 cm / cuboid 版删 mesh / 边界连续性列为落地必测）。本文记录动机、可行性分析、约束、逐模块改法。
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

### 1.1 关键：GT 正确性闸门 `motion_stays_in_free` 只吃 voxmap

这一节要讲透一件事——**决定「一条轨迹能不能算合格 GT」的那道关卡，数据全部来自 voxmap 这张三态图，完全不碰 cuRobo 的碰撞世界**。这正是「细化 voxmap ⇒ 良率提升」这条因果链的落点。

**逐词拆解**（对照 `swept.py:180` 的 `motion_stays_in_free(handle, voxmap, q_from, q_to)`）：

- **`motion_stays_in_free` 是什么**：判断一段运动 `q_from → q_to` 的整臂扫掠体积是否**全部落在 FREE 体素内**。返回 `(ok, n_nonfree)`：`ok=True` 表示整臂扫掠 ⊆ FREE。
- **「跑在 voxmap 上」是什么意思**：看它的实现（`swept.py:189-196`），从头到尾只出现 `voxmap`：
  1. `cells = swept_volume(handle, voxmap, q_from, q_to)` —— 算出整臂扫过哪些体素下标；
  2. `states = voxmap.get(cells)` —— 直接查 **voxmap** 每个格是什么状态；
  3. `n_nonfree = (states != FREE).sum()` —— 数有几个格不是 FREE。
  全程没有 `h_expl`、没有 `sync_collision_world`、没有 cuRobo 的任何调用。所以「跑在 voxmap 上」＝**它的输入和判据全在 voxmap 这张网格里，cuRobo 那张网格改不改都影响不到它**。

- **「GT 正确性闸门」是什么意思**：「闸门（gate）」＝一道**放行/拦截**的关卡。它是决定「这条规划出来的轨迹能不能被采纳为一条合格 GT」的最后一关：
  1. STOMP（或 cuRobo）规划出一条候选轨迹 `q_from → q_to`；
  2. `motion_stays_in_free` 把整条臂沿这条轨迹**扫掠**一遍（碰撞球体素化，`swept_volume`）；
  3. 判断这片扫掠体积是不是全部落在 FREE 格里：
     - 全 FREE → `ok=True` → **放行**，这段运动被采纳为 GT；
     - 只要有一个格是 OCCUPIED 或 UNKNOWN → `ok=False` → **拦截**，这条候选被丢弃。
  之所以叫「**正确性**闸门」：GT 的定义是「**只在已确认自由的空间里运动**」（保守探索）。UNKNOWN 也算不合格（`swept.py:185` 注释：越界/未知按非 FREE 计入），因为「没看过 ＝ 不敢保证安全」。这道闸门保证产出的 GT 名副其实。

**为什么这件事是整份方案的关键**——它把「细化 voxmap ⇒ 良率提升」这条因果链钉死在一个点上：

- 决定良率（多少候选被放行）的，**就是这道闸门**；
- 这道闸门**只读 voxmap**；
- 所以**只要把 voxmap 在焊缝周围变精（4 cm→1 cm），闸门的判据就变精**，完全不需要动 cuRobo。

精度体现在两处（对应 §4 Row 3，此处先给结论，Row 3 再展开）：
1. **FREE 空间的量化粒度**：carve 用 1 cm 雕，能雕出 4 cm 雕不出的细自由通道；
2. **扫掠占据的「整格代偿」**：`voxelize_spheres`（`swept.py:70`）本身是**几何精确**的球-体素相交判据（`swept.py:90-91`，用 `half=0.5·vs`），但占据是**以整格为单位**记的——球擦到某格一个角，这**整格**就算被扫掠占用。格越大代偿越猛：vs=0.04 时量级 ≈`0.5·vs·√3 ≈ 3.5 cm`，vs=0.01 时降到 ≈`0.87 cm`。
   （注意：`swept.py:81` 的 `pad = 0.5·vs·√3` **不是**这份代偿的来源——它只用来框候选 AABB 保证不漏，精确判据随后会把多框的剔掉；真正的过度膨胀来自「整格判定」这一步。详见 Row 3。）

**后果**：现在很多「其实臂能安全穿过、只因 4 cm 量化把相邻格误判成非 FREE」的贴缝位姿被闸门误杀 → 良率低。voxmap 缝周细化后，这批位姿被救回。这就是「良率低是因为判据太粗，而判据只跟 voxmap 有关」的完整论证，也是这份方案值得做的根本理由。

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
- **fine（细网格）**：`vs=0.01`，覆盖**整条焊缝的包围盒（AABB）再各方向外扩 10 cm**（不是「焊缝中点 ±10 cm」——焊缝是一段线，按中点取盒会把两端露在 coarse 里仍被 4 cm 误杀）。焊缝长 30 cm 时盒约 `0.5 m×0.2 m×0.2 m`，随缝长变；成本很低（0.2 m 边 @1 cm 仅 20³≈8000 格/维）。
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

### Row 2 — `stomp_iface.world_from_voxmap_auto`：两张 cuboid 世界 union 成一个 STOMP world

**已定：只做 cuboid 版，删 mesh 版。** 当前 `world_from_voxmap`（mesh 版，152 行）连同 `world_from_voxmap_auto` 里的 mesh 分支、`cfg.stomp_params['voxel_world']` 这个 mesh/cuboid 开关、以及 mesh 相关调试全部删除；只留 `world_from_voxmap_cuboid`（224 行）。选 cuboid 的理由：合盒对 coarse/fine 各自合完再 union，拼接干净；mesh 版两套 marching-cubes 面在 fine 盒边界处可能穿插。

`world_from_voxmap_cuboid` 当前读一张 voxmap 的 `non_free_mask` + 单一 `vs` 产一批 cuboid。改成：

1. **fine 子网格** → 照原样跑 → 一批 **1 cm** 的细 cuboid（焊缝周围精细障碍）。
2. **coarse 子网格** → 先把 fine 盒内的格子从 mask 剔掉（§3「挖空」），再跑 → 一批 **4 cm** 的粗 cuboid，fine 盒处留空。
3. **合并**：`WorldConfig(cuboid = coarse_cuboids + fine_cuboids)`。**cuRobo 一个 world 放多个 cuboid 合法**——这是 STOMP-only 能做多分辨率的根本。

**不膨胀（inflate=0，已落地）**：膨胀单位是「体素层数」，多分辨率下同「膨 1 层」在粗/细网格物理厚度不同（4 cm vs 1 cm），语义不一致；且细化已把 GT 闸门的 `√3/2·vs` 过近似缝从 ≈3.5 cm 收窄到 ≈0.87 cm，膨胀需求随之消失。`config.voxel_inflate_voxels` 已加 assert 强制 =0（读到非 0 即报错），`world_from_voxmap_cuboid` 里的 `binary_dilation` 分支成为死代码，可随删除一并清掉。

### Row 3 — `swept.motion_stays_in_free`：GT 正确性闸门变精（收益最大）

本行是整份方案收益最大的一处，因为 §1.1 已论证：**良率由这道闸门决定，而闸门只读 voxmap**。这里把「voxmap 变精 ⇒ 闸门变精 ⇒ 良率升」的机理讲到可落地。

**调用链**：`motion_stays_in_free`（`swept.py:180`）→ `swept_volume`（`swept.py:107`）→ `voxelize_spheres(voxmap, spheres)`（`swept.py:70`）。整臂沿 `q_from→q_to` 细插值出多个路点，每个路点 FK 出整臂碰撞球 `(M,4)=xyz+r`，把这些球体素化成一批体素下标 `cells`，最后 `voxmap.get(cells)` 查是否全 FREE。

**精度由两个量决定，都随 vs 线性变**：

1. **FREE 空间的量化粒度**。carve（Row 4）用多大的 vs 去雕，FREE 就有多细。vs=0.04 时，一条宽度 3 cm 的真实自由缝隙可能连一个完整 FREE 格都放不下（格宽 4 cm），于是这条缝隙在 voxmap 里根本不存在、整段是非 FREE；vs=0.01 时同一条缝隙能雕出 2~3 个 FREE 格，臂就能「看见」并被判可穿行。

2. **球体素化的保守 pad**。见 `voxelize_spheres`（`swept.py:79-81`）：
   - `half = 0.5·vs`：体素半边长，用于**精确**的球-AABB 相交判据（`swept.py:90-91`，逐轴 clamp 后 `∑d² ≤ r²`）。这一步本身不保守，是精确的。
   - `pad = 0.5·vs·√3`：**半个体素的体对角线长度**（边长 vs 的立方体，体对角线 = `vs·√3`，半条即 `0.5·vs·√3`）。它**只用来框候选 AABB**（`swept.py:84-85`，`R = r + pad`），保证「可能相交的体素」这个候选集是**超集、不漏**，再由上面的精确判据剔掉「只擦到角外」的格。
   - 那么保守性从哪来？——**FREE/非 FREE 是以整格为单位判定的**。一个球哪怕只碰到某格的一个角，只要精确判据通过，这**整格**就算被扫掠占用；这格若不是 FREE，整段就判失败。格越大（vs 越大），「一个角被擦到 ⇒ 整格 4 cm 被计为占用」的过度膨胀就越严重。vs=0.04 时，这种「整格代偿」量级 ≈`0.5·vs·√3 ≈ 3.5 cm`；vs=0.01 时降到 `0.5·0.01·√3 ≈ 0.87 cm`。

**改法**：`voxelize_spheres` 不能再依赖单一 `voxmap.world_to_voxel` / `voxmap.voxel_size`（Row 1 归为 B 类歧义方法），拆成**按子网格各跑一遍**：
- 球（含半径 r）的影响范围触及 fine 盒 → 在 **fine 网格**上算候选 AABB、用 fine 的 `vs` 和 `pad`、精确判据后查 `fine.get`；
- 其余的球 → 走 **coarse 网格**同一套流程；
- **保守取并**：任一子网格判出非 FREE → 整段 `ok=False`。这条保证「拆两张网格」不会比「单张网格」更宽松，只会更严或持平——安全性不降级。

**收益举例**：一条贴着焊缝、离工件 2 cm 的候选臂姿。vs=0.04 时，3.5 cm 的整格代偿让臂的碰撞球「擦」到工件那一侧的相邻非 FREE 格，闸门判失败、候选被丢——**但臂其实并没碰到工件**，是 4 cm 量化误杀。vs=0.01 后代偿降到 0.87 cm，同一姿态的球不再擦到非 FREE 格，候选被放行。缝周成百上千个这类贴缝位姿由此被救回。这就是「良率低不是规划器不行、而是 GT 判据在缝周太粗」这一诊断的正解，也是把细化范围**精准放在焊缝周围**（而非全局细化、徒增显存）的依据。

### Row 4 — `sensor.carve_observe`：观测两张网格各雕一遍

当前 `_roi_centers(voxmap)`（244 行）一次性生成整张网格所有体素中心，warp kernel 对这些中心做视锥雕刻。改成：
- `_roi_centers` 产**两套**中心：coarse 全体中心（排除 fine 盒内）+ fine 全体中心；缓存 key 仍 `id(voxmap)`。
- kernel 对两套中心**分别**投影 + 雕刻，`near`、`0.5·vs` 按各自网格 vs 取。
- 返回的 `free_idx / occ_idx` 标明属于哪张网格，调用方 `_observe` 按网格写回对应子网格。

**成本**：fine 盒小（0.5 m）但 1 cm ⇒ 50³≈12.5 万格；coarse 覆盖大但 4 cm ⇒ 格数少。相机视锥筛选（`infr` 掩码）后实际参与 warp 的更少，可控。
**坑**：fine 盒边界处，coarse 中心被排除、fine 接管，要保证视锥筛选/雕刻在边界**连续、不留缝、不重复**——实现时专门测。

### Row 5 — STOMP-only：删除 cuRobo 规划后端 + VOXEL 碰撞世界一整套（已定）

**已定走 STOMP-only，cuRobo 不再当规划后端**，故不止是「gate 掉 sync」，而是直接删除下面这些「只服务 cuRobo VOXEL 规划」的支路。**注意删除边界**——STOMP 全程踩在 cuRobo 上（吃 cuRobo `WorldConfig`、内部用 cuRobo IKSolver、FK/运动学走 `curobo_iface`），cuRobo 的运动学地基一行都不能删。

**可删（不走的 cuRobo 规划路径）：**
- `main_loop.py:807` 里 `planner_backend != 'stomp'` 的 else 分支（在 VOXEL 世界上直接 `ci.plan_to_config`）；
- VOXEL 碰撞世界整套：`sync_collision_world`（`main_loop.py:803`）、`collision_sync.py`、`curobo_occupied_centers`、调试 `_debug_viz_curobo`（`main_loop.py:188`）；
- `voxel_inflate_voxels`（只服务 VOXEL 世界膨胀，已 assert=0，连同 `binary_dilation` 死分支一并删）。

**不可删（STOMP 的运行地基）：**
- `curobo_iface` 的运动学/FK/机器人碰撞球（`swept.py` FK 出整臂碰撞球靠它，GT 闸门离不开）；
- `WorldConfig / Cuboid / CollisionCheckerType`（建 STOMP 世界必需）；
- cuRobo IKSolver（STOMP 内部解 IK）；
- `h_expl` handle 本身（相机 FK、`explain_endpoints` 诊断仍用）。

---

## 5. 改动面汇总

| 文件 | 改动 |
|---|---|
| `gt_gen/voxmap.py` | 加 `MultiResVoxelMap` 壳（A 类透明分派 + B 类留给下面三处） |
| `gt_gen/stomp_iface.py` | 只留 cuboid 版：`world_from_voxmap_auto` 改「两网格各建 + 挖空 coarse + union」；**删 mesh 版** `world_from_voxmap` + mesh 分支 + inflate 死分支 |
| `gt_gen/swept.py` | `voxelize_spheres` 从「整张网格一遍」改「两子网格各一遍再取并」（球跨界取并） |
| `gt_gen/sensor.py` | `_roi_centers` / `carve_observe` 从「整张一遍」改「两子网格各一遍」 |
| `gt_gen/main_loop.py` | **删** stomp 下不走的 cuRobo 规划支路：else 规划分支 + `sync_collision_world` + `_debug_viz_curobo` |
| `gt_gen/collision_sync.py` | **整文件删**（只服务 VOXEL 世界） |
| `gt_gen/curobo_iface.py` | **保留**（FK/运动学/IK/WorldConfig 是 STOMP 地基）；只删 VOXEL 世界专用件（如 `curobo_occupied_centers`） |
| `configs/default.yaml` + `config.py` | 加 fine 盒的 center / dims / voxel_size_m 参数；删 `voxel_inflate_voxels` |

---

## 6. 待定项（落地前需确认）

1. ~~**确认走 STOMP-only**~~ → **已定**：走 STOMP-only，cuRobo 不再当规划后端，按 Row 5 删除 VOXEL 世界一整套（注意保留 cuRobo 运动学地基）。
2. ~~**粗网格覆盖多大 / fine 盒多大**~~ → **已定**：coarse 仍覆盖整个 ROI（罩住 retract→焊缝整段扫掠）；fine 盒 = **整条焊缝 AABB 各方向外扩 10 cm**（随缝长变，非固定立方）。
3. ~~**mesh 版还是 cuboid 版**~~ → **已定**：**cuboid 版**做多分辨率 union，删 mesh 版。
4. **fine 盒边界连续性（唯一遗留，落地时重点测）**：这不是设计选择题，是「实现别忘了测的坑」，两处——
   - **Row 4 观测端**：fine 盒边界那半格（coarse 中心在盒外、体积探进盒内）谁雕？规则要保证**不留缝、不重复**。
   - **Row 3 判据端「球跨界取并」**：一个碰撞球圆心在盒外、球体探进 fine 盒时，**不能按圆心二选一**——fine 盒内的 coarse 格已挖空，只在 coarse 上算会漏掉探进盒里那段。必须**两张网格都体素化、取并**（任一非 FREE 即整体失败）。
   - → 造「正好压在盒面上的球 / 正好跨边界的臂姿」写单测验证。

---

## 7. 附：voxel_size 当前来源（现状）

- `roi.voxel_size_m: 0.04`（`configs/default.yaml:74`）是探索 voxmap + cuRobo VOXEL 世界的**单一真源**。
- `config.voxel_size_m`（`config.py:61-62`）读它；`build_roi_voxmap`（`voxmap.py:150`）与 `init_curobo`（`curobo_iface.py:137`）都用它。
- 勿与另几个 `voxel_size_m`（`plan_init_pose.*`=0.01、`plan_init_pose_kejian2.*`=0.01）混淆——那些是**工件 ESDF** 用途，与探索三态图无关。
