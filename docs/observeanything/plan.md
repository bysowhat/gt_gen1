# 在 USD 环境物体表面上找焊缝直线并计算焊缝信息 — 方案(修订)

## 目标
从一个工厂 USD 环境(如 `full_warehouse.usd`)里，自动找出**所有贴在物体表面、由两面相交形成的直线棱线**(候选焊缝)，
并对每条计算与 `ifc_analyzer` 一致的**焊缝信息**。
不再输入指定长度，改为**输入最短/最长焊缝长度 [Lmin, Lmax] 用于过滤**。

## 焊缝信息(与 ifc_analyzer 对齐，语义源自 `/home/a/Projects/xiaoyu/ifc_analyzer/step3_vis.py`)
对每条焊缝直线段(边方向 `edge_dir`、中点 `edge_mid`)输出：

- **p0, p1**：焊缝起点/终点(世界系，单位 m)。ifc 里来自 `match.json` 预匹配段；本场景无 match，
  **起终点即我们检测到的特征棱线直段两端**。
- **boundary_dirs `[d0, d1]`**：边两侧面各一个方向。`d = normalize(cross(face_normal, edge_dir))`，
  翻向使其从 `edge_mid` 指向该面质心 —— 沿面、垂直于焊缝、朝面内的单位向量。
- **boundary_normals `[n0, n1]`**：两侧面的法向量。
- **bisector**：`normalize(d0 + d1)`，两 boundary_dir 的角平分线，指向张开的楔形(外部空间)，单位向量。
- **gap_deg**：两 boundary_dir 间的楔形张角(度)。`>180°` 侧为反射角，丢弃。
- **is_inside / inside_ratio**：楔形内撒 25 点(5 角向×5 径向, ≤1cm)，`igl.fast_winding_number`
  判断是否在实体内。`ratio>0.5` 记为 inside。真焊缝坡口应朝外(OUT)。

鲁棒性(照搬 ifc)：每个面方向同时探测 `±d`，用 `probe_on_face`(沿方向 0–1cm 采样、到网格 <1mm 计数
≥ 阈值)只保留真正贴面的方向，解决三角形绕序造成的法向/方向朝向歧义。

## 与 ifc_analyzer 的差异(关键)
- ifc 输入是**单个已分割、近水密的工件 OBJ** + `match.json`(告诉它哪些边是焊缝)。
- 本任务输入是**整个 USD 场景**，没有 match：候选焊缝边要**我们自己检测**(特征棱线)，
  且 winding-number 的"实体内外"判定要**逐 prim**(每个物体自身)做。
- warehouse 网格常**非水密**，`fast_winding_number` 的内外判定会不稳 → 见确认点 1。

## 处理流程
1. **加载与提取**：纯 `pxr` 打开 USD，`Stage.Traverse()` 遍历所有 `UsdGeom.Mesh`；
   `ComputeLocalToWorldTransform` 把 points 变到世界系(m)；多边形扇形三角化 →
   每个 prim 一个 `trimesh.Trimesh`，记录 prim 路径。**不起 SimulationApp**。
2. **抽特征棱线**：逐 prim 用 `trimesh.face_adjacency` + `face_adjacency_angles`
   取相邻面二面角偏差 ≥ `θ_min` 的边作为候选焊缝边。
3. **接链成直段**：把首尾相连、切向共线(角度容差内)的特征边连成折线，
   在方向拐折处断开 → 得到一条条**直线段**。每段两端即 p0/p1。
4. **长度过滤**：只保留 `Lmin ≤ 段长 ≤ Lmax` 的直段。
5. **角度过滤**：候选边二面角偏差须落在 `[θmin, θmax]`(命令行传入)。
6. **算焊缝信息**：对每条直段(取其中点作 edge_mid、整段方向作 edge_dir)照搬 ifc 流程：
   `get_face_in_plane_dirs` → `probe_on_face` 筛 ±d → `sort_dirs_by_angle` →
   `classify_regions`(`fast_winding_number`，见第 7 步的 winding 网格) → 取朝外楔形的
   bisector / gap_deg / boundary_dirs / boundary_normals / inside_ratio。
   **只保留朝外(OUT)楔形**，inside 的直接丢弃。
7. **winding 网格(全场景合并，可分 N 组)**：
   - 默认 `N=1`：把全场景所有 prim 网格合并成**一个大 mesh**，所有焊缝的内外都对它算
     ("inside = 落在场景任一物体内部")。
   - `N>1`：按 prim 质心聚成 N 组(整 prim 不切开)，每组合并成一个 mesh；
     每条焊缝用**其所属 prim 那一组**的 mesh 算 winding。用于场景过大时控显存/内存。
   - 坑：分组后，跨组边界附近的焊缝可能漏掉邻组物体的贡献 → N 尽量小、按空间聚类以让相邻物体同组。
8. **去重/控密**：近重复段按端点距离聚类去除。
9. **输出**：仅 `json`。每条含 `p0,p1,length,prim_path,edge_dir,bisector,gap_deg,`
   `boundary_dirs,boundary_normals,is_inside,inside_ratio`。世界系、单位 m。

## 环境
- 仅用 `pxr` + `trimesh` + `igl`；虚拟环境 `env_isaaclab`(均已具备，igl 已验证)。

## 最终参数(已确认)
命令行输入：
- `--usd <path>`：USD 场景路径。
- `--min <m> --max <m>`：焊缝直段长度过滤区间(米)。短于 `--min` 的段丢弃。
- `--angle-min <deg> --angle-max <deg>`：候选边二面角偏差过滤区间。
- `--nwind <N>`：winding 分组数，默认 1(全场景一个大 mesh)。
- `--out <path.json>`：输出 json。

固定策略(已定)：
1. **只保留朝外(OUT)** 焊缝(严格，与 ifc 一致)。
2. 二面角**上下限均由参数控制**(`--angle-min/--angle-max`)。
3. 接链共线容差 **5°**；最短段 = `--min`。
4. winding 用**全场景合并 mesh**，过大时分 **N** 组(`--nwind`)。
5. 长度 `--min/--max` 命令行传入(m)。
6. **只存 json**，不出 npy、不出可视化图。
