# find_surface_welds.py 改造方案：曲线焊缝 + bisector 强制朝外

针对 `scripts/find_surface_welds.py` 的两项改造：

1. **焊缝不限于直线**——支持曲线、圆弧、闭合圆环等。
2. **焊缝信息(bisector/gap_deg/boundary_*)一律朝外**——例如立方体每条凸边都是焊缝，其张角应为
   270°（bisector 指向立方体外部的空气侧），而不是内部实体的 90°。

> 本文只描述方案，不含实现。下游消费者：`scripts/viz_surface_welds_isaacsim.py`、
> `scripts/viz_weld_json.py`（消费 `corrected_p0/p1`、`bisector`、`boundary_dirs`）。 

---

## 一、决策（已与用户确认）

| 决策点 | 结论 |
| --- | --- |
| 曲线焊缝几何 | 输出 polyline 顶点数组；焊缝信息 **逐点数组** 输出（bisector/gap_deg/boundary_* 沿线逐顶点变化） |
| 接链断开条件 | **仅硬拐角断开**：相邻特征边转角 > 阈值时断开；平滑弯曲（圆弧）保持为一条链。阈值做成 `--chain-break-deg` 参数 |
| 闭合环 | 标记 `closed=true`，**不设端点**，几何为环，length 为周长 |

---

## 二、诉求 2：bisector 强制朝外（逻辑独立、优先落地）

### 2.1 根因

`enumerate_wedges()` 中有一行：

```python
if gap > np.pi:      # 反射角侧，跳过
    continue
```

- 立方体一条凸边：两侧面各朝面内的方向 `d0`/`d1`，把 360° 分成两个互补扇形——
  材料内侧 **90°** 与外部空气 **270°**。
- 现在只枚举 90° 扇形 → winding 判定为 inside → 被 `ratio > 0.5` 丢弃
  → **凸边完全产不出焊缝**；而用户要的恰恰是被跳过的外部 270° 扇形。
- 且 `bisector = (d0+d1)/‖d0+d1‖` 永远指向小角（内侧）平分线，对外部扇形方向相反。

### 2.2 改法：楔形按「绕边角度」参数化，允许反射角

重写 `enumerate_wedges()`：

1. **不再跳过反射角**。排序后相邻方向对 `(i, j)` 的张角 `gap` 允许 > 180°。
2. **角度参数化采样**代替 `d0→d1` 线性插值：
   - 取绕边正交基 `(u, w)`，`u = sdirs[0]`，`w = normalize(cross(edge_dir, u))`。
   - 采样方向 `dir(θ) = cosθ·u + sinθ·w`，θ 从 `angle_i` 扫到 `angle_j`（反射角走大弧）。
   - 这样反射角扇形的 25 个 winding 采样点落在**正确的那一侧**（原线性插值只能覆盖 ≤180° 扇形）。
3. `bisector` = 中间角度 `(angle_i + angle_j)/2` 对应的 `dir(θ_mid)`——天然指向该扇形内部；
   外部扇形即指向外。
4. `gap_deg = degrees(gap)`，可 > 180（立方体凸边 → 270）。
5. `boundary_dirs = [d0, d1]`、`boundary_normals = [n_i, n_j]` 仍是扇形两条边界（含义不变）。

winding 判定（Pass B/C 中 `ratio > 0.5 → inside → 丢弃`）**保持不变**：枚举出内外两个互补扇形后，
自动只留下 winding≈0 的外部扇形。

### 2.3 预期结果

- 立方体凸边：输出 bisector 朝外、`gap_deg ≈ 270`。
- L 形 / T 形内角：照常得到 < 180° 的朝外楔形。
- 下游 `viz_*` **无需改动**（字段语义不变，只是数值方向修正）。

---

## 二之补、winding 判据失效根因 + 双法向点积替换（**已落地，取代上文 winding**）

> 上文 2.2/2.3 及后文 3.3 里"用 `igl.fast_winding_number` 判内外、`ratio>0.5` 丢弃"的做法，
> 在本项目的钣金场景**整体失效**。实际实现已把 winding 判据整段删除，换成「双法向点积」判据。
> 本节记录失效根因与替换方案，视为对 2.2/3.3 的**修订**。

### 2.4 现象

跑 `warehouse.usdz` 全扫（1252 条焊缝），发现 **78.8%（987/1252）** 的焊缝 bisector 指向**材料内侧**
（bisector 与两侧面法向点积均 < 0），13.5% 朝外，7.7% 异号。可视化时 w00999 的 bisector 明显朝里。

### 2.5 复合根因

**① winding 判据在薄壳场景整体失效**

- 场景全是薄板开壳，`watertight=False`。`fast_winding_number` 对开放薄壳**两侧都 ≈ 0**，
  `wn>0.5` 永远不成立 → 一条边的两个互补楔形（朝里 90° + 朝外 270°）**都被判"朝外"**保留。
  证据：`crop≤1m` 时两楔形径向均值 `[0.5, 0.06, 0.24, ...]` / `[0.5, 0, 0.08, ...]`，
  除了 `r=0` 落在棱线上的边界值 0.5，其余全 ≈ 0。
- `crop_radius=3.5` 更糟：圈进 381 prim / 36.8 万面，探针被周围结构包住，wn 被压成**负值**
  （−0.5 ~ −0.68），`wn>0.5` 照样全 false。缩小半径只是把"负值"变"≈0"，两楔形依旧都保留。

**② `filter_similar_welds` 把同边两反向楔形当重复删掉**

- 它只比 `length / edge_dir / 中点`，**不看 bisector**。同一条边的两楔形这三项完全相同
  → 必被判"相似" → 只留枚举顺序里第一个 = 锐角 90° = **朝里那个**。
  所以①放行两楔形后，②又偏偏保留了朝里的，最终产出大面积朝内 bisector。

### 2.6 改法：用面法向判朝外，不用 winding

诊断显示 `winding_consistent=True`（法向一致且朝材料外），用「bisector 与两侧面法向的点积」判定极干净：

| 楔形 | dot(b, n0) | dot(b, n1) |
| --- | --- | --- |
| 朝里 90°（旧逻辑误留） | −0.707 | −0.707 |
| 朝外 270°（应留） | +0.707 | +0.707 |

**判据：楔形朝外 ⟺ `dot(bisector, n0) > 0` 且 `dot(bisector, n1) > 0`。**

- 对薄壳 / 非水密照样成立，直接替换 `wn>0.5`。
- 每条边只留真正朝外的楔形后，②的 dedup 副作用**自动消失**（同边只剩一个朝外楔形）。
- 剩 7.7% 点积异号的（非正交 / 退化折角）用「**点积之和最大**」兜底。

### 2.7 实现落点（`find_surface_welds.py`）

- 新增 `_wedge_outward_score(bisector, boundary_normals)` → `(is_outward, score)`：
  `is_outward = 所有点积 > 1e-3`，`score = 点积之和`。
- `find_welds` **删掉整个 winding 链路**：Pass B 邻域拼接、`_neighborhood_mesh`、
  `igl.fast_winding_number`、per-prim AABB、`_slice` 采样记账全部移除（顺带大幅提速）。
  Pass A 收集楔形后 `w.pop("Q")`（缠绕采样点不再需要）。
- 新 Pass B：端点空间哈希去重（不变）→ 对每条边所有楔形算双法向点积，
  保留 `is_out=True`；若全非朝外则兜底取 `score` 最大者。
- JSON 字段：`inside_ratio` → `normal_score`（下游无消费者读 `inside_ratio`，安全）；
  `is_inside` 恒 `False`；其余 `bisector/corrected_p0/p1/edge_dir/boundary_dirs/boundary_normals/gap_deg` 不变。
- `--crop-radius` / `--crop-watertight` 标 `[已弃用]`、不再生效（签名保留避免调用方报错）。
- `--debug-weld` 改为打印每个楔形的两点积、score 与朝外判定（不再是缠绕数）。

### 2.8 遗留注意

- **weld 数量可能略增**：旧逻辑靠 winding 丢弃"完全内埋"的边，现无此筛除——
  每条通过夹角 + 长度 + 贴面探测的特征边至少产出 1 条朝外焊缝。若发现内埋边混入需另加遮挡判定。
- **依赖法向朝外**：判据成立前提是面法向朝材料外（本场景 `winding_consistent=True` 已满足）。
  换场景若出现整片朝里的 prim，需 `fix_normals` 或靠兜底。

---

## 三、诉求 1：支持曲线 / 圆焊缝

### 3.1 根因

`build_straight_chains()` 只把**首尾相连且共线(< `COLLINEAR_TOL_DEG`=5°)** 的特征边接成直段。
圆弧 / 曲线焊缝会被切成一堆碎直段，再被 `filter_similar_welds` 误删或散落。

### 3.2 改法：共线接链 → 平滑连续接链

新增 `build_smooth_chains(vertices, edge_vids, break_deg)`（替换或与 `build_straight_chains` 并列）：

- 贪心接链逻辑保留（`v2e` 邻接、`used` 标记、双向 `extend`）。
- **接续判据从「共线」改为「平滑」**：相邻边方向夹角 < `break_deg`（默认约 45°，`--chain-break-deg`）
  即可接续；> 阈值视为硬拐角，断开。
- **分叉点断开**：某顶点连接 ≥ 3 条特征边（如立方体顶点三边交汇）时在该点断链，避免歧义把不同焊缝接成一条。
- **闭合检测**：延伸回到起点顶点时，标记 `closed=True` 并停止（不重复首顶点）。
- 链以 polyline 顶点序列 `vids` 表示，可为直线、曲线或闭合环。

### 3.3 焊缝信息：逐顶点计算

焊缝量原来在「链中点代表边」上算一次单值。曲线上朝向沿线变化，改为**逐顶点/逐边**计算：

- 对 polyline 每条内部边（或每个内部顶点邻域），沿用现有逐边流程：
  `two_face_dirs → probe_on_face_batch → sort_dirs_by_angle → enumerate_wedges(角度参数化) → 双法向点积选外部`。
  （原文写「winding 选外部」，**已按 2.6 改为双法向点积判据**；下面 winding 采样 / `_slice` 记账一段作废。）
- 每个采样点产出一组 `{bisector, gap_deg, boundary_dirs, boundary_normals, normal_score}`，
  组装成沿线数组。
- ~~winding 采样点仍统一攒进所属 winding 组做批量 `igl.fast_winding_number`~~
  （**已删除**，见 2.5 失效根因 / 2.7 实现落点）。

### 3.4 长度、去重、过滤的适配

- `length`：polyline **累积弧长**（闭合环为周长），替换直线 `‖p1-p0‖`。
- `filter_similar_welds`：现依赖 `length + edge_dir`（单一方向）。曲线无单一方向，改为
  按 `length + 首尾质心/AABB` 或 polyline 采样点集合近似比较；闭合环单独按周长+质心判重。
  （细节实现时定，语义仍是「同 prim 内近似重复只留先出现者」。）
- `edge_dir`：曲线无单一 `edge_dir`，输出改为逐顶点切向数组 `edge_dirs`；直线退化为单元素/两端一致。

---

## 四、输出 JSON 结构（新）

每条焊缝：

```jsonc
{
  "prim_path": "...",
  "closed": false,                 // 新增：是否闭合环
  "polyline": [[x,y,z], ...],      // 新增：世界系(米)顶点序列；闭合环不重复首点
  "corrected_p0": [x,y,z],         // 兼容：非闭合链首点；闭合环省略或置 null
  "corrected_p1": [x,y,z],         // 兼容：非闭合链尾点；闭合环省略或置 null
  "length": 0.0,                   // 弧长/周长(米)
  "samples": [                     // 新增：逐顶点焊缝信息（朝向沿线变化）
    {
      "point":            [x,y,z],
      "edge_dir":         [x,y,z], // 该点切向
      "bisector":         [x,y,z], // 朝外
      "gap_deg":          270.0,   // 可 >180
      "inside_ratio":     0.0,
      "is_inside":        false,
      "boundary_dirs":    [[...],[...]],
      "boundary_normals": [[...],[...]]
    }
  ],
  // 顶层兼容字段（下游取代表值时用；取链中点样本）：
  "bisector":         [x,y,z],
  "gap_deg":          270.0,
  "boundary_dirs":    [[...],[...]],
  "boundary_normals": [[...],[...]]
}
```

> **下游兼容**：`viz_surface_welds_isaacsim.py` / `viz_weld_json.py` 仍读 `corrected_p0/p1` 与顶层
> `bisector/boundary_dirs`，直线焊缝行为不变；曲线焊缝可后续升级 viz 读取 `polyline` + `samples`。
> 顶层单值取「链中点样本」作代表，保证旧 viz 不崩。

---

## 五、CLI 变更

新增参数：

```
--chain-break-deg  FLOAT  相邻特征边转角 > 此值(度)则断链，默认 ~45（越大越容易把曲线连成一条）
```

保留：`--min/--max`（弧长过滤）、`--angle-min/--angle-max`（二面角偏差）、`--nwind`、`--out`。

---

## 六、改动清单（按落地顺序）

1. **`enumerate_wedges()`**：角度参数化 + 允许反射角 + bisector 用中角方向（诉求 2，独立可先验证）。
2. **`build_smooth_chains()`**：平滑接链、硬拐角/分叉点断开、闭合检测（替代/并列 `build_straight_chains`）。
3. **`find_welds()` 主循环**：逐顶点算焊缝信息、`_slice` 记账扩展为逐采样点、组装 `polyline+samples+兼容字段`。
4. **`filter_similar_welds()`**：适配 polyline（弧长 + 质心/点集近似判重，闭合环单列）。
5. **`main()` / argparse**：加 `--chain-break-deg`；docstring 更新。
6. **回归**：先用立方体 USD 验证凸边 → 12 条边、每条 `gap_deg≈270`、bisector 朝外；再用含圆弧/圆环的 USD 验证 polyline 与 closed。

---

## 七、验证要点

- **立方体**：12 条边各成一条焊缝，`gap_deg≈270`，bisector 指向立方体外部（远离体心）。
- **L 形板内角**：朝外楔形 < 180°，bisector 指向凹口外的空气侧。
- **圆环 / 圆弧焊缝**：输出单条 `closed=true` 的 polyline，`length≈周长`，`samples` 沿环逐点，
  bisector 逐点朝外。
- **下游 viz 不崩**：直线焊缝的 `corrected_p0/p1` + 顶层 `bisector` 与改动前一致（仅方向修正）。
