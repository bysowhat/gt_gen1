# 反折焊缝（270° 外部二面角）观测位姿求解 —— 诊断与修复

## 一、问题现象

`ObserveAnythingScene` 跑 seam **319**（demo `--task type3`）时，
`compute_goal_pose()` 稳定返回 `None`（无观测位姿解），`finish=0/8`。
可视化上看这条焊缝明明能被看到，理应有解。

## 二、定位过程

在 `scene_pose2.py` 里加**临时诊断**，按代价分量统计逐点通过率（cost≈0），
并把 orientation 的三个子项 tgt / pl / nm 的均分、加权分、通过率单独打印。

三轮诊断结论：

| 代价分量 | 通过率 | 是否瓶颈 |
|---|---|---|
| collision | ~46% | 否 |
| insides（FOV 装下整条缝） | ~22–24% | 次瓶颈 |
| block（遮挡） | ~50–57% | 否 |
| **orientation（朝向）** | **1.4%** | **是** |

orientation 内部再拆（`orientation = 0.3·tgt + 0.2·pl + 0.5·nm`）：

| 子项 | 含义 | 均分 | 通过率 |
|---|---|---|---|
| tgt | 视线与焊缝走向成 30–60° | 4.5 | 53.8% |
| **pl** | 视线在垂直于焊缝的平面内投影，离 bisector 的方位角 | **17.9** | **4.1%** |
| nm | 视线与两面法向各成 45°±15° | 0.0 | 100% |

**真凶是 pl，通过率仅 4.1%。**

### 关键澄清：nm 不是瓶颈

用户一度以为「两个 seam_normal 夹角 270°，nm 不可能满足」。
实则 nm 计算用了 `.abs()`（`scene_pose2.py:634/642`），把法向当作**无向直线**，
所以 270° 与 90° 对 nm 完全等价，nm 一直 100% 通过、从没扣分。
关掉 nm（`use_nm=False`）对 seam 319 毫无帮助，验证了这一点。

## 三、根因：几何 `soll_dist` 只适配「内部观测」，反折焊缝要「外部观测」

pl 代价（`scene_pose2.py:604-616`）：

```
mid_vector = normalize(d1 + d2)                 # bisector，指向观测侧（可视化已确认正确）
dist       = ∠(视线在垂面内的投影, bisector)     # 相机偏离 bisector 的方位角
soll_dist  = ∠(bisector, d2)                    # 几何自动算出的“半锥角”
pl_cost    = clamp(dist - soll_dist, 0)          # 超出锥就扣分
```

因为 `∠(d1,d2)` 经 `arccos` 折叠到 [0,180]，270° 外部二面角与 90° 内部二面角
**数值上无法区分**，都得到 `∠(d1,d2)=90°` ⇒ `soll_dist=45°`。

- `soll_dist=45°` 是**内部观测**（站在 90° 材料楔形那侧看）的半锥角。
- 这条焊缝要的是**外部观测**：像从立方体外部看一条棱，外部自由空间是 **270°**，
  相机可站在以 bisector 为中心的 **270° 弧**内任意位置，
  其半锥角 = 270°/2 = **135°**，而不是 45°。

诊断佐证：能真正看到焊缝的相机 dist **平均 ~100°**，落在 (45°,135°) 这段
**合法但被 45° 锥误杀**的区间里。材料背面那 90° 楔形对应 dist∈[135°,180°]，才是真正该拒的。

> 术语提示：内部二面角 θ ⇒ 内部观测半锥角 = θ/2；外部观测半锥角 = 180° − θ/2。
> 本例 θ=90° ⇒ 外部半锥角 = 180° − 45° = **135°**。

## 四、修复：新增两个开关，用 `pl_limit_deg` 覆盖几何半锥角

在调用链上加了两个参数，默认都**不改变原行为**（向后兼容）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `use_nm` | `True` | `False` → orientation 去掉 nm，只剩 `0.3·tgt + 0.2·pl` |
| `pl_limit_deg` | `None` | `None`=用几何 `soll_dist`（内部观测）；设值=覆盖为固定半锥角 |

透传链路：

```
demo_scene.py  compute_pose_and_plan_path(..., use_nm=, pl_limit_deg=)
   └─ scene.py   compute_pose_and_plan_path  → compute_goal_pose(use_nm=, pl_limit_deg=)
        └─ scene.py   compute_goal_pose  → scene2.use_nm / scene2.pl_limit_deg
             └─ scene_pose2.py  visionOrientation：
                  if pl_limit_deg is not None: soll_dist = full_like(soll_dist, pl_limit_deg)
```

`compute_goal_pose` 也支持从 yaml `compute_goal_pose` 段读默认
（`use_nm`、`pl_limit_deg`），显式传参优先。

### 对 seam 319 的设置

```python
# scripts/demo_scene.py，obstacle_type3_demo_main 内
scene.compute_pose_and_plan_path(hand, max_stomp_try=1, max_init_pose=2,
                                 use_nm=False, pl_limit_deg=135)
```

- `pl_limit_deg=135`：外部观测的物理正解，只拒绝钻进材料背面 90° 楔形的相机。
- `use_nm=False`：nm 对反折焊缝无意义（.abs() 折叠），关掉更干净（可选）。

pl 放宽后：`pl_cost = clamp(dist - 135, 0)`，dist~100° 的合法外部视角不再被扣分，
pl 通过率、orientation 通过率随之大幅上升。

## 五、注意事项 / 后续

1. **135° 很宽松**：它几乎接受除材料背面外的所有方位。若发现相机贴着焊缝面「擦边」
   （dist 接近 135°、视角很掠），可回收到 ~120° 留裕度。
2. **朝向放行后，下一个瓶颈很可能是 `insides`（~22%，最低）**：那是 FOV 装不下整条缝的
   问题，跟朝向无关，需从相机距离 / 焊缝裁剪半径处理，不是本次修复范围。
3. `∠(n1,n2)`、`∠(d1,d2)` 因 `arccos` 折叠，**无法区分凹/凸（90° vs 270°）**。
   若将来要**自动**判定该用内部还是外部半锥角，需引入带方向的判据（如材料侧法向），
   目前靠人工按可视化设 `pl_limit_deg`。
4. **临时诊断代码**（`scene_pose2.py` 的 `_diag*`、`computeCollisionCost` /
   `computeVisionCost_1` 里的累计，`visionOrientation` 多返回的三子项，
   以及 `scene.py` 的分量通过率打印）在定位完成后可按需清理；
   `use_nm` / `pl_limit_deg` 两个参数是**正式功能**，保留。

## 六、相关代码位置

- `stomp_planner/scene_pose2.py`
  - `visionOrientation`（~577）：pl / tgt / nm 三子项；`pl_limit_deg` 覆盖在 ~608；
    `use_nm` 门在 ~652。
  - `__init__`：`self.use_nm`、`self.pl_limit_deg` 默认值。
- `gt_gen/scene.py`
  - `compute_goal_pose`（~1102）：参数、yaml 读取、`_method_keys`、`scene2.*` 赋值。
  - `compute_pose_and_plan_path`（~1278）：透传 `use_nm` / `pl_limit_deg`。
- `scripts/demo_scene.py`
  - `obstacle_type3_demo_main`（~317）：seam 319 的调用点。
