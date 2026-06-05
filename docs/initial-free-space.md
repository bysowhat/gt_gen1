# 初始引导 FREE 空间（initial free space）

## 动机

相机装在机械臂末端(Link6)，**初始视野极小**。保守探索从 `retract`（固定 home 构型）起步时，
voxmap 全 UNKNOWN。Step6/Step7 的判据是"整臂扫掠体积 ⊆ FREE 才允许这一步运动"
（`gt_gen/swept.py::motion_stays_in_free`、`gt_gen/reach_b.py::compute_reach_pt`）。
全 UNKNOWN 时整臂扫掠体积里必然含 UNKNOWN 体素 → 任何一步都判不通过 → `reach_pt = 0`，
**机械臂第一步就动不了**，相机也就无法转动去观测更多空间，探索死锁。

实测（122 条 `tmp/plan_seam` 轨迹）：全 UNKNOWN 下 `reach_idx` 全为 0，证实死锁。

因此需要一个**初始引导 FREE 空间**：在 retract 邻域预先标一小块"已知自由"活动空间，
让机械臂能起步、小幅旋转/移动各关节（带动末端相机扫视），逐步观测把可行区往外扩。

## 空间定义

函数 `gt_gen/init_free.py::set_initial_free_space`：

- 以 `retract` 为中心，对**每个关节**做 **±dq** 单关节扰动，得到 `2 × DOF` 个邻域构型；
- 对每个 `retract → 扰动构型` 段做**整臂扫掠**（`gt_gen/swept.py::swept_volume`，碰撞球细插值体素化）；
- 所有段扫掠体素**并集去重**后，在 voxmap 上置 `FREE`。

直观上这就是"机械臂站在 retract 原地、把每个关节小幅来回摆动一下，整条臂扫过的那团体积"。
它**只依赖 retract 与 dq，与具体工件/场景无关**，故 Step7 自测与正式 GT 生成可共用同一份定义。
只用 handle 的运动学做 FK，**不需要 MESH 碰撞世界**。

### 为什么这样标 FREE 是安全的
`retract` 是固定安全 home 构型——机械臂能停在这里，说明其本体邻域本就无障碍。
工件焊缝目标都在远处（靠移动臂去够），不会侵入 retract 的小邻域。标定脚本提供
`--check-workpiece` 抽样核查 blob 是否贴近任一工件表面以佐证这一点。

## 参数标定（dq 取多少）

`dq`（各关节活动半幅，rad）决定空间大小：太小起步不了，太大违反"尽量小"且有误标风险。
用 `scripts/calibrate_init_free.py` 扫多个 dq，对每条已规划轨迹 P* 跑 `compute_reach_pt`，
统计起步成功率（reach_idx≥1）、能走多远、blob 体素数。

实测结果（122 条 `tmp/plan_seam` 轨迹，voxel_size=0.04m）：

| dq (rad) | blob FREE 体素 | reach_idx min/med/max | EE 行进 med/max (m) | reach≥1 占比 |
|---|---|---|---|---|
| 基线（全 UNKNOWN） | 0 | **0 / 0 / 0** | — | 0% |
| 0.05 | 1534 | 4 / 7 / 14 | 0.013 / 0.038 | **100%** |
| 0.10 | 1934 | 4 / 7 / 19 | 0.017 / 0.052 | **100%** |
| 0.15 | 2337 | 4 / 7 / 20 | 0.019 / 0.052 | **100%** |
| 0.20 | 2757 | 4 / 8 / 24 | 0.022 / 0.052 | **100%** |
| 0.30 | 3601 | 4 / 9 / 24 | 0.027 / 0.070 | **100%** |

**结论：**
- 即便最小 `dq=0.05rad` 也让 **122/122** 条轨迹起步（reach_idx 最小 4、中位 7）。
  （早期轨迹运动很小——P[1]=P[0]、前几路点 <0.01rad——故小邻域即覆盖前若干段。）
- reach_idx 随 dq 增大收益递减：blob 主要作用是**让机械臂起步**，不是沿某条 P* 走远；
  起步后由传感观测继续往外扩可行区。
- **默认取 `dq = 0.10rad`**：blob≈1900 体素（≈0.12 m³），在 0.05 最小值上留一档余量，
  既保证 100% 起步、给相机留出可观测的小幅转动量，又仍然很小。

配置项：`configs/default.yaml`
```yaml
init_free:
  dq_rad: 0.10
```
读取：`gt_gen/config.py::Config.init_free_dq`。

## 调用方式

运行时（GT 生成起步、Step7 自测）：
```python
from gt_gen.voxmap import build_roi_voxmap
from gt_gen.init_free import set_initial_free_space

vm = build_roi_voxmap(cfg)                       # 全 UNKNOWN
set_initial_free_space(handle, vm, config=cfg)   # retract 邻域标 FREE，机械臂可起步
```

- Step7 自测：`scripts/verify_step7_my.py::verify_compute_reach_pt`（方式2）断言
  全 UNKNOWN → reach_idx=0、加初始 FREE 空间 → reach_idx≥1。
- 复核/重标定：换机器人、改 retract、或 `tmp/plan_seam` 轨迹集变化后，重跑
  `scripts/calibrate_init_free.py` 确认 dq 是否仍合适。

## 相关文件
- `gt_gen/init_free.py` — `set_initial_free_space`（运行时函数）
- `scripts/calibrate_init_free.py` — 标定/复核脚本
- `configs/default.yaml` `init_free` 段、`gt_gen/config.py::init_free_dq`
- `gt_gen/reach_b.py::compute_reach_pt`、`gt_gen/swept.py::swept_volume`、`gt_gen/voxmap.py`
