# traj_sampler 接入 Scene 轨迹 —— 方案

> 目标：让 `traj_sampler`（关键帧下采样）消费 `scripts/demo_scene.py` 产出的新 Scene pkl 轨迹。
> 本文档为**待确认方案**，确认后再动代码。
>
> **2026-07-20 用户已拍板 7 条决策（见 §3），只剩 1 项待确认（见 §5）。**

---

## 1. 现状梳理（已读代码确认）

### 1.1 老 GT 生成全链路（三级）

```
stomp_planner/main_controller_sp_pose.py   →  output/<工件>/pose/seam_N.pkl   （候选观测位姿）
stomp_planner/main_controller_sp_traj.py   →  output/<工件>/traj/seam_N.pkl   （STOMP 稠密关节轨迹）
traj_sampler/main_controller_sp.py         →  就地给 traj/seam_N.pkl 追加 actions/valid（关键帧下采样）
```

- `traj_sampler` 是**最后一级**：不生成轨迹，只对已有稠密轨迹做**关键帧稀疏采样**（SE(3) 运动度量 + 图搜索）。
- 当前 `traj_sampler/main_controller_sp.py` 只做一件事：多 GPU 并行调 `down_sampling.py`，输入 `data_root=/media/a/新加卷/tempt/5/output`，遍历每个工件文件夹的 `traj/` 子目录。

### 1.2 `traj/seam_N.pkl` 老格式（源：`stomp_planner/optimize_traj.py:203-211`）

> ⚠️ 本方案**不再产出此格式**（决策 D1）。此表仅为理解老 `down_sampling` 逻辑保留。

`list[dict]`，每个 dict（一条 STOMP 批量轨迹）：

| 字段 | 形状 | 含义 |
|------|------|------|
| `trajectory` | (B, D=6, M) | B=num_batch 条候选轨迹，6 自由度，M 时间步 |
| `idx_3d` | (M,) int | **每个 STOMP 段的最后一帧=1，其余=0**（到达观测位姿即"拍 3D"的标记） |
| `robot_pose` / `piece_pose_original` / `robot_pose_original` | (7,) | 机器人/工件位姿（下游训练用） |
| `seam_line` | (20,3) | 焊缝折线 |
| `seam_limits` | (20,2,3) | 焊缝限界 |
| `actions` | (B',L',D+1) | ← `down_sampling` 追加：采样出的关键帧（关节角+idx_3d） |
| `valid` | (B',) | ← `down_sampling` 追加：批内有效轨迹下标 |

### 1.3 下采样核心（`traj_sampler/down_sampling.py` + `geometry_sampling/`）

`down_sampling.run()` 对每个 dict：
1. `trajectory` (B,D,M) → permute (B,M,D)；`idx_3d` expand 成 (B,M)。
2. `UR12e_t(B*M).forward_cam_pose(...)` —— **用写死的 DH + 相机外参**做正运动学，把关节角 → 相机 4×4 位姿 (B,M,4,4)。
3. `build_action_matrix` → SE(3) 累计距离矩阵 M_action (B,M,M)。
4. `get_segmented_keyframes` → 图搜索 + **跨批次统计筛选** → `actions`、`valid`。
5. 只读 `trajectory`/`idx_3d`，其余字段原样保留后写回。

> ⚠️ **写死的相机外参 + DH**：`geometry_sampling/kinematics.py:10-11` 硬编码 `camera_to_j6link_pos/rot`，
> `:17-24` 硬编码 DH 臂长。这必须与产生关节角的机器人一致，否则相机位姿算错、SE(3) 距离错、采样点错。
> 记忆 `camera_single_source_config` 记录过写死旧相机导致的 bug（详见 §4.3、§5）。

### 1.4 新 Scene pkl 格式（源：`gt_gen/scene.py` save/merge_trajectory_entries；已反序列化确认）

`Scene.save()` 存的顶层 state dict，其中 `trajectories = {seam_id: {(hand,index): [entry,...]}}`。
一条成功轨迹 `entry`（`status=="reached"`）：

| 字段 | 形状/类型 | 含义 |
|------|------|------|
| `positions` | (T, 6) float64 | 关节角序列（**6 DOF，与 UR12e_t 一致**）——**采样只依据此列**（D2） |
| `observe` | (T,1) int64 {0,1} | 该帧是否被相机拍过（"拍 3D"）——**保留为输出列 + 强制纳入**（D3/D4） |
| `goal` | (T,1) int64 {0,1} | 该帧是否为某段到达目标——**保留为输出列 + 强制纳入**（D3/D4） |
| `status` | str | "reached" 为成功 |
| `goal_joints`/`cur_joints` | (6,) | 末/起关节角 |
| `goal_index`/`variant`/`segments`/`info` | 杂项 | 元信息 |

- 遍历成功轨迹：`Scene.load(pkl).summarize_trajectories()` 返回 `items=[(seam_id,hand,index),...]`。
- 示例 `..._type2_seam0.pkl`：seam 0 有 2 条成功轨迹（forehand / backhand），各自 `positions` 长度不同（如 301）。
- **老 `idx_3d`（每段末=1）语义上相当于新数据的 `goal`**；本方案把它拆成 `observe` + `goal` 两列分别保留（D3）。

### 1.5 无效代码（保留，不清理 —— D7）

README 主推的是**另一套**基于 RGB-D 图像+共视率的采样（需 `info.npy`/深度图），当前链路完全没用到，但**按决策保留不动**：
- `traj_sampler/mapanything_sampling/`、`test_segmented_sampling.py`、`test_*.py`
- `traj_sampler/samplers/`、`generate_pairs.py`、`core/`、`utils/`、`traj_visualize.py`、`inspect_info.py`、`check_mapanything_interface.py`
- 一堆 `configs/exp_*.yaml`（只 `geometry_sampling.yaml` 在用）

---

## 2. 核心难点：批次语义不匹配

| | 老（STOMP） | 新（Scene/curobo） |
|--|--|--|
| 一个 seam 下轨迹数 | B=num_batch（如 24）条**等长同时间线**变体 | 每 (seam,hand) **1 条**独立轨迹，长度各异 |
| 段标记 | 全批共享一条 `idx_3d` (M,) | 每条轨迹各自的 `observe`/`goal` (T,) |
| `get_segmented_keyframes` 的 `valid`/`res_ratio` 筛选 | 靠**跨 B 变体统计**（`count_sort[4:-4]` 需 B>8）剔除离群/失败变体 | curobo 轨迹已是验证过的 GT，无需批内互筛；且 B=1 时该统计会 NaN/报错 |

**结论**：不能把新轨迹硬塞进 (B,D,M) 批张量。应**逐条轨迹处理**（B=1），并改用一个不依赖跨批统计的关键帧提取（见 §4.2、附录 A.4）。

---

## 3. 已定决策（2026-07-20 用户确认）

- **D1 不兼容旧训练格式**：输出**不再**沿用 `output/<工件>/traj/seam_N.pkl` 的目录布局与字段；下游训练代码由用户自行修改，采样器只管产出新格式。
- **D2 采样只依据 positions**：关键帧图搜索**只用关节角 positions**（→FK→SE(3)→图搜索）挑帧，**不再传入 `idx_3d`**。
- **D3 observe / goal 保留为两列**：老 `idx_3d`（每段末=1）语义相当于新数据的 `goal`；现在拆成 `observe`、`goal` 两个独立通道，均保留到输出。
- **D4 强制纳入 observe/goal 帧**：图搜索挑完关键帧后，若某帧 `observe==1` 或 `goal==1` 恰好被采样丢弃，则**补回该帧**（关键帧集合与这些帧取并集）。
- **D5 输出 (L,8)**：每条成功轨迹产出一个 (L,8) 数组，L=最终关键帧数；列 = **[q1..q6（6 关节角）, observe, goal]**。
- **D6 相机外参 + 机器人 DH 全取自 config，不写死**：作废 `kinematics.py:10-11` 写死外参、`:17-24` 写死 DH（机械臂已换）。相机外参读 `sensor.camera` 的 `extrinsic_pos`/`extrinsic_quat_wxyz`/`mount_link`；**DH/机器人本体走 `robot.cfg_path`**（见下方注 + §4.3）。
- **D7 无效代码保留**：不清理 §1.5 列出的无关代码。
- **D8 输出另存到指定目录（不覆盖原 pkl）**：采样结果加进 Scene 后**另存**到用户给定的输出目录 `--out-dir`，**保留原文件名**（`<out_dir>/<原pkl文件名>`），原输入 pkl 不动。新增 `Scene.sampled_trajectories`（与 `self.trajectories` 平行），存进整份 state 后 `scene.save(<out_dir>/<原名>)`，仍可 `Scene.load` 加载。结构镜像 `trajectories`：`{seam_id: {(hand,index): [每条 entry 的 (L,8) 或 None]}}`，与 `trajectories[sid][(hand,index)]` 的 entry 列表 1:1 对齐（`reached` 的存 (L,8)，其余 None）。

> 注（D6 落地）：`default.yaml` **未直接列 DH 数字**，只给 `robot.cfg_path`（curobo `ur12e_full.yml`）。故"DH 从 config 读" = **用 `robot.cfg_path` 的机器人模型算 FK**（加载 curobo 运动学取相机 link 位姿，或从该 yml 解析连杆），与规划器同源，外参+臂长均无写死值。

---

## 4. 拟定方案（据上述决策）

### 4.1 新增脚本 `scripts/traj_downsample.py`

对单个 Scene pkl：
1. `Scene.load(pkl)` → `summarize_trajectories()` 取所有 `status=="reached"` 的成功轨迹。
2. 逐 (seam_id, hand, index) 取 `entry`，拿 `positions` (T,6)、`observe` (T,)、`goal` (T,)。
3. 调用 `sample_keyframes_single(positions, observe, goal, ...)`（见 §4.2）→ 得 (L,8) 的 `actions`。
4. 写回 `scene.sampled_trajectories[seam_id][(hand,index)]`（镜像 `trajectories` 结构，与其 entry 列表 1:1 对齐；非 `reached` 置 None）。
5. `scene.save(<out_dir>/<原pkl文件名>)` **另存到指定输出目录**（D8），保留原文件名、不覆盖输入 pkl；仍可 `Scene.load` 加载。

> 说明：`Scene.save/load` 是显式 state dict，故 `sampled_trajectories` 需在 `gt_gen/scene.py` 三处登记（`__init__` 初始化 / `save()` 存进 state / `load()` 读回，见 §6）。
> 本步须走完整 `Scene.load`（因要 `scene.save` 回写整份 state），env_isaaclab 环境即可，不必 GPU。

### 4.2 单轨迹关键帧提取 `sample_keyframes_single()`

输入单条轨迹（B=1），流程：

1. **FK**：`positions` (T,6) 经正运动学 → 相机位姿 (T,4,4)。相机外参/机器人模型取自 config（§4.3）。
2. **SE(3) 距离**：复用 `build_action_matrix` → M_action (1,T,T)。
3. **图搜索挑帧（只用 positions）**：复用 `get_segmented_keyframes` 的**段内图搜索**
   （`_search_path_in_segment` + 50 倍数固定节点 + 阈值内合并），得关键帧下标集合 `S`。
   - **移除**依赖 B>8 的 `count_sort[4:-4]` / `valid` 跨批剔除逻辑（附录 A.4）；B=1 时直接返回该条路径。
   - 对很短轨迹加保护：`T<50` 时固定节点只用首尾。
4. **强制纳入 observe/goal 帧（D4）**：`S = sort( S ∪ {t | observe[t]==1} ∪ {t | goal[t]==1} )`。
5. **拼输出 (L,8)（D5）**：`actions = concat( positions[S] (L,6), observe[S] (L,1), goal[S] (L,1) )`。

要点：
- **不传 `idx_3d`**（D2）——采样只看 positions 的运动量；observe/goal 只在第 4/5 步作为"必留帧 + 输出列"参与。
- 不改动原 `get_segmented_keyframes` / `_update_paths_with_nodes`（避免影响其它可能的旧数据消费者），新函数独立实现或以 `B=1` 分支复用其可靠部件。

### 4.3 FK 相机位姿：真源与一致性（全取自 config，D6）

- **相机外参**：改读 `configs/default.yaml → sensor.camera`（`extrinsic_pos`、`extrinsic_quat_wxyz`(w,x,y,z)、`mount_link=Link6`）。作废 `kinematics.py:10-11` 写死值。
  - 实测差异（写死 vs config）：平移 (-0.076,-0.035,0.162) → (0.050,0.100,0.148)，差约 **0.19 m**；朝向 quat 完全不同。→ 不改必错。
  - 注意 quat 顺序：config 明确 (w,x,y,z)；接线时须与 `math_traj.matrix_from_quat` 约定对齐。
- **机器人本体/DH**：作废 `kinematics.py:17-24` 写死 DH。改用 `robot.cfg_path`（curobo `ur12e_full.yml`）的机器人模型算 FK：
  - 首选：加载 curobo `CudaRobotModel`，直接取相机所在 link 的位姿（与规划器同源，最稳；需 curobo+GPU，env_isaaclab 已具备）。
  - 备选：从 `ur12e_full.yml` / URDF 解析连杆参数喂给现有 DH-FK（若想保持 CPU 轻量）。
  - 二者皆无写死值；具体走哪条按实现便利定，输出等价。

### 4.4 输出：另存到指定目录，写进 `sampled_trajectories`（D8）

- **不覆盖原 pkl**，把采样结果加进 Scene 后**另存**到用户给定目录 `--out-dir`，保留原文件名（`<out_dir>/<原pkl文件名>`）。`Scene.load` 仍可加载。
- 新增 `Scene.sampled_trajectories`，与 `self.trajectories` 平行、结构镜像：
  `{seam_id: {(hand,index): [每条 entry 的 actions]}}`，与 `trajectories[sid][(hand,index)]` 的 entry 列表 **1:1 对齐**——
  `reached` 的 entry 存其 (L,8) 数组，其余存 `None`（保持下标对应）。
- 每个 (L,8)：列 = [q1..q6, observe, goal]（D5）。
- 需改 `gt_gen/scene.py` 三处让新字段随 save/load 往返（§6）。

### 4.5 调度 `main_controller_sp.py`

- 扫描输入目录下的 Scene pkl（如 `/media/a/新加卷/tempt/5/*.pkl`）→ 逐个跑 `scripts/traj_downsample.py`（`load`→采样→写 `sampled_trajectories`→`save` 另存到 `--out-dir`）。
- 命令行参数：`--in-dir`（输入 pkl 目录）、`--out-dir`（输出目录，另存、不覆盖原文件）。
- 数据量不大时单进程顺序即可；保留多进程能力（不必强绑 8×8 GPU 调度）。
- **不再**调用老 `down_sampling`、不产老格式（D1）。

---

## 5. 待确认事项

决策已全部落定（§3 D1–D8）。文档确认无误即可开工，无遗留待选项。

---

## 6. 交付物（确认后）

1. `scripts/traj_downsample.py`：`Scene.load` → 逐条 (L,8) actions（[q1..q6, observe, goal]）→ 写 `sampled_trajectories` → `save` 另存到 `--out-dir`（不覆盖原 pkl）。
2. `traj_sampler/geometry_sampling/sampling.py`：新增 `sample_keyframes_single()`（不改原函数）。
3. `traj_sampler/geometry_sampling/kinematics.py`：相机外参 + DH/FK 全改读 config（`sensor.camera` + `robot.cfg_path`），删写死值。
4. `gt_gen/scene.py`：新增 `sampled_trajectories` 字段——`__init__`(约392) 初始化、`save()`(约989) 存进 state dict、`load()`(约1039) 读回。
5. 改造后的 `traj_sampler/main_controller_sp.py`：以 Scene pkl 为输入的调度（load→采样→save 回写；不产老格式）。
6. 一次端到端小样跑通验证（用 `/media/a/新加卷/tempt/5/*.pkl`）：跑完后 `Scene.load` 确认 `sampled_trajectories` 正确、(L,8) 列义正确、observe/goal 帧确被纳入。

---

## 附录 A：关键帧稀疏采样原理详解（SE(3) 运动度量 + 图搜索）

> 面向无背景读者，从零讲清 §1.3「关键帧稀疏采样」到底做什么、为什么这么做。

### A.0 要解决的问题

规划器产出的轨迹**非常稠密**：一条可能 301 帧，每帧是机械臂 6 个关节的角度：

```
帧0:   [ 0.10, -1.20,  1.50, -0.30,  1.57,  0.00 ]
帧1:   [ 0.101, -1.201, 1.501, ... ]   ← 跟上一帧几乎一样
帧2:   [ 0.102, -1.202, 1.502, ... ]
...
帧300: [ 0.90, -0.50,  0.80, ... ]
```

相邻帧差别极小，**大部分帧是冗余的**。目标：从 301 帧里挑出约 20 帧「关键帧」，连起来仍能代表整条轨迹的运动。

**关键帧稀疏采样 = 从稠密轨迹里抽稀，只留代表性的帧。**
难点在于「凭什么说哪帧重要、哪帧能扔」——需要一把「衡量两帧之间动了多少」的尺子。

### A.1 第一步：正运动学（关节角 → 相机位姿）

轨迹存的是**关节角**，但「动了多少」我们关心的是**末端相机在空间里移动/旋转了多少**，不是关节转了几度。

同样关节动 5°：
- 靠近底座的大臂关节动 5° → 相机可能在空间划过 **30 cm**
- 手腕末端关节动 5° → 相机可能只挪 **1 cm**

所以不能直接比关节角，要先用**正运动学（FK）**把每帧 6 个关节角算成**相机的 4×4 位姿矩阵**（位置 xyz + 朝向）：

```
帧0 关节角 --FK--> 相机位姿 T0   帧1 --FK--> T1  ...
```

对应代码 `UR12e_t.forward_cam_pose`，依赖 DH 参数（臂长）+ 相机外参（相机装末端何处）。
→ **这就是 §4.3 / §5 强调外参与机器人模型必须对齐的原因**：外参或 DH 错则相机位置错，后续全错。
（注：本方案里这一步**只吃 positions**，不涉及 observe/goal——决策 D2。）

### A.2 第二步：SE(3) 运动度量（衡量两帧动了多少的尺子）

**SE(3)** = 三维刚体位姿 = 位置(3) + 姿态/朝向(旋转)，一个 4×4 矩阵即一个 SE(3) 位姿。

「SE(3) 运动度量」衡量从位姿 A 到 B 一共动了多少（配置 `w_trans=1.0, w_rot=0.1`）：

```
distance(A, B) = w_trans × ‖平移‖ + w_rot × |旋转角|
```

例：相机平移 0.2 m、转 30°(≈0.52 rad) → `1.0×0.2 + 0.1×0.52 = 0.252`。
权重含义：**更看重相机挪多远，不太看重转多少**。

`build_action_matrix` 据此算出**累计距离矩阵** `M_action`：

```
M_action[i][j] = 从第 i 帧走到第 j 帧，累计运动量

        帧0    帧10   帧20   帧30
帧0  [   0    0.05   0.11   0.18 ]   ← M_action[0][20]=0.11：帧0→帧20 累计动了 0.11
帧10 [   -      0    0.06   0.13 ]
```

### A.3 第三步：图搜索（用尺子挑关键帧）

**核心规则：让相邻关键帧之间的运动量都 ≈ 目标值 `D_target`（=0.1）。**
即每挑一帧就是相机又动了约 0.1 的地方——运动快处挑得密，运动慢处挑得稀，每帧携带信息量相当。

链式跳转（贪心）：

```
从 帧0：查表找 M_action[0][?]≈0.1  → 帧23    ✓ 选中
从 帧23：查表找 M_action[23][?]≈0.1 → 帧58   ✓ 选中（这段运动快，只走35帧）
从 帧58：查表找 M_action[58][?]≈0.1 → 帧140  ✓ 选中（这段几乎没动，走82帧才凑够0.1）
...
→ 关键帧序列 [0, 23, 58, 140, ...]（空间等间距，而非时间等间距）
```

代码对应：
- `next_idx = argmin(|M_action - D_target|)` = 「找累计运动量最接近 0.1 的那帧」。
- `_search_path_in_segment` = 「不断跳到 next_idx 直到走完」的链式跳转。

**硬性节点（50 的倍数）**：纯按运动量挑，若某段长时间几乎不动会一口气跳过几百帧、漏掉转折点。
故强制在 帧 0、50、100、150… 各放一个关键帧当保底锚点（`nodes`）。
`_update_paths_with_nodes` 再清理：贪心帧若离锚点太近（运动量 < `D_target/3`）就合并掉，避免锚点旁挤一堆。

**（本方案新增）强制纳入 observe/goal 帧**：图搜索挑完后，再把 `observe==1`、`goal==1` 的帧并进关键帧集合
（决策 D4）——这些是"拍 3D"/"到达目标"的语义关键帧，无论运动量多少都不能丢。

### A.4 与本方案的关系（为什么 B=1 要新写函数）

§2「跨批次统计」就在 `_update_paths_with_nodes` 末尾：

```python
count_sort = torch.sort(count)[0]
mean_count = torch.mean(count_sort[4:-4].float())   # 砍掉最高4/最低4再平均
valid = (count > mean_count * (1 - res_ratio))      # 保留关键帧数够多的轨迹
```

- 老数据一条 seam 有 **B=24 条等长变体**，这段拿 24 条**互相比**，剔除关键帧数偏少的离群/失败变体；`count_sort[4:-4]` 砍头尾各 4，**要求 B>8**。
- 新 Scene 数据一条 (seam,hand) **只有 1 条**（B=1）：`count_sort[4:-4]` 对长度 1 切片 = 空 → 求平均得 **NaN** → 崩；且 curobo 轨迹已是验证过的 GT，**无需互比剔除**。

→ 故 §4.2：**新写 `sample_keyframes_single()`，复用 FK + SE(3) 距离 + 贪心图搜索（对 B=1 全部有效），仅移除末尾「跨批剔除」**，B=1 时直接返回挑好的关键帧，再并入 observe/goal 帧、拼成 (L,8)。

### A.5 一句话总结

> 先把每帧关节角经 FK 换成相机空间位姿（**为何 SE(3)**），再用「平移+旋转」加权算任意两帧间运动量（**运动度量**），
> 然后从头每凑够 0.1 运动量选一帧、并每 50 帧保底选一帧（**图搜索**），最后补上所有 observe/goal 关键帧，
> 把 301 帧稠密轨迹压成一个 (L,8) 的关键帧数组（列 = 6 关节角 + observe + goal）。
</content>
</invoke>
