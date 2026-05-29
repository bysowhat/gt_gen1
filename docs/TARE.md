# TARE 算法详解（零基础版）

TARE（**T**raveling salesman based **A**daptive **R**eceding-horizon **E**xploration）是 CMU 的 Cao 等人 2021 年发表的探索规划算法。它的特点是把"该往哪走"建模成**旅行商问题（TSP）** 来解，因此每个周期不是输出"下一步去哪"，而是**输出一整条路线**。

本文从零开始讲，前面已经在 [DSVP.md](DSVP.md) 和 [GBPlanner2.md](GBPlanner2.md) 介绍过的基础概念会快速带过，重点讲 TARE 引入的新概念。

---

## 一、必须先懂的 4 个新概念

> 占据栅格、Raycasting、Frontier、Viewpoint、Information Gain 这 5 个概念已在 [DSVP.md 第一章](DSVP.md#一5-个必须先懂的概念) 详细解释，这里不再重复。

### 概念 1：旅行商问题 (TSP, Traveling Salesman Problem) —— TARE 的核心数学工具

**问题描述**：

> 一个推销员要拜访 N 个城市，每两个城市之间的距离已知。问：从某城市出发、每个城市恰好访问一次、最后回到起点，**总路程最短**的访问顺序是什么？

```text
4 个城市的 TSP 例子（数字是距离）:

       A ─── 5 ─── B
       │  ╲    ╱   │
      10   3  4    7
       │  ╱    ╲   │
       D ─── 8 ─── C

候选 A→B→C→D→A: 5+7+8+10 = 30
候选 A→B→D→C→A: 5+4+8+10 = 27 (差点)
候选 A→C→B→D→A: 3+7+4+10 = 24
候选 A→D→C→B→A: 10+8+7+5 = 30
最短: A→C→B→D→A, 距离 24.
```

TSP 的真正难度在于城市数变多时（比如 100 个城市）枚举所有顺序不可能（100! ≈ 10^158），需要专门的求解器。

**TARE 用 LKH-3 求解器**（Lin-Kernighan-Helsgaun，业界最强 TSP 求解器之一）。对几百个节点，几毫秒到几十毫秒就出最优解。**对你来说当黑盒函数用就行**：

```text
order = LKH3.solve(distance_matrix, start_city)
# distance_matrix[i][j] = i 到 j 的距离
# start_city = 必须从哪个城市开始
# 返回值 order 是城市访问顺序列表
```

### 概念 2：定向越野问题 (Orienteering Problem, OP) —— TSP 的"带奖励"变种

普通 TSP 强制访问**所有**城市。TARE 用的是它的变种：

> 每个城市除了距离，还有一个"奖励值"（比如游客打卡积分）。在**总路程不超过 L** 的约束下，从起点出发，**选一些城市访问**（不必全部），使**总奖励最大**。

```text
5 个景点的 OP 例子（节点上的数字 = 奖励, 边上的数字 = 距离）:

       A(0)──5──B(20)
        │       │
       10      4
        │       │
       D(15)──6──C(30)
                │
               8
                │
               E(50)

约束: 总路程 ≤ 20.
候选 A→B→C→A: 路程=5+4+...回不来; 假设可以
候选 A→B→C→E→C→A: 5+4+8+8+...超 20.
候选 A→D→C→B→A: 路程=10+6+4+5=25, 超.
候选 A→D→C→A: 10+6+...回不来.
候选 A→B→C→A: 路程=5+4+(C到A的距离), 假设 14+5=24 等等.
... 求解器返回最优.
```

**TARE 把"探索"建模成 OP**：

- 城市 = 候选 viewpoint
- 距离 = 两 viewpoint 之间机器人路径长度
- 奖励 = 那个 viewpoint 能扫到的 surface frontier 数量
- 总路程约束 = 一个周期能跑多远（比如 20 m）

求解结果 = "我应该走的最值得的路线"。

### 概念 3：Surface Frontier（表面边界） vs Voxel Frontier（体素边界）

DSVP 和 GBPlanner2 用的是 **voxel frontier**：known free 与 unknown 之间的体素。

```text
体素 frontier 示意（每个 □ 是一个体素）:

  □ □ □ □ ? ? ? ?
  □ □ □ □ ? ? ? ?     F = voxel frontier（free 邻接 unknown）
  □ □ □ F ? ? ? ?
  □ □ □ F ? ? ? ?
  □ □ □ F ? ? ? ?
  □ □ □ □ ? ? ? ?
```

问题：体素 frontier 很多（动辄几万个），TSP 当节点处理不了。

TARE 用 **surface frontier**：从相机扫到的**点云表面**上提取边界点，**不是从体素提取**。

```text
点云示意:
   云  云  云  云  云  云  云
   云  云  云  ●云  云  云  云     ● = 在表面边缘的点（旁边没有更多点）
   云  云  云  云  云  云  云
   云  云  ●  云  云  ●  云
   云  云  云  云  云  云  云

surface frontier = 表面边缘的稀疏点 (●)
```

数量级对比：

| 类型 | 典型数量 |
|---|---|
| Voxel frontier | 几万 |
| Surface frontier | 几百到几千 |

**少 1–2 个数量级 → TSP 可解 → TARE 才能跑**。

### 概念 4：Subspace（子空间） —— 大场景分而治之

地面机器人探索的环境可能是几十米到几百米。一次 TSP 处理几千个 viewpoint 太慢，TARE 把世界切成大块（subspace）做粗粒度规划：

```text
俯视：把世界切成 50m × 50m 的网格

  +---+---+---+---+
  | A | B | C | D |     每个 cell 称为一个 subspace.
  +---+---+---+---+
  | E |[F]| G | H |     [F] = 当前所在 subspace.
  +---+---+---+---+
  | I | J | K | L |
  +---+---+---+---+

  每个 subspace 状态:
    - explored: 内部 frontier 已经清光了吗?
    - centroid: subspace 中心位置
    - frontiers: 该 subspace 内还剩多少 surface frontier
```

机械臂工作空间一共就 1 m³ 左右，**subspace 概念在 arm 上几乎没必要用**——可以直接禁用 global 层。所以本文重点讲 local，global 简略说明。

---

## 二、TARE 要解决什么问题

机器人在开放或半开放环境中探索，要求：

1. 不仅"找到下一个最值得去的点"，还要规划**接下来一段时间的整条路径**——避免抖动、急转弯、来回拉锯。
2. 路径要"扫到尽可能多的 frontier"——覆盖效率最大化。
3. 在大场景下也能跑（几百到几千平方米）。

DSVP 每次只看一步，GBPlanner2 用 Dijkstra 找单条路径——它们都是**点对点最优**。
TARE 不一样，它问：

> "未来 20 米我应该按什么顺序经过哪些 viewpoint，使总收益最大？"

这是**路线规划**问题，所以用 TSP/OP 求解。

---

## 三、TARE 的两层结构

```text
                  地图 + frontier 持续更新
                          │
                          ▼
              ┌───────────────────────┐
              │   Local TSP Layer     │   ← 默认运行
              │  (当前 box 内最优 tour) │
              └─────┬──────────────┬──┘
                    │              │
              当前 subspace 还有 frontier   当前 subspace 探完
                    │              │
                    ▼              ▼
                 沿 tour 走  ┌──────────────────────────┐
                            │   Global TSP Layer       │
                            │  (subspace 间最优巡游)    │
                            └─────────┬────────────────┘
                                      │
                                      ▼
                                到下一个 subspace
                                回 Local Layer
```

两层都是 TSP，只是粒度不同：

- **Local 层**：节点是 viewpoint（精细，~1 m 间距），求 OP 得到一条多 viewpoint 的探索路线。
- **Global 层**：节点是 subspace（粗，几十米间距），求 TSP 得到访问顺序。

---

## 四、Local 层详细步骤

### 步骤 1：定义 Local Volume

以当前位置为中心定义一个盒子（论文中 20 m × 20 m × 5 m）。在 arm 上是 0.5–1.0 m 立方。

### 步骤 2：在 Local Volume 内**密集均匀**采样 viewpoint

不像 DSVP 用 RRT\* 树，TARE 是**栅格 + 微抖动**采样：

```text
俯视:

  V V V V V V V V V V        每隔 1m 放一个 viewpoint, 加 ±0.2m 抖动.
  V V V V V V V V V V        约 100~500 个候选.
  V V V V●V V V V V V        ●=机器人当前位置
  V V V V V V V V V V
  V V V V V V V V V V
```

每个 viewpoint 包含位置 + 一个或多个候选朝向。

### 步骤 3：过滤候选 viewpoint

对每个 viewpoint $v$：

1. **位置必须在已知 free 体素内**（否则机器人去不了）。
2. **不能离障碍太近**（保留 clearance）。
3. **至少能看到 N 个 surface frontier**（否则没用，丢弃）。

剩下的 viewpoint 才是合格候选，比如从 500 个降到 50 个。

### 步骤 4：算每个 viewpoint 的 coverage（即奖励）

对每个 viewpoint，模拟相机视锥扫一遍：

```text
coverage(v) = 这个 viewpoint 视锥内能看到的 surface frontier 数量
```

注意是 **surface frontier**，不是体素 unknown。

### 步骤 5：算 viewpoint 之间的距离（即 OP 的边权）

对任意两个候选 viewpoint $v_i, v_j$，需要知道机器人从 $v_i$ 走到 $v_j$ 要花多少路径长度。

实现方式：在 known free 空间内跑 A\* 或 Dijkstra：

```text
distance[i][j] = AStar(occupancy_grid, v_i.position, v_j.position).path_length
```

得到一个距离矩阵 distance[N][N]（N = 候选 viewpoint 数）。

### 步骤 6：求解 Orienteering Problem

把数据塞给 LKH-3：

```text
输入:
  - N 个 viewpoint
  - 起点 = 机器人当前位置
  - reward[i] = coverage(v_i)
  - distance[i][j] = 上一步算的
  - 路径预算 L_max (比如 20m)

输出:
  tour = [v_start, v_3, v_7, v_2, v_9, ..., v_start]
  total_reward = sum of coverage along tour
  total_length ≤ L_max
```

这条 tour 就是一条**首尾闭合的路线**——我从这里出发、走过一些 viewpoint、最后回到起点。

> 为什么要回到起点？因为 OP 的标准定义是闭合的；TARE 实际只执行 tour 的前一段，回起点的部分自然不会执行。

### 步骤 7：路径增强（TARE 的"adaptive"特色）

直接执行 tour 还不够好，TARE 还做两件事：

#### 7a. 沿路径插入额外 viewpoint

在 tour 路径上，每隔一段距离往侧向偏移采样几个候选 viewpoint，看是否能再扫到更多 frontier。如果能，加进去。

#### 7b. 路径平滑

把折线 tour 用 spline 拟合，避免急转。

### 步骤 8：执行 tour 的前一段

机器人沿 tour 走 5–10 m（不一次走完），然后回步骤 1 重算 tour。

> 这就是"receding horizon"（滚动时域）：每个周期重做一次完整规划，但只执行前面一小段。

---

## 五、Global 层详细步骤（地面机器人才会用上）

### 步骤 1：维护 subspace 网格

世界开始时切成均匀网格。每个 subspace 维护：

```text
subspace[i,j,k]:
  centroid: 中心位置
  state: explored | exploring | unexplored
  frontiers: 该 cell 内剩余 surface frontier 列表
  reachability: 是否和当前位置在已知 free 中连通
```

机器人每移动一次，更新所在 subspace 的状态。

### 步骤 2：触发条件

Local 层连续 N 次找不到值得 tour 的 viewpoint（说明当前 subspace 探完了）→ 切到 Global。

### 步骤 3：识别"需要去的 subspace"

```text
todo = [s for s in subspace_grid 
        if s.state != explored 
        and s.frontiers > 阈值 
        and s.reachable_from_now]
```

### 步骤 4：求解 subspace 间的 TSP

```text
节点 = todo 中的 subspace 中心
起点 = 当前位置
距离矩阵 = 在已知 free 上跑 A* 算两两之间的真实路径长度
求解: 标准 TSP (访问所有, 不带奖励, LKH-3 一样能解)
```

### 步骤 5：导航到 tour 第一个 subspace

走过去（沿 keypose graph 上的已知路径），到达后切回 Local 层。

---

## 六、完整例子

设：地面机器人探索一个仓库，但因为 subspace 在 arm 上不实际，这里用混合例子——用 arm 探索五面开口盒**只跑 Local 层**：

末端起点 $q_0 = (24.4, 0.5, 9.8)$，目标焊缝 $P_{weld} = (24.0, 0.16, 11.4)$。

### Cycle 1（Local Layer）

```text
1. Local Volume = q_0 周围 0.6m 立方
2. 撒 5×5×5 = 125 个候选 viewpoint, 每个 8 朝向 → 1000 个 (位置, 朝向) 候选.
3. 过滤掉 unreachable / 离障碍太近 / coverage < 5 的, 剩 80 个.
4. 算 coverage:
   - 朝盒子开口的 v_a 等候选: coverage=120 (大量 surface frontier)
   - 朝外部空旷的 v_b 等候选: coverage=15
   - 朝侧面被立柱挡死的: coverage=0 (已被过滤)
5. 算距离矩阵 (80×80 矩阵, A* 跑了 80×79/2 次).
6. LKH-3 解 OP, 路径预算 0.5m, 得到 tour:
   q_0 → v_a → v_a' → v_a'' → v_a''' → q_0
   (4 个高 coverage 的 viewpoint 串成的回路)
7. 在 tour 沿路径侧向偏移采样, 多塞进去 1 个高收益点 → 5 viewpoint tour.
8. 执行 tour 前 0.1m, 到 q_0 → v_a 的中间点.
9. 地图更新, 看到了盒子内部一大片新区域, 立柱位置确认.
```

### Cycle 2-N（继续 Local Layer）

```text
每个 cycle 重算 tour. 因为已经看过的 surface frontier 不再算 coverage,
tour 自然在每个周期偏向新区域.

机器人沿"最值得的回路"前段不断推进, 中间会出现"为了串起多个 viewpoint
而绕的轻微弯路"——但总体是高效覆盖.

直到某 cycle, P_weld 进入视野, 视线全 free → 任务完成.
```

### 与 DSVP/GBPlanner2 的可视化对比

```text
DSVP 移动模式:
  q_0 →(贪心一步)→ q_1 →(贪心一步)→ q_2 → ...
  每步独立, 路径可能曲折.

GBPlanner2 移动模式:
  q_0 →(Dijkstra 出最短路)→ q_3 (经过 q_1, q_2)
  路径直, 但每步还是单一目标.

TARE 移动模式:
  q_0 →(沿 tour)→ v_a →(顺路扫一眼侧)→ v_b →(继续 tour)→ v_c → ...
  多目标串成一条路, 一次性扫多个 frontier.
  适合开阔环境, 类似邮递员设计的最优投递路线.
```

---

## 七、超精简伪代码

```python
def TARE_LOCAL(q_start, P_target, octomap, surface_frontiers):
    while not target_visible(q_start, P_target, octomap):
        # ===== Step 1-3: 采样 + 过滤 =====
        V_local = bounding_box(q_start, size=(0.6, 0.6, 0.6))
        candidates = uniform_sample_with_jitter(V_local, spacing=0.1)
        candidates = [v for v in candidates 
                      if in_free_space(v, octomap) 
                      and clearance(v, octomap) > 0.05
                      and coverage(v, surface_frontiers) > MIN_COV]
        
        # ===== Step 4: 计算 reward =====
        for v in candidates:
            v.reward = coverage(v, surface_frontiers)
        
        # ===== Step 5: 计算距离矩阵 =====
        N = len(candidates)
        dist = numpy.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = astar(candidates[i].pos, candidates[j].pos, octomap)
                dist[i, j] = dist[j, i] = d
        
        # ===== Step 6: 解 OP =====
        tour = LKH3.solve_OP(
            nodes=candidates,
            start=q_start,
            distance=dist,
            rewards=[v.reward for v in candidates],
            budget=L_MAX
        )
        
        # ===== Step 7: 路径增强 + 平滑 =====
        tour = insert_side_viewpoints(tour, surface_frontiers, octomap)
        tour = spline_smooth(tour)
        
        # ===== Step 8: 执行前段, 滚动 =====
        execute(tour[0:STEP_LENGTH])
        q_start = tour[STEP_LENGTH]
        update_octomap_from_camera(q_start)
        update_surface_frontiers(surface_frontiers, octomap)
```

---

## 八、对机械臂场景的关键改动

1. **关节空间 vs 笛卡尔空间**：候选 viewpoint 在末端笛卡尔空间撒，但每个候选必须 IK 可解 + 整条手臂碰撞通过。这一步过滤会刷掉大部分候选。
2. **距离矩阵改用 cuRobo trajectory cost**：A\* 在末端 3D 空间不准确（因为机械臂的可达性不是简单的笛卡尔距离），用 cuRobo 算两个构型间真实的轨迹长度。
3. **Local Volume 0.5–0.8 m 立方**：超过 UR12 可达范围（~0.85 m）就没意义。
4. **N_SAMPLES 不要太大**：候选 ≤ 50，否则距离矩阵 50×50 = 2500 次 cuRobo 调用太慢。可以两层过滤：先笛卡尔 distance 粗筛，再做精确 trajectory cost。
5. **Global 层完全去掉**：arm 工作空间小，不需要 subspace 划分。
6. **加目标偏置**：`reward(v) = coverage(v) + α · max(0, viewable(P_weld, v))`，看到目标的 viewpoint 给大加成。

---

## 九、超参清单

| 超参 | 含义 | 典型值 (arm) | 调大效果 | 调小效果 |
|------|------|--------------|----------|----------|
| `local_box_size` | Local Volume 边长 | 0.6 m | 候选多、距离矩阵慢 | 探索局部 |
| `viewpoint_spacing` | 撒点栅格间距 | 0.1 m | 候选稀、tour 粗 | 候选密、计算重 |
| `MIN_COV` | 候选最低 coverage 门槛 | 5 | 过滤激进、候选少 | 候选多、TSP 慢 |
| `L_MAX` | OP 路径预算 | 0.5–1.0 m | tour 长、覆盖广 | tour 短、保守 |
| `STEP_LENGTH` | 每周期执行多长 | 0.1 m | 周期少、地图过期风险 | 周期多、计算多 |
| `α` (目标偏置) | viewable($P_{weld}$) 加权 | 50–200 | 直奔目标 | 自由覆盖 |

---

## 十、与 DSVP / GBPlanner2 的对比

| 维度 | DSVP | GBPlanner2 | TARE |
|------|------|------------|------|
| 数据结构 | RRT\* 树 + keypose chain | 全局图（带环） | viewpoint 网格 + subspace 网格 |
| 求解形式 | 贪心 max gain | Dijkstra 单目标 | OP/TSP 多目标 tour |
| 输出 | 一条边 | 一条最短路径 | 一条 tour（多 viewpoint 串行） |
| 求解器 | 自实现 | 自实现 | LKH-3 (TSP) |
| 大场景能力 | 弱 | 中 | 强（subspace 分层） |
| 实现工作量 | 1 周 | 2–3 周 | 3–4 周 |
| 调参敏感度 | 中 | 中 | 高（OP budget 很关键） |
| arm 适用度 | 高 | 高（窄空间最稳） | 中（局部够用，subspace 没用） |
| 最适合的场景 | 快速 baseline | 半封闭工件、管腔 | 开阔大空间、需高覆盖率 |

**给焊缝场景的建议**：

- 如果你的工件大致**开阔、机械臂需要绕到工件多个面探索**：TARE 的 tour 思路有用，因为它会自动设计"顺路扫多个面"的路线。
- 如果工件是**窄盒/管腔/有立柱**：选 GBPlanner2，TARE 的优势用不上。
- 如果只是**baseline 起步**：选 DSVP，TARE 实现成本是它的 3 倍。

---

## 十一、参考资料

- 原论文：C. Cao, H. Zhu, H. Choset, J. Zhang, *TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments*, RSS 2021.
- 仓库：<https://github.com/caochao39/tare_planner>
- TSP 求解器 LKH-3：<http://webhotel4.ruc.dk/~keld/research/LKH-3/>
- 同组前作 DSVP：<https://github.com/HongbiaoZ/dsv_planner>
