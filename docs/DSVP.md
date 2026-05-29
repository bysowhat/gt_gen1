# DSVP 算法详解（零基础版）

DSVP（Dual-Stage Viewpoint Planner）是一种用于"未知空间在线探索 + 朝目标推进"的规划算法。本文从零开始解释，配套机械臂焊缝观测场景的具体使用方式。

---

## 一、5 个必须先懂的概念

### 概念 1：地图怎么存 —— 占据栅格 (Occupancy Grid)

把空间切成小方块（体素 voxel），每个体素只有 3 种状态：

```text
■  occupied    有东西（墙、桌子、障碍）
□  free        肯定空着
?  unknown     还没观察过
```

OctoMap 是这种栅格的高效实现（八叉树）。

机器人开机时，整个地图都是 `?`（unknown）。

### 概念 2：相机怎么把 `?` 变成 `□` 或 `■` —— 光线投射 (Raycasting)

相机给一帧 RGB-D（彩色 + 深度）。深度图告诉每个像素"前方多远撞到了东西"。

对每个像素做一次 raycast：

```text
相机 ●───□───□───□───□───■   ← 像素深度=2.0m, 撞到障碍
                          ?
                          ?  ← 障碍后面仍是 unknown

相机 ●───□───□───□───□───□  ← 像素深度=∞ 或超出 2m, 标这一段全 free
                            (再远不写, 保持 unknown)
```

每帧把视锥里所有像素 raycast 一遍 → OctoMap 更新。

### 概念 3：什么值得看 —— 边界 (Frontier)

边界 = "已知 free" 和 "unknown" 紧挨着的体素。它是"刚好能走到、能看新东西"的地方。

```text
□ □ □ □ ? ? ? ?
□ □ □ F ? ? ? ?     F = frontier（free, 但邻居有 unknown）
□ □ □ F ? ? ? ?
□ □ □ F ? ? ? ?
```

走到 frontier 处把相机朝向 unknown，新一帧就能把 `?` 变成 `□`/`■`，地图扩大。

### 概念 4：候选位置和它的"价值" —— viewpoint 与 information gain

**viewpoint** = 相机的一个候选位姿（位置 + 朝向 + 视锥）。

**information gain（信息增益）** = 从这个 viewpoint 看出去，预计能把多少 unknown 体素变成已知。

计算方法：从 viewpoint 模拟一个相机视锥，对视锥内每个方向做 raycast（**只在地图上模拟，不动机器人**），数路径上 unknown 体素的个数。

```text
viewpoint A ───────?─?─?       gain ≈ 3
viewpoint B ───■               gain = 0  (一开始就被挡)
viewpoint C ───?─?─?─?─?─?     gain ≈ 6  (开阔方向)
```

**gain 越大 → 这个 viewpoint 越值得去**。

### 概念 5：怎么探索到候选位置 —— 随机扩张树 (RRT*)

RRT\* 是一种规划算法。直观理解：

> 从当前位置出发，往空间里随机撒点，每个撒进去的点尝试**连一条直线**回到树上最近的点。如果这条直线穿过的全是 free 空间，就把这个新点加入树。重复几百次，得到一棵伸向四面八方的树。

```text
            ●            ●
           /            /
          ●           ●
         / \         /
        ●───●───●──●──● 当前位置 (root)
                   \
                    ●
                     \
                      ●
```

- **节点**：一个候选位置（在 arm 上就是一个末端位姿 / 关节构型）。
- **边**：两节点之间的直线段，**必须穿过的全是 free 空间**（用碰撞检测验证）。
- **从 root 到任一节点**：沿树走，得到一条已验证可行的路径。

"\*" 表示它会优化路径长度（不重要，先忽略）。

> RRT\* 的本质就是：**在已知 free 空间里快速生成一堆"我能走得到"的候选位置**。

---

## 二、DSVP 要解决什么问题

机器人现在在 A 点，目标 P 在远处（比如 2 米外），中间一片地图都是 unknown。
**它必须一边走一边看，逐步把 unknown 变 known，最终走到能看见 P 的位置。**

DSVP 的回答：每个时刻做一件事——

> "在我周围长一棵 RRT\* 树，挑一个最值得去的节点，走过去。"

走过去之后地图变了，重复。

---

## 三、DSVP 的两个工作状态

```text
          地图更新
             │
             ▼
     ┌───────────────┐
     │  Exploration  │   ←── 默认状态
     │  (附近找一步)  │
     └───┬───────┬───┘
         │       │
   找到好节点  连续找不到
         │       │
         ▼       ▼
       执行   ┌───────────────┐
              │  Relocation   │
              │  (跳到远处)    │
              └───────┬───────┘
                      │
                      ▼
                  跳完, 回到 Exploration
```

- **Exploration（探索）态**：默认。在当前位置周围长 RRT\*，挑最优节点执行一步。
- **Relocation（重定位）态**：紧急状态。当前局部已经没东西可看了，跳到远处历史 frontier 重新开工。

---

## 四、Exploration 态的详细步骤

设当前末端位置 $q_0$，相机参数已知。

### 步骤 1：初始化空树

```text
T = { root: q_0 }
gain_max = 0
v_best = root
```

### 步骤 2：循环采样（重复 N 次，比如 N=300）

每次循环干 4 件事：

#### 2a. 随机采样一个候选点 $q_{rand}$

在以 $q_0$ 为中心、半径 $r$（比如 0.5 m）的球内随机撒一个点。

> 进阶：DSVP 还会**有偏采样**——把一部分采样偏向"frontier 附近"或"目标方向"，提高效率。新手版可以先全随机。

#### 2b. 找树上最近的节点 $q_{near}$

```text
q_near = argmin_{v ∈ T} ‖v − q_rand‖
```

#### 2c. 沿 $q_{near} → q_{rand}$ 方向走一小段（比如 0.1 m），得到 $q_{new}$

```text
q_new = q_near + step · (q_rand − q_near) / ‖q_rand − q_near‖
```

#### 2d. 检查这条边能不能走、能看到什么

1. **碰撞检测**：从 $q_{near}$ 到 $q_{new}$ 的直线段，是否全程在 free 体素内？arm 上要做完整 IK + 全身碰撞，但概念上就是 yes/no。
   - **不行** → 丢掉这次采样，继续下一次循环。
   - **可以** → 把 $q_{new}$ 加入树 $T$，加边 $(q_{near}, q_{new})$。

2. **算 information gain**：在 $q_{new}$ 处模拟相机看出去（朝向可以用"指向 unknown 中心"或"指向目标 P"），raycast 数视锥里的 unknown 体素数。

   ```text
   info_gain(q_new) = unknown 体素数
   ```

3. **算节点综合得分**：

   ```text
   gain(q_new) = info_gain(q_new) · exp(−λ · cost_root_to_q_new)
   ```

   - `cost_root_to_q_new` = 沿树从 root 走到 $q_{new}$ 的总长度。
   - `exp(−λ · cost)` 是惩罚远的节点（走太远不划算）。
   - λ 是调参，比如 0.5。

   举例：

   - 节点 A：info=20，距离 0.2 m → gain = 20 · exp(−0.1) ≈ 18.1
   - 节点 B：info=30，距离 1.0 m → gain = 30 · exp(−0.5) ≈ 18.2
   - 节点 C：info=50，距离 2.5 m → gain = 50 · exp(−1.25) ≈ 14.3

   平衡"看得多"和"走得近"。

4. 如果 `gain(q_new) > gain_max`，更新：

   ```text
   gain_max = gain(q_new)
   v_best = q_new
   ```

### 步骤 3：判断是否值得执行

循环跑完后看 `gain_max`：

- **`gain_max > g_high`**（高阈值，比如 15）：这一步很值得执行 → 进步骤 4。
- **`gain_max < g_low`**（低阈值，比如 3）：周围都没东西看 → **切到 Relocation 态**。
- 中间：再多采样一些（继续循环）或者执行（看实现选择）。

### 步骤 4：执行从 root 到 v_best 的**第一条边**

不是一次走到 v_best！只走树上的第一条边（root 到它的子节点）。

```text
root ─── child_1 ─── child_2 ─── v_best
        ↑
     只执行这一段, 比如 0.1 m
```

为什么只走第一段：走了之后地图会变，再远的边可能不再最优甚至撞墙。

执行完到了新位置 $q_0' \approx \text{child\_1}$，更新地图。

### 步骤 5：把当前位置作为新 root，**重置整棵树**，回到步骤 1

DSVP 不复用旧树（实现简单，地图变化大时旧树本就不可信）。

但**有一个东西要保留**：keypose graph（见下文 Relocation）。

---

## 五、Relocation 态的详细步骤

### 触发条件

Exploration 步骤 3 中 `gain_max < g_low`，连续几次都这样 → 当前局部已经"看完了"。

### 步骤 1：维护一个 keypose graph $G$（其实在 Exploration 中就一直在维护）

每次机器人执行完一段走到新位置 $q_0'$，把 $q_0'$ 加入 $G$ 作为节点，并和上一个 keypose 加一条边。

```text
keypose_0 ── keypose_1 ── keypose_2 ── ... ── keypose_now
              \
               keypose_2'  ← 之前来过的分叉
```

每个 keypose 节点还**记录该位置当时未尽的 frontier**（探索方向）。

### 步骤 2：在全局 OctoMap 上找所有当前 frontier 体素

把它们聚成一组 frontier cluster（连通分量），每个 cluster 取中心 $f_i$ 和大小 $|f_i|$。

### 步骤 3：对每个 frontier cluster $f_i$，估算"过去找它要花多少代价"

- 先在 $G$ 中找离 $f_i$ 最近的 keypose $k_i$（就是从哪个 keypose 出发能去到这个 frontier）。
- 在 $G$ 上跑 Dijkstra，从当前位置到 $k_i$ 的最短路径距离 $d_i$。

### 步骤 4：选最划算的 frontier

```text
f* = argmin_{f_i} [ d_i − μ · |f_i| ]
```

意思是：路径短 + frontier 大 = 优先去。

### 步骤 5：沿 $G$ 上路径走过去

这一段走的是**之前已经验证过**的 keypose 走廊（已知 free），所以可以一次规划完，不用边走边探。

到达 $k_i$ 附近后 → 切回 Exploration 态，从这里重新长 RRT\*。

---

## 六、完整例子（用焊缝观测场景的真实数据）

数据：机器人末端起点 $q_0 = (24.4, 0.5, 9.8)$，目标 $P_{weld} = (24.0, 0.16, 11.4)$，相距 ≈ 1.8 m。

### Cycle 1（Exploration）

```text
1. 当前 OctoMap 几乎全 unknown, 只在初始视锥内有一小块 known free.
2. RRT* 从 q_0 长 300 个节点, 都在 0.5m 球内.
   - 大部分节点的边因为穿过 unknown 被判为可走 (取决于策略, 这里假设保守: unknown 当障碍, 那大部分被丢)
   - 实际工程上 Exploration 的碰撞检测会"乐观一点"——unknown 认为可走, 但仍由后续严格检查.
3. 候选节点 v_1: 朝相机右上方 0.3m, info_gain=45 (扫出一片 unknown).
   gain = 45 · exp(−0.15) ≈ 38.7.
4. 38.7 > g_high=15 → 执行 root 到 v_1 路径的第一段 (0.1m).
5. 移动. 地图更新, 多了一片 known free 和右上方的某个柱子被发现 = occupied.
6. keypose graph: q_0 ─ q_0'.
```

### Cycle 2（Exploration）

```text
1. 在新位置 q_0' = (24.32, 0.55, 9.85) 重新长 RRT*.
2. 这次节点能往更多方向长 (因为周围多了 known free).
3. 信息增益最高的节点指向 P_weld 方向 (因为往那边的 unknown 体素最多).
   注: 这里没有显式目标偏置, 但 unknown 大块在 P_weld 方向, 所以自然会朝那走.
   如果加目标偏置, 收敛更快.
4. 选最高 gain 节点, 执行 0.1m.
5. keypose graph: q_0 ─ q_0' ─ q_0''.
```

### Cycle 3–10（继续 Exploration）

```text
每个 cycle 推进 0.1m, 地图持续扩张, P_weld 方向的 unknown 不断被 raycast.
直到某一帧, P_weld 进入相机视锥, 视线全程已知 free → 任务完成.
```

### 假设 Cycle 5 卡住了（Relocation 触发）

```text
Cycle 5 在 q_0''' 长 RRT* 后, 所有节点 info_gain 都 < 3 (周围被柱子+墙包围).
gain_max < g_low → 切到 Relocation.

1. 查 OctoMap 全局 frontier, 找到 3 个 cluster:
   f_1: 在 q_0' 的右上方, 大小 200 体素, 距离 0.8m.
   f_2: 在 q_0 的左方, 大小 80 体素, 距离 0.6m.
   f_3: 在 q_0''' 后下方, 大小 30 体素, 距离 0.5m.

2. 算 cost_i − μ·|f_i| (μ=0.005):
   f_1: 0.8 − 1.0 = −0.2  (越小越好)
   f_2: 0.6 − 0.4 = 0.2
   f_3: 0.5 − 0.15 = 0.35

3. f_1 胜出, 沿 keypose graph 走 q_0''' → q_0'' → q_0' → 接近 f_1.
4. 到达 f_1 附近 → 回 Exploration, 重新长 RRT*.
```

---

## 七、超精简伪代码

```python
def DSVP(q_start, P_target, octomap):
    keypose_graph = Graph()
    keypose_graph.add_node(q_start)
    q = q_start

    while not target_visible(q, P_target, octomap):
        T = RRTStarTree(root=q)
        for _ in range(N_SAMPLES):
            q_rand = sample_in_ball(q, radius=0.5)
            q_near = T.nearest(q_rand)
            q_new  = step_toward(q_near, q_rand, step=0.1)
            if not collision_free(q_near, q_new, octomap):
                continue
            T.add(q_new, parent=q_near)
            ig    = information_gain(q_new, octomap)
            cost  = T.path_length(q_new)
            score = ig * exp(-LAMBDA * cost)
            T.tag(q_new, score)

        v_best, gmax = T.best_node()

        if gmax > G_HIGH:
            first_edge = T.first_edge_toward(v_best)
            execute(first_edge)
            q = first_edge.end
            update_octomap_from_camera(q)
            keypose_graph.add_node(q)
            keypose_graph.connect_last_two()

        elif gmax < G_LOW:
            f_star = best_global_frontier(octomap, keypose_graph, q)
            path = dijkstra(keypose_graph, q, f_star)
            execute(path)
            q = path.end
            update_octomap_from_camera(q)

        # else: 中间值, 多采样几轮再决策, 这里省略
```

---

## 八、对机械臂场景的关键改动

1. **节点不再是 3D 位置，而是关节角向量** $(\theta_1, \dots, \theta_6)$，或末端 SE(3) + IK 解。
2. **碰撞检测**用 cuRobo / Isaac Lab 提供的接口（整条手臂对 OctoMap）。
3. **采样球**用关节空间 6-D 球（每个关节 ±0.3 rad）或末端笛卡尔球。
4. **information gain** 时相机朝向先尝试"指向 $P_{weld}$"，效果最好。
5. **加目标偏置**：score 加一项 $-w \cdot \|q_{new} - P_{weld}\|$，否则可能在原地打转。
6. **Relocation** 在 arm 这种小空间几乎用不上，可以先不实现，只保留 Exploration + 简单回退（往上一个 keypose 退）。

---

## 九、超参清单

| 超参 | 含义 | 典型值 | 调大效果 | 调小效果 |
|------|------|--------|----------|----------|
| `N_SAMPLES` | RRT\* 每周期采样数 | 200–500 | 节点更密、决策更准、计算变慢 | 节点稀、可能漏掉好候选 |
| `radius` | 采样球半径 | 0.3–0.5 m | 可远跳，单步进展大但失败率高 | 步子小、保守、慢 |
| `step` | 每条新边长度 | 0.05–0.15 m | 树扩张快、节点稀 | 树稠密、计算多 |
| `LAMBDA` | 距离惩罚系数 | 0.3–1.0 | 偏好近节点、保守 | 偏好远节点、激进 |
| `G_HIGH` | 立即执行阈值 | 10–20 | 更挑剔，反复扩张 | 急于动 |
| `G_LOW` | 触发 Relocation 阈值 | 1–5 | 容易 relocate（不耐心） | 死磕局部 |
| `μ` | frontier 大小奖励 | 0.001–0.01 | 偏好大 frontier | 偏好近 frontier |
| `w` (目标偏置) | 距 $P_{weld}$ 惩罚 | 0.1–1.0 | 直奔目标 | 自由探索 |

---

## 十、参考资料

- 原论文：H. Zhu et al., *DSVP: Dual-Stage Viewpoint Planner for Rapid Exploration by Dynamic Expansion*, IROS 2021.
- 仓库：<https://github.com/HongbiaoZ/dsv_planner>
- 同组后续工作 TARE：<https://github.com/caochao39/tare_planner>
