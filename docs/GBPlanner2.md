# GBPlanner2 算法详解（零基础版）

GBPlanner2（Graph-Based Exploration Planner v2）是 NTNU（挪威科技大学）开发的探索规划算法，原本针对地下矿洞、隧道、管腔等**狭长半封闭环境**。在 DARPA Subterranean Challenge（CERBERUS 队夺冠）实战验证。它对你的"五面开口盒内焊缝"或"管腔焊缝"这类场景特别合适。

本文从零开始讲，参考前文 [DSVP.md](DSVP.md) 中已经介绍过的概念可以直接跳过对应章节。

---

## 一、必须先懂的 7 个概念

前 5 个概念和 DSVP 完全相同（占据栅格、Raycasting、Frontier、Viewpoint/Information Gain、RRT\*），不再展开。**新引入的 2 个**是 GBPlanner2 的核心：

### 概念 6：图 (Graph) vs 树 (Tree) —— 这是 GBPlanner2 与 DSVP 的本质区别

**树**：每个节点只有**一个父节点**，形状像家谱。

```text
        ●  root
       / \
      ●   ●
     / \   \
    ●   ●   ●
```

从根到任一节点只有**一条**路径。DSVP 用的就是树。

**图**：节点之间可以**互相多条边连接**，形状像渔网或蜘蛛网。

```text
        ●─────●
       /│\   /│
      ● │ ● ● │
       \│/ \ │
        ●───●
```

从一个节点到另一节点可以有**多条**路径（绕路）。GBPlanner2 用的是图。

**为什么这个区别很重要？**

- 走到死胡同时，**树**只能原路回退（沿父节点链）。
- **图**因为有环（多条路径），可以"绕一圈到分叉口换路"。

```text
[树版本]
  入口 ── A ── B ── 死胡同
  机器人在死胡同, 必须沿 B → A → 入口 一步步退回, 慢且必然撞自己走过的路.

[图版本]
  入口 ── A ── B ── 死胡同
         │      │
         └──C───┘
  机器人在死胡同, 可以走 死胡同 → C → A → 入口, 不用倒退.
```

在窄走廊或管腔场景这个能力是必须的。

### 概念 7：Dijkstra 算法 —— 在图上找最短路径

给定图 $G$ 和起点 $s$、终点 $t$，Dijkstra 算法返回**沿图边能走的、总长度最短**的那条路。

> 直观理解：从起点出发，每次走"目前知道的最便宜的下一步"，像水从中心流出来一样慢慢扩散，第一次到 $t$ 的路径就是最短路径。

GBPlanner2 在两个地方用 Dijkstra：

1. 在局部图里找"从当前位置到收益最大节点"的最短路径。
2. 在全局图里找"从当前位置回到某个旧 frontier"的最短路径（dead-end 回退）。

**只需要知道 Dijkstra 是个黑盒函数**：

```text
path = dijkstra(graph, source, target)
```

---

## 二、GBPlanner2 要解决什么问题

机器人在**复杂半封闭环境**中（管道、矿洞、箱体内部），从起点 $q_0$ 出发，目标可能在远处或被障碍包围。要求：

1. 边走边建图（占据栅格）。
2. 走过的路要记住，**走过的拓扑（图结构）也要保留**，不能像 DSVP 那样每周期重置。
3. 遇到死胡同要会自动从分叉口绕过去。
4. 能保证不会无限循环、不会撞墙。

GBPlanner2 的回答：

> "维护一张**会一直长大**的全局图，每个周期在局部小盒子里多撒些节点丰富它，然后在图上跑 Dijkstra 找最值得走的路径。"

注意"图一直长大"——这是和 DSVP（每次重置树）最大的实现差别。

---

## 三、四个核心特性详解

第二章列出的四点是 GBPlanner2 与 DSVP 的根本性差别，单独看每条都很模糊，下面逐条讲清楚。

### 1. "边走边建图" 是什么意思

#### 关键事实：机器人开机时没有任何地图

不像 Google 地图给你一份完整的世界，机器人开机时整个空间在它眼里都是 unknown：

```text
机器人开机的瞬间, OctoMap 里:

      ? ? ? ? ? ? ? ? ?
      ? ? ? ? ? ? ? ? ?
      ? ? ? ? ? ? ? ? ?
      ? ? ? ? ●(我) ? ? ?     ←  整个世界都是问号
      ? ? ? ? ? ? ? ? ?
      ? ? ? ? ? ? ? ? ?
      ? ? ? ? ? ? ? ? ?
```

#### 每移动一帧，地图被一点点"擦亮"

相机看到的那一锥形区域（视锥）里的体素被更新：

```text
第 1 帧 (相机朝右上):

      ? ? ? ? ? ? ? ? ?
      ? ? ? ? ? □ □ ? ?
      ? ? ? ? □ □ □ ? ?
      ? ? ? ? ●━━━□□■ ?     ●发出射线, 一直到 ■(墙) 之间全标 free
      ? ? ? ? ? □ □ ? ?
      ? ? ? ? ? ? ? ? ?

第 5 帧 (机器人移动+相机扫描后):

      ? ? ? ? ? ? ? ? ?
      ? ? ? □ □ □ □ ? ?
      ? ? □ □ □ □ □ ? ?
      ? ? □ ●━━━□ □ ■ ?
      ? □ □ □ □ □ □ ? ?
      ? ? □ □ ? ? ? ? ?     越多帧, ? 越来越少
```

#### "边走" = 机器人物理上在动；"建图" = 这个动作的副产品

```text
时间轴:
 t=0:   位置=q_0,  OctoMap 99% 是 ?
 t=0.1: 位置=q_0+δ, OctoMap 95% 是 ?  ← 移动那一瞬间也在更新地图
 t=0.2: 位置=q_1,   OctoMap 90% 是 ?
 ...
 t=10:  位置=q_n,   OctoMap 30% 是 ?
```

**重点**：建图不是单独一个步骤，而是机器人每动一下、相机每出一帧，OctoMap 就被自动更新一次。规划器读到的地图永远是当前最新的版本。

GBPlanner2 和 DSVP 在"建图"这件事上**完全一样**——它们都基于当前地图做决策。区别在第 2、3、4 点。

### 2. "保留走过的拓扑"是什么意思

先定义两个词：

- **路**：你实际机器人脚（或末端）走过的轨迹。
- **拓扑**：你**验证过**的"哪些点之间能直接走"。这是一张抽象的连接关系图，节点是位置，边是"我已经验证从 A 到 B 这条直线全程不撞墙"。

#### DSVP 的做法

每个 cycle，DSVP 在当前位置长一棵 RRT\* 树。这棵树包含了几百条**验证过**的边。

```text
DSVP cycle 1 长出来的树:

         ●
         │
      ●──●──●
      │     │
   ●──●     ●──●
                │
                ●

每条边都做过碰撞检测, 是验证过的"安全连接".
```

执行完一段后，DSVP **把这棵树扔掉**，只在 keypose graph 上加一个新节点：

```text
DSVP cycle 1 结束后保留的:

   q_0 ── q_1     ← 仅此而已！整棵树被丢弃
```

下个 cycle 在 q_1 重新长树，又是几百条新边。**之前验证过的几百条边的工作被浪费了**。

#### GBPlanner2 的做法

每个 cycle 在当前位置周围长出来的节点和边，**全部塞进同一张全局图，永不删除**。

```text
GBPlanner2 cycle 1 结束后保留的（全部！）:

         ●─────●
         │  ╲  │
      ●──●───●─●
      │  ╳   │
   ●──●─────●──●
                │
                ●
```

cycle 2 在 q_1 周围长新节点时，新节点和**老节点的边**也会建立连接：

```text
GBPlanner2 cycle 2 后:

         ●─────●─────●(新)
         │  ╲  │      │
      ●──●───●─●──●(新)
      │  ╳   │      │
   ●──●─────●──●(新)
                │
                ●

老节点和新节点之间多了好几条连接边.
```

#### 为什么"拓扑"重要

举个例子：你 30 分钟前从 A 走到 B 的时候，验证了 A→B 这条直线段没有障碍。现在你在 X 处需要回到 B：

- **DSVP**：A→B 那条边早扔了。要么沿 keypose 链一步步回退（慢），要么重新做 RRT\* 重新发现并验证。
- **GBPlanner2**：A→B 还在图里。直接 Dijkstra 一下：`X → A → B`，每条边都已知 free，一次规划，直接走。

**拓扑就是"我之前花碰撞检测算出来的路况报告"。GBPlanner2 把这份报告全部存档，DSVP 只存当前位置的链表。**

### 3. "遇到死胡同自动从分叉口绕过去"

#### 什么是死胡同（dead-end）？

死胡同 = 当前位置周围已知 free 的体素全部探完了，但**还没看到目标**。

```text
俯视图（机械臂末端在五面开口盒里）:

  ┌─────────────────────────────┐
  │      盒子内部 (有立柱)         │
  │           │                  │
  │           │   ●(末端在这里)    │   ← 周围已知 free 都看完了
  │       立柱│                  │
  │           │   P_weld想看到这里, │
  │           │   但被立柱挡死      │
  └─────────────────────────────┘
                    ╳ 开口
```

末端发现：在这个角落，朝任何方向看都是 occupied（柱子或墙）或已知 free。**必须挪到能看到 P_weld 的地方**。

#### DSVP 用 keypose 链回退（很笨）

DSVP 维护的是一条"我走过的脚印链"：

```text
入口 ── q_1 ── q_2 ── q_3 ── q_4 ── q_5(死胡同)
```

要从死胡同回退到 q_2 找新路：必须**沿原路一步步退**：q_5 → q_4 → q_3 → q_2。

```text
DSVP 必须这样退:

  ●─────●─────●─────●─────●
  入口  q_1  q_2  q_3  q_4  q_5(我)
                              ↓
                        往回 ←
                              ↓
                        往回 ←
                              ↓
                        往回 ←
                              ↓
                       到 q_2
```

每一步都是一次完整的"plan + execute"，慢且占空间。

#### GBPlanner2 用全局图直接绕过去（聪明）

GBPlanner2 在走的过程中**不断在不相邻的节点之间加交叉边**（只要碰撞检测过）：

```text
GBPlanner2 累积的全局图:

     ●─────●─────●─────●─────●
     入口  q_1  q_2  q_3  q_4  q_5(我)
            │   │     │
            └───╳─────┘     ← 这些"对角"边是 cycle 中陆续加入的
                │
              ●──●─●        ← 还有其他随机采样的内部节点
                │
                ●

q_5 和 q_2 之间可能有 cycle 时撒进去的节点 c_1, c_2, c_3 形成的旁支:
     q_5 ─ c_1 ─ c_2 ─ q_2
```

要从死胡同到 q_2，**Dijkstra 直接给出最短路径**——可能是 `q_5 → c_2 → c_1 → q_2`，根本不沿来路退。

#### 配一个更直观的例子（管腔工件）

```text
俯视一个 U 形管腔，焊缝 P_weld 在 U 形的另一臂底部:

           入口
            │
       ┌────┴────┐
       │         │
       │   ┌─────┘  ← q_1 ─ q_2 ─ q_3(死胡同, 看不到 P_weld)
       │   │
       │   └─────┐
       │         │
       └────┬────┘
            │
        P_weld

DSVP: q_3 → q_2 → q_1 → 入口 → ... 慢慢摸到 P_weld 那条腿
GBPlanner2: 如果某个 cycle 里在交叉口位置撒过节点 c, 形成了
            q_3 ─ c ─ (P_weld 那臂的某节点)
            就能直接从 q_3 切到另一条腿, 跳过回退.
```

### 4. "不会无限循环、不会撞墙"是怎么保证的

这是两件不同的事，分开讲。

#### 4a. 为什么不会撞墙

只要做到一件事：**只在已验证安全的边上移动**。

GBPlanner2 的规则：

```text
图中的每一条边 (a, b), 在加进图的那一刻已经做过:
  "从 a 到 b 的直线段, 是否完全在 OctoMap 已知 free 体素内?"
  是 → 加边
  否 → 不加

所以图里的每条边都是绿灯.
```

执行时：

```text
Dijkstra 给的路径 = 一系列图中的边
机器人沿这些边移动 = 100% 走在已验证 free 上
→ 不撞墙
```

**唯一的例外**：边加入图的那一刻地图是 t1 时刻的，到 t2 时刻地图可能更新了（比如发现以前以为 free 的体素其实是动态障碍）。GBPlanner2 处理方式：执行前**重新检查**第一条边，过期就丢弃 + Dijkstra 重新规划。

#### 4b. 为什么不会无限循环

这点更微妙。机器人**有可能**在一片区域里转来转去看似没进展，但它不会**真的**无限循环，因为有三道保险：

##### 保险 1：地图状态单调减少 unknown

每次相机出帧，OctoMap 只能让 unknown 体素**减少**（变成 free 或 occupied），不会回头：

```text
t=0:  unknown 体素数 = 100000
t=1:  unknown 体素数 = 95000  (减少)
t=2:  unknown 体素数 = 92000  (减少)
...
t=N:  unknown 体素数 = 8000   (饱和)
```

只要 unknown 还能减少，机器人就在做有效工作。

##### 保险 2：节点的 gain 会枯竭

节点 v 的 gain 是 "从 v 看出去能扫多少 unknown"。一旦那个区域被探索过，gain 自然降到接近 0。

```text
节点 v 的 gain 历史:
  cycle 1 创建: gain=80
  cycle 5: gain=20  (其他视角顺手扫到了 v 周围)
  cycle 10: gain=5  (v 周围基本探完)
  cycle 15: gain=0  (完全探完)
```

GBPlanner2 评分用 `gain · exp(-λ·cost)`：gain=0 的节点永远不会被选。

**所以图变大不要紧，但"还值得去"的节点只会越来越少**。算法本质上是 **monotonic（单调）** 的。

##### 保险 3：stuck_count 计数器 + Global 触发

如果连续几个 cycle 都找不到 gain > g_low 的节点，`stuck_count` 累加。超过阈值（比如 3） → 切到 Global Planning Step。

Global Step 在**整张图**上找仍有 gain 的 frontier 节点。如果整张图 frontier 也空了 → 探索完成（或目标不可达），算法**终止**，不会无限跑。

```text
终止条件:
  P_weld 已经能看到 → 任务成功, 终止
  全图所有节点 gain 都 < 阈值 → 整个连通空间探完, 终止
  连续 N 次 Global 都失败 → 报错终止
```

这三个机制叠加，**有限步内**算法一定停止。

#### 4c. 类比理解

把整个过程类比成"蒙眼摸黑探房间"：

| 危险 | 保护机制 |
|------|---------|
| 撞墙 | 只走"我手已经摸过证实是空的"的方向 |
| 在原地打转 | 每摸一次就在地上画 X，不去重复画过 X 的位置 |
| 永远摸不完 | 房间是有限大的，X 画满就停 |

GBPlanner2 的图、gain 衰减、stuck 计数器分别对应这三个机制。

### 5. 总结

四个性质之间的关系：

```text
  边走边建图  ←──────  这是基础, GBPlanner2 和 DSVP 都做
       │
       ↓
  保留全局图  ←──────  GBPlanner2 多做这一步, DSVP 不做
       │
       ↓
  死胡同绕行  ←──────  因为有全局图带环, 自然实现
       │
       ↓
  不撞墙、不死循环  ←  因为图边都验证过, 加上 gain 单调衰减 + stuck 触发 Global
```

DSVP 只到第 1 步；GBPlanner2 走完 4 步。这就是 GBPlanner2 在窄空间、复杂工件场景里更稳的根本原因。

---

## 四、GBPlanner2 的两个工作阶段

```text
                  地图 + 图持续更新
                          │
                          ▼
              ┌───────────────────────┐
              │  Local Planning Step  │   ← 默认状态
              │  (当前 box 内长图)     │
              └─────┬──────────────┬──┘
                    │              │
              找到好节点      连续多次没收益
                    │              │
                    ▼              ▼
                 执行   ┌──────────────────────────┐
                        │  Global Planning Step    │
                        │  (跳到历史 frontier)      │
                        └─────────┬────────────────┘
                                  │
                                  ▼
                            到达后回 Local
```

- **Local Planning Step（局部）**：默认。在当前位置周围的 box 内增长图，找最优执行边。
- **Global Planning Step（全局）**：紧急。当前局部已榨干，跳到历史上还有未尽方向的旧节点重新开工。

> 与 DSVP 不同点：**两阶段操作的是同一张全局图**。Local 只是在图上往当前位置周围加新点；Global 是在整张图上跑 Dijkstra。

---

## 五、Local Planning Step 详细步骤

### 步骤 0：维护全局图 G(V, E)

GBPlanner2 启动时初始化空图：

```text
V = { v_0 = 当前起点 }
E = { }
```

之后每一步只**追加**节点和边，从不清空。

### 步骤 1：定义当前 Local Volume（局部盒子）

以当前位置为中心，定义一个 axis-aligned 盒子 $V_{local}$：

```text
[x_min, x_max] × [y_min, y_max] × [z_min, z_max]
比如 ±5m × ±5m × ±3m  (地面机器人)
比如 ±0.4m × ±0.4m × ±0.4m  (机械臂)
```

> 在管腔/走廊场景，盒子可以**沿走廊主轴拉长**（比如 ±10m × ±2m × ±2m），让节点分布更合理。
> GBPlanner2 论文中盒子是**自适应**的：当一段时间没找到好节点，盒子会自动扩大。

### 步骤 2：在 $V_{local}$ 内增量增长图

重复以下子步骤 N 次（比如 N = 100）：

#### 2a. 在 $V_{local}$ 内随机采样一个点 $p_{rand}$

#### 2b. 检查这个点是不是"好位置"

- 必须在已知 free 体素内（不是 occupied，也不是 unknown）。
- 周围有足够的 clearance（碰撞安全距离）。

如果不满足，丢弃，继续下一次采样。

#### 2c. 把 $p_{rand}$ 连接到附近现有节点

找出 $V$ 中所有距 $p_{rand}$ 在阈值（比如 1.5 m）内的节点 $\{v_i\}$，对每一个尝试加边：

- **碰撞检测**：从 $v_i$ 到 $p_{rand}$ 的直线段是否完全穿过已知 free？
  - **是** → 加边 $(v_i, p_{rand})$ 到 $E$。
  - **否** → 不加。

如果一条边都加不上（$p_{rand}$ 太孤立）→ 丢弃 $p_{rand}$。
如果至少加上一条边 → 把 $p_{rand}$ 加入 $V$。

```text
图扩张可视化（每次循环）：

   ●───●               ●───●
   │   │               │   │
   ●   ●     →         ●───●───● p_rand
   │   │               │   │   │
   ●───●               ●───●───●
                            └─新边
```

> **注意**：这一步是 GBPlanner2 和 RRT\* 的核心区别——RRT\* 每个新节点只连**一条**边给最近邻；GBPlanner2 连**所有**距离阈值内可达的节点。这就是为什么图能形成环（多条路径）。

### 步骤 3：计算每个新节点的 Volumetric Gain

对步骤 2 中加入的每个新节点 $p_{rand}$：

```text
volumetric_gain(p_rand) = 从 p_rand 处, 模拟相机在多个朝向下,
                         视锥能覆盖的 unknown 体素总数
```

具体计算：

1. 在 $p_{rand}$ 处假设相机有 K 个候选朝向（如 8 个 yaw 方向 + 上下 pitch）。
2. 对每个朝向算视锥内 raycast 到 unknown 的体素数。
3. 取最大值或加权和作为 $p_{rand}$ 的 gain。

> 实践上每个节点存"最佳朝向 + 该朝向 gain"，后面执行时直接用这个朝向。

### 步骤 4：在图 G 上找最优路径

定义节点综合得分：

```text
score(v) = volumetric_gain(v) · exp(−λ · path_length(q_now, v))
```

其中 `path_length(q_now, v)` 是 Dijkstra 在 $G$ 上从当前位置到 $v$ 的最短路径长度。

```text
v_best = argmax_v score(v)
gain_max = score(v_best)
```

### 步骤 5：判断是否值得执行

- **`gain_max > g_high`**：这一步值得 → 执行。
- **`gain_max < g_low`**：当前 box 探完了 → 切到 Global Planning Step。
- 中间：再多采样一些（步骤 2 继续），或者执行（看实现）。

### 步骤 6：执行从 $q_{now}$ 到 $v_{best}$ 的路径**第一段**

跟 DSVP 一样，**只走第一条边**，不一次性走全程。

到达新位置后：

- 把新位置加入图 V（并和旧位置连边）。
- 用相机数据更新 OctoMap。
- 回到步骤 1（重新定义 Local Volume）。

### 步骤 7：路径优化（GBPlanner2 特色）

执行前还会对路径做两件事：

#### 7a. 路径平滑

把 Dijkstra 的折线路径用 spline 拟合一下，避免机器人突然急转弯。

#### 7b. 顺路 viewpoint 增强

沿路径每隔一段距离，挑相机朝向使 raycast 看到的 unknown 最多——**顺手探更多**。

---

## 六、Global Planning Step 详细步骤

### 触发条件

Local 步骤 5 中 `gain_max < g_low`，连续 M 次都这样（M 是耐心系数，比如 3） → 当前局部已经"看完了"，需要远跳。

### 步骤 1：识别"残留 frontier 节点"

遍历**全局图** $V$ 中**所有历史节点**（不是只看当前 box 内的）：

```text
对每个 v ∈ V:
  重新算 v 的 volumetric_gain (用当前最新地图)
  if volumetric_gain(v) > 阈值:
    标记 v 为"frontier 节点"
```

为什么要重新算？因为之前算的 gain 是当时地图状态的，现在地图已经更新过，有些节点的 unknown 已经被别的位置看清了。

> 直观理解：把历史上每个走过的位置重新看一眼，"这个地方现在还有没有什么没看完的方向？"

### 步骤 2：对每个 frontier 节点估算去那里的代价

```text
对每个 frontier 节点 f_i:
  d_i = dijkstra(G, q_now, f_i).length
```

这一步是在 **已经验证 free 的全局图**上跑 Dijkstra，得到的是**保证可走**的路径长度。

### 步骤 3：选最划算的 frontier 节点

```text
f* = argmax_{f_i}  [ volumetric_gain(f_i) · exp(−λ · d_i) ]
```

或者最小化：

```text
f* = argmin  [ d_i − μ · gain(f_i) ]
```

两种公式效果接近。

### 步骤 4：沿全局图路径走过去

用 Dijkstra 给出的路径，**整段直接执行**。这一段路是历史已验证 free 的，不需要边走边探（但相机仍持续更新地图，顺手看一眼）。

### 步骤 5：到达 $f^*$ 后切回 Local Planning Step

当前位置变成 $f^*$，重新定义 Local Volume 做局部探索。

---

## 七、完整例子（机械臂 + 五面开口盒场景）

设：工件放在五面开口盒里（前面开口、其他五面有钢板），机器人末端起点在盒子前方，目标焊缝 $P_{weld}$ 在盒子内部右后角。

```text
俯视图:

         ┌────────────────┐
         │       盒子内部   │
         │                │
         │      P_weld●   │
         │                │
         │                │
         └────────────────┘
                  ╳ 开口
                  
                 ●  q_0 (机器人末端起点)
```

### Cycle 1（Local）

```text
1. q_0 周围长图: 在 0.4m 立方体内撒 100 个候选, 大部分能加进图 (开阔区域)
2. 算各节点 volumetric_gain:
   - 朝盒子开口的节点 v_a: gain=120 (能看进盒子大片 unknown)
   - 朝侧面的节点 v_b: gain=30
   - 朝外的节点 v_c: gain=10 (背向工件)
3. score(v_a) = 120 · exp(-0.05) ≈ 114  ← 最高
4. 114 > g_high → 执行 q_0 → v_a 第一段, 移动 0.1m
5. 图 G: 节点 = q_0, v_a, 新位置 q_1; 边 = (q_0, v_a), (q_0, q_1), 还有 v_a 周围的内部边
```

### Cycle 2-4（Local，逐步进入盒子开口）

```text
机器人沿盒子开口慢慢进入, 相机持续把盒子内部 unknown 变 known.
图 G 持续在前方累积新节点, 形成网络.
keypose 形成: q_0 ── q_1 ── q_2 ── q_3
但 GBPlanner2 还会加横向连接: q_2 ── 旁边节点 ── q_1 形成环.
```

### Cycle 5（Local，进入盒子内）

```text
进入盒子内部, 相机视野里能看到 P_weld 方向, 但盒子右边有立柱遮挡视线.
RRT*-like 风格的话只能从进来的方向继续往里钻.
GBPlanner2 因为是图, 能采样到右下绕柱的节点, 自然加了一条绕行路径.
```

### Cycle 6（卡住，Global 触发）

```text
机器人在盒子里走到一个角落, 三面是墙, 一面是已经探过的区域.
Local 连续 3 个周期 gain_max < g_low.

Global Planning Step:
1. 重新算所有历史节点的 volumetric_gain.
2. 发现 q_2 处 (盒子刚进来时) 朝侧上方还有未探索方向, gain=80.
3. dijkstra(G, q_now, q_2): 因为图有环, 找到一条从角落直接绕回 q_2 的路径,
   比沿来路退回去短.
4. 执行该路径, 到达 q_2 附近.
5. 切回 Local, 朝 q_2 处的未尽方向探索, 看到了之前被柱子挡住的另一面.
```

### Cycle 7+（Local，看到 P_weld）

```text
新视角下, P_weld 进入相机视锥, 视线全程已知 free → 任务完成.
```

---

## 八、超精简伪代码

```python
def GBPlanner2(q_start, P_target, octomap):
    G = Graph()
    G.add_node(q_start)
    q = q_start
    stuck_count = 0
    
    while not target_visible(q, P_target, octomap):
        # ===== Local Planning Step =====
        V_local = bounding_box(q, size=(0.4, 0.4, 0.4))
        new_nodes = []
        
        for _ in range(N_SAMPLES):
            p = sample_in_box(V_local)
            if not in_free_space(p, octomap):
                continue
            neighbors = G.nodes_within(p, radius=0.2)
            valid_edges = [v for v in neighbors 
                           if line_collision_free(v, p, octomap)]
            if not valid_edges:
                continue
            G.add_node(p)
            for v in valid_edges:
                G.add_edge(v, p)
            new_nodes.append(p)
        
        for v in new_nodes:
            v.gain = volumetric_gain(v, octomap)
        
        # 在图上找最优路径
        best_score = 0
        best_node = None
        for v in G.nodes:
            cost = dijkstra_length(G, q, v)
            score = v.gain * exp(-LAMBDA * cost)
            if score > best_score:
                best_score = score
                best_node = v
        
        if best_score > G_HIGH:
            path = dijkstra(G, q, best_node)
            path = smooth_and_enhance_viewpoints(path, octomap)
            execute(path[0:1])      # 只走第一段
            q = path[1]
            G.add_node(q)
            G.connect_to_neighbors(q)
            update_octomap_from_camera(q)
            stuck_count = 0
        elif best_score < G_LOW:
            stuck_count += 1
            if stuck_count >= PATIENCE:
                # ===== Global Planning Step =====
                frontier_nodes = []
                for v in G.nodes:
                    v.gain = volumetric_gain(v, octomap)  # 重新评估
                    if v.gain > FRONTIER_THRESHOLD:
                        frontier_nodes.append(v)
                
                f_star = max(frontier_nodes, 
                    key=lambda f: f.gain * exp(-LAMBDA * dijkstra_length(G, q, f)))
                
                path = dijkstra(G, q, f_star)
                execute(path)        # 这次走全程, 因为是已知安全路径
                q = f_star
                update_octomap_from_camera(q)
                stuck_count = 0
        # else: 继续采样, 不执行也不切换
```

---

## 九、对机械臂场景的关键改动

1. **节点是关节构型 $\theta \in \mathbb{R}^6$**，不是 3D 点。Local Volume 是关节空间的小立方体（每个关节 ±0.3 rad）或末端笛卡尔小立方体 + IK。
2. **边的"碰撞检测"** 用 cuRobo 的 trajectory check：模拟从 $\theta_a$ 插值到 $\theta_b$，每个中间构型都不撞 OctoMap 的 occupied 体素。
3. **Local Volume 自适应**：在工件附近用小盒子（0.3 m），远离工件用大盒子（0.6 m），探索效率高。
4. **Volumetric gain 加目标方向偏置**：除了数 unknown 体素，再加 $w \cdot \cos(\angle(\text{相机朝向}, P_{weld} - p))$，鼓励朝目标看。
5. **Frontier 节点判定**：除了体素 unknown 数，还要考虑"是否在朝 $P_{weld}$ 的方向"。
6. **Global 阶段几乎一定会触发**：因为五面开口盒/管腔这类场景，Local 在死角必然卡住。这是 GBPlanner2 比 DSVP 强的地方。

---

## 十、超参清单

| 超参 | 含义 | 典型值 (arm) | 调大效果 | 调小效果 |
|------|------|--------------|----------|----------|
| `bounding_box` | Local Volume 尺寸 | 0.3–0.6 m 立方 | 节点分布广、采样浪费多 | 探索局部、可能进度慢 |
| `N_SAMPLES` | 每周期采样次数 | 50–200 | 图增长快、计算慢 | 图稀疏、可能漏好路径 |
| `radius` | 节点连接距离阈值 | 0.1–0.3 m | 图边密、环多、内存大 | 图退化为树、回退困难 |
| `LAMBDA` | 距离惩罚系数 | 0.3–1.0 | 偏好近节点、保守 | 偏好远收益高节点 |
| `G_HIGH` | 立即执行阈值 | 10–30 | 更挑剔、慢 | 急于动 |
| `G_LOW` | 触发 Global 的阈值 | 1–5 | 容易切 Global、更耐心建图 | 死磕 Local |
| `PATIENCE` | 连续多少次低 gain 才切 Global | 3–5 | 更不容易切 Global、稳 | 频繁切 Global |
| `FRONTIER_THRESHOLD` | 历史节点判 frontier 的 gain 下限 | 5–15 | frontier 少、Global 选择少 | frontier 多、可能选到无效目标 |

---

## 十一、与 DSVP 的对比（决定何时选哪个）

| 维度 | DSVP | GBPlanner2 |
|------|------|------------|
| 数据结构 | RRT\* 树 + 简单 keypose graph | 完整图（带环） |
| 每周期是否重置 | 重置树 | 图持续累积 |
| Local 找路径 | 树上贪心 | Dijkstra 全图 |
| Global 找路径 | keypose graph + 体素 frontier | 全局图节点上的 frontier + Dijkstra |
| 走过的拓扑可被复用 | 部分 | 完全 |
| Dead-end 回退 | 沿原路退 | 走环绕过去 |
| 最适合的场景 | 开阔环境、快速 baseline | 窄走廊、管腔、半封闭工件 |
| 实现工作量 | 1 周左右 | 2–3 周 |

**给焊缝场景的建议**：

- **梁/柱类敞开焊缝**（你 sample 的 BEAM）：DSVP 够用，简单。
- **箱体内部 / 管腔焊缝 / 复杂工装包围**：GBPlanner2 必要。
- **不知道哪种工件占比多**：先实现 DSVP 跑通，把"keypose graph 加更多边"升级成 GBPlanner2 风格的全局图，能拿到大部分好处。

---

## 十二、参考资料

- 原论文 v1：T. Dang, M. Tranzatto, et al., *Graph-based subterranean exploration path planning using aerial and legged robots*, JFR 2020.
- 改进版：M. Kulkarni, T. Dang, K. Alexis, *Aerial Field Robotics*, 2022（含 GBPlanner2）。
- 仓库：<https://github.com/ntnu-arl/gbplanner_ros>
- DARPA SubT 实战论文：M. Tranzatto et al., *CERBERUS in the DARPA Subterranean Challenge*, Science Robotics 2022.
