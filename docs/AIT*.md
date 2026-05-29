# AIT\* 算法详解（零基础版）

AIT\*（**A**daptively **I**nformed **T**rees）是 Strub 和 Gammell 2020 年发表的采样式路径规划算法。它在 BIT\*（同实验室 2015 年作品）基础上加了"自适应启发式"，是目前**最优采样规划算法**里收敛最快的之一。

> 原论文：M. P. Strub & J. D. Gammell, *Adaptively Informed Trees (AIT\*): Fast Asymptotically Optimal Path Planning through Adaptive Heuristics*, ICRA 2020.
> 仓库（OMPL 内置）：<https://ompl.kavrakilab.org/classompl_1_1geometric_1_1AITstar.html>

本文从零开始讲，前面已经在 [DSVP.md](DSVP.md) 等文档介绍过的概念会快速带过。

---

## 一、必须先懂的 6 个概念

### 概念 1：构型空间 (Configuration Space, C-space)

机器人**所有可能状态**的集合。

```text
- UAV 平动: C = ℝ³  (位置 x, y, z)
- 6-DoF 机械臂: C = ℝ⁶  (6 个关节角)
- 移动机器人: C = SE(2) (x, y, yaw)
```

路径规划 = 在 C 中找一条从 q_start 到 q_goal 的连续曲线，全程不撞障碍。

每个 q ∈ C 是一个"构型点"。整个 C 被划分成两类：

```text
C_free: 自由区, 机器人在这些 q 不撞墙
C_obs:  障碍区, 机器人在这些 q 会撞墙
```

我们要找的路径必须**整条都在 C_free** 里。

### 概念 2：启发式 (Heuristic) 和 A\* 算法

**A\*** 是经典图搜索算法，每次从待探索节点中选最有"潜力"的：

```text
f(v) = g(v) + h(v)

g(v) = 从 start 走到 v 的真实最短代价 (已知)
h(v) = 从 v 走到 goal 的估计代价 (启发式)
```

A\* 每次取 f 最小的节点扩展。如果 `h` **可采纳 (admissible)** —— 即 h(v) ≤ 真实代价 —— A\* 保证找到最优解。

**启发式好坏决定速度**：

```text
h(v) = 0:               A* 退化为 Dijkstra (盲搜, 慢)
h(v) = 直线距离:         不考虑障碍, 中等速度
h(v) = 真实最短路径代价:   完美, 一步到位 (但事先算不出, 这就是悖论)
```

```text
俯视图: start → goal 之间有一堵墙

   start ●─────────●─────────● goal
              ║      
              ║墙        
              ║       
   绕路真实代价: 1.5 (要绕)
   直线距离 h:   1.0 (无视墙, 太乐观)

A* 用直线 h 会先尝试穿墙的方向, 撞了墙才绕.
更好的 h 应该是 1.5, 让 A* 一开始就绕.
```

**AIT\* 的核心创新就是让 h 自动变得越来越准。**

### 概念 3：边碰撞检测的高昂代价

```text
点检测  q 是否撞墙: 查 OctoMap 一个体素 → 1 μs
边检测  q_a → q_b 直线段是否全 free: 沿线段插 K 个点全部点检测 → K μs

K 通常 = 10-50, 所以边检测比点检测贵 1-2 个数量级.
```

采样规划每加一条边就要做一次边检测。**总开销几乎全在边检测上**。AIT\* 优化的核心就是"少做边检测"。

### 概念 4：Lazy 评估（推迟做）

如果一条边可能根本不会被最终路径用上，**为什么要花时间检测它？**

```text
Eager: 一条边加进图就立即做碰撞检测
Lazy:  先不检测, 等真的要用 (路径需要它) 才检测
```

PRM\* 是 eager 的，Lazy PRM\* 是 lazy 的。AIT\* 是**两者混合**：反向树 lazy，前向树 eager。

### 概念 5：渐近最优性 (Asymptotic Optimality)

```text
样本数 N → ∞ 时, 算法找到的解 → 真实最优解
```

简单的 RRT 不渐近最优（贪心连接，路径常常很弯）。RRT\* / BIT\* / AIT\* 都渐近最优——核心是**rewire**（发现更短路径就改父节点）。

### 概念 6：双树思想（forward + reverse）

```text
前向树 (forward): 从 start 长向 goal
反向树 (reverse): 从 goal 长向 start

       start ●─●─●─●         前向树 (T_F)
                    ╲
                     ●─●─●  goal      反向树 (T_R)
                          
       两棵树之间还没接通, 需要继续生长.
```

为什么要两棵树？

- 单树 RRT-Connect 也用双树，但只为了"让它们相遇"。
- AIT\* 用双树是为了让**反向树作为前向树的 h**。这是它和别人的根本区别。

---

## 二、AIT\* 要解决什么问题

> 给定 q_start, q_goal, 障碍物地图（OctoMap / ESDF），在 C_free 中找一条**渐近最优**的路径，使**碰撞检测开销最小**。

### 与 BIT\* 的区别

BIT\*（前作）的限制：

- A\* 启发式 `h(v) = 直线距离 v→goal`，**无视障碍**。
- 障碍密集时，h 严重低估真实代价，A\* 会反复探索"看起来近实际上撞墙"的方向。

AIT\* 的回答：

> "让反向树作为前向树的启发式，每次撞墙就更新反向树，h 越来越准。"

---

## 三、整体结构（双树协作）

```text
                ┌────────────────────────────────────┐
                │  撒一批 N 个样本 (V_samples)       │
                └────────────────┬───────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                     ▼
   ┌─────────────────────┐               ┌─────────────────────┐
   │  反向树 T_R          │               │  前向树 T_F          │
   │  从 goal 长向 start  │  ←─启发式──   │  从 start 长向 goal  │
   │  Lazy: 不做碰撞检测  │  ─碰撞反馈→  │  Eager: 真做碰撞     │
   └─────────────────────┘               └─────────────────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                ┌────────────────────────────────────┐
                │  前向树到达 goal? 输出当前最优解.  │
                │  Informed 收紧椭球, 撒下一批.      │
                └────────────────────────────────────┘
```

两棵树之间通过两条"通信线":

1. **反向树 → 前向树**：反向树中节点 `v` 到 goal 的最短路径长度，作为前向树用的 `h(v)`。
2. **前向树 → 反向树**：前向树发现某条边 `(u, w)` 撞墙时，告诉反向树。反向树**重新算**受影响节点的最短路径。

**协作的妙处**：每发现一次碰撞，启发式就变得更准，下次 A\* 就不再走那个方向。**自适应**就是这个意思。

---

## 四、反向树 T_R 详细：lazy，提供启发式

### 4.1 数据结构

```text
T_R: 节点是 V_samples 的子集, 边是这些节点之间的"虚拟连线"
     每条边只算欧氏距离, 不做碰撞检测 (lazy)
     根节点 = q_goal
```

### 4.2 构建步骤

```text
1. 初始化: T_R = { q_goal }, 给 q_goal 一个 g_R = 0 (从 goal 走到自己代价为 0)

2. 撒 N 个样本 V_samples (start 和 goal 也包含)

3. 对 V_samples 中每个节点 v, 找它在样本集里的 k 近邻 (k 由 PRM* 公式决定)
   形成隐式图 G

4. 在 G 上跑 Dijkstra (从 q_goal 出发):
   - 边代价 = 欧氏距离 (没碰撞检测!)
   - 给每个节点算 g_R(v) = T_R 中从 q_goal 到 v 的最短距离
   - 这个值就是前向树要用的 h_F(v)

5. 反向树暂停, 让前向树来取启发式
```

### 4.3 反向树的核心特性

- 每个节点 v 都有一个 `h_F(v) = g_R(v)`，这是"假装没障碍下从 v 到 goal 的最短距离"。
- 因为忽略障碍，**h 总是低估**或等于真实代价 → admissible，符合 A\* 的最优性要求。
- 一旦前向树报告"某条边碰撞"，反向树**对应的边代价变成 ∞**，重跑 Dijkstra → 受影响节点的 h 变大（更接近真实）。

---

## 五、前向树 T_F 详细：eager，找真实路径

### 5.1 数据结构

```text
T_F: 节点是 V_samples 的子集, 边经过碰撞检测 (eager)
     根节点 = q_start
     每个节点 v 有 g_F(v) = T_F 中从 q_start 到 v 的真实代价
     边队列 Q_E: 待评估的边 (按 f = g_F + h_F 排序)
```

### 5.2 工作步骤

```text
1. 初始化: T_F = { q_start }, g_F(q_start) = 0
   把 q_start 出发的所有候选边 (q_start, v) 入队 Q_E
   
2. 从 Q_E 取 f 值最小的边 (u, w):
   f((u, w)) = g_F(u) + dist(u, w) + h_F(w)
   
3. 三种情况:
   
   a) 这条边的 f >= 当前最优解的代价: 
      → 不可能更优, 整个 batch 结束, 撒下一批
   
   b) g_F(u) + dist(u, w) >= g_F(w) (现在已知 w 的更短路径): 
      → 这条边没用, 丢
   
   c) 否则 → 真做碰撞检测:
      ✓ 通过 → 加边到 T_F, 更新 g_F(w), 把 w 出发的边入队 Q_E
      ✗ 失败 → 这条边在反向树里也无效, 通知反向树!
              反向树重算受影响节点的 g_R, h_F 自动更新
              继续从 Q_E 取下一条边
```

### 5.3 找到解时

当前向树扩展到 q_goal（或近似），从 q_start 沿父节点回溯到 q_goal，得到当前路径。

---

## 六、关键：两树的协作机制（adaptive 的来源）

```text
                            ┌─────────────────┐
                            │  反向树 T_R     │
                            │  Dijkstra 跑出  │
                            │  h_F(v) for all │
                            └────────┬────────┘
                                     │
                              h_F(v) 给前向树
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  前向树 T_F     │
                            │  A* 用 f=g+h    │
                            │  取边 (u, w)    │
                            └────────┬────────┘
                                     │
                              碰撞检测 (u, w)
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                     ✓ 通过                ✗ 失败
                          │                     │
                          ▼                     ▼
                  T_F 加边               通知 T_R: (u, w) 不可用
                  扩张前向树              T_R 把这条边代价设 ∞
                                          重跑受影响节点的 Dijkstra
                                          → h_F 更新, 更准确
                                          → 前向树下次决策更对
```

### 一个具体例子说明 adaptive

```text
2D 场景, start 在左, goal 在右, 中间有 ┃ 墙.

第 1 次:
  T_R 用直线距离算: h_F(start) = 1.0 (笔直距离)
  T_F: A* 选朝右的边, 撞墙!
  通知 T_R, 把"穿墙"的边设 ∞.
  T_R 重新算: 现在到 goal 必须绕墙 → h_F(start) = 1.5

第 2 次:
  T_F: A* 看到 h_F(start) = 1.5, 只考虑值更低的方向 → 不再朝墙撞, 直接绕.
  效率显著提高.

第 3 次, 第 4 次:
  反复修正, h_F 越来越接近真实最优代价.
```

这就是"adaptively informed"——**自适应、被信息驱动**。

---

## 七、完整算法主循环

```text
batch_count = 0

while not converged:
    # === 撒样本 ===
    if 第一次:
        V_samples = 均匀撒 N 个点
    else:
        V_samples = informed 椭球内撒 N 个点  # BIT* 同款
    
    # === 反向树 ===
    在 V_samples 上跑 lazy Dijkstra (从 goal 出发)
    得到每个节点的 h_F(v)
    
    # === 前向树 ===
    清空 Q_E, 把 q_start 出发的边入队
    
    while Q_E 非空:
        取 f 最小的边 (u, w)
        if f >= c_best: break  # 不可能更优
        if g_F(u) + dist(u, w) >= g_F(w): continue  # 已有更短
        
        if collision_check(u, w) == OK:
            T_F.add_edge(u, w)
            g_F(w) = g_F(u) + dist(u, w)
            把 w 的出边入队 Q_E
            if w == q_goal:
                c_best = g_F(q_goal)
                继续优化或退出
        else:
            # 通知反向树
            T_R.invalidate_edge(u, w)
            T_R.repair()  # 局部重算 g_R 和 h_F
            # 边队列 Q_E 中相关边的优先级更新
            Q_E.update_priorities()
    
    batch_count += 1
```

---

## 八、完整例子：UAV 室内导航

### 场景

```text
俯视 5m × 5m 房间:

   ●(start, 1m,1m)              
                                
                   ┃            
                   ┃ 墙          
                   ┃ (3m, 1.5m 到 3m, 4m)
                                
                                    ●(goal, 4m, 4m)
```

### Batch 1（100 样本）

```text
反向树 T_R:
  Dijkstra 在 100 样本图上, 边权 = 欧氏距离.
  h_F(start) ≈ 4.2  (start 到 goal 的隐式图最短路径,
                     此时还没碰撞信息, 包括穿墙的边)

前向树 T_F:
  Q_E 取出 (start, v_1=(2,2)), f = 0 + 1.4 + 2.8 = 4.2
  碰撞检测: OK → 加入树, g_F(v_1) = 1.4
  
  Q_E 取出 (v_1, v_2=(3.2, 3)), f = 1.4 + 1.5 + 1.5 = 4.4
  碰撞检测: 边穿墙! 失败.
  → 通知 T_R 把 (v_1, v_2) 设 ∞, T_R 重算.
  → h_F(v_1) 从 2.8 升到 3.5 (因为现在必须绕墙)

  继续从 Q_E 取下一条边...
  最终找到一条绕墙路径 [start → v_1 → v_3 → ... → goal], c_best = 5.5
```

### Batch 2（再 100 样本，informed）

```text
椭球: 焦点 (start, goal), 长轴 = 5.5
新 100 点只在椭球内撒.

T_R 重新跑 Dijkstra (新边都 lazy 加入)
   - 因为之前有不少边被设 ∞, h_F 已经反映了墙的存在
   - 新撒的 100 点有些可能提供更短的绕墙路径

T_F:
   开始时启发式已经很准, A* 几乎不走错路
   找到改进解: c_best = 5.2
```

### 收敛

随着 batch 增加，c_best 单调下降，解收敛到真实最优。每个 batch 比 BIT\* 更快，因为启发式越来越准。

---

## 九、超精简伪代码

```python
def AITstar(q_start, q_goal, octomap, max_batches=20):
    V = [q_start, q_goal]
    T_R = ReverseTree(root=q_goal)         # lazy
    T_F = ForwardTree(root=q_start)         # eager
    c_best = inf
    
    for batch_id in range(max_batches):
        # === 撒样本 ===
        if c_best == inf:
            V += uniform_sample(N)
        else:
            V += informed_sample(N, q_start, q_goal, c_best)
        
        # === 反向树构建 (lazy Dijkstra) ===
        T_R.build_lazy(V, k=k_PRM(N))       # 不做碰撞检测
        h_F = {v: T_R.shortest_path_length(v, q_goal) for v in V}
        
        # === 前向树搜索 ===
        Q_E = PriorityQueue()
        for v in V:
            if connectable(q_start, v):
                Q_E.push((q_start, v), priority=h_F[v])
        
        while not Q_E.empty():
            (u, w), priority = Q_E.pop()
            f = T_F.g(u) + dist(u, w) + h_F[w]
            
            if f >= c_best:
                break  # 整个 batch 不可能更优
            
            if T_F.g(u) + dist(u, w) >= T_F.g(w):
                continue  # 没改进
            
            if collision_check(u, w, octomap):
                T_F.add_edge(u, w)
                if w == q_goal or near(w, q_goal):
                    c_best = T_F.g(q_goal)
                # 把 w 的出边入队
                for x in V:
                    if connectable(w, x):
                        Q_E.push((w, x), priority=T_F.g(w) + dist(w, x) + h_F[x])
            else:
                # 关键: 自适应启发式更新
                T_R.invalidate_edge(u, w)
                affected = T_R.repair()
                for v in affected:
                    h_F[v] = T_R.shortest_path_length(v, q_goal)
                Q_E.update_priorities(h_F)
    
    return T_F.path_to(q_goal)
```

---

## 十、与其他算法的对比

| 维度 | RRT-Connect | RRT\* | BIT\* | **AIT\*** |
|------|-------------|-------|-------|-----------|
| 双树 | ✓（仅相向相遇）| ✗ | ✗ | ✓（启发式协作）|
| 渐近最优 | ✗ | ✓ | ✓ | ✓ |
| Anytime | ✗ | ✓ | ✓ | ✓ |
| 启发式 | 无（贪心）| 无 | 直线距离（固定）| **自适应** |
| 边检测策略 | 立即 | 立即 | Lazy + A\* 引导 | Lazy + adaptive A\* |
| 障碍密集场景速度 | 中 | 慢 | 快 | **最快** |
| 实现复杂度 | 低 | 中 | 高 | **更高** |
| OMPL 名 | RRTConnect | RRTstar | BITstar | AITstar |

### 何时选 AIT\*

- **障碍多、空间复杂**：启发式自适应价值最大。
- **要求最优解 + 时间预算有限**：anytime + 收敛快。
- **高维（机械臂 6-DoF+）**：维度越高，启发式越关键。
- **离线规划或软实时**：单次求解时间稍长（建反向树有开销），不适合 100Hz 高频重规划。

### 何时不选 AIT\*

- **要求毫秒级出第一条路径**：用 RRT-Connect。
- **环境近乎无障碍**：BIT\* 够用，少一层复杂度。
- **算力极限**：AIT\* 维护两棵树，内存占用约 BIT\* 的 1.5 倍。

---

## 十一、超参清单

| 超参 | 含义 | 典型值 | 调大效果 | 调小效果 |
|------|------|--------|----------|----------|
| `N` | 每 batch 样本数 | 100–500 | 单 batch 慢, 解可能更早出 | batch 多, 总开销可能上升 |
| `k_PRM` | 每节点连接近邻数 | `k(N) = 2e·log(N)` | 图密, 内存大 | 图稀, 可能不连通 |
| `max_batches` | 最大 batch 数 | 20–50 | 解更优, 时间更长 | 早停, 解可能次优 |
| `step_resolution` | 边碰撞检测离散步长 | 0.01–0.05 m | 精确, 慢 | 快, 可能漏小障碍 |
| 椭球收缩 | informed 区域更新 | 自动（依 c_best）| — | — |

OMPL 默认值通常已经合理，主要调 `N` 和 `max_batches` 平衡速度与质量。

---

## 十二、对你机械臂场景的适配

如果你想用 AIT\* 做你的焊缝场景中的"段间运动规划"：

1. **节点 = 关节构型 θ ∈ ℝ⁶**，不是末端笛卡尔位置。
2. **碰撞检测**接 cuRobo：每条边沿插值做整臂 vs OctoMap 检测。
3. **距离度量**用关节空间加权 L2 或 Riemannian 度量，**不要**用末端笛卡尔距离（不准）。
4. **q_goal 不是单点**：是"满足合格观测条件的所有 viewpoint 集合"，需要 multi-goal AIT\*（OMPL 支持）。
5. **OctoMap 在线更新**：地图变化时，把变化区域的边在反向树里 invalidate，避免重头跑。

可以直接调 OMPL 的 `ompl::geometric::AITstar`，配上 cuRobo 的碰撞检测插件，几百行代码就能用。

---

## 十三、参考资料

- **原论文**：M. P. Strub & J. D. Gammell, *Adaptively Informed Trees (AIT\*): Fast Asymptotically Optimal Path Planning through Adaptive Heuristics*, ICRA 2020. <https://arxiv.org/abs/2002.06599>
- **改进版（含 EIT\*）**：M. P. Strub & J. D. Gammell, *AIT\* and EIT\*: Asymmetric bidirectional sampling-based path planning*, IJRR 2022.
- **前作 BIT\***：J. D. Gammell, S. S. Srinivasa, T. D. Barfoot, *Batch Informed Trees (BIT\*)*, ICRA 2015.
- **OMPL 文档**：<https://ompl.kavrakilab.org/classompl_1_1geometric_1_1AITstar.html>
- **作者实验室**：Estimation, Search, and Planning (ESP) Group, U. Oxford <https://robotic-esp.com/>

---

## 总结

```text
RRT-Connect:  双树相向相遇, 不最优, 但快
RRT*:         单树 + rewire, 最优但慢
BIT*:         批量样本 + A* + 直线启发式 + lazy → 快很多
AIT*:         BIT* + 反向树自适应启发式 → 障碍密集场景再快一截
```

AIT\* 是把 **"碰撞信息"**这种以前被丢掉的副产品**反哺给启发式**，让 A\* 越走越聪明。一句话：**它从撞过的墙学到怎么绕墙**。这是它和所有前辈的根本区别。
