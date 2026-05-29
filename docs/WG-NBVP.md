# WG-NBVP 算法详解（零基础版）

WG-NBVP（**W**eighted-**G**ain **N**ext-**B**est-**V**iew **P**lanner）是 Naazare 等人 2022 年发表在 IEEE RA-L 的算法，专门为**带机械臂的移动机器人 (mobile manipulator)** 设计。它针对一类特殊场景：

- 不只要建图（探索 exploration）
- 还要重点观察预先已知的"高重要度区"（巡检 inspection）

典型应用是化工厂泄漏点观察、核设施污染检查——已经先有一份大致的"哪里有污染"地图，机器人去把那些地方拍清楚，同时把整个环境也建出来。

> 论文：M. Naazare et al., *Online Next-Best-View Planner for 3D-Exploration and Inspection With a Mobile Manipulator Robot*, IEEE RA-L 2022.
> 代码：<https://github.com/fkie/fkie-nbv-planner>
> 演示视频：<https://youtu.be/nsJ_LCio0h0>

本文从零开始讲。前面已在 [DSVP.md](DSVP.md) 介绍过的概念会快速带过。

---

## 一、必须先懂的 5 个新概念

> 占据栅格、Raycasting、Frontier、Viewpoint、Information Gain 这 5 个基础概念已在 [DSVP.md 第一章](DSVP.md#一5-个必须先懂的概念) 详细解释，本节只讲新概念。

### 概念 1：移动机械臂 (Mobile Manipulator)

底盘（车）+ 机械臂的复合系统。

```text
           ┌──┐
           │  │← 相机 (eye-in-hand)
        ┌──┘  │
        │  机械臂
        │  │
   ┌────┴──┴────┐
   │   底盘     │ ← 在地面 SE(2) 移动
   └────────────┘
```

- **底盘**：在地面跑 (x, y, yaw)，慢、不太精确，启停代价大。
- **机械臂**：6 或 7 个关节，相对底盘运动，快、精确。
- **相机**：装在末端附近，叫 "eye-in-hand"。

**为什么不能像无人机当成自由 SE(3)？** 因为底盘和机械臂运动特性差太多——能用机械臂就别动底盘。

### 概念 2：兴趣区域 (ROI) 与强度图 I

WG-NBVP 与其他探索算法（DSVP/GBPlanner2/TARE）最大的输入差别——**它有先验信息**：

```text
强度图 I (3D 体素栅格):

  z=2: │  0   0   0   0   0   │
  z=1: │  0  10  50 100  20   │  ← 数值越大 = 越重要
  z=0: │  0   5  30  80  10   │
       └─────────────────────┘
            x →
```

每个体素 m_i 是一个标量（例如辐射强度、化学浓度）。来源是事先的人工或机器人快速踏勘。机器人会**优先把相机对准高强度体素**。

ROI = 强度图中数值高的连通区域。

### 概念 3：多目标优化 (MOO) 与加权求和

机器人探索同时要满足多个目标：

1. 多扫新空间（**G_f**：自由空间增益）
2. 多看高重要度区（**G_m**：测量增益）
3. 别重复看同一个地方（**G_v**：访问惩罚）

这些目标会冲突。MOO 的处理方式是**加权求和**：把多个评分按权重叠加成一个总评分。

```text
G(q) = w_f · G_f(q) + w_m · G_m(q) + w_v · G_v(q)
```

权重不同 → 行为不同：

| 权重 | 行为 |
|------|------|
| w_m 大、w_f 小 | 优先扫 ROI（inspection 模式） |
| w_f 大、w_m 小 | 优先扫地图（exploration 模式） |
| w_v 大 | 严格不重复（防止原地打转） |

### 概念 4：传感器位姿空间 vs 关节构型空间

机械臂规划有两种空间：

- **关节构型空间**：直接对 6/7 个关节角规划，维度 = 关节数。
- **传感器位姿空间 (SE(3))**：对相机的 (位置, 朝向) 规划，由 IK 反推关节角。维度 = 6。

WG-NBVP 选**传感器位姿空间**：

- 信息增益只取决于相机看哪儿，跟关节怎么摆没关系。
- 同一个相机位姿可能由多组关节角实现（IK 多解），不必固定。
- 维度更小、搜索更快。

代价：要做 IK 验证，确认这个相机位姿确实有合法关节解、并且整条手臂不撞墙。

### 概念 5：Approach 动作（够不到就尽量靠近）

如果"最佳 viewpoint" q\* 当前底盘位置 + 机械臂**根本伸不到**：

- **选项 A**（笨）：开车过去，重新规划 base 路径。慢。
- **选项 B**（聪明）：先用机械臂朝 q\* 方向尽量伸长，**到不了 q\* 但能到 q\* 附近的某个 q'**。q' 处的视角已经有部分增益。

WG-NBVP 优先选项 B：先伸臂；伸不动了，再开车。这是 mobile manipulator 特有的策略。

---

## 二、WG-NBVP 要解决什么问题

> **给定**：移动机械臂、未知环境、相机装在臂末端、先验强度图 I。
> **求**：边走边建图，**优先扫高重要度区**，整个过程**少开车、多动臂**、**少重复**。
> **输出**：实时增长的 OctoMap + ROI 已扫过的体素。

### 与前三个算法的核心差异

| 维度 | DSVP / GBPlanner2 / TARE | WG-NBVP |
|------|--------------------------|---------|
| 平台 | UAV / 地面机器人 | UGV + 机械臂 |
| 输入先验 | 只有空地图 | 地图 + ROI 强度图 |
| 评分目标 | 单一信息增益 | 多目标加权（探索 + 巡检 + 去重）|
| 搜索空间 | 3D 位置 / 关节空间 | 传感器位姿 SE(3) |
| 运动决策 | 单一执行器 | 臂优先 → 必要时 base |

---

## 三、整体循环结构（Algorithm 1）

每个 cycle t 做以下事：

```text
                    ┌─────────────────────────────────────────────┐
                    │         开始 cycle t                         │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │ 1. 沿上 cycle 计划的位姿序列 Q_t 走         │
                    │    每个位姿处用相机拍 RGB-D                  │
                    │    更新 OctoMap → M_{t+1}                    │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │ 2. 在 M_{t+1} 已知 free 中长一棵 RRT          │
                    │    起点 = 当前相机位姿 q_t                   │
                    │    候选 = 半球壳采样的 viewpoint              │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │ 3. 对每个节点算加权 G                         │
                    │    G = w_f·G_f + w_m·G_m + w_v·G_v           │
                    │    选最大的 q_{t+1}                          │
                    └─────────────────┬───────────────────────────┘
                                      │
                            G(q_{t+1}) > g_min ?
                                      │
                              ┌───────┴───────┐
                              │ Yes           │ No
                              ▼               ▼
                ┌─────────────────────┐    finished = True
                │ 4. 取 RRT 上从 q_t  │    (没什么再值得探的)
                │    到 q_{t+1} 的分支│
                │    抽出"机械臂可达" │
                │    的最长前缀 R      │
                └─────────┬───────────┘
                          │
                    R 非空 ?
                          │
                  ┌───────┴────────────┐
                  │ Yes                │ No
                  ▼                    ▼
           ┌──────────────┐  ┌─────────────────────────┐
           │ Q_{t+1} = R  │  │ 5. 开车: 把 q_{t+1}     │
           │ 只动臂       │  │    投到地面给底盘导航   │
           └──────┬───────┘  │    Q_{t+1} = approach   │
                  │          └────────────┬────────────┘
                  └────────────┬──────────┘
                               ▼
                       cycle t+1 开始
```

四个关键模块（步骤 2、3、4/5）的细节在下面分述。

---

## 四、模块详细：RRT 扩展（半球面采样）

### 4.1 起点和搜索空间

- **起点**：当前相机位姿 q_t（SE(3)）。
- **搜索空间**：以**机械臂底座**为中心、半径稍大于臂最大可达（论文中 1.3 m）的**球壳**。地面机器人通常只用上半球。
- **限制**：节点位置必须在 OctoMap 的已知 free 体素中，否则丢弃。

```text
俯视图:

          球壳采样区
        ┌───────────┐
       /  ●  ●   ●   \    每个 ● = 候选 viewpoint
      /    ●  ●  ●    \   都在球壳上
     / ●           ●   \
    |    [机械臂底座]   |  ← 球心
     \                 /
      \  ●    ●   ●   /
       \_____________/

特点: 候选 viewpoint 总在臂"够得到"的距离上.
```

### 4.2 单次采样的 4 步（每个 cycle 重复 N_max = 600 次）

#### 4.2a 在球壳上均匀采样 q_rand 的位置

只采位置，不采朝向——朝向单独算（4.2c）。

#### 4.2b 找最近邻 q_near

```text
q_near = RRT 中现有节点里离 q_rand 位置最近的那个
```

#### 4.2c 算新节点 q_new 的位置 + 朝向

位置：

```text
q_new.position = q_near.pos + l · (q_rand − q_near.pos) / ‖q_rand − q_near.pos‖
```

`l` 是固定步长（论文 0.5 m）。

**朝向（关键设计）**——根据模式不同：

- **Inspection 模式**：朝向 = 强度图 I 在 q_new 处的**正梯度方向**（指向更高强度）。
- **Exploration 模式**：朝向 = 在 q_new 处尝试多个候选朝向，选 G_f 最大的一个。

> 这样保证相机始终"望向有用方向"，而不是随机看。这是 WG-NBVP 比纯随机采样更高效的关键。

#### 4.2d 三重检查（决定是否加入 RRT）

1. **q_near.pos → q_new.pos 直线段全在已知 free 内**（碰撞检测过）。
2. **q_new 不与任何现有节点过近**（间距 < d 则丢，避免冗余）。
3. **q_new 在 exploration_bounds 内**（不要跑出指定区域）。

满足全部 → 加入 RRT；否则丢，下次采样。

### 4.3 缓存节点 + 变量阈值（escape local minima）

只看当前 cycle 的 RRT 容易在局部转圈。WG-NBVP 借鉴 [Witting 2018, Selin 2019] 引入跨 cycle 缓存：

```text
缓存集 C:
  - 每 cycle 把 RRT 中得分高的若干节点存入 C
  - 下个 cycle 重新评估 C 里每个节点的当前 G (地图变了, gain 也变)
  - 只保留 top-k 个高分节点
  - 把这 top-k 的最低分作为下个 cycle 的 g_min 阈值
```

效果：

- 历史好节点不会被遗忘。
- **g_min 自适应**：地图越接近探完，剩下高 gain 节点越少，g_min 自动上升，算法变得更挑剔，最后到处都没东西可看时自然终止。

> 注意：缓存集 C（用于阈值自适应）和"已访问位姿集合 Q̂"（用于 G_v 重复惩罚）是**两个不同的东西**，不要混淆。

---

## 五、模块详细：加权求和信息增益

每个候选 viewpoint q 的总分：

```text
G(q, M, I) = w_f · G_f(q, M)         ← 探索新空间
           + w_m · G_m(q, I)         ← 巡检高重要度区
           + w_v · G_v(q, Q̂)         ← 去重 (惩罚已访问)
```

### 5.1 G_f：自由空间增益（探索）

模拟相机在 q 处的视锥发射射线，**累加射线穿过的 unknown 体素体积**。

公式：

```text
gdV(r,θ,φ) = dV(r,θ,φ)   if M(r,θ,φ) is unknown
           = 0            otherwise

dV(r,θ,φ) = (2r²·Δr + Δr³/6) · Δθ · sin(φ) · sin(Δφ/2)

         ψ+fov_θ/2  fov_φ/2  max_or_hit
G_f =      ∑           ∑         ∑       gdV(r,θ,φ)
       θ=ψ-fov_θ/2  φ=-fov_φ/2  r=0
```

直观理解：

```text
相机 ──●─?─?─?─?─?─■   一束射线
       └─────────┘
        穿过 5 个 unknown 体素 → +5 个 dV 累加
        (越远 dV 越大, 因为视锥截面越大)
        撞到 ■ (occupied) 就停止
```

举例：

- viewpoint A 朝盒子开口看：射线穿 2 m unknown → G_f 很大。
- viewpoint B 朝墙看：射线立即撞 occupied → G_f ≈ 0。

### 5.2 G_m：测量增益（巡检）

```text
G_m(q, I) = I(q.position)
```

直接读 q 位置在强度图 I 中的值：

- q 在化学桶旁 → I = 100 → G_m = 100。
- q 在远离 ROI 的草地 → I = 0 → G_m = 0。

> 这里只看相机所在体素的 I 值，不是看相机能看到的体素的 I 值。这是论文的简化处理。实际工程上可扩展为视锥内 I 值积分（更合理但更慢）。

### 5.3 G_v：访问惩罚（去重）

```text
G_v(q, Q̂) = -1   if q ∈ 历史已访问 Q̂
           = 0    otherwise
```

权重 w_v 通常很大（论文用 500），等于"已访问的 viewpoint 直接被扣 500 分" → 几乎不可能再被选中。

### 5.4 论文使用的权重组

| 模式 | w_f | w_m | w_v |
|------|------|------|------|
| Exploration（纯探索） | 5.0 | 0.0（关闭）| 500 |
| Inspection（探索+巡检）| 1.0 | 5.0 | 500 |

Inspection 模式下 `w_m / w_f = 5`，强烈偏向高强度区。

---

## 六、模块详细：臂优先、底盘最后

### 6.1 决策树

每 cycle 选出最佳 viewpoint q_{t+1}（RRT 中 G 最大的）后：

```text
if G(q_{t+1}) >= g_min:
    branch = RRT 中从 q_t 到 q_{t+1} 的边序列 = [q_t, q_a, q_b, ..., q_{t+1}]
    
    R = 这个分支中"底盘不动, 只用机械臂能伸到"的最长连续前缀
    
    if R 非空:
        # 优先方案: 只动臂
        Q_{t+1} = R     # 下个 cycle 顺次访问 R 中的每个位姿
    else:
        # 备用方案: 开车
        Q_{t+1} = [approach(q_{t+1}, M)]    # 让车开过去 + 臂朝 q_{t+1} 方向
else:
    finished = True
```

### 6.2 "可达前缀" R 怎么算

对分支上每个 q_i（按 q_t → q_{t+1} 顺序）：

```text
1. IK(q_i) → 求出关节角 θ_i (失败 → 截断)
2. 用 θ_i 在当前 OctoMap M 上做整条手臂碰撞检测 (失败 → 截断)
3. 通过 → R.append(q_i), 检查下一个 q_{i+1}
```

如果**第一个 q_a 就不可达** → R 为空 → 必须开车。

### 6.3 为什么这套机制有用（实验数据）

论文 Table V：

| 配置 | 60 min ROI 探了 | 60 min 总跑动距离 |
|------|----------------|-------------------|
| Stationary arm（不动臂只开车）| 50.2% | 309.6 m |
| **Mobile arm（臂优先）** | **82.6%** | **167.1 m** |

**多动臂的版本探得更多、跑得更少**。原因：

- 动臂比开车快得多（孔径角变化几十度只要 1 秒）。
- 一次开车的时间能换 50–100 次臂动。
- 底盘转向、启停、加减速代价很大。

---

## 七、完整循环例子

设：机器人开车到化工厂门口，前方 3 m 有几个化学桶（已知有泄漏，强度图 I 在桶处值大）。

### Cycle 0（初始化）

```text
Q_0 = [q_0]       # 开机时相机位姿
M_0 = 空 OctoMap
缓存 C = {}
访问 Q̂ = {q_0}
```

### Cycle 1

```text
1. 在 q_0 拍 RGB-D, 更新 M_1: 看到 4 m 范围内大致空旷, 化学桶是 occupied 的一团.
2. 在 q_0 周围长 RRT, 600 次采样:
   - 朝化学桶方向: 候选 v_1, v_2, ... 朝向用 I 的梯度 → 都瞄准桶
   - 朝侧面无污染区: 候选 v_a, v_b 朝向用 max-G_f → 朝外看
3. 算 G (Inspection 权重 w_f=1, w_m=5, w_v=500):
   - v_1: G = 1·30 + 5·100 + 500·0 = 530   (朝桶, I 值 100)
   - v_a: G = 1·45 + 5·5   + 500·0 = 70    (朝外, I 值 5)
4. q_1 = v_1, G(q_1)=530 > g_min=2.
5. 分支 q_0 → v_1 中: q_a (中间节点) IK 通过, v_1 IK 失败.
   R = [q_a]
6. Q_2 = [q_a], 下 cycle 让臂伸到 q_a.
7. 把 v_1, v_a 的 RRT 节点入缓存 C, 重算下个 cycle 的 g_min.
```

### Cycle 2（继续，只动臂）

```text
1. 臂走到 q_a, 拍照, 更新 M_2: 看到桶后面一小段地面, 桶的边缘细节也补全.
2. 重新评估缓存 C: 
   - v_1 现在更近, IK 可能可解了; G_m 不变 (I 是先验), G_f 略降.
3. 长新 RRT, 起点是 q_a. 半球壳已经偏移到 q_a 附近.
4. 选最佳 q_2: 可能是 v_1 (因为 v_1 现在 IK 可达).
5. R = [v_1] (只一步), 执行.
```

### Cycle 5（开车触发）

```text
机器人现在末端探完了化学桶正面, 想看背面, 但臂伸到极限也碰不到.

1. RRT 中 q_5* 是桶后方的 viewpoint, G 最高.
2. 分支 q_t → q_5* 的可达前缀 R = 空 (第一个中间节点 IK 就失败).
3. 走备用方案:
   approach_pose = 把 q_5* 投影到地面 + 安全距离, 比如 (x_桶后, y_桶后, 0).
   teb_local_planner 接管, 开车过去.
4. 到达后臂朝 q_5* 方向摆好, Q_6 = [approach_pose].
5. cycle 6 开始重新长 RRT.
```

### 终止条件

```text
某 cycle, RRT 最佳节点 max G < g_min, 缓存 C 中也没有 G > g_min 的节点.
→ finished = True, 输出最终 OctoMap M_T.
```

---

## 八、超精简伪代码

```python
def WG_NBVP(q_0, intensity_map_I, mode='inspection'):
    M = OctoMap()
    cache = {}                     # 缓存: 高分 RRT 节点
    Q = [q_0]                       # 上 cycle 规划的位姿序列
    visited = {q_0}                 # 历史已访问 (用于 G_v)
    finished = False
    g_min = 2.0                     # 初始阈值
    
    if mode == 'inspection':
        w_f, w_m, w_v = 1.0, 5.0, 500
    else:                            # 'exploration'
        w_f, w_m, w_v = 5.0, 0.0, 500
    
    while not finished:
        # ===== Step 1: 执行上 cycle 序列 + 建图 =====
        for q in Q:
            depth = capture(q)
            M.integrate(depth, q)
        q_now = Q[-1]
        visited.update(Q)
        
        # ===== Step 2: 长 RRT =====
        rrt = RRT(root=q_now)
        for _ in range(N_MAX):
            q_rand_pos = sample_in_hemisphere(arm_base, R_max)
            q_near = rrt.nearest_position(q_rand_pos)
            q_new_pos = step_toward(q_near.pos, q_rand_pos, l)
            
            if not in_free(q_new_pos, M): continue
            if not edge_collision_free(q_near.pos, q_new_pos, M): continue
            if too_close(q_new_pos, rrt, d): continue
            
            # 朝向: 模式相关
            if mode == 'inspection':
                ori = positive_gradient(intensity_map_I, q_new_pos)
            else:
                ori = max_Gf_orientation(q_new_pos, M)
            q_new = SE3(q_new_pos, ori)
            rrt.add_node(q_new, parent=q_near)
        
        # ===== Step 3: 评分 + 缓存更新 =====
        for q in rrt.nodes:
            q.G = w_f*G_f(q, M) + w_m*G_m(q, I) + w_v*G_v(q, visited)
        # 把 RRT 高分节点合并入缓存
        merge_cache(cache, rrt.top_k_nodes(K_CACHE))
        # 重算缓存中旧节点的 G (地图变了)
        for q in cache:
            q.G = w_f*G_f(q, M) + w_m*G_m(q, I) + w_v*G_v(q, visited)
        cache = top_k(cache, K_CACHE)
        g_min = min(q.G for q in cache)
        
        # ===== Step 4: 选最佳 NBV =====
        q_next = max(rrt.nodes, key=lambda q: q.G)
        if q_next.G < g_min:
            finished = True
            break
        
        # ===== Step 5: 决定臂动还是车动 =====
        branch = rrt.path_from_root_to(q_next)
        R = []
        for q_i in branch[1:]:
            theta = inverse_kinematics(q_i)
            if theta is None or arm_collides(theta, M): break
            R.append(q_i)
        
        if R:
            Q = R                    # 只动臂
        else:
            ground_goal = project_to_ground(q_next)
            drive_to(ground_goal)    # 开车
            Q = [approach_pose(q_next, M)]
    
    return M
```

---

## 九、对你焊缝场景的适配

WG-NBVP 与你焊缝观测场景的对应关系：

| 论文场景 | 你的焊缝场景 |
|----------|-------------|
| 移动机器人 + 7-DoF 臂 | 固定底座 + 6-DoF UR12e |
| 污染强度图 I | "焊缝重要度图"——P_weld 附近高，远处低 |
| ROI = 化学桶 | ROI = 焊缝中点周围小球壳 |
| 探索 25×25 m 区域 | 探索 1–2 m 工件附近 |
| 臂优先, 必要时开车 | **底座不动, 完全省掉 base motion** |

### 关键改动

1. **删除 base motion 模块**：你的底座固定，`approach()` 函数也不需要——`R` 为空就直接报"不可达"，跳过该 NBV。
2. **强度图 I 改成目标软球**：`I(p) = max(0, R - ‖p - P_weld‖)`，越靠近焊缝值越高。或用 Gaussian。
3. **半球壳采样 → 工作球**：以臂底座为中心、半径 0.85 m 的球（UR12 可达），所有 IK 不可解的位姿直接丢。
4. **G_m 升级**：原版 `G_m = I(q.pos)` 看的是相机所在体素，对你不太合理（你想要的是"相机能不能看到 P_weld"）。改成"视锥内 P_weld 周围 ROI 体素的可见数"，更贴合任务。
5. **w_m 调高到 50–200**：让"找焊缝"主导决策，比纯探索快得多。
6. **碰撞检测**：用 cuRobo（你已经在工程里有），把 OctoMap 喂进去做整臂碰撞过滤。

### 与四种算法的横向对比

| 算法 | 适配难度 | 主要特长 | 适合你的场景? |
|------|----------|----------|----------------|
| DSVP | 低（1 周） | 简单, 好上手 | △（缺目标先验）|
| GBPlanner2 | 中（2–3 周）| 半封闭/管腔最稳 | △（缺目标先验，但适合管腔）|
| TARE | 高（3–4 周）| 大空间高覆盖 | ✗（杀鸡用牛刀）|
| **WG-NBVP** | **中（2 周）** | **目标驱动 + 多目标加权** | **✓（最贴合）**|

WG-NBVP 是这四个里**最匹配焊缝场景**的：

- 它本来就有"先验目标位置"（ROI），直接对应你 `P_weld`。
- 你不需要 base motion 部分 → 实际实现量比论文版小一半。
- 多目标加权能直接把"避障 + 探索 + 看目标"统一在一个评分里。

---

## 十、超参清单

| 超参 | 含义 | 论文值 | 焊缝场景建议 |
|------|------|--------|--------------|
| `l` | RRT 步长 | 0.5 m | 0.05–0.1 m（工件小）|
| `N_max` | RRT 最大节点 | 600 | 100–300 |
| 球壳半径 | 采样球半径 | 1.4 m（>1.3 max reach）| 0.9 m（>0.85 max reach）|
| `d` | 节点最小间距 | < l | 0.02 m |
| FOV | 相机视场角 | 86° × 57° | 你相机参数 |
| `r_max` | 深度有效距离 | 1.5 m | 2.0 m |
| `w_f` | 自由空间权重 | 5（explore）/ 1（inspect）| 1.0 |
| `w_m` | 测量权重 | 0（explore）/ 5（inspect）| 50–200 |
| `w_v` | 重复访问惩罚 | 500 | 500 |
| `K_CACHE` | 缓存节点数 | 不详 | 30–50 |
| `g_min` | 终止阈值 | 自适应 | 自适应 |

---

## 十一、与四种算法的总览

| 维度 | DSVP | GBPlanner2 | TARE | WG-NBVP |
|------|------|------------|------|---------|
| 数据结构 | RRT* 树（重置）| 全局图（带环）| viewpoint 网格 + subspace | 单 RRT + 缓存集 |
| 目标先验 | ✗ | ✗ | ✗ | ✓（强度图 I）|
| 多目标加权 | 单目标 | 单目标 | 单目标 | ✓（3 项加权）|
| 局部最优逃脱 | 重置 RRT | Global 阶段 | Subspace TSP | 缓存 + 自适应 g_min |
| 平台 | UAV/UGV | UAV/UGV | UGV | UGV + 机械臂 |
| 实现复杂度 | 低 | 中 | 高 | 中 |
| 开源 | ✓ | ✓ | ✓ | ✓ |
| 适合焊缝场景 | △ | △ | △ | ✓ |

---

## 十二、参考资料

- **原论文**：M. Naazare, F. Garcia Rosas, D. Schulz, *Online Next-Best-View Planner for 3D-Exploration and Inspection With a Mobile Manipulator Robot*, IEEE RA-L 7(2), 2022.
- **DOI**：<https://doi.org/10.1109/LRA.2022.3146558>
- **代码仓库**：<https://github.com/fkie/fkie-nbv-planner>
- **演示视频**：<https://youtu.be/nsJ_LCio0h0>
- **核心引用工作**：
  - Bircher 2016 *Receding Horizon NBV* (RH-NBVP)：RRT 长树 + 单目标信息增益的鼻祖。
  - Selin 2019 *AEP*：本文 baseline，加了 frontier-based 全局策略。
  - Witting 2018 *History-aware Exploration*：缓存节点思想来源。
- **多目标优化基础**：R. T. Marler & J. Arora, *The weighted sum method for multi-objective optimization: New insights*, Struct. Multidisciplinary Optim. 41(6), 2010.
