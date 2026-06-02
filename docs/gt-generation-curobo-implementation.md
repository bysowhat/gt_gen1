# 基于 cuRobo 的 GT 生成实现流程 

本文档给出用 **cuRobo** 落地 [`gt-generation-design.md`](./gt-generation-design.md)
的具体步骤与代码骨架。

> 阅读顺序：先读设计文档（问题定义、保守探索的难点、特权专家思想），本文是其
> cuRobo 实现版。
>
> ⚠️ **API 版本提示**：cuRobo 的类名 / 方法签名随版本演进，下文凡涉及**精确
> 调用**处均标注「【按版本核对】」。**流程与数据流是确定的**，照此实现即可，
> 具体函数名以你安装的 cuRobo 版本文档为准。

---

## 0. 职责边界（再次明确）

| 层 | 谁负责 |
|----|--------|
| 建图、IK、无碰撞轨迹规划、碰撞检测、轨迹平滑 | **cuRobo 现成** |
| **未知=障碍** 的体素世界同步 | 你写（cuRobo 提供 voxel 世界接口） |
| **探索大脑**：主循环 / 前沿·NBV / 主动环顾 / 特权专家 / GT 导出 | **你写**（cuRobo 不提供） |

本文重点是把「你写」的部分串成可落地的流程。

---

## 1. 环境与前置

- **硬件**：NVIDIA GPU（CUDA）。
- **安装**：cuRobo（含 torch / warp 依赖）；如走 nvblox 路线再装 nvblox。
- **仿真/真值源**：Isaac Sim（推荐，渲染深度 + 提供真值场景）或你自有的
  带真值网格的仿真器。
- **机器人描述**：URDF + cuRobo 的 robot 配置（`*.yml`，定义碰撞球近似、关节
  限位、末端 link、相机相对末端的外参）。
- **关键标定**：**手眼外参**（相机相对末端 link 的变换）必须正确写入配置，
  否则观测投影错位。

---

## 2. 数据结构

### 2.1 三态体素图（你维护，核心）
在一个**有界 ROI 工作区**内：

```
voxmap[i,j,k] ∈ { FREE, OCCUPIED, UNKNOWN }，初始全 UNKNOWN
```

- 选定 ROI 包围盒（覆盖机臂可达 + 目标区域）与体素分辨率（算力足可取细，
  如 1~2 cm）。
- 这是「悲观约束」的载体：**喂给 cuRobo 的碰撞世界 = 所有非 FREE 体素当占据**。

### 2.2 真值场景（特权专家用，不参与移动约束）
完整真值网格/体素，仅用于：① 模拟相机观测（raycast）；② NBV 信息增益评估；
③ 最终碰撞校验。

---

## 3. cuRobo 初始化

用 **VOXEL 碰撞世界**（不用 BLOX/nvblox），因为悲观「未知=障碍」需要你完全
掌控每个体素占据状态，voxel 世界最直接。

```python
# 【按版本核对】导入路径与类名
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig,
)
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

tensor_args = TensorDeviceType()
robot_cfg = load_yaml(join_path(get_robot_configs_path(), "my_arm.yml"))["robot_cfg"]

# 初始世界：一个覆盖 ROI 的空 voxel 网格（初始全部当占据=UNKNOWN）
world_cfg = WorldConfig()   # 【按版本核对】voxel 网格的构造方式

motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg, tensor_args,
    collision_checker_type=CollisionCheckerType.VOXEL,   # 关键：体素碰撞
    interpolation_dt=0.02,                               # 输出轨迹时间步
    # 还可设：碰撞激活距离、轨迹优化迭代数、种子数等
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()                                      # 预热（编译 kernel）
```

> cuRobo 的轨迹**自带速度/加速度/jerk 限制与平滑**，所以设计文档里的「后处理
> 平滑 + 时间参数化」这一步**省掉**——`plan_single` 的结果直接是可执行轨迹。

---

## 4. 两个基础原语（你写）

### 4.1 观测并更新三态图
```python
def observe_and_update(voxmap, cam_pose, truth_scene):
    # 从相机位姿对真值场景做 raycast（带 FOV / 量程）
    rays = sample_camera_rays(cam_pose, intrinsics, max_range)
    for ray in rays:
        hit, t = raycast(truth_scene, ray)
        mark_FREE(voxmap, along ray until t)     # 视线穿过的空体素 → FREE
        if hit:
            mark_OCCUPIED(voxmap, point_at(ray, t))
    # 其余仍为 UNKNOWN
```

### 4.2 把三态图同步进 cuRobo 碰撞世界（桶 C，核心接入点）
```python
def sync_collision_world(motion_gen, voxmap):
    # 悲观：非 FREE（OCCUPIED ∪ UNKNOWN）= 占据
    occ = (voxmap != FREE)
    # 【按版本核对】通过 motion_gen.world_coll_checker 更新 voxel 占据/ESDF
    motion_gen.world_coll_checker.update_voxel_data(occ, ...)
    # 或重建 WorldConfig 后 motion_gen.update_world(world_cfg)
```

> 这是 cuRobo 与 MoveIt 的唯一接入差异点：voxel 世界让你**逐体素**设占据，
> 实现「未知=障碍」干净直接。

---

## 5. 探索大脑原语（你写）

### 5.1 主动环顾（破眼在手死锁）
```python
def look_around(motion_gen, config, voxmap, truth_scene):
    # 在当前确认自由区内，生成一组小幅腕/关节摆动候选
    for view_cfg in small_wrist_sweeps(config):
        # 用 cuRobo 规划到该候选，确认整条运动留在 FREE 区
        res = plan_to_config(motion_gen, config, view_cfg)
        if res.success:
            config = view_cfg
            observe_and_update(voxmap, camera_pose(config), truth_scene)
            sync_collision_world(motion_gen, voxmap)
            record(config)
    return config
```

### 5.2 特权 NBV：用真值选下一最优视点

> 📖 **完整详述见 [`privileged-nbv.md`](./privileged-nbv.md)**（oracle-path 打分、
> 候选生成、边界处理、正当性分析）。下面仅为骨架。

```python
def best_next_view_using_oracle(motion_gen, config, voxmap, truth_scene, goal_pose):
    best, best_gain = None, -1
    for cand in sample_reachable_views(config, voxmap):   # 候选须在 FREE 区可达
        # 用真值精确算：该视点能把多少「通往目标走廊」的未知体素揭成已知
        gain = expected_revealed_toward_goal(cand, voxmap, truth_scene, goal_pose)
        if gain > best_gain and plan_to_config(motion_gen, config, cand).success:
            best, best_gain = cand, gain
    return best
```

### 5.3 规划封装（调 cuRobo）
```python
def plan_to_pose(motion_gen, start_cfg, goal_pose):
    start = JointState.from_position(to_tensor(start_cfg))
    result = motion_gen.plan_single(
        start, goal_pose,
        MotionGenPlanConfig(max_attempts=K, enable_graph=True),  # 【按版本核对】
    )
    return result   # result.success / result.get_interpolated_plan()

def plan_to_config(motion_gen, start_cfg, goal_cfg):
    # 关节空间到关节空间：可用 IK 反推 pose，或用 cuRobo 的关节目标规划接口
    ...  # 【按版本核对】是否有 plan_single 的 joint-goal 形式
```

---

## 6. 主循环（GT 生成，你写）

```python
def generate_gt(truth_scene, goal_pose, retract_config):
    voxmap = init_unknown_voxmap(ROI, resolution)
    motion_gen = init_curobo(...)                       # §3

    config = retract_config                              # 起点=已知自由
    traj = [config]
    observe_and_update(voxmap, camera_pose(config), truth_scene)
    sync_collision_world(motion_gen, voxmap)

    # 目标位姿→关节角（cuRobo IK，碰撞感知）
    goal_config = solve_ik(motion_gen, goal_pose)        # 【按版本核对】IKSolver

    while True:
        # 1) 主动环顾，扫开周边未知
        config = look_around(motion_gen, config, voxmap, truth_scene); traj += ...

        # 2) 尝试在已确认自由区内直达目标
        res = plan_to_pose(motion_gen, config, goal_pose)
        if res.success:
            path = res.get_interpolated_plan()
            assert_no_collision_against_truth(path, truth_scene)   # 校验
            traj += path; break                          # 完成

        # 3) 到不了 → 特权 NBV 选子目标，移过去再观测
        nbv = best_next_view_using_oracle(motion_gen, config, voxmap,
                                          truth_scene, goal_pose)
        if nbv is None:
            handle_stuck(); continue                     # 扩大环顾/标记不可行
        sub = plan_to_config(motion_gen, config, nbv)
        config = sub.path[-1]; traj += sub.path
        observe_and_update(voxmap, camera_pose(config), truth_scene)
        sync_collision_world(motion_gen, voxmap)

    return export_gt(traj)
```

**大障碍绕行自动发生**：障碍近面标 OCCUPIED → cuRobo 轨迹优化在确认自由区内
绕开 → 边探索边完善绕行路径。无局部极小死磕。

---

## 7. GT 导出

每条轨迹记录（对齐项目现有格式）：
- 关节角序列 + 时间戳（cuRobo 输出含 `interpolation_dt`）；
- 可选：每步的相机位姿、当时的 voxmap 快照（若模型需要观测输入）；
- 起点 `retract_config`、目标 `goal_pose`、是否成功、总步数/时长。

> 与项目记忆一致：起点用固定 `retract_config`；目标位姿对应「焊接到位姿」，
> 不是 pkl 里的初始姿态。

---

## 8. 批量产 GT（cuRobo 的主场）

GT 数据集 = 很多场景 × 每场景一条轨迹，规划调用量巨大。利用 cuRobo 的
**GPU batched** 能力提吞吐：

- **多种子并行**：单次规划本就并行多种子，提升成功率与轨迹质量。
- **跨场景并行**：在数据生成脚本层面并行多个场景实例（受显存约束）。
- 因不要求实时，可把 `max_attempts`、优化迭代数调大，**换更高质量 GT**。

---

## 9. 待确认参数（落地前拍板）

- [ ] **旋钮 A**：自由约束作用于**整臂本体**还是**仅末端/工具**
      → 决定碰撞球配置与 §4.2 占据判定（**最关键，先定**）。
- [ ] ROI 包围盒范围 + 体素分辨率。
- [ ] 相机内参 / FOV / 量程（决定每步揭开范围与 NBV 评估）。
- [ ] 「到达目标」判据（位姿误差阈值）与接近段约束。
- [ ] `look_around` 的摆动幅度与候选数；`handle_stuck` 策略。
- [ ] cuRobo 版本 → 核对 §3~§6 标注的 API。

---

## 10. 与设计文档的对应关系

| 设计文档（§） | 本文实现 |
|---|---|
| §5 主循环骨架 | §6 主循环（cuRobo 版） |
| §6.1 未知=障碍 | §4.2 `sync_collision_world` + VOXEL 碰撞世界 |
| §4 特权专家 | §5.2 `best_next_view_using_oracle` |
| §3-B 主动环顾 | §5.1 `look_around` |
| §7 技术栈 | §3 cuRobo 初始化 + §8 批量 |
