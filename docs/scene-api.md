# API 重组：`Scene` 状态类 + `SceneVisualizer` 可视化类

> 把散落在 `scripts/` 的编排逻辑收敛成两个可编程对象：`Scene`（持有当前世界 + 机械臂 + 焊缝
> 状态）和 `SceneVisualizer`（用 open3d / isaacsim 可视化 Scene）。
>
> **本文档随 API 增量更新。** 当前是**第一阶段**：只落地 `Scene` 的构造 + `plan_init_pose`，
> 以及 `Open3DSceneVisualizer.show_init_poses`；其余字段/方法已占位，后续补充。
>
> **核心原则**：只做「API 方向的结构包装」，**不改各底层模块/脚本本身在做的事**。
> 例如 `Scene.plan_init_pose` 只是把 `scripts/plan_init_pose.py:InitPoseLookupSolver` 的求解
> 流程包成一个方法（输入/输出/算法完全一致），不复制、不改动其几何逻辑。

---

## 1. `Scene`（`gt_gen/scene.py`）

一个 `Scene` = **一条焊缝**（`weld_json` 里第 `seam_id` 条）+ 它所定义的世界/机械臂状态。

### 1.1 字段（按组）

```python
class Scene:
    # —— 静态输入 ——
    cfg            : Config                 # 配置（configs/default.yaml）
    workpiece_obj  : str                    # 工件 mesh 路径（_part.obj / _watertight.obj）
    weld_json      : str                    # 焊缝信息文件（_weld_angle3.json）
    seam_id        : int                    # 用 weld_json 中的第几条焊缝
    seam           : dict                   # 选中那条焊缝（load_welds：p0/p1/mid/bisector/boundary_dirs/raw）
    workpiece_pose : np.ndarray | None      # 工件在 base 系下的 pose7；plan_init_pose 选定后填入
    goal_user      : tuple | None           # 用户【直接输入】的 goal ((x,y,z),(qw,qx,qy,qz))，base 系（后续 API 用）

    # —— 世界状态（3D）——
    init_pose            : InitPoseCandidate | None   # 当前工件↔臂相对 pose (R,t)；None=尚未求解/应用
    init_pose_candidates : list[InitPoseCandidate]    # plan_init_pose 产出的全部候选（已排序）
    obstacles            : list             # 已放障碍（ObstacleSpec）——后续
    truth_scene          : trimesh | None   # 工件+障碍（base 系），raycast 几何源——后续
    voxmap               : ThreeStateVoxelMap | None  # 三态记忆——后续

    # —— 机械臂状态 ——
    cur_cfg        : list[float]            # 当前关节角；所有用法起步均为 retract（与初始位姿无关）

    # —— cuRobo 世界（惰性构建，脏标记缓存）—— 本期占位 ——
    _h_truth, _h_plan, _world_plan, _h_expl, _camera_model
    _solver        : InitPoseLookupSolver   # 候选求解器句柄（按 (工件,n_per_dof) key 复用查表）
    _solver_key    : tuple
    _dirty         : set                    # {"worlds","truth_scene","goal"} 缓存失效标记
```

> **`init_pose` 不是机械臂起步关节角**：它是工件 mesh world ↔ base 的相对 pose `(R,t)`
> （`p_base = R·p_world + t`），定义工件在 base 系的摆放；机械臂起步关节角恒为 `retract`，
> 候选的 `q` 仅是「焊枪头落在该焊缝 standoff 落点」的可达参考构型。

### 1.2 构造

```python
from gt_gen.scene import Scene

scene = Scene(
    cfg="configs/default.yaml",       # Config 实例 / yaml 路径 / None（None→load_config 默认）
    workpiece_obj="/.../BEAM_..._part_watertight.obj",
    weld_json="/.../BEAM_..._weld_angle3.json",
    seam_id=0,                         # 用 weld_json 第 0 条焊缝
)
```

构造期：归一 `cfg`，用 `plan_init_pose.load_welds` 读 `weld_json` 并按 `weld['idx']==seam_id`
选中焊缝存入 `self.seam`，`cur_cfg` 置为 `retract`，其余世界字段占位为 `None`/空。
**不**触发任何 GPU/cuRobo 导入（求解器是惰性的）。

### 1.3 `plan_init_pose(n_per_dof=None, diagnostic=False, rebuild=False) -> list[InitPoseCandidate]`

计算「工件 ↔ 机械臂」候选初始位姿，等价于 `scripts/plan_init_pose.py --solve` 对本焊缝跑一遍：

1. `InitPoseLookupSolver(cfg, workpiece_obj, collision_tolerance, voxel_size, n_per_dof)`
   → `precompute_joint_table()`（与工件无关、仅一次；按 `(workpiece_obj, n_per_dof)` 缓存复用，
   `rebuild=True` 强制重建）。
2. `solve_one_weld_lookup(self.seam, rot_x, rot_y, rot_z)`——角度采样 / standoff 等全部读
   `cfg`（`default.yaml` 的 `plan_init_pose` 段），口径与脚本完全一致。

产出：全部候选（已按 `combined_score` 降序 + 真实焊缝中点 `x>0` 优先）。
副作用：`self.init_pose` = best（`candidates[0]`，失败为 `None`）、`self.init_pose_candidates`
= 全部候选、`self.workpiece_pose` 由 best 的 `(R,t)` 填入。

```python
cands = scene.plan_init_pose()         # 需 GPU
print(len(cands), "个候选；best 关节角 =", scene.init_pose.q)
print("工件在 base 的 pose7 =", scene.workpiece_pose)
```

### 1.4 `InitPoseCandidate`（dataclass）

包一条 solver 解：`R,t,q,ee_pos_in_base,mid_in_base,ee_x_in_base,rot_{x,y,z}_deg,
d_link,d_retract,align_score,combined_score`，外加便捷属性
`T_workpiece_in_base`（4×4）与 `workpiece_pose7`。字段 `raw` 保留原始解 dict，
`to_solution()` 还原它，供可视化函数原样复用（**不改可视化逻辑**）。

---

## 2. `SceneVisualizer`（`gt_gen/scene_viz.py`）

```python
class SceneVisualizer:           # 基类：__init__(self, scene)，持有一个 Scene
class Open3DSceneVisualizer(SceneVisualizer):    # open3d 后端（本机开窗）
class IsaacSimSceneVisualizer(SceneVisualizer):  # isaacsim 后端（后续补充）
```

### 2.1 `Open3DSceneVisualizer.show_init_poses()`

逐个可视化该 Scene 焊缝的**候选初始位姿**（工件相对机械臂的摆放）。前提：先
`scene.plan_init_pose()`。内容与 `scripts/plan_init_pose.py` 一致：整臂碰撞球 + init_free 盒 +
base 坐标架 + 按 `(R,t)` 摆放的工件网格 + 绿色焊缝线/中点 + 红色 standoff 落点；同一窗口按
**C 键**切下一个候选。底层直接复用 `plan_init_pose.show_lookup_solutions`。

```python
from gt_gen.scene_viz import Open3DSceneVisualizer

scene.plan_init_pose()
Open3DSceneVisualizer(scene).show_init_poses()   # 按 C 翻看候选
```

---

## 3. 用法示例

```python
from gt_gen.scene import Scene
from gt_gen.scene_viz import Open3DSceneVisualizer

obj = "/media/a/新加卷/hanfeng/segment/A3Changfang/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part_watertight.obj"
wj  = "/media/a/新加卷/hanfeng/segment/A3Changfang/BEAM_1aEEYa00Ed5Z4sE34qDJKu_weld_angle3.json"

scene = Scene(cfg="configs/default.yaml", workpiece_obj=obj, weld_json=wj, seam_id=0)
scene.plan_init_pose()                          # 求候选初始位姿（需 GPU）
Open3DSceneVisualizer(scene).show_init_poses()  # 逐个可视化
```

等价的 CLI（行为相同）：
`conda run -n env_isaaclab python -u scripts/plan_init_pose.py --solve --obj <obj> --weld-json <wj> --welds 0 --viz`
（见 `.vscode/launch.json` 的 `plan_init_pose`）。

---

## 4. 待补充（占位接口，后续按需实现）

| 方向 | 计划 |
|------|------|
| Scene | `set_goal` / `apply_init_pose` / 障碍（type1/2/3）/ `reset_voxmap` / 惰性世界 `_ensure_worlds` / `plan_global` / `generate_gt_explore` / `run_goal_sequence` / 导出与 npz 互转 |
| SceneVisualizer | open3d & isaacsim 对「3D 世界 / 机械臂 / 焊缝 / 已观测区域」等的可视化 |

> 历史方案（更全的 `Scene + recipes` 设想，已搁置）见 `docs/api-重组方案.废弃.md`，仅作参考。
