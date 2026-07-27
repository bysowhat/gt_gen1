# 渲染 type3（ObserveAnything）轨迹 —— 逐帧渲染方案

> 目标：对 **type3 类型**（`ObserveAnythingScene` 产出）的下采样轨迹 pkl 做逐帧左右目 RGB + 深度 + 分割渲染，
> 复用 type2 已有的渲染/多卡调度基础设施，但适配 type3 的"整场景 USD"几何模型。
>
> 本文只描述方案，尚未落地实现。关联文档：[渲染整条采样轨迹-方案.md](渲染整条采样轨迹-方案.md)（type2 渲染）、
> [render_info-数据字段说明.md](render_info-数据字段说明.md)。  

---

## 0. 背景与输入

- **type2 轨迹**：`gt_gen/scene.py` 的 `Scene`。先算机械臂与工件（含障碍）的相对位姿，再把它们放进场景。
  一个 pkl = 一个工件 obj + 若干焊缝轨迹。
- **type3 轨迹**：`gt_gen/observe_scene.py` 的 `ObserveAnythingScene`。在整个 USD 场景里采样机械臂 base 位姿，
  确定用哪个位置就把机械臂悬空放进 USD 场景的那个位置。没有"单个工件"，工件是整场景 USD 的一部分。

两者的轨迹都要先经 `scripts/traj_downsample.py` 关键帧下采样，再逐帧渲染。

**本方案的输入**（示例，均为本地路径）：

| 项 | 路径 |
|---|---|
| type3 原始轨迹 pkl | `/media/a/新加卷/tempt/7/full_warehouse_seam1_r3.5_type3_seam2.pkl` |
| type3 下采样后 pkl（渲染输入） | `/media/a/新加卷/tempt/7_ds/full_warehouse_seam1_r3.5_type3_seam2.pkl` |
| 场景 USD | `/media/a/upan/others/full_warehouse.usdz` |
| 焊缝 json | `/media/a/新加卷/tempt/7/full_warehouse.json` |

> 生成命令（参考）：
> `scripts/demo_scene.py --task type3 --usd <usdz> --weld-js <json>`

---

## 1. 核心结论：为什么现有 `render/render_trajectory.py` 不能直接跑 type3

现有 `render/render_trajectory.py` 写死了 type2 的几何模型，与 type3 场景模型有三处冲突：

| 环节 | render_trajectory.py（type2 写法） | type3 需要 |
|---|---|---|
| 加载器 | `from gt_gen.scene import Scene` + `Scene.load()`（`collect_render_jobs`, L143/145） | `ObserveAnythingScene.load()` |
| 场景几何 | 通用装饰性 warehouse USD 当背景 + **单个工件 obj**（`WORKPIECE_OBJ` 转 USD 摆到 `wp_pose7`） | **真实场景 USD 整体**（如 full_warehouse.usdz），机械臂悬空放进去 |
| 输出命名 | `part_stem` 来自工件 obj 名 | 应来自 `usd_path` 名（如 `full_warehouse`） |

若直接把 type3 pkl 喂给它：`Scene.load` 与 `ObserveAnythingScene.__init__`（签名 `usd_path/welds_json`）不兼容会报错；
即便绕过，渲出来也只是焊缝旁一小块**裁剪块**，机械臂相机看到的真实场景（货架、地面、远处结构）全缺失
→ **RGB / depth / seg 全错**。

---

## 2. 关键决策：渲染时**直接引用 USD**，不拆 mesh

`gt_gen/scene_viz.py` 的 `ObserveSceneVisualizer._spawn_environment` 处理 type3 时是把整场景**拆成 per-prim trimesh** 再逐个 spawn。
**但那是 viz 的历史选择，渲染不该照抄**：

- viz 拆 mesh 的原因 ①：复用父类 `spawn_mesh(points/faces)` 管线，图代码统一。
- viz 拆 mesh 的原因 ②：规避 **pxr 双加载崩溃**——该约束仅针对"在启动 SimulationApp **之前**"经
  `find_surface_welds._bootstrap_pxr` 自举 pxr（见 `docs/observeanything`、记忆 `observe_viz_type3_crash_fix`）。
  **在 Isaac 起来之后**用 `add_reference_to_stage` / `UsdFileCfg` 直接引用 USD，走的是 kit 自己的 USD 库，
  不经过那个自举，**不会崩**。

而 `render_trajectory.py` 本身**已经在直接引用 USD**——type2 的背景 warehouse 就是这么加载的（`build_scene`, L452-457）：

```python
bg = sim_utils.UsdFileCfg(
    usd_path=warehouse_usd,
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
bg.func(wh, bg, translation=(off[0], off[1], 0.0))
```

**因此 type3 的做法**：把这个"通用 warehouse 背景"替换成真正的场景 USD（`scene.usd_path`），
并把它的 xform 摆到 base 系（`inv(T_base_world)`，即 `workpiece_pose7` 那一套，与现在 `set_prim_pose(wp7)` 摆工件同理）。
**不再需要单独工件 obj，也不需要拆 mesh。**

### 直接引用 USD 的收益

- **保留原始材质/纹理/法线** → RGB 的 GT 更真实（拆 mesh 只剩几何，颜色被硬编码成灰 `0.72`）。
- **省显存**：USD 引用可走实例化（instancing），比每 env 复制一份 trimesh 顶点更省。
- **快**：省掉 `_load_scene_meshes` 遍历所有 prim 建 trimesh。
- **代码更简单**：`warehouse_usd` 换成 `scene.usd_path` + 摆 xform 即可。

**不做拆 mesh 兜底**：usdz 引用不通就**直接报错、停止运行**，不回退拆 mesh。

---

## 3. 关键技术约束

1. **`ObserveAnythingScene.load()` 故意不解析整场景 mesh**（`observe_scene.py:130-147`，`load_meshes=False`）。
   这正好契合 render_trajectory 的两阶段架构：
   - **阶段一**（起 Isaac 前，纯 numpy）：`ObserveAnythingScene.load` 只复原数据状态（不碰 mesh/pxr）+ 抽轨迹/位姿 numpy + 带出 `usd_path`。
   - **阶段二**（起 Isaac 后）：直接 `UsdFileCfg` 引用 `usd_path`（kit 的 USD 库，安全）。
2. **坐标一致性天然成立**：type3 的 `T_workpiece_in_base` 实际是 `inv(T_base_world)`（世界系-米 → base 系）。
   整场景 USD、裁剪块、障碍、焊缝线都在同一世界系-米，套同一 `T`/`workpiece_pose7` 即与机械臂/焊缝管/相机严丝合缝。
   （现有代码里障碍已用 `verts @ R_T.T + t_T` 这么变换，整场景引用改设 xform 即等价刚体变换。）
3. **API 全部继承**：`ObserveAnythingScene` 继承 `Scene`，`set_init_pose` / `_obstacle_solid_trimeshes` /
   `_seam_frame` / `sampled_trajectories` / `seams` 均可直接复用（type3 一般无障碍，`_obstacle_solid_trimeshes` 返回空即可）。

---

## 4. 实现方案（两个新文件，不动远程、不动现有 type2 脚本/代码）

### 改动 1：新增 `render/render_trajectory_observe.py`（复制 `render_trajectory.py` 后改动）

- **`collect_render_jobs`**：
  - 改 `ObserveAnythingScene.load`。
  - 每条轨迹仍取 `set_init_pose` 的 `T_workpiece_in_base` / `workpiece_pose7`、`observe/goal`、`sampled`、
    `_seam_frame`（焊缝管）、`_obstacle_solid_trimeshes`（type3 通常空）。
  - **新增**：把 `scene.usd_path` 一并带出（供阶段二引用）。
- **`build_scene` / `prepare_job`**：
  - 删掉"spawn 单个工件 obj"（不再需要 `asset_convert.convert_workpiece_obj`）。
  - 改成每 env 用 `UsdFileCfg` 引用 `scene.usd_path`（`collision_enabled=False`），并把该引用根 prim 的 xform
    设成 `workpiece_pose7`（＋env 偏移）→ 摆到 base 系。
  - **逐 prim 打语义标签**（见 §6 已定）：`UsdFileCfg` 引用后，`Usd.PrimRange` 遍历引用根子树，
    对每个 `UsdGeom.Mesh` 单独 `add_update_semantics(prim, label)`（label 用 prim 名/路径），
    使 `instance_segmentation_fast`（`colorize=False`）给每个 prim 输出各自的整数 id。
  - 机械臂（整体 `robot`）、双 pass 焊缝管（`seam`，seg-only）、左右相机、`render_info` / `_traj_meta` 逻辑**全部不动**。
  - **逐帧 id↔类别映射照旧**：type2 已在 render_info 逐帧存 `seg_id_to_label`（`{整数id: 类别串}`）
    与整条轨迹的 `seg_id_to_label_list`（`render_trajectory.py:767/803`），type3 逐 prim 标签后天然写进这套，无需改。
- **输出 `part_stem`**：改用 `usd_path` 的 stem（如 `full_warehouse`）。
- **不做兜底**：usdz 引用不通时**直接抛异常、终止进程**（不回退拆 mesh），让调度器按失败处理、日志留崩因。

### 改动 2：新增 `scripts/bash/v1/render_trajectory_scheduler3.sh`（复制现有调度器，只改路径）

- `DS_DIR` → type3 下采样目录；`RENDER_SCRIPT` → `render/render_trajectory_observe.py`；`OUT_DIR` 改名。
- 多卡调度 / 超时 / `RENDER_TRAJECTORY_DONE` 完成标记 / 进程组强杀 / 断点续跑（per-traj `_DONE_`）逻辑**完全复用**。

---

## 5. 本地 vs 远程执行

- 现有 `render_trajectory_scheduler.sh` 里的路径（`/kpfs_dataset_ssd/...`、`/workspace/isaaclab/_isaac_sim/python.sh`）
  是**远程 debug 节点**的（`ssh debug`）。**不改远程文件。**
- 本地跑：`DS_DIR` 用本地 `/media/a/新加卷/tempt/7_ds/`，`PYTHON_BIN` 用本地 `env_isaaclab`。
- ⚠️ **显存**：整场景（full_warehouse）+ 多 env 并行很吃显存（记忆 `plan_explore_path_gpu_oom`：本地 11.6G 卡易 OOM）。
  建议先 `--max-envs 1` 单 pkl 手动跑通，再上调度器/上调 env 数。

---

## 6. 已定的设计点

1. **语义分割（seg）粒度：逐 prim 实例 + 焊缝，与 type2 一致。**
   - usdz 引用进来后其内部每个 prim 仍存在于 stage，可 `Usd.PrimRange` 遍历子树、逐 `UsdGeom.Mesh`
     单独打 semantic tag → `instance_segmentation_fast`（`colorize=False`）给每个 prim 各自的整数 id。
     （"只能对引用根打一个 tag"是 type2 的省事写法，不是能力上限。）
   - 类别集合 = **各 prim 实例 + `robot`（机械臂整体一个）+ `seam`（焊缝管，走 pass2 seg-only）**。
   - **每帧输出 id↔类别对应 dict**：沿用 type2 的 `seg_id_to_label`（`{整数id: 类别串}`，逐帧存于 render_info）
     与 `seg_id_to_label_list`（整条轨迹），无需新增字段。
   - 注：prim 的 label 串取 prim 名/路径，是**几何分块**，USD 本身不含"哪个是待焊工件"的语义，下游若需赋义自行处理。
2. **整场景全渲**：忠实 `UsdFileCfg` 引用整场景 USD（GT 正确）。显存压力大，靠 §5 的 `--max-envs` 控制。

---

## 7. 建议的落地顺序（先验证再成型）

1. **可行性验证**：拿 `7_ds/full_warehouse_..._type3_seam2.pkl`，写最小脚本在 Isaac 里直接 `UsdFileCfg` 引用
   `full_warehouse.usdz`、摆到 base 系、渲一帧 —— 确认：
   - usdz 能否直接引用（个别 usdz 打包/材质路径有坑）；
   - 场景 USD 自带 `PhysicsScene`/碰撞是否需 `collision_enabled=False` 或 `_disable_physics`；
   - 整场景套 `workpiece_pose7` 后与机械臂/焊缝管/相机是否严丝合缝；
   - 逐 prim 遍历子树打 semantic tag 后，`instance_segmentation_fast` 是否给各 prim 输出不同整数 id、
     `seg_id_to_label` 映射是否正确。
2. **成型**：验证通过后落成完整 `render/render_trajectory_observe.py`（usdz 引用失败即报错停止，不做拆 mesh 兜底）。
3. **调度**：新增 `scripts/bash/v1/render_trajectory_scheduler3.sh`，先本地 `--max-envs 1` 小批跑，再放量。

---

## 8. 输出结构（与 type2 一致，见 render_trajectory.py docstring）

```
<out>/<usd_stem>/seam{sid}_{hand}{index}_traj{j}/
    left/  {k}_rgb.jpg  {k}_depth.exr  {k}_seg.png  render_info.npy
    right/ {k}_rgb.jpg  {k}_depth.exr  {k}_seg.png  render_info.npy
    _traj_meta.npy
    _DONE_...            # 断点续跑哨兵
```

`render_info.npy` 字段沿用 type2（见 [render_info-数据字段说明.md](render_info-数据字段说明.md)），无需改动。
