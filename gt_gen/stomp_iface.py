"""STOMP 规划后端：封装本项目 stomp_planner/stomp_planning_api（与 curobo_iface 并列）。

stomp_planner/ 是原参考项目 gt_overall 中 STOMP 核心(stomp_planning_api / plan_path_stomp /
plan_path_stomp_obstacle / stomp_utils_traj)的 vendoring 副本，已并入本项目、不再外部引用。

当 configs/default.yaml 的 planner.backend == 'stomp' 时，obstacle_placement 的
「默认轨迹 / 绕行解」改走这里；碰撞判定(check_state)、扫掠 FK、三条件验证逻辑均不变。

设计要点（见对话确认）：
- 固定朝向：STOMP 的 IK 解【全位姿】(pos+quat)，不支持放开末端 roll（cuRobo free_pose_metric 才支持）；
  obstacle_placement 传进来的 metric 在 STOMP 分支被忽略。
- buffer 不重复叠加：STOMP 自身 buffer≈0.001（activation_distance）；间隙完全来自障碍物理膨胀
  （obstacle_buffer_m，工件不膨胀）。两后端在【同一个 world】里规划——默认=仅工件、绕行=障碍已膨胀——
  只换规划器。
- 目标位姿就是 seam_ee_pose 返回的 (pos, quat) 元组：stomp_planning_api._normalize_pose 原生支持。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

import numpy as np


def _ensure_api(cfg):
    """把 stomp_planner 目录注入 sys.path 并 import stomp_planning_api（懒加载，避免无 STOMP 时报错）。

    目录优先级：环境变量 STOMP_PLANNER_DIR > config.stomp_params['stomp_planner_dir']
    （后者默认 = 本项目 stomp_planner/）。
    """
    d = os.environ.get("STOMP_PLANNER_DIR") or cfg.stomp_params["stomp_planner_dir"]
    if d and d not in sys.path:
        sys.path.insert(0, d)
    import stomp_planning_api  # noqa: E402
    return stomp_planning_api


def _plan_kwargs(cfg, buffer, checker_type):
    """组装透传给 stomp_planning_api.plan_to_pose_* 的公共参数。"""
    sp = cfg.stomp_params
    if buffer is None:
        buffer = float(sp["buffer_m"])
    return dict(
        robot_yml=cfg.robot_cfg_path, checker_type=checker_type, buffer=float(buffer),
        num_iterations=int(sp["num_iterations"]), num_batch=int(sp["num_batch"]),
        num_timesteps=int(sp["num_timesteps"]), delta_t=float(sp["delta_t"]),
        collision_weight=float(sp["collision_weight"]),
    )


def plan_pose_single(cfg, world, cur_cfg, goal_pose, *, buffer: Optional[float] = None,
                     checker_type: Any = None) -> Optional[np.ndarray]:
    """用 STOMP 规划 cur_cfg -> goal_pose，返回【单条最优】避障轨迹（内部先 IK 解目标关节角）。

    调 stomp_planning_api.plan_to_pose_single：在所有「无碰撞(n_collision_steps==0)且在限位
    (in_limit)」的候选里取 state_cost 最小那条；无合格候选返回 None（合格判定全在 API 内部，
    本层不再解析 info）。

    入参
    ----
    cfg          : gt_gen.config.Config（读 robot_cfg_path 与 stomp_params）。
    world        : cuRobo WorldConfig（默认轨迹=仅工件；绕行=障碍已膨胀、工件不膨胀）。
    cur_cfg      : 起点关节角（list/序列，长度=dof）。
    goal_pose    : (pos[3], quat_wxyz[4]) 元组 或 [x,y,z,qw,qx,qy,qz]（base 系）。
    buffer       : STOMP 碰撞激活距离(m)；None 时取 cfg.stomp_params['buffer_m']（≈0.001）。
    checker_type : cuRobo CollisionCheckerType；None 时由 stomp_planning_api 按 world 内容自动推断。

    返回：(T,dof) ndarray 或 None。
    """
    api = _ensure_api(cfg)
    _t0 = time.perf_counter()
    traj = api.plan_to_pose_single(
        list(map(float, cur_cfg)), goal_pose, world,
        **_plan_kwargs(cfg, buffer, checker_type))
    print(f"[计时] plan_pose_single（建planner+IK+STOMP）{time.perf_counter() - _t0:.3f}s "
          f"-> {'None' if traj is None else 'ok'}")
    return None if traj is None else np.asarray(traj, float)


def plan_pose_multi(cfg, world, cur_cfg, goal_pose, *, buffer: Optional[float] = None,
                    checker_type: Any = None, ik_position_threshold: Optional[float] = None,
                    ik_rotation_threshold: Optional[float] = None,
                    showik: bool = False) -> list:
    """用 STOMP 规划 cur_cfg -> goal_pose，返回【所有合格】避障轨迹（按 state_cost 升序）。

    调 stomp_planning_api.plan_to_pose_multi：返回所有「无碰撞且在限位」的候选轨迹；这些候选
    都指向同一个 IK 目标关节角（plan_pose 先解一次 IK 再批量规划），差在中段绕障路线、终点相同。
    去重（按路线偏差挑互不相同的若干条）由调用方 obstacle_placement.detour_exists 负责。

    ik_position_threshold / ik_rotation_threshold：透传给 STOMP 内部 cuRobo IKSolver 的位置/朝向
    收敛门限（None 时用 stomp_planning_api 默认 0.005/0.05）。这两个参数仅 detour_exists 的绕行
    规划会传入（来自 cfg.obstacle_placement 的 detour_ik_*_threshold），plan_pose_single 不受影响。

    showik：调试开关（调用方写死）。True 时在跑 STOMP 前，于【同一 world + 同一阈值】独立解一次
    IK（与 plan_to_pose_multi 内部同逻辑），取 StompPlanner.ik 选/退回的目标关节角，用 Open3D 画出
    该姿态下的整臂碰撞球（品红线框）+ world 内 mesh（工件灰 / 障碍橙）+ goal 坐标系，用于诊断
    「[ik][warn] 所有 IK 解都进入碰撞」时终点姿态到底插进了哪里。仅可视化，不改变返回值。

    入参其余同 plan_pose_single。返回：list[(T,dof) ndarray]（无合格候选时为 []）。
    """
    api = _ensure_api(cfg)
    kw = _plan_kwargs(cfg, buffer, checker_type)
    if ik_position_threshold is not None:
        kw["ik_position_threshold"] = float(ik_position_threshold)
    if ik_rotation_threshold is not None:
        kw["ik_rotation_threshold"] = float(ik_rotation_threshold)
    if showik:
        _visualize_ik(cfg, world, goal_pose, checker_type=checker_type, buffer=buffer,
                      ik_position_threshold=ik_position_threshold,
                      ik_rotation_threshold=ik_rotation_threshold)
    trajs = api.plan_to_pose_multi(
        list(map(float, cur_cfg)), goal_pose, world, **kw)
    return [np.asarray(t, float) for t in trajs] if trajs else []


def plan_joint_single(cfg, world, cur_cfg, target_cfg, *, buffer: Optional[float] = None,
                      checker_type: Any = None) -> Optional[np.ndarray]:
    """用 STOMP 规划 cur_cfg -> target_cfg（关节空间目标），返回【单条最优】避障轨迹。

    调 stomp_planning_api.plan_to_joint_single：跑 num_batch 条并行 STOMP，在所有「无碰撞
    (n_collision_steps==0) 且在限位(in_limit)」的候选里取 state_cost 最小那条；无合格候选返回 None。

    入参同 plan_pose_single，但目标是关节角 target_cfg（长度=dof）而非位姿。
    返回：(T,dof) ndarray 或 None。
    """
    api = _ensure_api(cfg)
    _t0 = time.perf_counter()
    traj = api.plan_to_joint_single(
        list(map(float, cur_cfg)), list(map(float, target_cfg)), world,
        **_plan_kwargs(cfg, buffer, checker_type))
    print(f"[计时] plan_joint_single（建planner+STOMP，无IK）{time.perf_counter() - _t0:.3f}s "
          f"-> {'None' if traj is None else 'ok'}")
    return None if traj is None else np.asarray(traj, float)


def plan_joint_multi(cfg, world, cur_cfg, target_cfg, *, buffer: Optional[float] = None,
                     checker_type: Any = None) -> list:
    """用 STOMP 规划 cur_cfg -> target_cfg（关节空间目标），返回【所有合格】避障轨迹（按 state_cost 升序）。

    调 stomp_planning_api.plan_to_joint_multi。入参同 plan_joint_single。
    返回：list[(T,dof) ndarray]（无合格候选时为 []）。
    """
    api = _ensure_api(cfg)
    trajs = api.plan_to_joint_multi(
        list(map(float, cur_cfg)), list(map(float, target_cfg)), world,
        **_plan_kwargs(cfg, buffer, checker_type))
    return [np.asarray(t, float) for t in trajs] if trajs else []


def world_from_voxmap(cfg, voxmap, inflate_voxels: Optional[int] = None):
    """三态体素图的「非 FREE」(OCCUPIED ∪ UNKNOWN) 区域 → 单个 mesh 的 WorldConfig（供 STOMP 用）。

    STOMP 不支持 VOXEL 碰撞检查（stomp_planning_api 只 MESH/PRIMITIVE），故把探索世界的障碍区
    转成一个 mesh：用 marching cubes 只网格化「自由泡」边界 + ROI 外壳（面数千量级，逐体素 cuboid
    在 UNKNOWN 占满 ROI 时不可行）。

    膨胀层数 inflate_voxels 缺省取 cfg.voxel_inflate_voxels（与 collision_sync.sync_collision_world
    同一单一来源）——保证 STOMP 与 cuRobo 在【相同障碍集】上规划，行为一致。

    顶点 base 系映射：matrix_to_marching_cubes 顶点 = 原矩阵索引×voxel_size（角点在索引 0），
    voxmap voxel i 中心在 origin+(i+0.5)·vs，故 world = origin + mc_verts + 0.5·vs（表面正好落在
    障碍体素外缘面上，与 voxmap 占据对齐、不错位）。

    返回 (WorldConfig, CollisionCheckerType.MESH)；非 FREE 全空时返回空 world（仅自碰撞）。
    """
    import gt_gen.compat  # noqa: F401  trimesh shim
    gt_gen.compat.apply_trimesh_shim()
    import trimesh
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType

    mask = np.asarray(voxmap.non_free_mask(), bool)
    if inflate_voxels is None:
        inflate_voxels = int(getattr(cfg, "voxel_inflate_voxels", 0))
    if inflate_voxels and mask.any() and not mask.all():
        from scipy import ndimage
        st = ndimage.generate_binary_structure(3, 3)             # 26 邻接，覆盖对角
        mask = ndimage.binary_dilation(mask, structure=st, iterations=int(inflate_voxels))
    if not mask.any():
        return WorldConfig(mesh=[]), CollisionCheckerType.MESH

    vs = float(voxmap.voxel_size)
    _t0 = time.perf_counter()
    tm = trimesh.voxel.ops.matrix_to_marching_cubes(mask, pitch=vs)
    verts = np.asarray(tm.vertices, float) + np.asarray(voxmap.origin, float) + 0.5 * vs
    mesh = Mesh(name="explore_obstacles", vertices=verts.tolist(),
                faces=np.asarray(tm.faces, np.int64).tolist(),
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    print(f"[计时] world_from_voxmap（非FREE体素={int(mask.sum())} -> "
          f"mesh 顶点={len(tm.vertices)} 面={len(tm.faces)}）{time.perf_counter() - _t0:.3f}s")
    return WorldConfig(mesh=[mesh]), CollisionCheckerType.MESH


def _greedy_boxes(sub):
    """贪心 3D 盒分解：把 bool 子掩码 sub(nx,ny,nz) 恰好覆盖成若干轴对齐大盒（不重叠、无近似）。

    每步取一个未认领的占据体素当种子，先沿 X 长到连续占据的最远处，再沿 Y（整行占据才扩）、
    再沿 Z（整板占据才扩），标记认领后继续。实心块 → 1 个大盒；三态体素的成片占据收缩极猛。
    返回 list[(i0,j0,k0,i1,j1,k1)]（含端点，相对 sub 原点的索引）。
    """
    claimed = ~np.ascontiguousarray(sub, dtype=bool)         # 非占据视为已认领
    nx, ny, nz = claimed.shape
    boxes = []
    xs, ys, zs = np.nonzero(~claimed)                        # 原始占据体素（C 序）
    for x0, y0, z0 in zip(xs.tolist(), ys.tolist(), zs.tolist()):
        if claimed[x0, y0, z0]:
            continue
        x1 = x0
        while x1 + 1 < nx and not claimed[x1 + 1, y0, z0]:
            x1 += 1
        y1 = y0
        while y1 + 1 < ny and not claimed[x0:x1 + 1, y1 + 1, z0].any():
            y1 += 1
        z1 = z0
        while z1 + 1 < nz and not claimed[x0:x1 + 1, y0:y1 + 1, z1 + 1].any():
            z1 += 1
        claimed[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
        boxes.append((x0, y0, z0, x1, y1, z1))
    return boxes


def world_from_voxmap_cuboid(cfg, voxmap, n: Optional[float] = None,
                             inflate_voxels: Optional[int] = None):
    """三态体素图「非 FREE」区 → 一批 cuRobo Cuboid 的 WorldConfig（PRIMITIVE，供 STOMP 用）。

    与 world_from_voxmap(mesh 版) 并列的另一种转法：
      - 仅转【以 base 原点(0,0,0)为心、半边长 n 的盒 [−n,n]³】与 ROI 交集内的非 FREE 体素；
        盒外 UNKNOWN 当 FREE（可穿行）——放宽「只走确认空域」保守性换速度（见 default.yaml 注释）。
      - 体素本就是立方体，逐格转 Cuboid 是占据的【精确】表示（无 marching-cubes 表面近似）；
        再用 _greedy_boxes 把成片占据【合并成少量大盒】，避免 PRIMITIVE 检查器线性扫几万个 cuboid。

    n 缺省取 cfg.stomp_params['local_box_m']；inflate_voxels 缺省同 cfg.voxel_inflate_voxels（同一来源）。
    返回 (WorldConfig(cuboid=[...]), CollisionCheckerType.PRIMITIVE)；盒内无占据时返回空 world。

    盒覆盖 voxel 索引 [i0..i1] → Cuboid 中心 = origin+(i0+i1+1)/2·vs，边长 = (i1−i0+1)·vs
    （voxel i 中心在 origin+(i+0.5)·vs，故盒面正好贴体素外缘，与 voxmap 占据精确对齐）。
    """
    import gt_gen.compat  # noqa: F401  trimesh shim
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.geom.sdf.world import CollisionCheckerType

    mask = np.asarray(voxmap.non_free_mask(), bool)
    if inflate_voxels is None:
        inflate_voxels = int(getattr(cfg, "voxel_inflate_voxels", 0))
    if inflate_voxels and mask.any() and not mask.all():
        from scipy import ndimage
        st = ndimage.generate_binary_structure(3, 3)
        mask = ndimage.binary_dilation(mask, structure=st, iterations=int(inflate_voxels))

    if n is None:
        n = float(cfg.stomp_params.get("local_box_m", 2.0))
    vs = float(voxmap.voxel_size)
    origin = np.asarray(voxmap.origin, float)
    grid = np.asarray(mask.shape, int)
    # [−n, n]^3（base 原点为心）→ index 子盒，与 ROI 取交：voxel i 中心 = origin+(i+0.5)·vs ∈ [−n,n]
    lo = np.ceil((-n - origin) / vs - 0.5).astype(int)
    hi = np.floor((n - origin) / vs - 0.5).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, grid - 1)
    if np.any(lo > hi):
        print(f"[计时] world_from_voxmap_cuboid（n={n}m 盒与 ROI 无交）-> 空 world")
        return WorldConfig(cuboid=[]), CollisionCheckerType.PRIMITIVE

    sub = mask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    n_occ = int(sub.sum())
    if n_occ == 0:
        print(f"[计时] world_from_voxmap_cuboid（n={n}m 盒内非FREE=0）-> 空 world")
        return WorldConfig(cuboid=[]), CollisionCheckerType.PRIMITIVE

    _t0 = time.perf_counter()
    boxes = _greedy_boxes(sub)
    cuboids = []
    for bi, (i0, j0, k0, i1, j1, k1) in enumerate(boxes):
        g0 = lo + np.array([i0, j0, k0])
        g1 = lo + np.array([i1, j1, k1])
        center = origin + (g0 + g1 + 1) * 0.5 * vs
        bdims = (g1 - g0 + 1).astype(float) * vs
        cuboids.append(Cuboid(name=f"vox_{bi}", dims=bdims.tolist(),
                              pose=center.tolist() + [1.0, 0.0, 0.0, 0.0]))
    print(f"[计时] world_from_voxmap_cuboid（n={n}m 盒内非FREE格={n_occ} -> 合并大盒={len(cuboids)}）"
          f"{time.perf_counter() - _t0:.3f}s")
    return WorldConfig(cuboid=cuboids), CollisionCheckerType.PRIMITIVE


def world_from_voxmap_auto(cfg, voxmap, inflate_voxels: Optional[int] = None):
    """按 cfg.stomp_params['voxel_world'] 选转法：'cuboid'→world_from_voxmap_cuboid，否则 mesh 版。

    步①/步⑤ 都经此把探索 voxmap 转成 STOMP 世界，保证两处用同一种转法。
    """
    mode = str(cfg.stomp_params.get("voxel_world", "mesh")).lower()
    if mode == "cuboid":
        return world_from_voxmap_cuboid(cfg, voxmap, inflate_voxels=inflate_voxels)
    return world_from_voxmap(cfg, voxmap, inflate_voxels=inflate_voxels)


def _visualize_ik(cfg, world, goal_pose, *, checker_type=None, buffer=None,
                  ik_position_threshold=None, ik_rotation_threshold=None) -> None:
    """showik=True 时调用：在同一 world 上独立解一次 IK，取【所有成功 IK 解】（不止 ik() 退回的
    那一个），按位置误差升序逐个用 Open3D 弹窗——每窗画一个解姿态下的整臂碰撞球（品红线框）+
    world mesh（工件灰/障碍橙）+ goal 坐标系；窗口标题标「当前第几/共多少」及 err/碰撞距离。

    说明：cuRobo IKSolver 用随机种子，这里重解出的解集与 plan_to_pose_multi 内部那次可能不完全
    相同，但同一 world/goal/阈值下碰撞性质一致，诊断意义相同。缺 open3d / 无显示时打印告警跳过，
    不影响主流程。
    """
    api = _ensure_api(cfg)
    sp = cfg.stomp_params
    buf = float(sp["buffer_m"]) if buffer is None else float(buffer)
    planner = api.StompPlanner(world, robot_yml=cfg.robot_cfg_path,
                               checker_type=checker_type, buffer=buf)
    # 收集【所有】成功 IK 解（不止 ik() 退回的那一个）：复用 planner 的 solver/oracle，按位置误差升序。
    pos_th = float(ik_position_threshold) if ik_position_threshold is not None else 0.005
    rot_th = float(ik_rotation_threshold) if ik_rotation_threshold is not None else 0.05
    num_seeds = 100
    solver = planner._get_ik_solver(num_seeds, pos_th, rot_th)
    from curobo.types.math import Pose
    pos, quat = api._normalize_pose(goal_pose)
    ta = planner.oracle.ta
    goal = Pose(position=ta.to_device(pos).view(1, 3),
                quaternion=ta.to_device(quat).view(1, 4))
    res = solver.solve_batch(goal, return_seeds=num_seeds)
    dof = len(planner.robot["joint_names"])
    sol = res.solution.reshape(-1, dof)
    errs = res.position_error.reshape(-1)
    rerrs = (res.rotation_error.reshape(-1)
             if getattr(res, "rotation_error", None) is not None else None)
    succ = res.success.reshape(-1)
    cands = []
    for i in range(sol.shape[0]):
        if not bool(succ[i].item()):
            continue
        q = sol[i]
        d = float(planner.oracle.distance(q.view(1, -1))[0].item())
        re = float(rerrs[i].item()) if rerrs is not None else float("nan")
        cands.append((q.detach().cpu().numpy().tolist(), float(errs[i].item()), re, d))
    cands.sort(key=lambda x: x[1])                           # 位置误差升序（与 ik() 选解顺序一致）
    if not cands:
        print("[showik] 无成功 IK 解，跳过可视化。")
        return
    n_free = sum(1 for _, _, _, d in cands if d <= 1e-4)
    N = len(cands)
    # 可视化前：先打印所有解与 goal 的平移/旋转误差
    # （平移=position_error 米；旋转=rotation_error 四元数测度 sin(θ/2)）
    print(f"[showik] 成功 IK 解 {N} 个（无碰 {n_free} / 碰撞 {N - n_free}），"
          f"与 goal 的误差（按平移误差升序）：")
    for k, (q, pe, re, d) in enumerate(cands):
        tag = "无碰" if d <= 1e-4 else "碰撞"
        qs = "[" + ", ".join(f"{v:.3f}" for v in q) + "]"
        print(f"[showik]   [{k + 1:>2}/{N}] 平移误差={pe:.4f} m  旋转误差={re:.4f}  "
              f"碰撞距离={d:.4f} [{tag}]  关节角={qs}")
    print("[showik] 依次弹窗，关闭当前窗口看下一个。")

    try:
        import open3d as o3d
    except Exception as e:                                   # noqa: BLE001  缺库 → 跳过
        print(f"[showik] open3d 不可用，跳过可视化：{e}")
        return
    from gt_gen import obstacle_placement as opl            # 延迟 import：复用 FK / pose 工具

    # 所有窗口共用：原点坐标系 + world mesh（工件灰/障碍橙）+ goal 坐标系
    base_geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
    for m in (getattr(world, "mesh", None) or []):
        try:
            tm = m.get_trimesh_mesh()
            verts = np.asarray(tm.vertices, float)
            T = opl._pose_to_T(list(map(float, m.pose)))     # wxyz，与 o3d 约定一致
            verts_w = (T[:3, :3] @ verts.T).T + T[:3, 3]
            o3m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts_w),
                o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
            o3m.compute_vertex_normals()
            o3m.paint_uniform_color([0.7, 0.7, 0.7] if m.name == "workpiece"
                                    else [0.95, 0.55, 0.15])
            base_geoms.append(o3m)
        except Exception as e:                               # noqa: BLE001
            print(f"[showik] mesh 加载失败 {getattr(m, 'name', '?')}: {e}")
    gp = goal_pose[0] if (isinstance(goal_pose, (tuple, list)) and len(goal_pose) == 2
                          and hasattr(goal_pose[0], "__len__")) else goal_pose[:3]
    gf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    gf.translate(np.asarray(gp, float)[:3])
    base_geoms.append(gf)

    # 逐个解弹窗：每窗画该解姿态下整臂碰撞球（品红线框）
    for k, (q, pe, re, d) in enumerate(cands):
        per_wp, _ = opl.compute_link_sweep(cfg, [q], cfg.collision_link_names)
        balls = []
        for _ln, s in per_wp.items():
            for c in np.asarray(s, float)[0]:                # (S,4) 取唯一路点
                cx, cy, cz, r = (float(v) for v in c)
                if r <= 1e-4:
                    continue
                ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
                ball.translate((cx, cy, cz))
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color([0.85, 0.1, 0.85])
                balls.append(ls)
        tag = "无碰" if d <= 1e-4 else "碰撞"
        title = (f"showik: IK 解 {k + 1}/{N}  平移={pe:.4f}m 旋转={re:.4f} "
                 f"dist={d:.4f} [{tag}]")
        o3d.visualization.draw_geometries(base_geoms + balls, window_name=title)
