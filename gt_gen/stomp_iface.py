"""STOMP 规划后端：封装参考项目 gt_overall/stomp_planning_api（与 curobo_iface 并列）。

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
from typing import Any, Optional

import numpy as np


def _ensure_api(cfg):
    """把 gt_overall 目录注入 sys.path 并 import stomp_planning_api（懒加载，避免无 STOMP 时报错）。

    目录优先级：环境变量 GT_OVERALL_DIR > config.stomp_params['gt_overall_dir']。
    """
    d = os.environ.get("GT_OVERALL_DIR") or cfg.stomp_params["gt_overall_dir"]
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
    traj = api.plan_to_pose_single(
        list(map(float, cur_cfg)), goal_pose, world,
        **_plan_kwargs(cfg, buffer, checker_type))
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
