"""SceneVisualizer —— 用 open3d / isaacsim 可视化 Scene 的各种信息（API 重组）。

设计：基类 SceneVisualizer 持有一个 Scene；两个后端子类：
  · Open3DSceneVisualizer  —— 用 open3d 在本机开窗可视化（需显示器，env_isaaclab 装了 open3d）。
  · IsaacSimSceneVisualizer —— 用 isaacsim/isaaclab 可视化（后续补充）。

与 Scene 一样，只做 **API 方向的结构包装**：本期 open3d 的「逐个可视化候选初始位姿」直接复用
scripts/plan_init_pose.py 的 show_lookup_solutions（整臂碰撞球 + init_free 盒 + 工件网格 + 焊缝线
+ standoff 落点），不改其渲染逻辑。

本期已实现：
  · Open3DSceneVisualizer.show_seam()            —— open3d 看指定焊缝：工件 + 焊缝红线 + 障碍物
  · Open3DSceneVisualizer.show_init_poses()      —— 逐个看 Scene 的候选初始位姿（按 C 切下一个）
  · Open3DSceneVisualizer.show_scene_isaacsim()  —— 用 isaacsim 可视化当前 3D 场景（工件 + 焊缝 + 障碍物类型2/3）
  · Open3DSceneVisualizer.show_joint_table_ee()  —— open3d 看关节表(.pt)落在 ee_xy/ee_z 范围内的末端 xyz 点云

后续补充（占位）：3D 世界 / 机械臂 / 焊缝 / 已观测区域 等的 open3d & isaacsim 可视化。
"""
from __future__ import annotations

from gt_gen.scene import Scene, _load_plan_init_pose


def _quat_wxyz_to_R(q):
    """四元数 wxyz -> 3×3 旋转矩阵（归一化后）。"""
    import numpy as np
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def _fk_arm_spheres(scene, joints):
    """机械臂在关节角 joints（长度=DOF，rad）下的整臂碰撞球，**base_link 系**。

    curobo CudaRobotModel 做 per-link FK，取每个 collision link 位姿，再套 robot yml 的
    collision_spheres（各球的 center/radius，link 局部系）变换到 base 系。robot base 在原点，
    故 base 系即渲染系。跳过 radius<=1e-4 的占位球。须在有 curobo 的进程调用。
    返回 (centers(N,3) ndarray, radii(N,) ndarray, link_names(list[str] 长 N))。
    """
    import numpy as np
    from gt_gen import compat as _compat
    _compat.apply_trimesh_shim()                        # warp/trimesh shim（须在 curobo 前）
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import load_yaml

    rd = load_yaml(scene.cfg.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    coll_links = list(kin["collision_link_names"])
    spheres_def = kin["collision_spheres"]
    kin["link_names"] = coll_links                       # 让 FK 输出所有碰撞 link 的位姿
    ta = TensorDeviceType()
    model = CudaRobotModel(RobotConfig.from_dict(rd["robot_cfg"], ta).kinematics)
    st = model.get_state(torch.tensor([list(joints)], dtype=torch.float32, device=ta.device))

    centers, radii, links = [], [], []
    for ln in coll_links:
        if ln not in spheres_def:
            continue
        pos = st.link_pose[ln].position[0].detach().cpu().numpy()
        R = _quat_wxyz_to_R(st.link_pose[ln].quaternion[0].detach().cpu().numpy())
        for s in spheres_def[ln]:
            r = float(s["radius"])
            if r <= 1e-4:
                continue
            centers.append(R @ np.asarray(s["center"], float) + pos)
            radii.append(r); links.append(ln)
    return np.asarray(centers, float), np.asarray(radii, float), links


def _goal_arm_collision(scene, goal_joints, wp_pose7, T):
    """到达 goal 观测位姿的关节角 goal_joints 的三分碰撞检测：自碰撞 / 碰工件 / 碰障碍。

    curobo CudaRobotModel 做 FK 得每颗碰撞球在 base 系的球心+半径（_fk_arm_spheres），再：
      · 自碰撞：球心距 < r_i+r_j（跳过同一 link 内球对 + self_collision_ignore 里成对的相邻 link）；
      · 工件/障碍：trimesh ProximityQuery.signed_distance(球心)+半径 > 0 即球体入网格。
    工件/障碍 mesh 均按 base 系摆放（工件用 wp_pose7、障碍实体各 apply T），与 FK 球同框
    （robot base 在原点，故 base 系=渲染系）。须在 SimulationApp 启动【之后】调用（curobo import 顺序）。
    返回 dict(self, workpiece, obstacle: bool; n_self, n_work, n_obs: int)。
    """
    import numpy as np
    import trimesh
    from curobo.util_file import load_yaml
    from gt_gen.sensor import load_truth_scene

    centers, radii, links = _fk_arm_spheres(scene, goal_joints)
    n = len(centers)

    # —— 自碰撞：球-球，跳过同 link + self_collision_ignore 相邻 link（对称）——
    kin = load_yaml(scene.cfg.robot_cfg_path)["robot_cfg"]["kinematics"]
    ignore = set()
    for a, nbrs in (kin.get("self_collision_ignore") or {}).items():
        for b in nbrs:
            ignore.add(frozenset((a, b)))
    n_self = 0
    for i in range(n):
        for j in range(i + 1, n):
            if links[i] == links[j] or frozenset((links[i], links[j])) in ignore:
                continue
            if float(np.linalg.norm(centers[i] - centers[j])) < radii[i] + radii[j]:
                n_self += 1

    # —— 工件 / 障碍：trimesh signed_distance（网格内为正）——
    def _n_hits(mesh):
        if mesh is None or n == 0:
            return 0
        sd = trimesh.proximity.ProximityQuery(mesh).signed_distance(centers)
        return int(np.count_nonzero(sd + radii > 0))

    n_work = _n_hits(load_truth_scene(scene.workpiece_obj, mesh_pose=wp_pose7))

    obs_tms = scene._obstacle_solid_trimeshes()          # open_cylinder 纯视觉，已自动跳过
    obs_mesh = None
    if obs_tms:
        tms = [tm.copy() for tm in obs_tms]
        for tm in tms:
            tm.apply_transform(np.asarray(T, float))     # 工件 mesh 系 → base 系（与工件同一 T）
        obs_mesh = trimesh.util.concatenate(tms) if len(tms) > 1 else tms[0]
    n_obs = _n_hits(obs_mesh)

    return {"self": n_self > 0, "workpiece": n_work > 0, "obstacle": n_obs > 0,
            "n_self": n_self, "n_work": n_work, "n_obs": n_obs}


def _pose7_to_T(pose7):
    """pose7 [x,y,z, qw,qx,qy,qz] → 4×4 齐次矩阵。"""
    import numpy as np
    p = np.asarray(pose7, float)
    t = p[:3]
    w, x, y, z = p[3:]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _build_scenepose2_debug(scene, include_obstacles=True, device=None):
    """【忠实复刻 Scene.compute_goal_pose 的 ScenePose2 搭建】，供碰撞可视化诊断复用。

    与 compute_goal_pose 同口径：读 configs/default.yaml 的 compute_goal_pose 段建 ConfigurationPose、
    seam 数据、robot_pose7=pose7(inv(T_workpiece_in_base))、piece=identity，建 ScenePose2 + reset
    （含障碍则 _inject_obstacles_into_scenepose2），再建 OptimizerPose + resetSeamData。
    返回 dict(scene2, optimizer, device, want_obs)。前提：先 set_init_pose。⚠ 会初始化 curobo/warp。"""
    import numpy as np
    import torch
    from gt_gen.scene import _load_compute_goal_poses2, _load_plan_init_pose

    if scene.cur_init_pose is None:
        raise RuntimeError("需要当前 init pose：请先 set_init_pose(hand, index)")

    sec = dict(scene.cfg.raw.get("compute_goal_pose", {}) or {})
    horizontal = int(sec.get("horizontal", 2))
    dev = device or str(sec.get("device", "cuda"))

    cgp = _load_compute_goal_poses2()
    Configuration, Optimizer, ScenePose2 = cgp.Configuration, cgp.Optimizer, cgp.Scene
    cfg = Configuration()
    cfg.usd_path = ""
    cfg.pc_path = ""
    for k, v in sec.items():
        if k in ("include_obstacles", "horizontal", "device"):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.num_randoms_all = cfg.num_randoms_new + cfg.num_randoms_old + 1
    cfg.num_envs = cfg.num_batches * (cfg.num_randoms_new + cfg.num_randoms_old)

    sl, st_, slim = scene._seam_data_arrays()
    seam_line = torch.as_tensor(sl, dtype=torch.float, device=dev)
    seam_tangent = torch.as_tensor(st_, dtype=torch.float, device=dev)
    seam_limits = torch.as_tensor(slim, dtype=torch.float, device=dev)

    pim = _load_plan_init_pose()
    T = np.asarray(scene.cur_init_pose.T_workpiece_in_base, float)
    robot_pose7 = np.asarray(pim.mat44_to_pose7(np.linalg.inv(T)), float)
    robot_pose_t = torch.as_tensor(robot_pose7, dtype=torch.float, device=dev)
    piece_pose_t = torch.as_tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=dev)

    scene2 = ScenePose2(cfg, num_envs=cfg.num_envs, device=dev,
                        obj_path=scene.workpiece_obj, robot_cfg_path=scene.cfg.robot_cfg_path)
    scene2.reset(robot_pose_t, horizontal, piece_pose_t)
    want_obs = bool(include_obstacles and scene.obstacles.get(scene.seam_id))
    if want_obs:
        scene._inject_obstacles_into_scenepose2(scene2)

    optimizer = Optimizer(cfg=cfg, scene=scene2, device=dev)
    optimizer.resetSeamData(seam_line, seam_tangent, seam_limits)
    return dict(scene2=scene2, optimizer=optimizer, device=dev, want_obs=want_obs)


def _ik_select_candidates(scene2, cam_poses_piece, joints_opt, device):
    """对一批候选相机位姿（piece 系，(M,7)）复刻 scene_pose2.getJoints 的 IK 解析+选解口径，
    逐候选返回 dict(cam(piece7), joints(6), ik_fail(bool))。用于挑一个"能到的" IK 构型来可视化。"""
    import numpy as np
    import torch
    import scene_pose2 as sp2

    M = cam_poses_piece.shape[0]
    K = 2
    robot_inv = scene2.robot_base_inv_pose.clone().repeat(M, 1)
    cam_quat, cam_pos = sp2.math.tf_combine(robot_inv[:, 3:], robot_inv[:, :3],
                                            cam_poses_piece[:, 3:], cam_poses_piece[:, :3])
    ik_solver = sp2.UR12e_t(num_envs=M, device=device)
    R_mat = sp2.math.matrix_from_quat(cam_quat)
    T_mat = scene2.generate_transformation(cam_pos, R_mat)
    ik = torch.nan_to_num(ik_solver.solve_fairino_ec(T_mat).float(), nan=-20)      # (M,8,6)
    outb = torch.any((ik < scene2.joints_lower_limit) | (ik > scene2.joints_upper_limit), dim=-1)
    rewards = -torch.norm(ik - joints_opt.unsqueeze(-2), dim=-1)
    rewards[outb] -= 500
    rmax, midx = torch.topk(rewards, k=K, dim=1)                                    # (M,K)
    ik_sel = torch.gather(ik, 1, midx.unsqueeze(-1).expand(-1, -1, 6))              # (M,K,6)
    outb_sel = torch.gather(outb, 1, midx)                                          # (M,K)
    # 首选 K 中未越界者；全越界则记 ik_fail
    best = []
    for m in range(M):
        kk = 0 if (not bool(outb_sel[m, 0])) else (1 if not bool(outb_sel[m, 1]) else 0)
        ikfail = bool(outb_sel[m].all()) or bool((rmax[m] <= -400).all())
        q = ik_sel[m, kk].detach().cpu().numpy()
        if ikfail:                                                                 # 兜底：用 retract
            q = joints_opt[0].detach().cpu().numpy()
        best.append(dict(cam=cam_poses_piece[m].detach().cpu().numpy(),
                         joints=[float(v) for v in q], ik_fail=ikfail))
    return best




class SceneVisualizer:
    """Scene 可视化基类：持有 Scene，具体渲染由后端子类实现。"""

    def __init__(self, scene: Scene):
        if not isinstance(scene, Scene):
            raise TypeError(f"SceneVisualizer 需要 Scene 实例，收到 {type(scene)}")
        self.scene = scene


class Open3DSceneVisualizer(SceneVisualizer):
    """open3d 后端可视化（本机开窗）。"""

    def show_init_poses(self, stride: int = 1, show_init_arm: bool = True, init_joints=None):
        """逐个可视化该 Scene 焊缝的【候选初始位姿】（工件相对机械臂的摆放）。

        前提：先调用 Scene.plan_init_pose() 求出候选。可视化复用 scripts/plan_init_pose.py 的
        _show_kejian2_results（正手→反手依次开窗：按 (R,t) 摆放的工件网格 + 焊缝/中点 + standoff 落点
        + 蓝色 bisector 正反手判据轴 + 【焊接位姿 sol["q"] 的整臂碰撞球，红色线框】）；同一窗口按
        **C 键** 切到下一个候选。

        show_init_arm=True（默认）：额外叠加机械臂在【初始关节角】(retract，缺省 scene.cur_cfg=
        cfg.retract_config) 的整臂碰撞球，**各 link 用不同颜色**（区别于焊接位姿那条红色臂），
        便于对照"起步 home 姿态"与工件/障碍的相对关系（如判断初始臂是否已插进工件——见诊断）。
        init_joints：显式指定要画的初始关节角（rad，长度=关节数）；None→用 scene.cur_cfg。

        若 Scene 已放障碍（add_obstacle_type2/type3），障碍随工件一起按各候选的 T_workpiece_in_base
        摆到 base 系一并显示（经 _show_kejian2_results 的 extra_geoms 钩子；open_cylinder 也含在内）。

        stride：每隔几个候选抽 1 个看（默认 1=逐个看全部）。
        """
        scene = self.scene
        cands = scene.init_pose_candidates.get(scene.seam_id, {})
        if not cands or not (cands.get("forehand") or cands.get("backhand")):
            raise RuntimeError(
                "无候选初始位姿可视化：请先调用 Scene.plan_init_pose()（且求解成功）")
        pim = _load_plan_init_pose()
        res = {"forehand": [c.raw for c in cands.get("forehand", [])],
               "backhand": [c.raw for c in cands.get("backhand", [])]}

        # extra_geoms 钩子：障碍（随工件按 (R,t) 摆放）+ 初始关节角机械臂碰撞球（base 系、与候选无关）
        factories = []
        if scene.obstacles.get(scene.seam_id):
            factories.append(self._obstacle_o3d_factory())
        if show_init_arm:
            factories.append(self._init_arm_o3d_factory(init_joints))
        factories.append(self._init_free_o3d_factory())
        extra = None
        if factories:
            def extra(R, t, _fs=factories):
                geoms = []
                for f in _fs:
                    geoms.extend(f(R, t))
                return geoms

        pim._show_kejian2_results(scene.cfg, scene.workpiece_obj, scene.seam, res,
                                  stride=stride, extra_geoms=extra)

    # 逐步过滤 8 个步骤的名字（与 plan_init_pose._kejian2_solve_weld 打印口径一致，n 从 1 起）
    _DEBUG_STEP_NAMES = [
        "① lookup 候选（绕末端轴采样）",
        "② 工作空间 + 朝向snap 命中",
        "③ 正面过滤(bisector base-z≥0)",
        "④ 焊缝中心 base-x>0",
        "⑤ 工件+障碍 vs init_free 空间无交集",
        "⑥ 底座-工件 XY 投影不相交",
        "⑦ 轻去重(同朝向 + 2cm 同位)",
        "⑧ snap后 retract-工件无碰撞复检（合格）",
    ]

    def show_init_poses_debug(self, n: int, stride: int = 1,
                              show_init_arm: bool = True, init_joints=None,
                              sort_by_seam_x: bool = True):
        """可视化【逐步过滤第 n 步通过】的候选位姿（n=1..8），其余可视化与 show_init_poses 完全一致。

        n 对应 _kejian2_solve_weld 打印的 8 个步骤（见 _DEBUG_STEP_NAMES）：
          1=lookup 原始候选(未 snap)、2=工作空间+朝向 snap、3=正面过滤、4=焊缝中心 base-x>0、
          5=工件最近点、6=底座-工件 XY 不相交、7=轻去重后(复检前)、8=复检后合格。
        前提：先调用 Scene.plan_init_pose()（求解成功后自动把逐步候选存进 scene.init_pose_debug_steps）。
        与 show_init_poses 一致：正手→反手依次开窗，工件 mesh + 焊缝/中点 + standoff 落点 + 蓝色
        bisector 轴 + 焊接位姿整臂碰撞球（红），障碍随工件摆放、初始关节角机械臂碰撞球(异色)一并叠加；
        按 C 切下一个、直接关窗退出。stride：每隔几个抽 1 个。
        sort_by_seam_x：True 时每手候选按【焊缝中点在 base-x 分量（seam_center_base[0]）】从大到小排序后再开窗。"""
        scene = self.scene
        steps = scene.init_pose_debug_steps.get(scene.seam_id)
        if not steps:
            raise RuntimeError(
                "无逐步 debug 候选：请先调用 Scene.plan_init_pose()（且求解成功）")
        n = int(n)
        if not (1 <= n <= len(steps)):
            raise ValueError(f"n 需在 1..{len(steps)}（8 个过滤步骤），收到 {n}")
        step_list = steps[n - 1]
        name = self._DEBUG_STEP_NAMES[n - 1]
        if not step_list:
            raise RuntimeError(f"第 {n} 步「{name}」无通过候选（0 个），无可视化")
        pim = _load_plan_init_pose()
        # 按 hand 分组喂 _show_kejian2_results（该函数依 forehand→backhand 顺序开窗）；debug 存的
        # 就是 raw 结果 dict（含 T_workpiece_in_base/joint_angles/bisector_base/… 可视化所需字段），
        # 无需 .raw / from_kejian2，直接使用。
        res = {"forehand": [c for c in step_list if c.get("hand") == "forehand"],
               "backhand": [c for c in step_list if c.get("hand") == "backhand"]}

        if sort_by_seam_x:
            def _seam_x(c):
                sc = c.get("seam_center_base")
                return float(sc[0]) if sc is not None else float("-inf")
            for _h in ("forehand", "backhand"):
                res[_h].sort(key=_seam_x, reverse=True)

        # extra_geoms 钩子：与 show_init_poses 相同（障碍随工件摆放 + 初始关节角机械臂碰撞球）
        factories = []
        if scene.obstacles.get(scene.seam_id):
            factories.append(self._obstacle_o3d_factory())
        if show_init_arm:
            factories.append(self._init_arm_o3d_factory(init_joints))
        factories.append(self._init_free_o3d_factory())
        extra = None
        if factories:
            def extra(R, t, _fs=factories):
                geoms = []
                for f in _fs:
                    geoms.extend(f(R, t))
                return geoms

        print(f"[viz] 逐步过滤第 {n} 步「{name}」通过候选 {len(step_list)} 个"
              f"（正手 {len(res['forehand'])} / 反手 {len(res['backhand'])}）")
        pim._show_kejian2_results(scene.cfg, scene.workpiece_obj, scene.seam, res,
                                  stride=stride, extra_geoms=extra)

    # 预筛三阶段的名字（与 plan_init_pose.solve_one_weld_lookup 漏斗打印一致，n 从 1 起）
    _PREFILTER_STEP_NAMES = [
        "端点∈工作空间",
        "+朝向粗筛(列夹角<3θ+15°)",
        "碰撞safe(link+retract vs 工件ESDF)",
    ]

    def show_init_pose_prefilter(self, stage, stride: int = 1,
                                 show_init_arm: bool = True, init_joints=None,
                                 sort_by_seam_x: bool = True):
        """可视化【预筛阶段】随机抽到的候选位姿（plan_init_pose(diagnostic=True) 时 lookup 内各阶段
        随机抽 ≤10 个），其余可视化与 show_init_poses / show_init_poses_debug 完全一致（画的是未 snap 的
        lookup 原始 (R,t) 位姿，goal 不定）。

        stage：阶段序号 1..3 或阶段名（见 _PREFILTER_STEP_NAMES）——
          1=端点∈工作空间、2=+朝向粗筛(列夹角<3θ+15°)、3=碰撞safe(link+retract vs 工件ESDF)。
        对应 solve_one_weld_lookup 打印的预筛漏斗；用于诊断候选在哪一层被筛光（如 safe=0 时看前两层）。
        前提：先 Scene.plan_init_pose(diagnostic=True)（非诊断模式各阶段为空 → 抛错）。

        与 show_init_poses 一致：正手→反手依次开窗，工件 mesh + 焊缝/中点 + standoff 落点 + 蓝色
        bisector 轴 + lookup 关节角整臂碰撞球(红)，障碍随工件摆放、初始关节角机械臂碰撞球(异色)、
        init_free 空间一并叠加；按 C 切下一个、直接关窗退出。stride：每隔几个抽 1 个。
        sort_by_seam_x：True 时每手候选按焊缝中点 base-x（seam_center_base[0]）从大到小排序后开窗。"""
        scene = self.scene
        steps = scene.init_pose_prefilter_steps.get(scene.seam_id)
        if not steps:
            raise RuntimeError(
                "无预筛抽样候选：请先 Scene.plan_init_pose(diagnostic=True)（仅诊断模式才抽样）")
        names = self._PREFILTER_STEP_NAMES
        if isinstance(stage, str):
            name = stage
            if name not in steps:
                raise ValueError(f"未知阶段名「{stage}」，可选：{list(steps.keys())}")
        else:
            n = int(stage)
            if not (1 <= n <= len(names)):
                raise ValueError(f"stage 需在 1..{len(names)}（预筛三阶段），收到 {n}")
            name = names[n - 1]
        step_list = steps.get(name, [])
        if not step_list:
            raise RuntimeError(
                f"预筛阶段「{name}」无抽样候选（0 个：该阶段幸存者为 0，或未开 diagnostic）")
        pim = _load_plan_init_pose()
        res = {"forehand": [c for c in step_list if c.get("hand") == "forehand"],
               "backhand": [c for c in step_list if c.get("hand") == "backhand"]}

        if sort_by_seam_x:
            def _seam_x(c):
                sc = c.get("seam_center_base")
                return float(sc[0]) if sc is not None else float("-inf")
            for _h in ("forehand", "backhand"):
                res[_h].sort(key=_seam_x, reverse=True)

        # extra_geoms 钩子：与 show_init_poses_debug 相同（障碍随工件摆放 + 初始臂碰撞球 + init_free 空间）
        factories = []
        if scene.obstacles.get(scene.seam_id):
            factories.append(self._obstacle_o3d_factory())
        if show_init_arm:
            factories.append(self._init_arm_o3d_factory(init_joints))
        factories.append(self._init_free_o3d_factory())
        extra = None
        if factories:
            def extra(R, t, _fs=factories):
                geoms = []
                for f in _fs:
                    geoms.extend(f(R, t))
                return geoms

        print(f"[viz] 预筛阶段「{name}」抽样候选 {len(step_list)} 个"
              f"（正手 {len(res['forehand'])} / 反手 {len(res['backhand'])}）")
        pim._show_kejian2_results(scene.cfg, scene.workpiece_obj, scene.seam, res,
                                  stride=stride, extra_geoms=extra)

    def show_joint_table_ee(self, path=None, show_out_of_range: bool = True,
                            show_range_shell: bool = True, show_init_arm: bool = True):
        """open3d 可视化【预计算关节表】(configs/plan_init_joint_table.pt) 里落在工作空间范围内的
        末端 xyz 点云（需 env_isaaclab 环境：装了 open3d + torch）。

        关节表由 precompute_joint_table 产出（与工件无关的 n^dof 关节采样），其中 ee_pos_t 是各采样
        关节角对应【焊枪末端 tip_link 在 base_link 系的位置】。本方法按 default.yaml plan_init_pose 的
        两组范围过滤并画成点云：
          · ee_xy_range_m=[lo,hi]：末端在 base xy 平面到原点距离 ∈ [lo,hi]（径向环）；
          · ee_z_range_m=[lo,hi]：末端 base-z ∈ [lo,hi]。
        命中点画绿色；show_out_of_range=True 时其余末端画淡灰便于对照；叠加 base 坐标系；
        show_range_shell=True 时叠加范围壳线框（内/外半径 xy_lo/xy_hi、z∈[z_lo,z_hi] 的两个竖直
        圆柱，标出被过滤的环形区域）；show_init_arm=True 时叠加【初始关节角】(retract) 整臂碰撞球
        线框（各 link 异色，见 _init_arm_o3d_factory），便于看末端点相对机械臂的位置。

        path：关节表 .pt 路径，None→cfg.plan_init_joint_table_path（= default.yaml
        plan_init_pose.joint_table_path，已解析为绝对路径）。
        """
        import os
        import numpy as np
        import open3d as o3d
        import torch

        cfg = self.scene.cfg
        fp = path or cfg.plan_init_joint_table_path
        if not os.path.isfile(fp):
            raise FileNotFoundError(
                f"关节表不存在：{fp}（请先 precompute_joint_table + save_joint_table）")
        try:
            payload = torch.load(fp, map_location="cpu", weights_only=False)
        except TypeError:                               # 老版 torch 无 weights_only 形参
            payload = torch.load(fp, map_location="cpu")
        ee = np.asarray(payload["ee_pos_t"], dtype=float)   # (N,3) 末端在 base 系
        if ee.ndim != 2 or ee.shape[1] != 3:
            raise RuntimeError(f"ee_pos_t 形状异常：{ee.shape}（应为 (N,3)）")

        xy = np.linalg.norm(ee[:, :2], axis=1)
        z = ee[:, 2]
        xy_lo, xy_hi = (float(v) for v in cfg.plan_init_ee_xy_range)
        z_lo, z_hi = (float(v) for v in cfg.plan_init_ee_z_range)
        in_mask = (xy >= xy_lo) & (xy <= xy_hi) & (z >= z_lo) & (z <= z_hi)
        n_in = int(in_mask.sum())
        print(f"[viz] 关节表末端点 {len(ee)} 个 <- {fp}")
        print(f"[viz] 范围内 {n_in} 个（xy∈[{xy_lo:.3f},{xy_hi:.3f}] & "
              f"z∈[{z_lo:.3f},{z_hi:.3f}]），范围外 {len(ee) - n_in} 个")
        if n_in == 0:
            raise RuntimeError("范围内末端点为 0：请放宽 ee_xy_range_m/ee_z_range_m 或检查关节表")

        geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

        if show_out_of_range and n_in < len(ee):
            pc_out = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(ee[~in_mask]))
            pc_out.paint_uniform_color([0.75, 0.75, 0.78])   # 淡灰
            geoms.append(pc_out)

        pc_in = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ee[in_mask]))
        pc_in.paint_uniform_color([0.10, 0.80, 0.20])        # 绿
        geoms.append(pc_in)

        if show_range_shell:
            h = max(1e-4, z_hi - z_lo)

            def _shell_cyl(radius):
                cyl = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=max(radius, 1e-4), height=h, resolution=48)
                cyl.translate((0.0, 0.0, (z_lo + z_hi) / 2.0))
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(cyl)
                ls.paint_uniform_color([0.0, 0.55, 0.85])    # 蓝
                return ls

            geoms.append(_shell_cyl(xy_lo))
            geoms.append(_shell_cyl(xy_hi))

            # 断面「方形」：在若干角向画 [xy_lo,xy_hi]×[z_lo,z_hi] 的矩形环线，直观展示被过滤区域
            for ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                cx, cy = np.cos(ang), np.sin(ang)
                pts = np.array([
                    [xy_lo * cx, xy_lo * cy, z_lo], [xy_hi * cx, xy_hi * cy, z_lo],
                    [xy_hi * cx, xy_hi * cy, z_hi], [xy_lo * cx, xy_lo * cy, z_hi],
                ], dtype=float)
                rect = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(pts),
                    o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]]))
                rect.paint_uniform_color([0.0, 0.55, 0.85])  # 蓝
                geoms.append(rect)

        if show_init_arm:
            # 初始关节角(retract) 整臂碰撞球线框；臂固定在 base 系，(R,t) 无关，取 factory 缓存即可
            geoms.extend(self._init_arm_o3d_factory()(None, None))

        o3d.visualization.draw_geometries(
            geoms, window_name=f"joint_table ee（范围内 {n_in}/{len(ee)}）")

    def _init_arm_o3d_factory(self, init_joints=None):
        """返回回调 (R,t)->list[o3d.geometry]：机械臂在【初始关节角】(retract，缺省 scene.cur_cfg)
        的整臂碰撞球，画成线框球，**各 link 一种颜色**（避开红色=焊接位姿臂，便于区分）。

        碰撞球由 compute_link_sweep 做 per-link FK 得到（base 系，robot base 在原点）；臂姿固定，
        与候选工件摆放 (R,t) 无关，故一次 FK 缓存、对所有候选返回同一套几何。"""
        import numpy as np
        import open3d as o3d
        from gt_gen.obstacle_placement import compute_link_sweep

        scene = self.scene
        q = [float(v) for v in (init_joints if init_joints is not None else scene.cur_cfg)]
        per_wp, _ = compute_link_sweep(scene.cfg, [q], scene.cfg.collision_link_names)

        # 每 link 一种颜色（避开红色 [0.85,0.1,0.1]=焊接位姿臂）
        palette = [
            [0.10, 0.45, 0.90],   # 蓝
            [0.10, 0.75, 0.75],   # 青
            [0.55, 0.25, 0.85],   # 紫
            [0.95, 0.60, 0.10],   # 橙
            [0.20, 0.70, 0.30],   # 绿
            [0.85, 0.75, 0.10],   # 黄
            [0.55, 0.40, 0.22],   # 棕
            [0.40, 0.40, 0.45],   # 灰
            [0.10, 0.30, 0.55],   # 深蓝
            [0.90, 0.45, 0.75],   # 粉
            [0.30, 0.80, 0.55],   # 蓝绿
            [0.65, 0.85, 0.15],   # 黄绿
        ]
        cached = []
        for li, (ln, s) in enumerate(per_wp.items()):
            col = palette[li % len(palette)]
            for c in np.asarray(s, float)[0]:            # (S,4)：取唯一路点
                cx, cy, cz, r = (float(v) for v in c)
                if r <= 1e-4:
                    continue
                ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
                ball.translate((cx, cy, cz))
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color(col)
                cached.append(ls)
        print(f"[viz] 初始关节角机械臂碰撞球：{len(cached)} 球 / {len(per_wp)} link（各异色线框，"
              f"q={np.round(q, 3).tolist()}）")

        def _factory(R, t, _g=cached):
            return _g          # 臂固定在 base 系，与候选 (R,t) 无关

        return _factory


    def _init_free_o3d_factory(self):
        """返回回调 (R,t)->list[o3d.geometry]：机械臂【初始 FREE 起步引导空间】（青色线框，base 系）。
        按 cfg.init_free_method_for_init 选形状（与 Scene 起步 FREE 一致）：
          box     → AABB 盒 [init_free_box_min_for_init, init_free_box_max_for_init]；
          cylinder→ 轴过 base 原点(x=y=0)、底面 z=init_free_cyl_z_min、半径 init_free_cyl_radius、
                    高 init_free_cyl_height 的竖直圆柱线框。
        该空间固定在 base 系、与候选工件摆放 (R,t) 无关，故对所有候选返回同一几何。"""
        import numpy as np
        import open3d as o3d

        cfg = self.scene.cfg
        col = (0.0, 0.75, 0.75)
        method = getattr(cfg, "init_free_method_for_init", "box")
        cached = []
        if method == "cylinder":
            radius = float(cfg.init_free_cyl_radius)
            height = float(cfg.init_free_cyl_height)
            z_min = float(getattr(cfg, "init_free_cyl_z_min", -0.02))
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height,
                                                            resolution=32)
            cyl.translate((0.0, 0.0, z_min + height / 2.0))   # 底面落在 z_min
            ls = o3d.geometry.LineSet.create_from_triangle_mesh(cyl)
            ls.paint_uniform_color(col)
            cached.append(ls)
            print(f"[viz] init_free 圆柱（青色线框）：半径={radius:.3f}m 高={height:.3f}m "
                  f"底面 z={z_min:.3f}（轴过 base 原点）")
        else:
            lo = np.asarray(cfg.init_free_box_min_for_init, float)
            hi = np.asarray(cfg.init_free_box_max_for_init, float)
            aabb = o3d.geometry.AxisAlignedBoundingBox(lo.tolist(), hi.tolist())
            aabb.color = col
            cached.append(aabb)
            print(f"[viz] init_free 盒（青色线框）：min={np.round(lo, 3).tolist()} "
                  f"max={np.round(hi, 3).tolist()} 尺寸={np.round(hi - lo, 3).tolist()}m")

        def _factory(R, t, _g=cached):
            return _g          # FREE 空间固定在 base 系，与候选 (R,t) 无关

        return _factory


    def _obstacle_o3d_factory(self):
        """返回回调 (R,t)->list[o3d.geometry]：把 scene.obstacles（工件 mesh 系）按 T_workpiece_in_base
        摆到 base 系。障碍→trimesh 复用 scene 的 _polygon_mesh_to_trimesh / _box_prim_to_trimesh
        （板 mesh + open_cylinder mesh 走前者，open_box 的 Box 原语走后者），再转 o3d、染 ObstacleSpec.color。"""
        import numpy as np
        import open3d as o3d
        from gt_gen.scene import (_polygon_mesh_to_trimesh, _box_prim_to_trimesh,
                                   _tube_prim_to_trimesh)

        obstacles = list(self.scene.obstacles.get(self.scene.seam_id, []))

        def _factory(R, t):
            T = np.eye(4)
            T[:3, :3] = np.asarray(R, float)
            T[:3, 3] = np.asarray(t, float)
            geoms = []
            for ob in obstacles:
                col = list(ob.color) if ob.color else [0.62, 0.64, 0.67]
                tms = []
                for mesh in ob.meshes:
                    tm = _polygon_mesh_to_trimesh(mesh)
                    if tm is not None:
                        tms.append(tm)
                for prim in ob.prims:
                    tms.append(_tube_prim_to_trimesh(prim) if hasattr(prim, "radius")
                               else _box_prim_to_trimesh(prim))
                for tm in tms:
                    m = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(tm.vertices, float)),
                        o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
                    m.transform(T)                       # 工件 mesh 系 → base 系（与工件同一 R,t）
                    m.compute_vertex_normals()
                    m.paint_uniform_color(col)
                    geoms.append(m)
            return geoms

        return _factory

    def show_goal_pose_collision(self, hand: str = "forehand", index: int = 0, cand=None,
                                 joints=None, include_obstacles: bool = True,
                                 show_cone: bool = True, compare: bool = True, device=None):
        """可视化 compute_goal_pose(ScenePose2) 内部的碰撞检测：把机械臂在【某候选观测相机位姿的
        IK 构型】下的整臂碰撞球画出来，按 stomp_planning_api._viz_collision 风格分色标出碰撞部位。

        动机：诊断"show_init_poses 里初始臂不碰工件，但 compute_goal_pose 报 world_hit≈100%"的矛盾。
        疑点在坐标系——ScenePose2 判碰时工件/障碍摆在 robot_base_inv_pose(=inv(robot_pose·base_transform)，
        含 base_transform 的 0.26m 平移 + 绕z 90°)，而 show_init_poses 摆在 T_workpiece_in_base。
        compare=True 开【两窗对照】：窗A=ScenePose2 摆放(重现判碰)、窗B=真值摆放(应无碰)，肉眼即可定位。

        参数：
          hand/index       : set_init_pose 选哪只手第几个候选。
          joints           : 直接指定要画的关节角(6)；缺省=从优化器初始化的候选相机位姿里解 IK 挑一个。
          cand             : 缺省从 8 个初始候选相机位姿自动挑(优先 IK 可达)；给定则用第 cand 个。
          include_obstacles: 是否并入障碍(默认 True，与 compute_goal_pose 同口径)。
          show_cone        : 是否标出 field 视锥自遮挡球(紫，仅 Link1/2/3)。
          compare          : True 开 A/B 双窗对照(强烈建议)。
          device           : None→cfg.compute_goal_pose.device。

        颜色：灰=未碰　红=碰工件/自碰低link　绿=自碰高link　橙=碰障碍　紫=视锥自遮挡；蓝球+蓝线=相机镜头/视线。
        ⚠ 会初始化 curobo/warp，须在【未启动 SimulationApp 的干净进程】里调用。缺 open3d/无显示则打印跳过。
        """
        import numpy as np
        import torch
        import trimesh

        scene = self.scene
        scene.set_init_pose(hand, index)
        ip = scene.cur_init_pose

        ctx = _build_scenepose2_debug(scene, include_obstacles=include_obstacles, device=device)
        scene2, optimizer, dev = ctx["scene2"], ctx["optimizer"], ctx["device"]
        import scene_pose2 as sp2   # _build_scenepose2_debug 已把 stomp_planner 加进 sys.path
        retract = torch.as_tensor([list(scene.cur_cfg)], dtype=torch.float, device=dev)

        # —— 选一个 IK 构型（+ 对应候选相机位姿，供视锥） ——
        cam_piece = None
        if joints is not None:
            q_list = [float(v) for v in joints]
            picked_desc = "用户指定 joints"
        else:
            optimizer.defineVariables()
            optimizer.initializePose()
            cams = optimizer.cam_pose_optimized.detach()             # (B,7) piece 系候选相机位姿
            cand_poses = cams[:8] if cams.shape[0] >= 8 else cams    # 前 8 = 8 个基姿(端点×±45°×flip)
            picks = _ik_select_candidates(scene2, cand_poses, retract, dev)
            if cand is not None:
                ci = int(cand) % len(picks)
            else:
                valid = [i for i, p in enumerate(picks) if not p["ik_fail"]]
                ci = valid[0] if valid else 0
            q_list = picks[ci]["joints"]
            cam_piece = torch.as_tensor(picks[ci]["cam"], dtype=torch.float, device=dev)
            picked_desc = (f"候选#{ci}/{len(picks)} ik_fail={picks[ci]['ik_fail']}"
                           + ("（无 IK 解→retract 兜底，与 getJoints 一致）" if picks[ci]["ik_fail"] else ""))
        print(f"[viz] 可视化构型：{picked_desc}  q={np.round(q_list, 3).tolist()}")
        q = torch.as_tensor([q_list], dtype=torch.float, device=dev)

        # —— FK 整臂碰撞球（base_link 系；即 ScenePose2 判碰所用） ——
        kin = scene2.rw.get_kinematics(q)
        sph = kin.link_spheres_tensor[0].detach().cpu().numpy()       # (S,4)
        kc = scene2.rw.kinematics.kinematics_config
        idx_map = kc.link_sphere_idx_map.detach().cpu().numpy()       # (S,) 每球所属 link idx
        idx2name = {int(v): k for k, v in kc.link_name_to_idx_map.items()}
        keep = sph[:, 3] > 1e-4
        centers, radii = sph[keep, :3], sph[keep, 3]
        link_ids = idx_map[keep]
        link_names = [idx2name.get(int(i), str(int(i))) for i in link_ids]
        S = len(centers)

        # —— 自碰撞球对（与摆放无关；跳过同 link + self_collision_ignore） ——
        from curobo.util_file import load_yaml
        kin_yml = load_yaml(scene.cfg.robot_cfg_path)["robot_cfg"]["kinematics"]
        ignore = set()
        for a, nbrs in (kin_yml.get("self_collision_ignore") or {}).items():
            for b in nbrs:
                ignore.add(frozenset((a, b)))
        self_lo, self_hi, self_pairs = set(), set(), []
        for i in range(S):
            for j in range(i + 1, S):
                if link_names[i] == link_names[j] or frozenset((link_names[i], link_names[j])) in ignore:
                    continue
                if float(np.linalg.norm(centers[i] - centers[j])) < radii[i] + radii[j]:
                    lo, hi = (i, j) if link_ids[i] <= link_ids[j] else (j, i)
                    self_lo.add(lo); self_hi.add(hi)
                    self_pairs.append((link_names[lo], link_names[hi]))

        # —— field 视锥自遮挡球（与摆放无关；仅 Link1/2/3；需候选相机位姿） ——
        cone_hit = np.zeros(S, bool)
        cam_apex = cam_axis = None
        if show_cone and cam_piece is not None:
            rbi = scene2.robot_base_inv_pose
            cq, cp = sp2.math.tf_combine(rbi[:, 3:], rbi[:, :3],
                                         cam_piece[3:].unsqueeze(0), cam_piece[:3].unsqueeze(0))
            cam_apex = cp[0].detach().cpu().numpy()
            cam_axis = sp2.math.matrix_from_quat(cq)[0, :, 2].detach().cpu().numpy()
            fmask = np.array([ln in sp2._FIELD_LINKS for ln in link_names])
            rel = centers - cam_apex
            tt = rel @ cam_axis
            radial = np.linalg.norm(rel - tt[:, None] * cam_axis, axis=-1)
            cone_r = np.clip(tt, 0, None) / sp2._FIELD_HEIGHT * sp2._FIELD_RADIUS
            cone_hit = fmask & (tt >= -radii) & (tt <= sp2._FIELD_HEIGHT + radii) & (radial <= cone_r + radii)

        # —— 工件/障碍原始 mesh（工件 mesh 系）——两窗只是施加的摆放 T 不同 ——
        wp_raw = trimesh.load(scene.workpiece_obj, force="mesh")
        obs_tms = scene._obstacle_solid_trimeshes() if ctx["want_obs"] else []

        def _placed(T4):
            wp = wp_raw.copy(); wp.apply_transform(np.asarray(T4, float))
            om = None
            if obs_tms:
                tms = [tm.copy() for tm in obs_tms]
                for tm in tms:
                    tm.apply_transform(np.asarray(T4, float))
                om = trimesh.util.concatenate(tms) if len(tms) > 1 else tms[0]
            return wp, om

        def _hits(mesh):
            if mesh is None or S == 0:
                return np.zeros(S, bool)
            sd = trimesh.proximity.ProximityQuery(mesh).signed_distance(centers)   # 网格内为正
            return (sd + radii) > 0

        _GRAY = [0.60, 0.60, 0.62]; _RED = [0.90, 0.10, 0.10]; _GREEN = [0.10, 0.80, 0.20]
        _ORANGE = [0.95, 0.55, 0.10]; _PURPLE = [0.60, 0.20, 0.80]

        def _colors(hit_w, hit_o):
            cols = [list(_GRAY) for _ in range(S)]
            for i in range(S):
                if cone_hit[i]:
                    cols[i] = list(_PURPLE)
                if hit_o[i]:
                    cols[i] = list(_ORANGE)
                if hit_w[i]:
                    cols[i] = list(_RED)
            for i in self_lo:
                cols[i] = list(_RED)
            for i in self_hi:
                cols[i] = list(_GREEN)
            return cols

        try:
            import open3d as o3d
        except Exception as e:                                        # noqa: BLE001
            print(f"[viz] open3d 不可用，跳过：{e}")
            return

        def _mesh_geom(tm, color):
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(tm.vertices, float)),
                o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
            m.compute_vertex_normals(); m.paint_uniform_color(color)
            return m

        def _window(T4, title):
            wp_mesh, obs_mesh = _placed(T4)
            hit_w, hit_o = _hits(wp_mesh), _hits(obs_mesh)
            cols = _colors(hit_w, hit_o)
            geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
            geoms.append(_mesh_geom(wp_mesh, [0.72, 0.72, 0.75]))
            if obs_mesh is not None:
                geoms.append(_mesh_geom(obs_mesh, [0.55, 0.50, 0.35]))
            for i in range(S):
                ball = o3d.geometry.TriangleMesh.create_sphere(radius=max(float(radii[i]), 1e-3), resolution=8)
                ball.translate(tuple(float(v) for v in centers[i]))
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color(cols[i])
                geoms.append(ls)
            if cam_apex is not None:                                  # 相机镜头(蓝球) + 视线轴(蓝线)
                cb = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                cb.translate(tuple(float(v) for v in cam_apex)); cb.compute_vertex_normals()
                cb.paint_uniform_color([0.1, 0.1, 0.9]); geoms.append(cb)
                axl = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([cam_apex, cam_apex + cam_axis * sp2._FIELD_HEIGHT]),
                    lines=o3d.utility.Vector2iVector([[0, 1]]))
                axl.paint_uniform_color([0.1, 0.1, 0.9]); geoms.append(axl)
            print(f"[viz] {title}\n      碰工件球={int(hit_w.sum())}/{S}  碰障碍球={int(hit_o.sum())}  "
                  f"视锥球={int(cone_hit.sum())}  自碰对={len(self_pairs)}")
            o3d.visualization.draw_geometries(geoms, window_name=title[:60])

        # 窗A：ScenePose2 摆放（工件/障碍 @ robot_base_inv_pose）——重现 compute_goal_pose 判碰
        poseA = scene2.robot_base_inv_pose[0].detach().cpu().tolist()
        _window(_pose7_to_T(poseA), "A: ScenePose2摆放(robot_base_inv_pose)=compute_goal_pose判碰")
        # 窗B：真值摆放（工件/障碍 @ T_workpiece_in_base）——与 show_init_poses 一致
        if compare:
            _window(np.asarray(ip.T_workpiece_in_base, float),
                    "B: 真值摆放(T_workpiece_in_base)=show_init_poses一致")

    def show_goal_pose(self, hand: str = "forehand", index: int = 0,
                       variant: int = 0, goal_index: int = 0,
                       include_obstacles: bool = True, show_camera: bool = True):
        """open3d 开窗可视化【compute_goal_pose 算出的最终观测位姿】：机械臂摆到 goal pose 的关节角，
        同屏画工件 + 障碍物，按碰撞把整臂碰撞球分色，肉眼即可判断该 goal 姿态是否有碰撞。

        与 show_goal_pose_collision 的区别：那个用【优化前】候选相机位姿解 IK 挑一个构型来画；本方法
        直接取 scene.goal_poses[seam_id]["joints"][variant, goal_index]（compute_goal_pose 收敛后的
        最终关节角）来摆臂——即真正会被规划/执行的那条观测姿态。

        坐标系：整臂碰撞球由 curobo FK 出，天然在 base_link 系（robot base 在原点）；工件/障碍按当前
        init pose 的 T_workpiece_in_base 摆到同一 base 系（= show_init_poses 的真值摆放），故与 FK 球
        同框、碰撞判定即物理真值。碰撞判定：工件/障碍用 trimesh signed_distance(球心)+半径>0；自碰撞
        用球-球距<半径和（跳过同 link + self_collision_ignore 相邻 link）。

        参数：
          hand/index       : 先 set_init_pose 选哪只手第几个候选（须与算 goal pose 时同一候选）。
          variant          : joints 第一维 K（近等价变体）索引，默认 0。
          goal_index       : joints 第二维 B（第几个观测位姿）索引，默认 0。
          include_obstacles: 是否画障碍并计入碰撞（默认 True，与 compute_goal_pose 同口径）。
          show_camera      : 是否在 goal 相机位姿处画蓝色镜头球 + 视线轴（cam_pose 经 T 变到 base 系）。

        goal pose 结果优先取 trajectory_goal_poses[(hand,index)] 快照（该候选已成功规划过时），否则取
        全局 scene.goal_poses[seam_id]。颜色：灰=未碰　红=碰工件/自碰低link　绿=自碰高link　橙=碰障碍；
        蓝球+蓝线=相机镜头/视线。⚠ 会初始化 curobo/warp，须在【未启动 SimulationApp 的干净进程】里调用。
        """
        import numpy as np
        import trimesh

        scene = self.scene
        ip = scene.set_init_pose(hand, index)
        T = np.asarray(ip.T_workpiece_in_base, float)

        # —— 取 goal pose 结果：优先该候选的快照，否则全局 goal_poses ——
        key = (scene.cur_init_hand, scene.cur_init_index)
        res = (scene.trajectory_goal_poses.get(scene.seam_id, {}) or {}).get(key)
        if res is None:
            res = scene.goal_poses.get(scene.seam_id)
        if res is None:
            raise RuntimeError(
                "无 goal pose 可视化：请先 compute_goal_pose()（或 compute_pose_and_plan_path）")

        jt = res["joints"]
        joints_all = jt.detach().cpu().numpy() if hasattr(jt, "detach") else np.asarray(jt)
        K, B = joints_all.shape[:2]                       # (K 变体, B 观测位姿, DOF)
        vi = max(0, min(int(variant), K - 1))
        bi = max(0, min(int(goal_index), B - 1))
        q_list = [float(v) for v in joints_all[vi, bi]]
        print(f"[viz] goal pose 关节角 variant#{vi}/{K} 观测#{bi}/{B}  q={np.round(q_list, 3).tolist()}")

        # —— FK 整臂碰撞球（base 系）——
        centers, radii, links = _fk_arm_spheres(scene, q_list)
        S = len(centers)

        # —— 自碰撞球对（跳过同 link + self_collision_ignore）；base 侧 link 红、tip 侧 link 绿 ——
        from curobo.util_file import load_yaml
        kin_yml = load_yaml(scene.cfg.robot_cfg_path)["robot_cfg"]["kinematics"]
        ignore = set()
        for a, nbrs in (kin_yml.get("self_collision_ignore") or {}).items():
            for b in nbrs:
                ignore.add(frozenset((a, b)))
        rank = {}                                         # link 首次出现次序≈base→tip（_fk_arm_spheres 按 coll_links 序）
        for ln in links:
            rank.setdefault(ln, len(rank))
        self_lo, self_hi, self_pairs = set(), set(), []
        for i in range(S):
            for j in range(i + 1, S):
                if links[i] == links[j] or frozenset((links[i], links[j])) in ignore:
                    continue
                if float(np.linalg.norm(centers[i] - centers[j])) < radii[i] + radii[j]:
                    lo, hi = (i, j) if rank[links[i]] <= rank[links[j]] else (j, i)
                    self_lo.add(lo); self_hi.add(hi)
                    self_pairs.append((links[lo], links[hi]))

        # —— 工件 / 障碍 mesh（经 T 摆到 base 系）——
        wp_mesh = trimesh.load(scene.workpiece_obj, force="mesh").copy()
        wp_mesh.apply_transform(T)
        obs_mesh = None
        if include_obstacles and scene.obstacles.get(scene.seam_id):
            obs_tms = scene._obstacle_solid_trimeshes()   # open_cylinder 纯视觉，已自动跳过
            if obs_tms:
                tms = [tm.copy() for tm in obs_tms]
                for tm in tms:
                    tm.apply_transform(T)
                obs_mesh = trimesh.util.concatenate(tms) if len(tms) > 1 else tms[0]

        def _hits(mesh):
            if mesh is None or S == 0:
                return np.zeros(S, bool)
            sd = trimesh.proximity.ProximityQuery(mesh).signed_distance(centers)   # 网格内为正
            return (sd + radii) > 0

        hit_w, hit_o = _hits(wp_mesh), _hits(obs_mesh)

        _GRAY = [0.60, 0.60, 0.62]; _RED = [0.90, 0.10, 0.10]
        _GREEN = [0.10, 0.80, 0.20]; _ORANGE = [0.95, 0.55, 0.10]
        cols = [list(_GRAY) for _ in range(S)]
        for i in range(S):
            if hit_o[i]:
                cols[i] = list(_ORANGE)
            if hit_w[i]:
                cols[i] = list(_RED)
        for i in self_lo:
            cols[i] = list(_RED)
        for i in self_hi:
            cols[i] = list(_GREEN)

        # —— 相机镜头/视线（cam_pose piece 系 → base 系）——
        cam_apex = cam_axis = None
        if show_camera and res.get("cam_pose") is not None:
            cp = res["cam_pose"]
            cam_all = cp.detach().cpu().numpy() if hasattr(cp, "detach") else np.asarray(cp)
            camT = T @ _pose7_to_T(cam_all[vi, bi])       # piece/mesh 系 → base 系
            cam_apex = camT[:3, 3]
            cam_axis = camT[:3, 2]                         # +z = 视线方向

        try:
            import open3d as o3d
        except Exception as e:                            # noqa: BLE001
            print(f"[viz] open3d 不可用，跳过：{e}")
            return

        def _mesh_geom(tm, color):
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(tm.vertices, float)),
                o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
            m.compute_vertex_normals(); m.paint_uniform_color(color)
            return m

        geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
        geoms.append(_mesh_geom(wp_mesh, [0.72, 0.72, 0.75]))
        if obs_mesh is not None:
            geoms.append(_mesh_geom(obs_mesh, [0.55, 0.50, 0.35]))
        for i in range(S):
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=max(float(radii[i]), 1e-3), resolution=8)
            ball.translate(tuple(float(v) for v in centers[i]))
            ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
            ls.paint_uniform_color(cols[i])
            geoms.append(ls)
        if cam_apex is not None:                          # 相机镜头(蓝球) + 视线轴(蓝线)
            cb = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            cb.translate(tuple(float(v) for v in cam_apex)); cb.compute_vertex_normals()
            cb.paint_uniform_color([0.1, 0.1, 0.9]); geoms.append(cb)
            axl = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([cam_apex, cam_apex + cam_axis * 0.4]),
                lines=o3d.utility.Vector2iVector([[0, 1]]))
            axl.paint_uniform_color([0.1, 0.1, 0.9]); geoms.append(axl)

        n_hit_total = int(hit_w.sum()) + int(hit_o.sum()) + len(self_pairs)
        print(f"[viz] goal pose 碰撞：碰工件球={int(hit_w.sum())}/{S}  碰障碍球={int(hit_o.sum())}  "
              f"自碰对={len(self_pairs)}"
              + (f"  自碰 link 对={sorted(set(self_pairs))}" if self_pairs else "")
              + ("  → 该 goal 姿态【无碰撞】" if n_hit_total == 0 else "  → 该 goal 姿态【有碰撞】"))
        title = (f"goal pose {scene.cur_init_hand}#{scene.cur_init_index} K#{vi} 观测#{bi}"
                 f"（工件{int(hit_w.sum())}/障碍{int(hit_o.sum())}/自碰{len(self_pairs)}）")
        o3d.visualization.draw_geometries(geoms, window_name=title[:70])

    def show_seam(self, seam_id: int):
        """在 **open3d** 中可视化【指定焊缝】：工件网格 + 该焊缝红色直线 + 障碍物（如有）。

        坐标系为【工件 mesh 系】（工件停在自身坐标 identity）：焊缝端点 p0_world↔p1_world、
        障碍（self.obstacles[seam_id]，本就在工件 mesh 系）与工件同框直接叠加，无需变换。
        焊缝以 p0→p1 的 **单条红色直线**（o3d.LineSet）表示。障碍网格来自 ObstacleSpec.meshes
        （棱柱/圆筒，_polygon_mesh_to_trimesh）与 prims（Box 原语，_box_prim_to_trimesh），
        各染 ObstacleSpec.color。不画机械臂（mesh 系无工件↔base 摆放）。
        """
        import numpy as np
        import open3d as o3d
        import trimesh
        from gt_gen.scene import (_polygon_mesh_to_trimesh, _box_prim_to_trimesh,
                                   _tube_prim_to_trimesh)

        scene = self.scene
        seams = scene.seams
        if not (-len(seams) <= int(seam_id) < len(seams)):
            raise IndexError(f"seam_id 越界：{seam_id}，共 {len(seams)} 条焊缝")
        sid = int(seam_id) % len(seams)
        w = seams[sid]

        geoms = []

        # 工件网格（mesh 系 identity，灰）
        tm = trimesh.load(scene.workpiece_obj, force="mesh")
        wp = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(tm.vertices, float)),
            o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
        wp.compute_vertex_normals()
        wp.paint_uniform_color([0.72, 0.72, 0.72])
        geoms.append(wp)

        # 焊缝红色直线 p0→p1：用细圆柱 mesh 表示（LineSet 线宽仅 1px 且贴面易被遮挡，看不清）
        p0 = np.asarray(w["p0_world"], float)
        p1 = np.asarray(w["p1_world"], float)
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            raise RuntimeError(f"焊缝#{sid} 端点重合（p0==p1），无法画直线")
        radius = max(0.001, L * 0.005)           # 半径随焊缝长度自适应，至少 1mm，保证可见
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=16)
        # 默认圆柱沿 +z、中心在原点 → 旋转 z 轴到 seam 方向，再平移到中点
        d_hat = seg / L
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, d_hat); s = float(np.linalg.norm(v)); c = float(np.dot(z, d_hat))
        if s < 1e-9:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        cyl.rotate(R, center=np.zeros(3))
        cyl.translate((p0 + p1) / 2.0)
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(cyl)

        # 障碍（mesh 系，直接叠加；open_cylinder 也含在内）
        obstacles = list(scene.obstacles.get(sid, []))
        for ob in obstacles:
            col = list(ob.color) if ob.color else [0.62, 0.64, 0.67]
            tms = []
            for mesh in ob.meshes:
                t = _polygon_mesh_to_trimesh(mesh)
                if t is not None:
                    tms.append(t)
            for prim in ob.prims:
                tms.append(_tube_prim_to_trimesh(prim) if hasattr(prim, "radius")
                           else _box_prim_to_trimesh(prim))
            for t in tms:
                m = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(t.vertices, float)),
                    o3d.utility.Vector3iVector(np.asarray(t.faces, np.int32)))
                m.compute_vertex_normals()
                m.paint_uniform_color(col)
                geoms.append(m)

        print(f"[viz] 焊缝#{sid}：工件 + 焊缝红线（p0→p1）+ {len(obstacles)} 个障碍")
        o3d.visualization.draw_geometries(geoms, window_name=f"seam #{sid}")

    def show_seam_all_isaacsim(self, seam_ids=None, headless: bool = False, radius: float = None):
        """用 **isaacsim** 一次性可视化【所有焊缝】：工件网格 + 每条焊缝一根【红色圆柱】。

        每根焊缝圆柱的 prim 路径含 `seam{seam_id}`（如 `/World/seam3`），在 stage 树里一眼就能
        分辨是哪条焊缝——无需在焊缝中心点画 id 文字。坐标系为【工件 mesh 系】（工件 identity 停放），
        焊缝端点 p0_world/p1_world 与工件同框直接叠加。

        ⚠ 工件 mesh 常停在远离世界原点处（如 x≈24、z≈12），而 isaacsim 默认相机看向原点，会让工件
        显得又小又偏、焊缝圆柱也难找。故本方法会把视口相机自动对准【工件/焊缝包围盒中心】。

        参数：
          seam_ids : 只画给定 id 列表（默认 None=全部焊缝）。
          headless : 无显示器自检（spawn 后跑几帧即退，打印摘要 + VIZ_SCENE_DONE）。
          radius   : 焊缝圆柱半径（米）；None=按【场景尺度】自适应（约包围盒对角线的 0.4%，至少 2cm，
                     保证在整件工件旁仍看得见）。

        须在【未初始化 curobo/torch】的干净进程里调用（SimulationApp 要最先启动）。
        """
        import os
        import numpy as np

        scene = self.scene
        seams = scene.seams
        n = len(seams)
        if n == 0:
            raise RuntimeError("Scene 无焊缝可可视化")
        ids = list(range(n)) if seam_ids is None else [int(i) % n for i in seam_ids]

        # —— SimulationApp 必须最先启动（在 import omni 之前）——
        try:
            import isaacsim  # noqa: F401  注册 omni.* 模块路径
        except ImportError:
            pass
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": bool(headless)})

        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.viewports import set_camera_view
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from scipy.spatial.transform import Rotation as Rsp

        def spawn_obj_mesh(pth, obj_path):
            import trimesh
            tm = trimesh.load(obj_path, force="mesh")
            verts = np.asarray(tm.vertices, float)
            faces = np.asarray(tm.faces, np.int64).reshape(-1, 3)
            stage = omni.usd.get_context().get_stage()
            mesh = UsdGeom.Mesh.Define(stage, pth)
            mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.72)])

        def spawn_workpiece(pth, obj_path):
            # 只在真存在 .usd 时才 reference；否则一律 trimesh（详见 show_scene_isaacsim 内同名函数注释）。
            usd_cands = []
            if obj_path.endswith("_watertight.obj"):
                usd_cands.append(obj_path[: -len("_watertight.obj")] + ".usd")
            usd_cands.append(os.path.splitext(obj_path)[0] + ".usd")
            usd_obj = next((c for c in usd_cands
                            if c.endswith(".usd") and os.path.exists(c)), None)
            if usd_obj is not None:
                add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
            else:
                spawn_obj_mesh(pth, obj_path)
            stg = omni.usd.get_context().get_stage()
            for pr in Usd.PrimRange(stg.GetPrimAtPath(pth)):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)

        def spawn_seam_cylinder(path, p0, p1, rad, color=(1.0, 0.0, 0.0)):
            """一根红色圆柱表示焊缝 p0→p1：UsdGeom.Cylinder 默认沿 +Z、中心在原点，
            设 height=L、radius=rad，再把 Z 轴旋到 seam 方向、平移到中点。"""
            p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
            seg = p1 - p0
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                return None
            stage = omni.usd.get_context().get_stage()
            cyl = UsdGeom.Cylinder.Define(stage, path)
            cyl.CreateAxisAttr("Z")
            cyl.CreateHeightAttr(float(L))
            cyl.CreateRadiusAttr(float(rad))
            cyl.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
            d_hat = seg / L
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, d_hat); s = float(np.linalg.norm(v)); c = float(np.dot(z, d_hat))
            if s < 1e-9:
                Rm = np.eye(3) if c > 0 else Rsp.from_euler("x", 180, degrees=True).as_matrix()
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            q = Rsp.from_matrix(Rm).as_quat()             # xyzw
            XFormPrim(path).set_world_pose(position=((p0 + p1) / 2.0).tolist(),
                                           orientation=np.r_[q[3], q[0], q[1], q[2]].tolist())
            return path

        print(f"工件     : {scene.workpiece_obj}")
        print(f"焊缝     : 共 {n} 条，本次画 {len(ids)} 条（红色圆柱，prim=/World/seam<id>）")

        # —— 场景包围盒（由所有待画焊缝端点求）：用于自适应圆柱半径 + 相机对准 ——
        pts = []
        for sid in ids:
            w = seams[sid]
            pts.append(np.asarray(w["p0_world"], float))
            pts.append(np.asarray(w["p1_world"], float))
        pts = np.asarray(pts, float)
        bb_min, bb_max = pts.min(0), pts.max(0)
        center = (bb_min + bb_max) / 2.0
        diag = float(np.linalg.norm(bb_max - bb_min)) or 1.0
        auto_rad = max(0.02, diag * 0.004)                # 按场景尺度，至少 2cm

        world = World(stage_units_in_meters=1.0)

        # 地面：置于所有焊缝最低处下方 1m
        world.scene.add_default_ground_plane(z_position=float(bb_min[2]) - 1.0)

        spawn_workpiece("/World/workpiece", scene.workpiece_obj)

        n_drawn = 0
        for sid in ids:
            w = seams[sid]
            p0 = np.asarray(w["p0_world"], float)
            p1 = np.asarray(w["p1_world"], float)
            rad = float(radius) if radius else auto_rad
            if spawn_seam_cylinder(f"/World/seam{sid}", p0, p1, rad) is not None:
                n_drawn += 1

        world.reset()

        # 相机对准工件/焊缝包围盒中心（否则默认看原点，工件常在 20+m 外显得很小）
        eye = center + np.array([diag * 0.9, -diag * 0.9, diag * 0.6], float)
        try:
            set_camera_view(eye=eye.tolist(), target=center.tolist())
        except Exception as e:                                    # noqa: BLE001
            print(f"[viz] set_camera_view 失败（忽略，可手动调视角）: {e}")

        if headless:
            for _ in range(3):
                world.step(render=False)
            print(f"已 spawn 工件 + {n_drawn} 条焊缝红色圆柱（prim=/World/seam<id>）。")
            print("VIZ_SCENE_DONE")
            simulation_app.close()
            return

        try:
            from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
            _set_lighting_mode("Grey Studio")
        except Exception:
            pass

        print(f"播放中（关闭窗口结束）：工件 + {n_drawn} 条焊缝红色圆柱；"
              f"在 stage 树按 prim 名 /World/seam<id> 辨别是哪条焊缝。")
        while simulation_app.is_running():
            world.step(render=True)
        simulation_app.close()

    def show_scene_isaacsim(self, headless: bool = False, goal_variant: int = 0,
                            trajectory=None, fps: int = 30, goal_arm_index: list = None,
                            observe=None, goal=None, flash_peak: float = 6e4,
                            flash_decay: int = None, goal_hold: int = None,
                            base_intensity: float = 250.0):
        """用 **isaacsim** 可视化当前 3D 场景：工件 + 障碍物（如有）+ 机械臂 + goal pose（如有）+ 当前焊缝红线（如有）。

        两种坐标系，取决于是否已 Scene.set_init_pose(hand, index) 选定当前 init pose：
          · 【未设 init pose】—— mesh 系（旧行为）：工件停在自身 mesh 坐标（identity），障碍/焊缝同框直接叠加，
            **不画机械臂**（无工件↔base 摆放无从摆臂）。供 demo_obstacle_type2/type3 用。
          · 【已设 init pose】—— base 系：机械臂 base 在原点、按 **retract 起始角** 摆姿；工件按当前候选的
            T_workpiece_in_base 摆到 base 系；障碍/当前焊缝红线/goal pose 视锥都随同一 T 变换后叠加。
            goal pose（若已 compute_goal_pose）以相机视锥（青→黄渐变）画出，cam_pose 经 T 从工件系变到 base 系；
            并在每个 goal pose 处放一个【真实 UsdGeom.Camera】（FOV 匹配视锥，看向焊缝），共 B 个。

        trajectory（可选，(T,DOF) 关节角序列，仅 base 系有意义）：给定则机械臂沿该序列【逐帧回放】
        （首尾各补 30 帧静止、播完保持 fps*2 帧后循环），用于看 plan_explore_path 的边走边看 GT；
        None 时保持原静态显示。fps=回放帧率。一般经 show_trajectory_isaacsim 转调，不直接传。

        observe / goal（可选，各 (T,) 或 (T,1) 的 0/1，与 trajectory 逐行对齐；merge_trajectory_entries 产出）：
          给定则进入【闪光回放】模式——基础照明压暗（不用 Grey Studio，改一盏低强度 DomeLight，
          intensity=base_intensity），在每个 goal 相机位姿处各放一盏 UsdLux.SphereLight（常态强度 0）。
          · observe[t]==1（该帧拍照）→ 对应 goal 相机的闪光灯瞬间拉到 flash_peak 白光，随后 flash_decay
            帧内线性衰减 → 相机位处爆闪（闪光从观测相机发出）。
          · goal[t]==1（着到某观测位姿那一刻）→ 机械臂在此【停顿 goal_hold 帧】，同时把该观测位姿的视锥
            染白高亮，便于肉眼确认"到位"，停顿结束再继续。
          闪光相机由 goal 列累计计数映射（第 n 个 goal==1 → 第 n 个视锥/相机）。flash_decay 缺省=fps//4，
          goal_hold 缺省=fps//2。无 goal 视锥（未 compute_goal_pose）时不闪光，仅普通回放。

        障碍以 Box 原语（open_box）与棱柱/圆筒 mesh（遮挡板 / open_cylinder）两种形态渲染，颜色取各
        ObstacleSpec.color。须在【未初始化 curobo/torch】的干净进程里调用（SimulationApp 要最先启动）。
        headless=True 时 spawn 后跑几帧即退（自检）。goal_variant 选 cam_pose 的第几个变体 K（默认 0；
        仅在 goal_arm_index=None 时生效，给了 goal_arm_index 则由其 n 覆盖）。

        goal_arm_index（可选，仅 base 系且已 compute_goal_pose）：
          · 选 goal 观测位姿 joints/cam_pose 第一维 **K**（变体）的索引，即 (160,1,6) 里 160 的下标。
          给定时把机械臂关节角从 retract 改成 **goal_poses[seam_id]["joints"][gk, 0]**（B=0 首观测位姿，无需 IK），
          同时 goal 视锥也画该第 gk 个变体；并打印其三分碰撞：自碰撞 / 碰工件 / 碰障碍
          （curobo FK 碰撞球 + trimesh signed_distance，见 _goal_arm_collision）。越界则夹取。
          标量=K 变体；兼容旧 [m,n] 写法：m 已废弃、忽略，取 n。None（默认）时机械臂仍摆 retract，行为同旧版。
        """
        import os
        import sys
        import numpy as np

        scene = self.scene
        obstacles = list(scene.obstacles.get(scene.seam_id, []))
        cur = scene.cur_init_pose
        base_frame = cur is not None

        # —— 解析 goal_arm_index → gk：选 goal 观测位姿的第一维 K（如 (160,1,6) 的 160）变体索引。
        #    goal_poses 现为单个 dict（不再是 list），故标量直接当 K 变体；兼容旧 [m,n] 写法：m 已废弃、
        #    忽略，取 n 作 K 变体（缺省退回 goal_variant）。——
        gk = int(goal_variant)
        if goal_arm_index is not None:
            if isinstance(goal_arm_index, (list, tuple)):
                gk = int(goal_arm_index[1]) if len(goal_arm_index) > 1 else int(goal_variant)
            else:
                gk = int(goal_arm_index)

        # —— 渲染坐标系变换 T（mesh 系 → 渲染系）：base 系用 T_workpiece_in_base，否则单位阵 ——
        if base_frame:
            T = np.asarray(cur.T_workpiece_in_base, float)
            wp_pose7 = np.asarray(cur.workpiece_pose7, float)      # 工件在 base 系 pose7（wxyz）
        else:
            T = np.eye(4)
            wp_pose7 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], float)
        R_T, t_T = T[:3, :3], T[:3, 3]

        def tf_pts(P):
            return np.asarray(P, float) @ R_T.T + t_T             # (…,3) mesh 系 → 渲染系

        # 当前焊缝折线（mesh 系）→ 渲染系
        cur_seam_line = None
        if getattr(scene, "seam", None) is not None:
            try:
                cur_seam_line = np.asarray(scene._seam_frame()[6], float)   # (N,3)
            except Exception as _e:
                print(f"[viz] 取当前焊缝失败（忽略）: {_e}")

        # —— SimulationApp 必须最先启动（在 import omni 之前）——
        try:
            import isaacsim  # noqa: F401  注册 omni.* 模块路径
        except ImportError:
            pass
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": bool(headless)})

        from omni.isaac.core import World
        from omni.isaac.core.objects import cuboid as _cuboid
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims import XFormPrim
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
        from scipy.spatial.transform import Rotation as Rsp

        def _compose_pose7(pose7):
            """把 mesh 系 pose7=[x,y,z,qw,qx,qy,qz] 经 T 变到渲染系，返回渲染系 pose7（wxyz）。"""
            p = np.asarray(pose7, float)
            Rc = Rsp.from_quat([p[4], p[5], p[6], p[3]])          # wxyz → xyzw
            Rw = Rsp.from_matrix(R_T) * Rc
            pos = R_T @ p[:3] + t_T
            q = Rw.as_quat()                                      # xyzw
            return np.r_[pos, q[3], q[0], q[1], q[2]]

        def spawn_obj_mesh(pth, obj_path):
            import trimesh
            tm = trimesh.load(obj_path, force="mesh")
            verts = np.asarray(tm.vertices, float)
            faces = np.asarray(tm.faces, np.int64).reshape(-1, 3)
            stage = omni.usd.get_context().get_stage()
            mesh = UsdGeom.Mesh.Define(stage, pth)
            mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.72)])

        def spawn_workpiece(pth, obj_path, pose7):
            """工件摆到渲染系 pose7（关物理当纯视觉）。usd 缺失则 trimesh 建 Mesh。"""
            # 候选 .usd：优先去掉 _watertight 后缀那份（如 _part_watertight.obj → _part.usd），
            # 再退回同名 .usd。只有【真正存在且后缀为 .usd】才走 add_reference；否则一律 trimesh。
            # 不能用 replace("_watertight.obj",".usd") 直接当结果——对 _part.obj 这类不含 _watertight
            # 的名字 replace 是 no-op，会把 .obj 本身当 usd 传给 add_reference（加载不出几何=工件消失）。
            usd_cands = []
            if obj_path.endswith("_watertight.obj"):
                usd_cands.append(obj_path[: -len("_watertight.obj")] + ".usd")
            usd_cands.append(os.path.splitext(obj_path)[0] + ".usd")
            usd_obj = next((c for c in usd_cands
                            if c.endswith(".usd") and os.path.exists(c)), None)
            if usd_obj is not None:
                add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
            else:
                spawn_obj_mesh(pth, obj_path)
            XFormPrim(pth).set_world_pose(position=np.asarray(pose7[:3], float).tolist(),
                                          orientation=np.asarray(pose7[3:7], float).tolist())
            stg = omni.usd.get_context().get_stage()
            for pr in Usd.PrimRange(stg.GetPrimAtPath(pth)):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)

        def spawn_segment(path, name, p0, p1, color, thick=0.01):
            p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
            seg = p1 - p0
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                return None
            d_hat = seg / L
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, d_hat); s = float(np.linalg.norm(v)); c = float(np.dot(z, d_hat))
            if s < 1e-9:
                Rm = np.eye(3) if c > 0 else Rsp.from_euler("x", 180, degrees=True).as_matrix()
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            quat = Rsp.from_matrix(Rm).as_quat()          # xyzw
            _cuboid.VisualCuboid(prim_path=path, name=name, position=(p0 + p1) / 2.0,
                                 orientation=np.r_[quat[3], quat[:3]], size=1.0,
                                 scale=np.array([thick, thick, L]), color=np.asarray(color, float))
            return path

        def spawn_seam_line(prefix, name0, seam_pts, color=(1.0, 0.0, 0.0), thick=0.01):
            """seam_pts 已在渲染系。"""
            for si in range(len(seam_pts) - 1):
                spawn_segment(f"{prefix}/seg{si}", f"{name0}_{si}",
                              seam_pts[si], seam_pts[si + 1], color, thick)

        def spawn_box_prim(path, name, prim, color):
            """障碍原语（mesh 系 pose）经 T 变到渲染系。Box→立方体，Tube→圆柱。"""
            pose7 = _compose_pose7(np.asarray(prim.pose, float))
            if hasattr(prim, "dims"):                             # Box
                _cuboid.VisualCuboid(prim_path=path, name=name,
                                     position=pose7[:3],
                                     orientation=pose7[3:7],       # wxyz
                                     size=1.0, scale=np.asarray(prim.dims, float),
                                     color=np.asarray(color, float))
            else:                                                 # Tube（轴沿局部 +Z）
                stage = omni.usd.get_context().get_stage()
                cyl = UsdGeom.Cylinder.Define(stage, path)
                cyl.CreateAxisAttr("Z")
                cyl.CreateHeightAttr(float(prim.height))
                cyl.CreateRadiusAttr(float(prim.radius))
                cyl.CreateDisplayColorAttr(
                    [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
                XFormPrim(path).set_world_pose(position=pose7[:3].tolist(),
                                               orientation=pose7[3:7].tolist())

        def spawn_mesh(path, mesh):
            """棱柱/圆筒 mesh（mesh 系 points）经 T 变到渲染系。"""
            stage = omni.usd.get_context().get_stage()
            m = UsdGeom.Mesh.Define(stage, path)
            pts = tf_pts(np.asarray(mesh["points"], float))
            m.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
            m.CreateFaceVertexCountsAttr(list(mesh["counts"]))
            m.CreateFaceVertexIndicesAttr(list(mesh["faces"]))
            col = mesh.get("color", [0.62, 0.64, 0.67])
            m.CreateDisplayColorAttr([Gf.Vec3f(float(col[0]), float(col[1]), float(col[2]))])
            m.CreateDoubleSidedAttr(True)

        # —— goal pose 视锥（scene_pose 同款 FOV 八顶点，+z 朝焊缝）——
        def _fov_corners():
            scl, scl_z = 7.0 / 11.0, 9.5 / 11.0
            near = np.array([[0.135 * scl, -0.20 * 5 / 6 * scl, 0.4],
                             [-0.135 * scl, -0.20 * 5 / 6 * scl, 0.4],
                             [-0.135 * scl, 0.20 * 5 / 6 * scl, 0.4],
                             [0.135 * scl, 0.20 * 5 / 6 * scl, 0.4]], float)
            dz = 0.140 * scl_z * scl; dy = 0.185 * 5 / 6 * scl_z * scl; z_far = 0.4 * (1 + scl_z)
            far = np.array([[0.135 * scl + dz, -0.20 * 5 / 6 * scl - dy, z_far],
                            [-0.135 * scl - dz, -0.20 * 5 / 6 * scl - dy, z_far],
                            [-0.135 * scl - dz, 0.20 * 5 / 6 * scl + dy, z_far],
                            [0.135 * scl + dz, 0.20 * 5 / 6 * scl + dy, z_far]], float)
            return near, far

        def _cam_intrinsics():
            """与 _fov_corners 同款：由近平面半尺寸给真实相机 FOV/clip 参数。
            返回 (half_w, half_h, near_z, far_z)——近平面半宽/半高（@z=near_z）与远平面 z。"""
            scl, scl_z = 7.0 / 11.0, 9.5 / 11.0
            near_z = 0.4
            far_z = 0.4 * (1 + scl_z)
            half_w = 0.135 * scl
            half_h = 0.20 * 5 / 6 * scl
            return half_w, half_h, near_z, far_z

        def spawn_camera(path, pos_w, R_w, half_w, half_h, near_z, far_z, focal=24.0):
            """在 goal pose 处放【真实 UsdGeom.Camera】，FOV 匹配线框视锥。
            USD 相机看本地 -Z、视锥 +z 朝焊缝 → 姿态绕本地 X 转 180° 对齐；
            horizontalAperture=2·focal·tan(hFOV/2)=2·focal·half_w/near_z（vertical 同理）；
            clippingRange 取近/远平面 z 使相机 gizmo 与线框贴合。"""
            stage = omni.usd.get_context().get_stage()
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(float(focal))
            cam.CreateHorizontalApertureAttr(float(2.0 * focal * half_w / near_z))
            cam.CreateVerticalApertureAttr(float(2.0 * focal * half_h / near_z))
            cam.CreateClippingRangeAttr(Gf.Vec2f(float(near_z), float(far_z)))
            Rx180 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], float)
            q = Rsp.from_matrix(np.asarray(R_w, float) @ Rx180).as_quat()   # xyzw
            XFormPrim(path).set_world_pose(position=np.asarray(pos_w, float).tolist(),
                                           orientation=np.r_[q[3], q[0], q[1], q[2]].tolist())

        def spawn_frustum(prefix, idx, pos_w, R_w, near, far, color):
            apex = np.asarray(pos_w, float)
            nw = apex + near @ R_w.T
            fw = apex + far @ R_w.T
            segs = []
            for k in range(4):
                segs.append(spawn_segment(f"{prefix}/near{k}", f"g{idx}_near{k}", nw[k], nw[(k + 1) % 4], color, 0.004))
                segs.append(spawn_segment(f"{prefix}/far{k}", f"g{idx}_far{k}", fw[k], fw[(k + 1) % 4], color, 0.004))
                segs.append(spawn_segment(f"{prefix}/side{k}", f"g{idx}_side{k}", nw[k], fw[k], color, 0.004))
                segs.append(spawn_segment(f"{prefix}/apex{k}", f"g{idx}_apex{k}", apex, nw[k], color, 0.004))
            return [s for s in segs if s]

        def spawn_flash_light(path, pos_w, radius=0.05):
            """在 goal 相机位姿处放一盏球形闪光灯（常态 intensity=0，回放时逐帧驱动）。返回 light schema。"""
            stage = omni.usd.get_context().get_stage()
            light = UsdLux.SphereLight.Define(stage, path)
            light.CreateRadiusAttr(float(radius))
            light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            light.CreateIntensityAttr(0.0)
            # 屏蔽外壳几何、只当发光体（避免遮挡视锥）
            try:
                UsdGeom.Imageable(light.GetPrim()).CreateVisibilityAttr("inherited")
            except Exception:
                pass
            XFormPrim(path).set_world_pose(position=np.asarray(pos_w, float).tolist())
            return light

        def set_segs_color(seg_paths, color):
            """把一组线段（视锥）的 DisplayColor 改成 color——用于 goal 到达时高亮。"""
            stage = omni.usd.get_context().get_stage()
            c = [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
            for p in seg_paths:
                pr = stage.GetPrimAtPath(p)
                if pr and pr.IsValid():
                    UsdGeom.Gprim(pr).GetDisplayColorAttr().Set(c)

        print(f"工件     : {scene.workpiece_obj}")
        print(f"当前 init pose : " + ("候选#%d（base 系，含机械臂 retract）" % scene.cur_init_index
                                       if base_frame else "未设定（mesh 系，不画机械臂）"))
        print(f"障碍      : 共 {len(obstacles)} 个 " +
              ", ".join(f"[{o.otype}:{o.kind}"
                        + (f"/{o.meta['candidate']}" if o.meta and 'candidate' in o.meta else "")
                        + "]" for o in obstacles))

        world = World(stage_units_in_meters=1.0)

        # —— 机械臂（仅 base 系）：base 在原点，稍后 retract 起始角 ——
        robot = None
        goal_joints = None         # 该 goal 位姿的关节角（joints[vi,gi]，无需 IK）
        goal_label = ""            # 打印用标签，如 "goal#0/7(变体#0)"
        if base_frame:
            try:
                from gt_gen import compat as _compat  # noqa: F401  warp shim（须在 curobo 前）
                from curobo.util_file import load_yaml
                import curobo as _curobo
                # helper.py 在 curobo 的 examples/isaac_sim（不在包内）。从 curobo 包位置推该目录，
                # 跨机器通用（本地 /home/a/...、远程 /kpfs_dataset/.../curobo）；本地路径留作回退。
                _curobo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(_curobo.__file__))))          # .../curobo/src/curobo → .../curobo
                for _isaac in (os.path.join(_curobo_root, "examples", "isaac_sim"),
                               "/home/a/Projects/Github/curobo/examples/isaac_sim"):
                    if os.path.isdir(_isaac) and _isaac not in sys.path:
                        sys.path.insert(0, _isaac)
                from helper import add_robot_to_scene
                robot_cfg = load_yaml(scene.cfg.robot_cfg_path)["robot_cfg"]
                robot, _ = add_robot_to_scene(robot_cfg, world)
                # goal_arm_index 给定：解析该 goal 观测位姿的关节角（已存于 goal_poses，无需 IK），
                # 稍后把这条唯一的机械臂摆到该关节角（而非 retract），并打印其三分碰撞。
                if goal_arm_index is not None and scene.goal_poses.get(scene.seam_id) is not None:
                    res = scene.goal_poses[scene.seam_id]     # 单个 dict（compute_goal_pose 产出）
                    jt = res["joints"]
                    jt = jt.detach().cpu().numpy() if hasattr(jt, "detach") else np.asarray(jt)
                    Kj, Bj = jt.shape[:2]                  # (K 变体, B 观测位姿, DOF)
                    vi = max(0, min(gk, Kj - 1))           # n → K 变体索引（越界则夹取）
                    goal_joints = [float(v) for v in jt[vi, 0]]   # 机械臂摆到首观测位姿 B=0
                    goal_label = f"goal/K#{vi}(共{Kj}变体,B={Bj})"
                elif goal_arm_index is not None:
                    print("[viz] 无 goal_poses（未 compute_goal_pose），跳过第二条臂")
            except Exception as e:
                print(f"[viz] 机械臂 spawn 失败（忽略，仅画工件/障碍/goal）: {e}")
                robot = None

        # 地面：置于渲染系中焊缝/工件最低处下方 1m
        obs_seam_lines = [tf_pts(ob.seam_line) for ob in obstacles if ob.seam_line is not None]
        zpool = []
        if cur_seam_line is not None:
            zpool.append(float(tf_pts(cur_seam_line)[:, 2].min()))
        for sl in obs_seam_lines:
            zpool.append(float(np.asarray(sl)[:, 2].min()))
        world.scene.add_default_ground_plane(z_position=(min(zpool) - 1.0) if zpool else -1.0)

        spawn_workpiece("/World/workpiece", scene.workpiece_obj, wp_pose7)

        # 当前焊缝红线（渲染系）
        if cur_seam_line is not None:
            spawn_seam_line("/World/seam/cur", "seam_cur", tf_pts(cur_seam_line), color=[1.0, 0.0, 0.0])

        # 障碍（经 T 变到渲染系）
        for oi, ob in enumerate(obstacles):
            for k, prim in enumerate(ob.prims):
                spawn_box_prim(f"/World/obs/o{oi}/box{k}", f"obs_{oi}_{k}", prim, ob.color)
            for k, mesh in enumerate(ob.meshes):
                spawn_mesh(f"/World/obs/o{oi}/mesh{k}", mesh)

        # goal pose 视锥（仅 base 系且已 compute_goal_pose）：gk 选 K 变体
        n_goal = 0
        frustum_segs = []          # 每个 goal 视锥的线段路径（闪光回放时高亮用）
        frustum_base_color = []    # 各视锥原色（高亮后复原）
        flash_lights = []          # 每个 goal 相机位姿处的球形闪光灯
        if base_frame and scene.goal_poses.get(scene.seam_id) is not None:
            res = scene.goal_poses[scene.seam_id]                # 单个 dict
            near, far = _fov_corners()
            half_w, half_h, near_z, far_z = _cam_intrinsics()
            cam_pose = np.asarray(res["cam_pose"])               # (K,B,7) piece 系 wxyz
            K = cam_pose.shape[0]
            vi = max(0, min(gk, K - 1))
            seq = cam_pose[vi]                                        # (B,7)
            B = seq.shape[0]
            for i in range(B):
                p7 = _compose_pose7(seq[i])                          # piece→base
                R_w = Rsp.from_quat([p7[4], p7[5], p7[6], p7[3]]).as_matrix()
                t = 0.0 if B <= 1 else i / (B - 1)
                col = [t, 1.0, 1.0 - t]
                frustum_segs.append(spawn_frustum(f"/World/goal/c{i}", i, p7[:3], R_w, near, far, col))
                frustum_base_color.append(col)
                spawn_camera(f"/World/goal/cam{i}", p7[:3], R_w, half_w, half_h, near_z, far_z)
                flash_lights.append(spawn_flash_light(f"/World/goal/flash{i}", p7[:3]))
            n_goal = B
        print(f"goal pose : " + (f"{n_goal} 个观测视锥（青→黄）+ {n_goal} 个真实相机"
                                  if n_goal else "无（未 compute_goal_pose 或非 base 系）"))

        # goal_arm_index 给定时：打印这条 goal 关节角的三分碰撞（自碰撞/工件/障碍）；
        # 不新画第二条臂，而是把下面唯一那条臂直接摆到 goal 关节角（见 set_joint_positions 处）。
        if goal_joints is not None:
            try:
                c = _goal_arm_collision(scene, goal_joints, wp_pose7, T)
                msg = (f"[goal-arm] {goal_label}: 自碰撞={c['self']} 碰工件={c['workpiece']} 碰障碍={c['obstacle']}"
                       f"（命中球 self={c['n_self']} work={c['n_work']} obs={c['n_obs']}）")
                print(msg)
                try:                                     # Isaac 接管 stdout 后 print 可能不可见，落一份文件便于核对
                    with open("/tmp/goal_arm_check.txt", "w") as _f:
                        _f.write(msg + "\n")
                except Exception:
                    pass
            except Exception as e:
                import traceback
                print(f"[goal-arm] 碰撞检测失败（忽略）: {e}")
                traceback.print_exc()

        world.reset()

        # 机械臂关节角：给了 goal_arm_index 则摆到 goal 关节角；否则给了 trajectory 用其首帧，
        # 再否则 retract 起始角（idx_list 供回放复用）。
        n_dof = len(scene.cfg.joint_names)
        traj = None if trajectory is None else np.asarray(trajectory, float).reshape(-1, n_dof)

        # —— 闪光回放：解析 observe/goal（各 (T,) 0/1，与 traj 逐行对齐），映射每行归属的 goal 相机 ——
        observe_row = None if observe is None else np.asarray(observe, np.int64).reshape(-1)
        goal_row = None if goal is None else np.asarray(goal, np.int64).reshape(-1)
        flash_mode = (traj is not None and len(flash_lights) > 0 and
                      (observe_row is not None or goal_row is not None))
        cam_of_row = None       # 每行归属 goal 相机下标（第 n 个 goal==1 → 相机 n）
        if flash_mode:
            n_rows = traj.shape[0]
            if observe_row is None:
                observe_row = np.zeros(n_rows, np.int64)
            if goal_row is None:
                goal_row = np.zeros(n_rows, np.int64)
            observe_row = observe_row[:n_rows]; goal_row = goal_row[:n_rows]
            gcum = np.cumsum(goal_row) - 1                     # -1=首个 goal 之前
            cam_of_row = np.clip(gcum, 0, len(flash_lights) - 1)
            fdecay = int(flash_decay) if flash_decay else max(1, int(fps) // 4)
            fhold = int(goal_hold) if goal_hold else max(1, int(fps) // 2)

        idx_list = None
        if robot is not None:
            try:
                if hasattr(robot, "initialize"):
                    robot.initialize()
                idx_list = [robot.get_dof_index(j) for j in scene.cfg.joint_names]
                if goal_joints is not None:
                    q0 = np.asarray(goal_joints, float)
                elif traj is not None:
                    q0 = traj[0]
                else:
                    q0 = np.asarray(scene.cur_cfg, float)
                robot.set_joint_positions(q0, idx_list)
            except Exception as e:
                print(f"[viz] 机械臂设关节角失败（忽略）: {e}")

        if headless:
            for _ in range(3):
                world.step(render=False)
            print(f"已 spawn 工件 + {len(obstacles)} 个障碍" +
                  (f" + 机械臂(retract) + {n_goal} 个 goal 视锥 + {n_goal} 个真实相机" if base_frame else "") +
                  (f" + 轨迹 {traj.shape[0]} 路点（逐帧回放）" if traj is not None else "") + "。")
            print("VIZ_SCENE_DONE")
            simulation_app.close()
            return

        # 灯光：闪光回放时压暗（换一盏低强度 DomeLight，让爆闪明显）；否则沿用 Grey Studio。
        if flash_mode:
            try:
                stg = omni.usd.get_context().get_stage()
                # 先把已有灯光（默认 distant/dome 等）压到极低，避免盖过爆闪；自家灯不动
                for pr in stg.Traverse():
                    p = pr.GetPath().pathString
                    if p.startswith("/World/goal/flash") or p == "/World/flashBaseDome":
                        continue
                    if pr.HasAPI(UsdLux.LightAPI) or pr.GetTypeName() in (
                            "DistantLight", "DomeLight", "SphereLight", "RectLight",
                            "DiskLight", "CylinderLight"):
                        try:
                            UsdLux.LightAPI(pr).GetIntensityAttr().Set(0.0)
                        except Exception:
                            pass
                dome = UsdLux.DomeLight.Define(stg, "/World/flashBaseDome")
                dome.CreateIntensityAttr(float(base_intensity))
                dome.CreateColorAttr(Gf.Vec3f(0.8, 0.85, 1.0))     # 略偏冷，衬托白色爆闪
            except Exception as e:
                print(f"[viz] 压暗基础照明失败（忽略）: {e}")
        else:
            try:
                from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
                _set_lighting_mode("Grey Studio")
            except Exception:
                pass

        # —— 闪光回放：机械臂沿 GT 逐帧运动，observe 帧相机爆闪、goal 帧停顿+视锥高亮 ——
        if flash_mode and robot is not None and idx_list is not None:
            pad = np.concatenate([np.tile(traj[0][None], (30, 1)), traj,
                                  np.tile(traj[-1][None], (30, 1))], axis=0)
            nL = len(flash_lights)
            n_rows = traj.shape[0]
            print(f"闪光回放（{pad.shape[0]} 帧，含首尾静止）：observe→相机位爆闪、goal→停顿{fhold}帧+视锥高亮。"
                  f"观测帧 {int(observe_row.sum())} 个、goal 到达 {int(goal_row.sum())} 个。关闭窗口结束。")
            i = 0                     # pad 索引
            prev_r = -999             # 上一处理的 traj 行（防 hold 期间重复触发）
            hold_remaining = 0
            held_cam = -1
            flash_timer = np.zeros(nL, float)
            end_hold = 0
            while simulation_app.is_running():
                # 每渲染帧驱动闪光灯强度（线性衰减）
                for c in range(nL):
                    inten = flash_peak * (flash_timer[c] / fdecay) if flash_timer[c] > 0 else 0.0
                    flash_lights[c].GetIntensityAttr().Set(float(inten))
                    if flash_timer[c] > 0:
                        flash_timer[c] -= 1
                world.step(render=True)
                if not world.is_playing():
                    continue
                if i < pad.shape[0]:
                    robot.set_joint_positions(pad[i], idx_list)
                    r = i - 30                                    # 对应 traj 行（<0 或 >=n_rows 为首尾静止）
                    if 0 <= r < n_rows and r != prev_r:           # 刚进入一条新 traj 行才触发
                        if observe_row[r] == 1:
                            flash_timer[int(cam_of_row[r])] = fdecay
                            print(f"[flash] frame {r}: OBSERVE 相机#{int(cam_of_row[r])} 爆闪")
                        if goal_row[r] == 1:
                            held_cam = int(cam_of_row[r])
                            set_segs_color(frustum_segs[held_cam], [1.0, 1.0, 1.0])
                            hold_remaining = fhold
                            print(f"[flash] frame {r}: GOAL#{held_cam} 到达 → 停顿+视锥高亮")
                        prev_r = r
                    if hold_remaining > 0:                         # 停顿：不推进 i，机械臂停在 goal
                        hold_remaining -= 1
                        if hold_remaining == 0 and held_cam >= 0:
                            set_segs_color(frustum_segs[held_cam], frustum_base_color[held_cam])
                            held_cam = -1
                    else:
                        i += 1
                else:
                    end_hold += 1
                    if end_hold > int(fps) * 2:                    # 播完保持 2s 后循环
                        i = 0; prev_r = -999; end_hold = 0
                        flash_timer[:] = 0.0
            simulation_app.close()
            return

        # —— 普通轨迹回放（无 observe/goal 或无 goal 视锥）：逐帧回放（首尾补静止帧、播完循环）——
        if traj is not None and robot is not None and idx_list is not None:
            pad = np.concatenate([np.tile(traj[0][None], (30, 1)), traj,
                                  np.tile(traj[-1][None], (30, 1))], axis=0)
            print(f"回放边走边看轨迹（{pad.shape[0]} 帧，含首尾静止；base 系：机械臂沿 GT 运动 + "
                  f"工件 + 障碍 + goal 视锥；关闭窗口结束）。")
            i = hold = 0
            while simulation_app.is_running():
                world.step(render=True)
                if not world.is_playing():
                    continue
                if i < pad.shape[0]:
                    robot.set_joint_positions(pad[i], idx_list)
                    i += 1
                else:
                    hold += 1
                    if hold > int(fps) * 2:
                        i = hold = 0
            simulation_app.close()
            return

        print("播放中（关闭窗口结束）。" +
              ("base 系：机械臂 retract + 工件 + 障碍 + 焊缝红线 + goal 视锥。" if base_frame
               else "mesh 系：工件 + 障碍 + 焊缝红线（未设 init pose，无机械臂）。"))
        while simulation_app.is_running():
            world.step(render=True)
        simulation_app.close()

    def show_init_poses_isaacsim(self, hand: str = None, top_n: int = 3, spacing: float = 2.5,
                                 joints: str = "reach", show_seam: bool = True,
                                 show_obstacles: bool = True, headless: bool = False):
        """用 **isaacsim** 把【多个候选初始位姿】同屏铺成网格：每格一个工件 + 一条机械臂。

        与 show_scene_isaacsim（只画当前选定的单个 init pose）不同：这里把 plan_init_pose() 求出的
        候选按网格铺开，一眼对比多种「工件↔机械臂」摆放。所有候选属于**同一条焊缝**（seam_id），
        故 mesh 系焊缝线各格共用、只是随各自 T_workpiece_in_base 变换。

        坐标系（base 系）：机械臂 base 在各自格心 off、工件在 off ⊕ T_workpiece_in_base；off 为纯平移，
        故机械臂 add_robot_to_scene(position=off)，工件 pose7[:3]+=off，焊缝/障碍点经 P@R.T+(t+off) 变换。

        参数：
          hand         : None（默认）取正手+反手；"forehand"/"backhand" 只取一只手。
          top_n        : 每只手取前 N 个候选（候选已按 xyz 多样性降序，差异大的在前）。默认 3。
          spacing      : 格心到格心的节距（米）。默认 2.5（UR12e 臂展+工件约 1.3~1.5m，1m 会重叠）。
          joints       : "reach"（默认，机械臂摆候选自带 q=焊接到位可达角）| "retract"（摆 cfg.retract 起步角）。
          show_seam    : 是否画各格焊缝红线（mesh 系经各格 T 变换）。
          show_obstacles: 是否画障碍（scene.obstacles[seam_id]，mesh 系经各格 T 变换）。
          headless     : 无显示器自检（spawn + 跑几帧即退，打印格数摘要 + VIZ_SCENE_DONE）。

        须在【未初始化 curobo/torch】的干净进程里调用（SimulationApp 要最先启动）——即 Scene.load 后另起进程。
        """
        import os
        import sys
        import math
        import numpy as np

        scene = self.scene

        # —— 选候选：按 hand 取正手/反手，每只手前 top_n（已按 xyz 多样性降序） ——
        cands = scene.init_pose_candidates.get(scene.seam_id, {})
        if not cands or not (cands.get("forehand") or cands.get("backhand")):
            raise RuntimeError("无候选初始位姿可视化：请先调用 Scene.plan_init_pose()（且求解成功）")
        hands = [hand] if hand else ["forehand", "backhand"]
        selected = []                       # [(label, cand), ...]
        n_by_hand = {}
        for h in hands:
            lst = list(cands.get(h, []))[:max(0, int(top_n))]
            n_by_hand[h] = len(lst)
            for j, c in enumerate(lst):
                selected.append((f"{h}#{j}", c))
        if not selected:
            raise RuntimeError(f"所选手别 {hands} 无候选：请先 plan_init_pose() 或换手别/加大 top_n")

        # —— 网格布局：近正方形，第 i 格格心 off=(col*spacing, row*spacing, 0) ——
        N = len(selected)
        cols = max(1, int(math.ceil(math.sqrt(N))))
        offs = []
        for i in range(N):
            r, c = divmod(i, cols)
            offs.append(np.array([c * float(spacing), r * float(spacing), 0.0], float))

        # 焊缝线（mesh 系，各格共用）
        seam_mesh = None
        if show_seam and getattr(scene, "seam", None) is not None:
            try:
                seam_mesh = np.asarray(scene._seam_frame()[6], float)     # (N,3)
            except Exception as _e:
                print(f"[viz] 取焊缝失败（忽略）: {_e}")
        obstacles = list(scene.obstacles.get(scene.seam_id, [])) if show_obstacles else []

        # —— SimulationApp 必须最先启动（在 import omni 之前）——
        try:
            import isaacsim  # noqa: F401  注册 omni.* 模块路径
        except ImportError:
            pass
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": bool(headless)})

        from omni.isaac.core import World
        from omni.isaac.core.objects import cuboid as _cuboid
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims import XFormPrim
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
        from scipy.spatial.transform import Rotation as Rsp

        # —— spawn 辅助（与 T 无关的直接照搬 show_scene_isaacsim；T 相关的把 R_T/t_T 作参数传入）——
        def spawn_obj_mesh(pth, obj_path):
            import trimesh
            tm = trimesh.load(obj_path, force="mesh")
            verts = np.asarray(tm.vertices, float)
            faces = np.asarray(tm.faces, np.int64).reshape(-1, 3)
            stage = omni.usd.get_context().get_stage()
            mesh = UsdGeom.Mesh.Define(stage, pth)
            mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.72)])

        def spawn_workpiece(pth, obj_path, pose7):
            # 只在真存在 .usd 时才 reference；否则一律 trimesh（详见 show_scene_isaacsim 内同名函数注释）。
            usd_cands = []
            if obj_path.endswith("_watertight.obj"):
                usd_cands.append(obj_path[: -len("_watertight.obj")] + ".usd")
            usd_cands.append(os.path.splitext(obj_path)[0] + ".usd")
            usd_obj = next((c for c in usd_cands
                            if c.endswith(".usd") and os.path.exists(c)), None)
            if usd_obj is not None:
                add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
            else:
                spawn_obj_mesh(pth, obj_path)
            XFormPrim(pth).set_world_pose(position=np.asarray(pose7[:3], float).tolist(),
                                          orientation=np.asarray(pose7[3:7], float).tolist())
            stg = omni.usd.get_context().get_stage()
            for pr in Usd.PrimRange(stg.GetPrimAtPath(pth)):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)

        def spawn_segment(path, name, p0, p1, color, thick=0.01):
            p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
            seg = p1 - p0
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                return None
            d_hat = seg / L
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, d_hat); s = float(np.linalg.norm(v)); c = float(np.dot(z, d_hat))
            if s < 1e-9:
                Rm = np.eye(3) if c > 0 else Rsp.from_euler("x", 180, degrees=True).as_matrix()
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            quat = Rsp.from_matrix(Rm).as_quat()          # xyzw
            _cuboid.VisualCuboid(prim_path=path, name=name, position=(p0 + p1) / 2.0,
                                 orientation=np.r_[quat[3], quat[:3]], size=1.0,
                                 scale=np.array([thick, thick, L]), color=np.asarray(color, float))
            return path

        def spawn_seam_line(prefix, name0, seam_pts, color=(1.0, 0.0, 0.0), thick=0.01):
            for si in range(len(seam_pts) - 1):
                spawn_segment(f"{prefix}/seg{si}", f"{name0}_{si}",
                              seam_pts[si], seam_pts[si + 1], color, thick)

        def compose_pose7(pose7, R_T, t_T):
            """mesh 系 pose7=[x,y,z,qw,qx,qy,qz] 经 (R_T,t_T) 变到渲染系，返回渲染系 pose7（wxyz）。"""
            p = np.asarray(pose7, float)
            Rc = Rsp.from_quat([p[4], p[5], p[6], p[3]])          # wxyz → xyzw
            Rw = Rsp.from_matrix(R_T) * Rc
            pos = R_T @ p[:3] + t_T
            q = Rw.as_quat()                                      # xyzw
            return np.r_[pos, q[3], q[0], q[1], q[2]]

        def spawn_box_prim(path, name, prim, color, R_T, t_T):
            pose7 = compose_pose7(np.asarray(prim.pose, float), R_T, t_T)
            if hasattr(prim, "dims"):                             # Box
                _cuboid.VisualCuboid(prim_path=path, name=name,
                                     position=pose7[:3], orientation=pose7[3:7],
                                     size=1.0, scale=np.asarray(prim.dims, float),
                                     color=np.asarray(color, float))
            else:                                                 # Tube（轴沿局部 +Z）
                stage = omni.usd.get_context().get_stage()
                cyl = UsdGeom.Cylinder.Define(stage, path)
                cyl.CreateAxisAttr("Z")
                cyl.CreateHeightAttr(float(prim.height))
                cyl.CreateRadiusAttr(float(prim.radius))
                cyl.CreateDisplayColorAttr(
                    [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
                XFormPrim(path).set_world_pose(position=pose7[:3].tolist(),
                                               orientation=pose7[3:7].tolist())

        def spawn_mesh(path, mesh, R_T, t_T):
            stage = omni.usd.get_context().get_stage()
            m = UsdGeom.Mesh.Define(stage, path)
            pts = np.asarray(mesh["points"], float) @ R_T.T + t_T
            m.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
            m.CreateFaceVertexCountsAttr(list(mesh["counts"]))
            m.CreateFaceVertexIndicesAttr(list(mesh["faces"]))
            col = mesh.get("color", [0.62, 0.64, 0.67])
            m.CreateDisplayColorAttr([Gf.Vec3f(float(col[0]), float(col[1]), float(col[2]))])
            m.CreateDoubleSidedAttr(True)

        print(f"工件     : {scene.workpiece_obj}")
        print(f"候选     : {N} 格（" + " / ".join(f"{h} {n}" for h, n in n_by_hand.items())
              + f"），关节角={joints}，节距={spacing}m，障碍/格={len(obstacles)}")

        world = World(stage_units_in_meters=1.0)

        # 地面：置于所有格焊缝最低处下方 1m
        zpool = []
        for i, (_lbl, cand) in enumerate(selected):
            if seam_mesh is not None:
                R = np.asarray(cand.R, float); toff = np.asarray(cand.t, float) + offs[i]
                zpool.append(float((seam_mesh @ R.T + toff)[:, 2].min()))
        world.scene.add_default_ground_plane(z_position=(min(zpool) - 1.0) if zpool else -1.0)

        # —— 加载 robot_cfg（一次）：推 curobo examples/isaac_sim 目录（跨机器通用，本地路径回退）——
        robot_cfg = None
        try:
            from gt_gen import compat as _compat  # noqa: F401  warp shim（须在 curobo 前）
            from curobo.util_file import load_yaml
            import curobo as _curobo
            _curobo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(_curobo.__file__))))
            for _isaac in (os.path.join(_curobo_root, "examples", "isaac_sim"),
                           "/home/a/Projects/Github/curobo/examples/isaac_sim"):
                if os.path.isdir(_isaac) and _isaac not in sys.path:
                    sys.path.insert(0, _isaac)
            from helper import add_robot_to_scene
            robot_cfg = load_yaml(scene.cfg.robot_cfg_path)["robot_cfg"]
        except Exception as e:
            print(f"[viz] robot_cfg 加载失败（仅画工件/焊缝/障碍，不画机械臂）: {e}")
            robot_cfg = None

        # —— 逐格 spawn：机械臂(off) + 工件(off⊕T) + 焊缝 + 障碍 ——
        robots = []                          # [(robot, q_i, label), ...]
        retract = np.asarray(scene.cur_cfg, float)
        for i, (label, cand) in enumerate(selected):
            off = offs[i]
            R = np.asarray(cand.R, float)
            toff = np.asarray(cand.t, float) + off       # 该格 mesh→渲染系平移

            if robot_cfg is not None:
                try:
                    robot_i, _ = add_robot_to_scene(
                        robot_cfg, world, subroot=f"/World/cell{i}/robot",
                        robot_name=f"ip_robot{i}", position=off)
                    q_i = (np.asarray(cand.q, float) if joints == "reach" else retract)
                    robots.append((robot_i, q_i, label))
                except Exception as e:
                    print(f"[viz] 第 {i} 格机械臂 spawn 失败（忽略）: {e}")

            wp7 = np.asarray(cand.workpiece_pose7, float).copy()
            wp7[:3] = wp7[:3] + off
            spawn_workpiece(f"/World/cell{i}/workpiece", scene.workpiece_obj, wp7)

            if seam_mesh is not None:
                spawn_seam_line(f"/World/cell{i}/seam", f"seam{i}",
                                seam_mesh @ R.T + toff, color=[1.0, 0.0, 0.0])

            for oi, ob in enumerate(obstacles):
                for k, prim in enumerate(ob.prims):
                    spawn_box_prim(f"/World/cell{i}/obs/o{oi}/box{k}",
                                   f"c{i}_obs_{oi}_{k}", prim, ob.color, R, toff)
                for k, mesh in enumerate(ob.meshes):
                    spawn_mesh(f"/World/cell{i}/obs/o{oi}/mesh{k}", mesh, R, toff)

        world.reset()

        # 机械臂关节角：逐台 initialize + set_joint_positions
        joint_names = scene.cfg.joint_names
        for robot_i, q_i, label in robots:
            try:
                if hasattr(robot_i, "initialize"):
                    robot_i.initialize()
                idx_list = [robot_i.get_dof_index(j) for j in joint_names]
                robot_i.set_joint_positions(np.asarray(q_i, float), idx_list)
            except Exception as e:
                print(f"[viz] {label} 机械臂设关节角失败（忽略）: {e}")

        if headless:
            for _ in range(3):
                world.step(render=False)
            print(f"已 spawn {N} 格（" + " / ".join(f"{h} {n}" for h, n in n_by_hand.items())
                  + f"），每格 工件 + 机械臂({joints}角) + "
                  + ("焊缝红线 + " if seam_mesh is not None else "")
                  + f"{len(obstacles)} 障碍。共 {len(robots)} 条机械臂。")
            print("VIZ_SCENE_DONE")
            simulation_app.close()
            return

        try:
            from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
            _set_lighting_mode("Grey Studio")
        except Exception:
            pass

        print(f"播放中（关闭窗口结束）：{N} 格候选初始位姿铺开，每格 工件 + 机械臂({joints}角)"
              + ("+ 焊缝红线" if seam_mesh is not None else "")
              + (f" + 障碍" if obstacles else "") + "。")
        while simulation_app.is_running():
            world.step(render=True)
        simulation_app.close()

    def show_trajectory_isaacsim(self, traj_index: int = -1, headless: bool = False,
                                 fps: int = 30, goal_variant: int = 0,
                                 flash_peak: float = 6e4, flash_decay: int = None,
                                 goal_hold: int = None, base_intensity: float = 250.0):
        """用 **isaacsim** 回放【边走边看轨迹】（Scene.plan_explore_path 产出）。

        base 系里机械臂沿 GT 关节序列逐帧运动，同屏显示工件 + 障碍物 + goal 观测视锥/真实相机
        （复用 show_scene_isaacsim(trajectory=...)，故场景摆放与那套完全一致）。参考 launch.json 的
        viz_placed_obstacle_isaacsim：首尾补静止帧、播完循环重播。会自动 set_init_pose 到该轨迹实际
        所属的 (手别,候选下标)，并还原该候选当时的 goal_poses 快照，故工件摆放/焊缝/视锥与所选轨迹一致
        （不依赖调用前 scene.cur_init_hand/index 恰好是哪个候选）。

        前提：先 plan_explore_path（其内部要求已 set_init_pose，故必为 base 系、会画机械臂）。跨进程时
        先 Scene.save→另起干净进程 Scene.load 再调用（compute/plan 会污染 warp，见 save/load 说明）。

        参数：
          traj_index  : 把 self.scene.trajectories[seam_id] 里各 init pose(手别,候选下标) 下的 list
                        按 key 插入顺序拼成一个扁平列表后，取第几条（默认 -1=最新；支持负索引）。
          headless    : 无显示器自检（spawn+跑几帧即退，打印路点数 + VIZ_SCENE_DONE）。
          fps         : 回放帧率。
          goal_variant: 画 goal 视锥用 cam_pose 的第几个变体 K（默认 0）。
          flash_peak / flash_decay / goal_hold / base_intensity:
              闪光回放参数——爆闪峰值强度 / 衰减帧数(缺省 fps//4) / goal 停顿帧数(缺省 fps//2) /
              压暗后基础 DomeLight 强度。entry 里有 observe/goal 时自动进入闪光模式（见 show_scene_isaacsim）。
        """
        import numpy as np

        scene = self.scene
        traj_map = scene.trajectories.get(scene.seam_id, {})
        trajs = [(key, e) for key, lst in traj_map.items() for e in lst]
        if not trajs:
            raise RuntimeError("无可回放轨迹：请先 Scene.plan_explore_path()")
        n = len(trajs)
        if not (-n <= int(traj_index) < n):
            raise IndexError(f"traj_index 越界：{traj_index}，共 {n} 条轨迹")
        key, entry = trajs[int(traj_index)]
        positions = np.asarray(entry["positions"], float)
        observe = entry.get("observe")
        goal = entry.get("goal")
        print(f"[viz] 回放轨迹 #{int(traj_index) % n}/{n}：init pose={key[0]}#{key[1]} "
              f"status={entry.get('status')} 路点={len(positions)} goal=观测位姿#{entry.get('goal_index')}"
              f"（变体#{entry.get('variant')}）")

        # 切到该轨迹实际所属的 init pose，并还原它当时的 goal_poses（视锥）快照——
        # 否则工件摆放/焊缝/视锥会停在 scene 当前的 cur_init_hand/index（多手别依次计算时，
        # 那始终是最后一次 compute_pose_and_plan_path 的候选，与本条轨迹不一致）。
        scene.set_init_pose(*key)
        snap = scene.trajectory_goal_poses.get(scene.seam_id, {}).get(key)
        if snap is not None:
            scene.goal_poses[scene.seam_id] = snap
        self.show_scene_isaacsim(headless=headless, goal_variant=goal_variant,
                                 trajectory=positions, fps=fps,
                                 observe=observe, goal=goal, flash_peak=flash_peak,
                                 flash_decay=flash_decay, goal_hold=goal_hold,
                                 base_intensity=base_intensity)


class IsaacSimSceneVisualizer(SceneVisualizer):
    """isaacsim/isaaclab 后端可视化（后续补充）。"""

    def show_init_poses(self):
        raise NotImplementedError("IsaacSimSceneVisualizer.show_init_poses 待补充")
