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

后续补充（占位）：3D 世界 / 机械臂 / 焊缝 / 已观测区域 等的 open3d & isaacsim 可视化。
"""
from __future__ import annotations

from gt_gen.scene import Scene, _load_plan_init_pose


def _goal_arm_collision(scene, goal_joints, wp_pose7, T):
    """到达 goal 观测位姿的关节角 goal_joints 的三分碰撞检测：自碰撞 / 碰工件 / 碰障碍。

    沿用 scripts/viz_collision_isaacsim.py 套路：curobo CudaRobotModel 做 FK，得每颗碰撞球在 base 系的
    球心+半径（球定义取 robot yml 的 collision_spheres），再：
      · 自碰撞：球心距 < r_i+r_j（跳过同一 link 内球对 + self_collision_ignore 里成对的相邻 link）；
      · 工件/障碍：trimesh ProximityQuery.signed_distance(球心)+半径 > 0 即球体入网格。
    工件/障碍 mesh 均按 base 系摆放（工件用 wp_pose7、障碍实体各 apply T），与 FK 球同框
    （robot base 在原点，故 base 系=渲染系）。须在 SimulationApp 启动【之后】调用（curobo import 顺序）。
    返回 dict(self, workpiece, obstacle: bool; n_self, n_work, n_obs: int)。
    """
    import numpy as np
    import trimesh
    from gt_gen import compat as _compat
    _compat.apply_trimesh_shim()                        # warp/trimesh shim（须在 curobo 前）
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import load_yaml
    from gt_gen.sensor import load_truth_scene

    def _quat_wxyz_to_R(q):
        w, x, y, z = [float(v) for v in q]
        n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)

    # —— FK 碰撞球（base 系）——
    rd = load_yaml(scene.cfg.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    coll_links = list(kin["collision_link_names"])
    spheres_def = kin["collision_spheres"]
    kin["link_names"] = coll_links                       # 让 FK 输出所有碰撞 link 的位姿
    ta = TensorDeviceType()
    model = CudaRobotModel(RobotConfig.from_dict(rd["robot_cfg"], ta).kinematics)
    st = model.get_state(torch.tensor([list(goal_joints)], dtype=torch.float32, device=ta.device))

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
    centers = np.asarray(centers, float)
    radii = np.asarray(radii, float)
    n = len(centers)

    # —— 自碰撞：球-球，跳过同 link + self_collision_ignore 相邻 link（对称）——
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




class SceneVisualizer:
    """Scene 可视化基类：持有 Scene，具体渲染由后端子类实现。"""

    def __init__(self, scene: Scene):
        if not isinstance(scene, Scene):
            raise TypeError(f"SceneVisualizer 需要 Scene 实例，收到 {type(scene)}")
        self.scene = scene


class Open3DSceneVisualizer(SceneVisualizer):
    """open3d 后端可视化（本机开窗）。"""

    def show_init_poses(self, stride: int = 1):
        """逐个可视化该 Scene 焊缝的【候选初始位姿】（工件相对机械臂的摆放）。

        前提：先调用 Scene.plan_init_pose() 求出候选。可视化复用 scripts/plan_init_pose.py 的
        _show_kejian2_results（正手→反手依次开窗：按 (R,t) 摆放的工件网格 + 焊缝/中点 + standoff 落点
        + 蓝色 bisector 正反手判据轴）；同一窗口按 **C 键**切到下一个候选。

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
        extra = self._obstacle_o3d_factory() if scene.obstacles.get(scene.seam_id) else None
        pim._show_kejian2_results(scene.cfg, scene.workpiece_obj, scene.seam, res,
                                  stride=stride, extra_geoms=extra)

    def _obstacle_o3d_factory(self):
        """返回回调 (R,t)->list[o3d.geometry]：把 scene.obstacles（工件 mesh 系）按 T_workpiece_in_base
        摆到 base 系。障碍→trimesh 复用 scene 的 _polygon_mesh_to_trimesh / _box_prim_to_trimesh
        （板 mesh + open_cylinder mesh 走前者，open_box 的 Box 原语走后者），再转 o3d、染 ObstacleSpec.color。"""
        import numpy as np
        import open3d as o3d
        from gt_gen.scene import _polygon_mesh_to_trimesh, _box_prim_to_trimesh

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
                    tms.append(_box_prim_to_trimesh(prim))
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
        from gt_gen.scene import _polygon_mesh_to_trimesh, _box_prim_to_trimesh

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
                tms.append(_box_prim_to_trimesh(prim))
            for t in tms:
                m = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(t.vertices, float)),
                    o3d.utility.Vector3iVector(np.asarray(t.faces, np.int32)))
                m.compute_vertex_normals()
                m.paint_uniform_color(col)
                geoms.append(m)

        print(f"[viz] 焊缝#{sid}：工件 + 焊缝红线（p0→p1）+ {len(obstacles)} 个障碍")
        o3d.visualization.draw_geometries(geoms, window_name=f"seam #{sid}")

    def show_scene_isaacsim(self, headless: bool = False, goal_variant: int = 0,
                            trajectory=None, fps: int = 30, goal_arm_index: list = None):
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
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
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
            usd_obj = obj_path.replace("_watertight.obj", ".usd")
            if not os.path.exists(usd_obj):
                usd_obj = os.path.splitext(obj_path)[0] + ".usd"
            if os.path.exists(usd_obj):
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
                return
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

        def spawn_seam_line(prefix, name0, seam_pts, color=(1.0, 0.0, 0.0), thick=0.01):
            """seam_pts 已在渲染系。"""
            for si in range(len(seam_pts) - 1):
                spawn_segment(f"{prefix}/seg{si}", f"{name0}_{si}",
                              seam_pts[si], seam_pts[si + 1], color, thick)

        def spawn_box_prim(path, name, prim, color):
            """Box 原语（mesh 系 pose）经 T 变到渲染系。"""
            pose7 = _compose_pose7(np.asarray(prim.pose, float))
            _cuboid.VisualCuboid(prim_path=path, name=name,
                                 position=pose7[:3],
                                 orientation=pose7[3:7],           # wxyz
                                 size=1.0, scale=np.asarray(prim.dims, float),
                                 color=np.asarray(color, float))

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
            for k in range(4):
                spawn_segment(f"{prefix}/near{k}", f"g{idx}_near{k}", nw[k], nw[(k + 1) % 4], color, 0.004)
                spawn_segment(f"{prefix}/far{k}", f"g{idx}_far{k}", fw[k], fw[(k + 1) % 4], color, 0.004)
                spawn_segment(f"{prefix}/side{k}", f"g{idx}_side{k}", nw[k], fw[k], color, 0.004)
                spawn_segment(f"{prefix}/apex{k}", f"g{idx}_apex{k}", apex, nw[k], color, 0.004)

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
                spawn_frustum(f"/World/goal/c{i}", i, p7[:3], R_w, near, far, [t, 1.0, 1.0 - t])
                spawn_camera(f"/World/goal/cam{i}", p7[:3], R_w, half_w, half_h, near_z, far_z)
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

        try:
            from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
            _set_lighting_mode("Grey Studio")
        except Exception:
            pass

        # —— 有轨迹：逐帧回放（首尾补静止帧、播完保持后循环）——
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
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
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
            usd_obj = obj_path.replace("_watertight.obj", ".usd")
            if not os.path.exists(usd_obj):
                usd_obj = os.path.splitext(obj_path)[0] + ".usd"
            if os.path.exists(usd_obj):
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
                return
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
            _cuboid.VisualCuboid(prim_path=path, name=name,
                                 position=pose7[:3], orientation=pose7[3:7],
                                 size=1.0, scale=np.asarray(prim.dims, float),
                                 color=np.asarray(color, float))

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
                                 fps: int = 30, goal_variant: int = 0):
        """用 **isaacsim** 回放【边走边看轨迹】（Scene.plan_explore_path 产出）。

        base 系里机械臂沿 GT 关节序列逐帧运动，同屏显示工件 + 障碍物 + goal 观测视锥/真实相机
        （复用 show_scene_isaacsim(trajectory=...)，故场景摆放与那套完全一致）。参考 launch.json 的
        viz_placed_obstacle_isaacsim：首尾补静止帧、播完循环重播。

        前提：先 plan_explore_path（其内部要求已 set_init_pose，故必为 base 系、会画机械臂）。跨进程时
        先 Scene.save→另起干净进程 Scene.load 再调用（compute/plan 会污染 warp，见 save/load 说明）。

        参数：
          traj_index  : self.scene.trajectories 里第几条（默认 -1=最新；支持负索引）。
          headless    : 无显示器自检（spawn+跑几帧即退，打印路点数 + VIZ_SCENE_DONE）。
          fps         : 回放帧率。
          goal_variant: 画 goal 视锥用 cam_pose 的第几个变体 K（默认 0）。
        """
        import numpy as np

        trajs = list(self.scene.trajectories.get(self.scene.seam_id, []))
        if not trajs:
            raise RuntimeError("无可回放轨迹：请先 Scene.plan_explore_path()")
        n = len(trajs)
        if not (-n <= int(traj_index) < n):
            raise IndexError(f"traj_index 越界：{traj_index}，共 {n} 条轨迹")
        entry = trajs[int(traj_index)]
        positions = np.asarray(entry["positions"], float)
        print(f"[viz] 回放轨迹 #{int(traj_index) % n}/{n}：status={entry.get('status')} "
              f"路点={len(positions)} goal=观测位姿#{entry.get('goal_index')}"
              f"（变体#{entry.get('variant')}）")
        self.show_scene_isaacsim(headless=headless, goal_variant=goal_variant,
                                 trajectory=positions, fps=fps)


class IsaacSimSceneVisualizer(SceneVisualizer):
    """isaacsim/isaaclab 后端可视化（后续补充）。"""

    def show_init_poses(self):
        raise NotImplementedError("IsaacSimSceneVisualizer.show_init_poses 待补充")
