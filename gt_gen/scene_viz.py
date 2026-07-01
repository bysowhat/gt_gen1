"""SceneVisualizer —— 用 open3d / isaacsim 可视化 Scene 的各种信息（API 重组）。

设计：基类 SceneVisualizer 持有一个 Scene；两个后端子类：
  · Open3DSceneVisualizer  —— 用 open3d 在本机开窗可视化（需显示器，env_isaaclab 装了 open3d）。
  · IsaacSimSceneVisualizer —— 用 isaacsim/isaaclab 可视化（后续补充）。

与 Scene 一样，只做 **API 方向的结构包装**：本期 open3d 的「逐个可视化候选初始位姿」直接复用
scripts/plan_init_pose.py 的 show_lookup_solutions（整臂碰撞球 + init_free 盒 + 工件网格 + 焊缝线
+ standoff 落点），不改其渲染逻辑。

本期已实现：
  · Open3DSceneVisualizer.show_init_poses()      —— 逐个看 Scene 的候选初始位姿（按 C 切下一个）
  · Open3DSceneVisualizer.show_scene_isaacsim()  —— 用 isaacsim 可视化当前 3D 场景（工件 + 焊缝 + 障碍物类型2/3）

后续补充（占位）：3D 世界 / 机械臂 / 焊缝 / 已观测区域 等的 open3d & isaacsim 可视化。
"""
from __future__ import annotations

from gt_gen.scene import Scene, _load_plan_init_pose


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
        if not scene.init_pose_candidates:
            raise RuntimeError(
                "无候选初始位姿可视化：请先调用 Scene.plan_init_pose()（且求解成功）")
        pim = _load_plan_init_pose()
        res = {"forehand": [c.raw for c in scene.init_pose_candidates if c.hand == "forehand"],
               "backhand": [c.raw for c in scene.init_pose_candidates if c.hand == "backhand"]}
        extra = self._obstacle_o3d_factory() if scene.obstacles else None
        pim._show_kejian2_results(scene.cfg, scene.workpiece_obj, scene.seam, res,
                                  stride=stride, extra_geoms=extra)

    def _obstacle_o3d_factory(self):
        """返回回调 (R,t)->list[o3d.geometry]：把 scene.obstacles（工件 mesh 系）按 T_workpiece_in_base
        摆到 base 系。障碍→trimesh 复用 scene 的 _polygon_mesh_to_trimesh / _box_prim_to_trimesh
        （板 mesh + open_cylinder mesh 走前者，open_box 的 Box 原语走后者），再转 o3d、染 ObstacleSpec.color。"""
        import numpy as np
        import open3d as o3d
        from gt_gen.scene import _polygon_mesh_to_trimesh, _box_prim_to_trimesh

        obstacles = list(self.scene.obstacles)

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

    def show_scene_isaacsim(self, headless: bool = False, goal_variant: int = 0,
                            trajectory=None, fps: int = 30):
        """用 **isaacsim** 可视化当前 3D 场景：工件 + 障碍物（如有）+ 机械臂 + goal pose（如有）+ 当前焊缝红线（如有）。

        两种坐标系，取决于是否已 Scene.set_init_pose(index) 选定当前 init pose：
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
        headless=True 时 spawn 后跑几帧即退（自检）。goal_variant 选 cam_pose 的第几个变体 K（默认 0）。
        """
        import os
        import sys
        import numpy as np

        scene = self.scene
        obstacles = list(scene.obstacles)
        cur = scene.cur_init_pose
        base_frame = cur is not None

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

        # goal pose 视锥（仅 base 系且已 compute_goal_pose）
        n_goal = 0
        if base_frame and scene.goal_poses:
            near, far = _fov_corners()
            half_w, half_h, near_z, far_z = _cam_intrinsics()
            cam_pose = np.asarray(scene.goal_poses[0]["cam_pose"])   # (K,B,7) piece 系 wxyz
            K = cam_pose.shape[0]
            vi = max(0, min(int(goal_variant), K - 1))
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

        world.reset()

        # 机械臂起始角：给了 trajectory 用其首帧，否则 retract 起始角（idx_list 供回放复用）
        n_dof = len(scene.cfg.joint_names)
        traj = None if trajectory is None else np.asarray(trajectory, float).reshape(-1, n_dof)
        idx_list = None
        if robot is not None:
            try:
                if hasattr(robot, "initialize"):
                    robot.initialize()
                idx_list = [robot.get_dof_index(j) for j in scene.cfg.joint_names]
                q0 = traj[0] if traj is not None else np.asarray(scene.cur_cfg, float)
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

        trajs = list(getattr(self.scene, "trajectories", []) or [])
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
