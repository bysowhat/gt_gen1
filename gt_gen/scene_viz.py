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

    def show_scene_isaacsim(self, headless: bool = False):
        """用 **isaacsim** 可视化当前 3D 场景：工件网格 + 焊缝红线 + 已添加的障碍物（类型2/3）。

        工件停在自身 mesh 坐标（identity），焊缝与障碍（Scene.add_obstacle_type2/type3 产出）均在同一
        工件 mesh 系（与 viz_weld_json 同框），故直接叠加即对齐。障碍以 Box 原语（open_box）与棱柱/圆筒
        mesh（遮挡板 / open_cylinder）两种形态渲染，颜色取各 ObstacleSpec.color。

        前提：先 Scene.add_obstacle_type2()/add_obstacle_type3() 放好障碍。须在【未初始化 curobo/torch】
        的干净进程里调用（SimulationApp 要最先启动）。headless=True 时 spawn 后跑几帧即退（自检）。
        """
        import os
        import numpy as np

        scene = self.scene
        obstacles = list(scene.obstacles)
        seam_lines = [ob.seam_line for ob in obstacles if ob.seam_line is not None]

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
            """工件摆在自身 mesh 坐标（identity），关物理当纯视觉。usd 缺失则 trimesh 建 Mesh。"""
            usd_obj = obj_path.replace("_watertight.obj", ".usd")
            if not os.path.exists(usd_obj):
                usd_obj = os.path.splitext(obj_path)[0] + ".usd"
            if os.path.exists(usd_obj):
                add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
            else:
                spawn_obj_mesh(pth, obj_path)
            XFormPrim(pth).set_world_pose(position=[0.0, 0.0, 0.0],
                                          orientation=[1.0, 0.0, 0.0, 0.0])
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

        def spawn_box_prim(path, name, prim, color):
            _cuboid.VisualCuboid(prim_path=path, name=name,
                                 position=np.asarray(prim.pose[:3], float),
                                 orientation=np.asarray(prim.pose[3:7], float),   # wxyz
                                 size=1.0, scale=np.asarray(prim.dims, float),
                                 color=np.asarray(color, float))

        def spawn_mesh(path, mesh):
            stage = omni.usd.get_context().get_stage()
            m = UsdGeom.Mesh.Define(stage, path)
            pts = np.asarray(mesh["points"], float)
            m.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
            m.CreateFaceVertexCountsAttr(list(mesh["counts"]))
            m.CreateFaceVertexIndicesAttr(list(mesh["faces"]))
            col = mesh.get("color", [0.62, 0.64, 0.67])
            m.CreateDisplayColorAttr([Gf.Vec3f(float(col[0]), float(col[1]), float(col[2]))])
            m.CreateDoubleSidedAttr(True)

        print(f"工件     : {scene.workpiece_obj}")
        print(f"焊缝      : weld_json 第 {scene.seam_id} 条")
        print(f"障碍      : 共 {len(obstacles)} 个 " +
              ", ".join(f"[{o.otype}:{o.kind}"
                        + (f"/{o.meta['candidate']}" if o.meta and 'candidate' in o.meta else "")
                        + "]" for o in obstacles))

        world = World(stage_units_in_meters=1.0)
        # 地面置于工件/焊缝最低处下方 1m
        zs = [float(np.asarray(sl, float)[:, 2].min()) for sl in seam_lines] or [0.0]
        world.scene.add_default_ground_plane(z_position=min(zs) - 1.0)

        spawn_workpiece("/World/workpiece", scene.workpiece_obj)

        for oi, ob in enumerate(obstacles):
            if ob.seam_line is not None:
                spawn_seam_line(f"/World/seam/o{oi}", f"seam_{oi}", ob.seam_line, color=[1.0, 0.0, 0.0])
            for k, prim in enumerate(ob.prims):
                spawn_box_prim(f"/World/obs/o{oi}/box{k}", f"obs_{oi}_{k}", prim, ob.color)
            for k, mesh in enumerate(ob.meshes):
                spawn_mesh(f"/World/obs/o{oi}/mesh{k}", mesh)

        world.reset()

        if headless:
            for _ in range(3):
                world.step(render=False)
            print(f"已 spawn 工件 + {len(obstacles)} 个障碍（类型2/3）。")
            print("VIZ_SCENE_DONE")
            simulation_app.close()
            return

        try:
            from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
            _set_lighting_mode("Grey Studio")
        except Exception:
            pass
        print("播放中（关闭窗口结束）。工件 + 焊缝红线 + 障碍物叠加显示在同一 mesh 系。")
        while simulation_app.is_running():
            world.step(render=True)
        simulation_app.close()


class IsaacSimSceneVisualizer(SceneVisualizer):
    """isaacsim/isaaclab 后端可视化（后续补充）。"""

    def show_init_poses(self):
        raise NotImplementedError("IsaacSimSceneVisualizer.show_init_poses 待补充")
