# -*- coding: utf-8 -*-
"""观测场景：把机械臂【悬空】放进 USD 场景里去观测焊缝。

设计见 docs/observeanything/plan_observe_scene.md。核心是「帧相对」等价：
`Scene` 的每个下游方法只认 `InitPoseCandidate.T_workpiece_in_base`（工件 mesh 系 ↔ base_link 系
的相对位姿）。焊接场景把工件摆到臂前；观测场景把机械臂悬空放到世界某处看焊缝。设机械臂 base 在世界
的位姿为 `T_base_world`，则

    T_workpiece_in_base == T_scene_in_base == inv(T_base_world)

即「动机械臂」与「动工件」在数学上是同一个量。故只要本类产出合法的 `InitPoseCandidate`，
`set_init_pose / compute_goal_pose / compute_pose_and_plan_path / _build_explore_world /
plan_explore_path` 全部整段复用父类、一行不改。

本类相对父类只做三件事：
  ① __init__：`load_scene_meshes(usd)` 缓存整场景 prim mesh（世界系/米）；
  ② `_crop_neighborhood_obj(seam_id)`：按焊缝中心半径 R 整块裁剪 prim → 拼接（或布尔并集）→
     导出世界系(米) .obj（缓存），当作该缝的 `workpiece_obj`；`_set_cur_seam` 切缝时重指；
  ③ `plan_init_pose_fast`：重写——不动工件，直接在世界系采样机械臂 base 的 xyz+yaw，
     换算 `T_workpiece_in_base = inv(T_base_world)`，过滤后产出与父类同结构的候选。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from gt_gen.config import Config, load_config, PROJECT_ROOT
from gt_gen.scene import Scene, InitPoseCandidate, _load_plan_init_pose


def _load_scene_meshes(usd_path: str, verbose: bool = True):
    """惰性导入 scripts/find_surface_welds.py 的 load_scene_meshes（世界系/米 per-prim trimesh）。

    该脚本模块级会跑 _bootstrap_pxr()：pxr 可直接导入时立即返回、无副作用（env_isaaclab 内即如此，
    见 [[pxr_pure_import_bootstrap]]）；故仅在需要时（本函数内）导入，避免非 pxr 环境下误触发自举。"""
    import sys
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import find_surface_welds  # noqa: E402
    return find_surface_welds.load_scene_meshes(usd_path, verbose=verbose)


def _aabb_dist_to_point(tm, c: np.ndarray) -> float:
    """点 c 到 trimesh 轴对齐包围盒(AABB)的最近距离（点在盒内则 0）。用于「球心 c 半径 R 内整块保留」判定。"""
    lo, hi = np.asarray(tm.bounds[0], float), np.asarray(tm.bounds[1], float)
    d = np.maximum(np.maximum(lo - c, c - hi), 0.0)   # 每轴超出盒的量（盒内=0）
    return float(np.linalg.norm(d))


class ObserveAnythingScene(Scene):
    """把机械臂悬空放进 USD 场景观测焊缝的 Scene 子类。

    构造示例：
        scene = ObserveAnythingScene(
            cfg="configs/default.yaml",
            usd_path="/media/a/upan/others/warehouse.usdz",
            welds_json="/tmp/welds1.json")
        scene._set_cur_seam(0)                 # 切到第 0 条焊缝（重指 workpiece_obj=该缝邻域块）
        cands = scene.plan_init_pose_fast()    # 世界系采样机械臂 base 位姿 → 候选（{"forehand":[...], "backhand":[...]}）
        # 之后 set_init_pose / compute_pose_and_plan_path 全部继承父类

    cfg 接受 Config 实例 / yaml 路径 / None。welds_json 为 find_surface_welds.py 输出（键
    corrected_p0/1 / bisector / boundary_dirs …，与父类 load_welds 对齐 → _load_seam 直接复用父类）。
    """

    # plan_init_pose_fast(debug=True) 逐步过滤 6 个快照的中文步名（与 init_pose_debug_steps 逐位对齐；
    # 供 Open3DSceneVisualizer.show_observe_init_poses_debug 显示，取代父类 kejian2 版 _DEBUG_STEP_NAMES）。
    _OBS_DEBUG_STEP_NAMES = [
        "① 采样候选(未过滤)",
        "② 正面过滤后(bis_base.z≥0)",
        "③ 端点在 ee 范围后",
        "④ 焊缝中点 base-x 后",
        "⑤ 场景 vs init_free 无交集后",
        "⑥ 去重后(合格)",
    ]

    def __init__(self,
                 cfg,
                 usd_path: Optional[str] = None,
                 welds_json: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        cfg = load_config(cfg)
        # 整场景 prim mesh（世界系/米），__init__ 只加载一次、缓存在内存供各缝裁剪复用。
        # usd_path 为空（load 复原 / 纯可视化，不再裁剪新缝）时跳过整场景 mesh 加载——
        # 裁剪块 .obj 已随 save 落盘，可视化只吃 workpiece_obj，用不到 _scene_prims。
        self.usd_path: Optional[str] = usd_path
        self._scene_prims = _load_scene_meshes(usd_path, verbose=verbose) if usd_path else None
        if not welds_json:
            raise ValueError("ObserveAnythingScene 需要 welds_json（find_surface_welds.py 输出）")
        # 邻域裁剪块缓存：{seam_id: obj_fp}；体素点缓存：{seam_id: (K,3) 世界系}
        self._crop_cache: Dict[int, str] = {}
        self._scene_pts_cache: Dict[int, np.ndarray] = {}
        # 正/反手分类（初始相机可见性，见 plan_observe_scene.md §7）：初始相机 base 位姿(常量)+内参 缓存；
        # 每缝裁剪块的 warp 遮挡 mesh 缓存 {seam_id: wp.Mesh}
        self._T_base_cam: Optional[np.ndarray] = None
        self._cam_model: Optional[dict] = None
        self._occluder_wp_cache: Dict[int, object] = {}
        self._cache_dir = cache_dir or os.path.join(
            PROJECT_ROOT, "render", "_assets_cache", "observe")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._verbose = bool(verbose)
        # 父类构造：workpiece_obj 先占位（父类只存不读；_set_cur_seam 时按缝重指为邻域块），
        # weld_json=welds_json → _load_seam 直接复用父类（键已对齐）。
        super().__init__(cfg, workpiece_obj="", weld_json=welds_json)

    # ------------------------------------------------------------------
    # 存/取：父类 save 的 state 不含 usd_path/welds_json 语义，这里补一层让 load 自描述
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """沿用父类 save（落 workpiece_obj=当缝裁剪块 / weld_json=welds_json），再补写 usd_path，
        使存盘自描述（load 时若需重建整场景 mesh 可用；纯可视化则无所谓）。"""
        import pickle
        p = super().save(path)
        with open(p, "rb") as f:
            state = pickle.load(f)
        state["usd_path"] = self.usd_path
        with open(p, "wb") as f:
            pickle.dump(state, f)
        return p

    @classmethod
    def load(cls, path: str) -> "ObserveAnythingScene":
        """从 save() 存盘复原（数据状态）供可视化。**不重跑整场景 USD mesh 加载**：
        usd_path 传存盘值（旧盘可能为 None），为空则 __init__ 跳过 _load_scene_meshes（裁剪用不到），
        workpiece_obj 由 _restore_state 用存盘的当缝裁剪块路径复原 → 直接喂
        Open3DSceneVisualizer.show_observe_scene_isaacsim。

        父类 Scene.load 会以 cls(workpiece_obj=..., weld_json=...) 构造，与本子类 __init__
        签名（usd_path/welds_json）不兼容，故必须在此覆盖。weld_json 路径须仍可读（重读焊缝）。
        """
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self = cls(cfg=state["cfg"], usd_path=state.get("usd_path"),
                   welds_json=state["weld_json"], verbose=False)
        self._restore_state(state)
        return self

    # ------------------------------------------------------------------
    # USD → 焊缝邻域裁剪世界系(米) .obj
    # ------------------------------------------------------------------
    def _crop_neighborhood_obj(self, seam_id: int) -> str:
        """以该缝 mid_world 为球心、半径 crop_radius_m 整块保留 prim → 拼接（或布尔并集）→ 导出 .obj。

        · 按 prim 整块取舍（AABB 到球心距离 ≤ R），绝不按三角形裁切（切三角会留破洞、破坏闭合性）。
        · 默认 concatenate（毫秒级、每壳闭合、curobo 碰撞/ESDF 够用）；crop.watertight=true 才走
          make_watertight 口径的布尔并集成单一 2-流形（更慢，贴合处可能反破闭合）。
        结果按 usd 名 + seam_id + R 命名缓存到 self._cache_dir。返回 obj 文件路径。"""
        if seam_id in self._crop_cache and os.path.isfile(self._crop_cache[seam_id]):
            return self._crop_cache[seam_id]

        import gt_gen.compat
        import trimesh as _trimesh
        gt_gen.compat.apply_trimesh_shim()

        mid = np.asarray(self.seams[seam_id]["mid_world"], dtype=np.float64)
        R = float(self.cfg.obs_crop_radius)
        kept = [tm for _pp, tm in self._scene_prims if _aabb_dist_to_point(tm, mid) <= R]
        if not kept:
            raise RuntimeError(
                f"焊缝 {seam_id} 半径 {R}m 邻域内无任何 prim mesh（mid={np.round(mid, 3).tolist()}），"
                f"请增大 observeanything.crop.crop_radius_m")

        merged = _trimesh.util.concatenate(kept)
        if self.cfg.obs_crop_watertight:
            # 布尔并集成单一流形（复用 watertight/make_watertight.py 口径）；失败回退拼接
            import sys
            wt_dir = os.path.join(PROJECT_ROOT, "watertight")
            if wt_dir not in sys.path:
                sys.path.insert(0, wt_dir)
            import make_watertight as _mw  # noqa: E402
            try:
                solids = _mw.components_as_solids(merged)
                union = _trimesh.boolean.union(solids, engine="manifold")
                union = _mw.drop_degenerate(union)
                union.merge_vertices()
                merged = union
            except Exception as e:
                if self._verbose:
                    print(f"[observe] 焊缝 {seam_id} 布尔并集失败({e})，回退拼接")

        usd_stem = os.path.splitext(os.path.basename(str(self.usd_path)))[0]
        obj_fp = os.path.join(self._cache_dir, f"{usd_stem}_seam{seam_id}_r{R:g}.obj")
        merged.export(obj_fp)
        self._crop_cache[seam_id] = obj_fp
        if self._verbose:
            print(f"[observe] 焊缝 {seam_id} 邻域裁剪：{len(kept)} 块 prim → "
                  f"{len(merged.vertices)}v/{len(merged.faces)}f → {obj_fp}")
        return obj_fp

    def _scene_points_for_seam(self, seam_id: int) -> np.ndarray:
        """该缝邻域块的稠密体素点（世界系/米），供「场景 vs init_free 无交集」过滤用（按缝缓存）。"""
        if seam_id in self._scene_pts_cache:
            return self._scene_pts_cache[seam_id]
        import gt_gen.compat
        import trimesh as _trimesh
        gt_gen.compat.apply_trimesh_shim()
        pim = _load_plan_init_pose()
        obj_fp = self._crop_neighborhood_obj(seam_id)
        tm = _trimesh.load(obj_fp, process=False, force="mesh")
        pts = pim._voxelize_mesh_points(
            np.asarray(tm.vertices, dtype=np.float64),
            np.asarray(tm.faces, dtype=np.int64).reshape(-1, 3),
            float(self.cfg.obs_fast_workpiece_x_voxel))
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self._scene_pts_cache[seam_id] = pts
        return pts

    # ------------------------------------------------------------------
    # 正/反手分类：以「初始关节角(retract)下末端相机可见性」为准（见 plan_observe_scene.md §7）
    # ------------------------------------------------------------------
    def _init_cam_pose_and_model(self):
        """初始相机在 base 系的 4x4 位姿 T_base_cam（常量：retract_config + 相机外参都固定）+ 相机内参 dict。
        懒算一次并缓存。需 curobo FK（build_kinematics）+ warp 环境（见 §7.6）。"""
        if self._T_base_cam is None:
            from gt_gen import sensor
            cm = sensor.load_camera_model(self.cfg)
            kin = sensor.build_kinematics(self.cfg)                       # 轻量 CudaRobotModel（仅 FK）
            T = sensor.camera_pose_from_config(kin, self.cfg.retract_config, cm)
            self._cam_model = cm
            self._T_base_cam = np.asarray(T, dtype=np.float64)
        return self._T_base_cam, self._cam_model

    def _occluder_wp_mesh(self, seam_id: int):
        """本缝裁剪块（世界系）的 warp 遮挡 mesh（供视线遮挡 raycast）。按缝缓存。
        复用 stomp_planner/scene_pose2 的模块级 warp 初始化（不建 ScenePose2/cuRobo，见 §7.3）。"""
        if seam_id in self._occluder_wp_cache:
            return self._occluder_wp_cache[seam_id]
        import sys
        stomp_dir = os.path.join(PROJECT_ROOT, "stomp_planner")
        if stomp_dir not in sys.path:
            sys.path.insert(0, stomp_dir)
        import scene_pose2 as _sp2                       # 顶层仅 torch/numpy，curobo 惰性，import 安全
        import warp as wp
        import trimesh as _trimesh
        import gt_gen.compat
        gt_gen.compat.apply_trimesh_shim()

        _sp2._ensure_warp()
        obj_fp = self._crop_neighborhood_obj(seam_id)
        tm = _trimesh.load(obj_fp, process=False, force="mesh")
        verts = np.asarray(tm.vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(tm.faces, dtype=np.int32).reshape(-1)
        mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device="cuda"),
                       indices=wp.array(faces, dtype=wp.int32, device="cuda"))
        self._occluder_wp_cache[seam_id] = mesh
        return mesh

    @staticmethod
    def _raycast_hits(wp_mesh, starts: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """批量首次命中：starts/dirs 为 (P,3)，返回命中点 (P,3)，无命中处填 inf
        （语义同 scene_pose2._raycast；复用其模块级 warp 内核）。"""
        import sys
        stomp_dir = os.path.join(PROJECT_ROOT, "stomp_planner")
        if stomp_dir not in sys.path:
            sys.path.insert(0, stomp_dir)
        import scene_pose2 as _sp2
        import warp as wp
        o = np.ascontiguousarray(starts, dtype=np.float32).reshape(-1, 3)
        d = np.ascontiguousarray(dirs, dtype=np.float32).reshape(-1, 3)
        n = o.shape[0]
        t_out = wp.zeros(n, dtype=wp.float32, device="cuda")
        wp.launch(_sp2._raycast_kernel(), dim=n,
                  inputs=[wp_mesh.id,
                          wp.array(o, dtype=wp.vec3, device="cuda"),
                          wp.array(d, dtype=wp.vec3, device="cuda"),
                          float(100.0), t_out],
                  device="cuda")
        t = t_out.numpy().reshape(-1, 1)                # (P,1)，miss=-1
        hit = o + t * d
        hit[t[:, 0] < 0] = np.inf
        return hit

    def _classify_hands(self, seam_id: int, results: List[dict]) -> None:
        """就地给每个候选写 result["hand"]：初始相机对该焊缝可见→forehand，全不可见→backhand（§7.4）。
        批量：所有候选 × N 焊缝采样点一次算 FOV（内参投影）+ 遮挡（warp raycast），单次 GPU 调用。"""
        if not results:
            return
        T_base_cam, cm = self._init_cam_pose_and_model()
        seam = self.seams[seam_id]
        p0 = np.asarray(seam["p0_world"], dtype=np.float64)
        p1 = np.asarray(seam["p1_world"], dtype=np.float64)
        Ns = max(2, int(self.cfg.obs_fast_visible_num_samples))
        ts = np.linspace(0.0, 1.0, Ns)[:, None]
        seam_world = (1.0 - ts) * p0[None, :] + ts * p1[None, :]          # (Ns,3) 固定（焊缝在世界系）

        # 每候选的相机世界位姿：T_world_cam = T_base_world @ T_base_cam，T_base_world = inv(T_workpiece_in_base)
        M = len(results)
        o_w = np.empty((M, 3), dtype=np.float64)
        RcT = np.empty((M, 3, 3), dtype=np.float64)                       # 光学→世界 旋转的转置（世界→光学）
        for i, r in enumerate(results):
            T_wc = np.linalg.inv(np.asarray(r["T_workpiece_in_base"], dtype=np.float64)) @ T_base_cam
            o_w[i] = T_wc[:3, 3]
            RcT[i] = T_wc[:3, :3].T

        # ① FOV（内参投影）：p_cam = R_cam^T (seam_world - o_w) → (M,Ns,3)
        diff = seam_world[None, :, :] - o_w[:, None, :]                   # (M,Ns,3)
        p_cam = np.einsum("mij,msj->msi", RcT, diff)                      # (M,Ns,3)
        z = p_cam[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cm["fx"] * p_cam[:, :, 0] / z + cm["cx"]
            v = cm["fy"] * p_cam[:, :, 1] / z + cm["cy"]
        in_fov = ((z > 0.0) & (z <= cm["max_depth"]) &
                  (u >= 0.0) & (u < cm["width"]) & (v >= 0.0) & (v < cm["height"]))   # (M,Ns)

        visible = np.zeros((M, Ns), dtype=bool)
        mi, si = np.nonzero(in_fov)                                       # 只对 FOV 内的点做遮挡（省算）
        if mi.size:
            r_disk = float(self.cfg.obs_fast_block_radius)
            Nb = max(0, int(self.cfg.obs_fast_num_block_pts))
            starts0 = o_w[mi]                                             # (P,3) 相机光心
            targets = seam_world[si]                                     # (P,3) 焊缝点
            main_dir = targets - starts0
            dist = np.linalg.norm(main_dir, axis=1, keepdims=True)
            dirn = main_dir / np.clip(dist, 1e-8, None)                   # (P,3)
            # 偏移起点：垂直视线的小圆盘上周向撒 Nb 点（照搬 generate_start_points 逻辑）
            ref = np.where((np.abs(dirn[:, [0]]) < 0.9), np.array([1., 0., 0.]), np.array([0., 1., 0.]))
            uu = np.cross(dirn, ref); uu /= np.clip(np.linalg.norm(uu, axis=1, keepdims=True), 1e-8, None)
            vv = np.cross(dirn, uu); vv /= np.clip(np.linalg.norm(vv, axis=1, keepdims=True), 1e-8, None)
            P = starts0.shape[0]
            starts_all = [starts0]                                        # 主射线起点 = 光心
            if Nb > 0:
                theta = np.linspace(0.0, 2.0 * np.pi, Nb)
                for th in theta:
                    starts_all.append(starts0 + r_disk * (np.cos(th) * uu + np.sin(th) * vv))
            starts_all = np.concatenate(starts_all, axis=0)              # ((Nb+1)*P, 3)
            tgt_rep = np.tile(targets, (Nb + 1, 1))                       # 同一焊缝点为目标
            rd = tgt_rep - starts_all
            rd /= np.clip(np.linalg.norm(rd, axis=1, keepdims=True), 1e-8, None)
            wp_mesh = self._occluder_wp_mesh(seam_id)
            hits = self._raycast_hits(wp_mesh, starts_all, rd)           # ((Nb+1)*P,3)，miss=inf
            dist_c = np.linalg.norm(hits - tgt_rep, axis=1).reshape(Nb + 1, P)   # 命中点到焊缝点距离
            blocked_each = ~(dist_c < 1e-2)                              # 未命中在焊缝点(含 miss=inf)=被挡（口径同 visionBlock）
            unblock = ~blocked_each.any(axis=0)                          # (P,) 全部射线都命中焊缝点才算未遮挡
            visible[mi, si] = unblock

        for i, r in enumerate(results):
            r["hand"] = "forehand" if bool(visible[i].any()) else "backhand"


    # ------------------------------------------------------------------
    # 切缝：把 workpiece_obj 重指向本缝邻域块
    # ------------------------------------------------------------------
    def _set_cur_seam(self, seam_id):
        """切到某缝时，先把 self.workpiece_obj 重指向本缝裁剪块（让 compute_goal_pose /
        _build_explore_world / plan_init_pose_fast 都吃到本缝邻域块），再调父类 _set_cur_seam。"""
        self.workpiece_obj = self._crop_neighborhood_obj(seam_id)
        super()._set_cur_seam(seam_id)

    # _load_seam：改名后 welds json 键与父类 load_welds 完全对齐 → 直接复用父类，不覆盖。

    # ------------------------------------------------------------------
    # 重写：世界系采样机械臂 base(xyz+yaw) → inv(T_base_world) → 过滤 → 候选
    # ------------------------------------------------------------------
    def plan_init_pose_fast(self,
                            rebuild: bool = False,
                            include_obstacles: bool = True,
                            verbose: bool = False,
                            debug: bool = False) -> Dict[str, List[InitPoseCandidate]]:
        """重写父类同名方法（见 docs/observeanything/plan_observe_scene.md §2）。

        不再 lay_flat / 移动工件；改为在世界系直接采样机械臂 base 的位置(xyz) + 朝向(yaw，臂竖直)：
          · 位置：以焊缝中点 mid_world 为原点，立方体 [-H,H]³（H=sample_cube_half_m）内按
            xy_step_m / z_step_m 取格点 pos_world；
          · 朝向：绕世界竖直轴 yaw = yaw_ref + 偏差，偏差从 yaw_min~yaw_max（半开）步长 yaw_step_deg；
            yaw_ref 为【base +x 正对焊缝】的基准朝向（bisector 在 base 系 y=0、x<0），故配置的 yaw_*_deg
            是相对该基准的偏差角（度，范围 -180~180），而非世界系绝对 yaw；
        每采样点得 T_base_world=[Rz(yaw) | pos_world]，由 T_workpiece_in_base=inv(T_base_world) 得
        候选帧 (R=Rz(yaw).T, t=-R·pos_world)。过滤链（base 系，p_base=R·p_world+t）：
          ① 正面 bis_base.z≥0（front_face_filter，朝向级闸门，兼作正/反手分类：bis_base.x<0=正手）；
          ② 端点在 ee 范围（xy 径向两端都须在；竖焊缝仅较低端点判 z，横焊缝两端都判 z）；
          ③ 焊缝中点 base-x > seam_center_x_min_m（取代父类「工件最近点 base-x」判据）；
          ④ 场景体素点 vs init_free 无交集（核心需求，_pts_in_init_free）；
          ⑤ 轻去重（同朝向 + 2cm 同位）。
        产出结构与父类逐位一致的候选，写入 self.init_pose_candidates[seam_id]。
        （去掉父类的「底座-工件 XY 投影相交」过滤——整仓库 footprint 必然盖住 base。）

        debug=True 时额外收集 6 个逐步过滤快照（① 未过滤～⑥ 合格，每步随机抽样封顶
        obs_fast_debug_max_per_step；① 保证每个 yaw 一个代表、含被正面过滤刷掉的 yaw）写入
        self.init_pose_debug_steps[seam_id] + 步名 self.init_pose_debug_step_names[seam_id]，
        随 Scene.save 落 pkl，供 Open3DSceneVisualizer.show_observe_init_poses_debug(n) 单步可视化。
        """
        pim = _load_plan_init_pose()
        c = self.cfg
        seam = self.seam
        seam_id = self.seam_id

        p0 = np.asarray(seam["p0_world"], dtype=np.float64)
        p1 = np.asarray(seam["p1_world"], dtype=np.float64)
        mid = np.asarray(seam["mid_world"], dtype=np.float64)
        bis = np.asarray(seam["bisector_world"], dtype=np.float64)
        bis = bis / (np.linalg.norm(bis) + 1e-12)

        H = float(c.obs_fast_sample_cube_half)
        xy_step = float(c.obs_fast_xy_step)
        z_step = float(c.obs_fast_z_step)
        yaw_min, yaw_max, yaw_step = c.obs_fast_yaw_deg
        xy_lo, xy_hi = c.obs_fast_ee_xy_range
        z_lo, z_hi = c.obs_fast_ee_z_range
        # 竖/横焊缝判据：焊缝起终点世界系高度差 > 阈值 ⇒ 竖焊缝（端点只判 xy 径向、不判 z）；否则横焊缝（xyz 全判）
        vertical_dz = float(c.obs_fast_vertical_seam_dz)
        is_vertical = abs(float(p0[2] - p1[2])) > vertical_dz
        x_min = float(c.obs_fast_seam_center_x_min)
        standoff = float(c.obs_fast_standoff)
        front_face = bool(c.obs_fast_front_face_filter)
        region = pim._init_free_region_from_cfg(c)
        retract_q = np.asarray(c.retract_config, dtype=np.float64)

        # 场景稠密体素点（世界系）——供 init_free 交集判定
        scene_pts = self._scene_points_for_seam(seam_id)   # (V,3) 世界系

        # 采样格点（世界系）= mid + [-H,H]³ 内网格
        def _axis(step):
            return np.arange(-H, H + 1e-9, float(step))
        gx, gy = _axis(xy_step), _axis(xy_step)
        gz = _axis(z_step)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        offsets = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)   # (N,3)
        G_world = mid[None, :] + offsets                                   # (N,3) base 原点候选（世界系）

        # yaw 采样：配置的 yaw_*_deg 不是世界系 yaw，而是相对【base +x 正对焊缝】基准朝向的偏差角（度，范围 -180~180）。
        # 基准朝向 yaw_ref：使 bisector 在 base 系满足 y=0 且 x<0（即 base +x 指向焊缝、bisector 指 base 的 -x）。
        # bis_base = Rz(yaw)ᵀ·bis ⇒ bis_base.y=0 得 yaw=atan2(bis.y,bis.x)（此时 bis_base.x=+|bis_xy|>0），
        # 再 +180° 翻到 bis_base.x<0 ⇒ yaw_ref = atan2(bis.y, bis.x) + 180°。世界 yaw = yaw_ref + 偏差。
        yaw_ref_deg = np.degrees(np.arctan2(float(bis[1]), float(bis[0]))) + 180.0
        yaws = yaw_ref_deg + np.arange(float(yaw_min), float(yaw_max), float(yaw_step))

        def _inxy(pb):   # (M,3) → 布尔 (M,)：径向 ∈ [xy_lo, xy_hi]
            r = np.hypot(pb[:, 0], pb[:, 1])
            return (r >= xy_lo) & (r <= xy_hi)

        def _inz(pb):    # (M,3) → 布尔 (M,)：base-z ∈ [z_lo, z_hi]
            return (pb[:, 2] >= z_lo) & (pb[:, 2] <= z_hi)

        results = []
        seen = set()
        n_yaw_kept = 0
        n_grid = len(G_world)
        n_ep = n_xmin = n_free = n_dedup = 0

        import random

        # ---- debug 逐步快照收集（debug=True 才建；每步随机抽样封顶 dbg_max，① 保证每 yaw 一代表）----
        DBG = bool(debug)
        dbg_max = int(c.obs_fast_debug_max_per_step)
        snap_reps: List[dict] = []   # ① 每 yaw 一代表（含被正面过滤刷掉的 yaw）
        snap_raw:  List[dict] = []   # ① 采样候选（保留 yaw 全格点抽样）
        snap_front: List[dict] = []  # ② 正面过滤后
        snap_ep:   List[dict] = []   # ③ 端点在 ee 范围后
        snap_xmin: List[dict] = []   # ④ 焊缝中点 base-x 后
        snap_free: List[dict] = []   # ⑤ init_free 无交集后（去重前）

        def _mk(R_, t_, bis_, seam_c_, hand_, quat_, oid_) -> dict:
            """按 kejian2 raw 结构造候选 dict（可视化所需字段全含）；数组均 copy，避免 debug 快照
            持有 t_all/mid_base 大数组视图导致内存驻留。"""
            seam_c_ = np.array(seam_c_, dtype=np.float64).reshape(3)
            ee = seam_c_ + standoff * bis_
            T = np.eye(4); T[:3, :3] = R_; T[:3, 3] = np.asarray(t_, dtype=np.float64)
            return {
                "workpiece_pose7": pim.mat44_to_pose7(T),
                "T_workpiece_in_base": T,
                "goal_pose7": np.concatenate([ee, quat_]),
                "joint_angles": np.asarray(retract_q, dtype=np.float64),
                "rot_x_deg": 0.0, "rot_y_deg": 0.0, "rot_z_deg": 0.0,
                "bisector_base": np.array(bis_, dtype=np.float64),
                "seam_center_base": seam_c_,
                "wpx_near_base": None,
                "orientation_id": int(oid_),
                "hand": hand_,
            }

        def _pick(arr, cap):   # 从索引数组随机抽 ≤cap 个（保序无所谓，可视化会 stride）
            arr = np.asarray(arr).reshape(-1)
            if len(arr) <= cap:
                return arr
            return arr[np.random.choice(len(arr), size=int(cap), replace=False)]

        for oid, yaw in enumerate(yaws):
            th = np.deg2rad(float(yaw))
            ct, st = np.cos(th), np.sin(th)
            Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            R = Rz.T                                    # world→base 旋转（p_base = R·p_world + t）

            bis_base = R @ bis
            bis_base = bis_base / (np.linalg.norm(bis_base) + 1e-12)
            # 旧 per-yaw 几何判据仅留作 debug 快照(①~⑤)的占位标签；真正的正/反手分类改用「初始相机可见性」，
            # 在候选全部产出后由 _classify_hands 批量判定（见 plan_observe_scene.md §7）。
            hand_dbg = "forehand" if float(bis_base[0]) < 0.0 else "backhand"
            goal_quat = pim.rotmat_to_quat_wxyz(pim._align_rotmat([1.0, 0.0, 0.0], bis_base))

            # ① 每 yaw 一代表：base 原点落焊缝中点（t=-R·mid → seam_center=0），含被正面过滤刷掉的 yaw
            if DBG:
                snap_reps.append(_mk(R, -(R @ mid), bis_base, np.zeros(3), hand_dbg, goal_quat, oid))

            # ② 正面过滤：bis_base 只随朝向 R 变、与平移无关 ⇒ 朝向级闸门
            if front_face and float(bis_base[2]) < 0.0:
                continue
            n_yaw_kept += 1

            Rp0, Rp1, Rmid = R @ p0, R @ p1, R @ mid
            RS = scene_pts @ R.T                        # (V,3) 场景点旋转部分，per-candidate 只 + t

            # t_all = -R·pos_world（对每个采样格点）
            t_all = -(G_world @ R.T)                    # (N,3)
            p0_base = Rp0[None, :] + t_all              # (N,3)
            p1_base = Rp1[None, :] + t_all
            mid_base = Rmid[None, :] + t_all

            # ③ 端点在 ee 范围：xy 径向两端点都须在；竖焊缝仅要求世界系较低端点 z 在范围内，横焊缝两端点 z 都须在
            m_ep = _inxy(p0_base) & _inxy(p1_base)
            if is_vertical:
                low_base = p0_base if float(p0[2]) <= float(p1[2]) else p1_base
                m_ep &= _inz(low_base)
            else:
                m_ep &= _inz(p0_base) & _inz(p1_base)
            # ④ 焊缝中点 base-x > x_min
            m_x = mid_base[:, 0] > x_min
            idx = np.nonzero(m_ep & m_x)[0]
            n_ep += int(m_ep.sum())
            n_xmin += int((m_ep & m_x).sum())

            if DBG:
                # ① 采样候选(全格点抽样) + ② 正面过滤后（同一存活总体）
                for k in _pick(np.arange(n_grid), dbg_max).tolist():
                    snap_raw.append(_mk(R, t_all[k], bis_base, mid_base[k], hand_dbg, goal_quat, oid))
                    snap_front.append(_mk(R, t_all[k], bis_base, mid_base[k], hand_dbg, goal_quat, oid))
                # ③ 端点在范围后
                for k in _pick(np.nonzero(m_ep)[0], dbg_max).tolist():
                    snap_ep.append(_mk(R, t_all[k], bis_base, mid_base[k], hand_dbg, goal_quat, oid))
                # ④ 焊缝中点 base-x 后
                for k in _pick(idx, dbg_max).tolist():
                    snap_xmin.append(_mk(R, t_all[k], bis_base, mid_base[k], hand_dbg, goal_quat, oid))

            n_free_yaw = 0   # 本 yaw 已收进 snap_free 的数量（每 yaw 封顶 dbg_max）
            for k in idx.tolist():
                t = t_all[k]
                # ⑤ 场景 vs init_free 无交集
                if pim._pts_in_init_free(RS + t, region).any():
                    n_free += 1
                    continue
                seam_center = mid_base[k]                        # 真实焊缝中点在 base
                if DBG and n_free_yaw < dbg_max:                 # init_free 后（去重前）快照
                    snap_free.append(_mk(R, t, bis_base, seam_center, hand_dbg, goal_quat, oid))
                    n_free_yaw += 1
                # ⑥ 轻去重：同朝向 + 2cm 同位
                key = (oid, round(float(t[0]), 2), round(float(t[1]), 2), round(float(t[2]), 2))
                if key in seen:
                    n_dedup += 1
                    continue
                seen.add(key)

                ee_pos = seam_center + standoff * bis_base
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
                results.append({
                    "workpiece_pose7": pim.mat44_to_pose7(T),
                    "T_workpiece_in_base": T,
                    "goal_pose7": np.concatenate([ee_pos, goal_quat]),
                    "joint_angles": np.asarray(retract_q, dtype=np.float64),
                    "rot_x_deg": 0.0, "rot_y_deg": 0.0, "rot_z_deg": 0.0,
                    "bisector_base": np.asarray(bis_base, dtype=np.float64),
                    "seam_center_base": np.asarray(seam_center, dtype=np.float64),
                    "wpx_near_base": None,
                    "orientation_id": int(oid),
                    "hand": None,                        # 由 _classify_hands 批量判定（§7）
                })

        # 正/反手分类：初始相机对该焊缝可见→forehand，全不可见→backhand（逐候选，批量一次 GPU 调用，§7.4）
        self._classify_hands(seam_id, results)
        fore = [r for r in results if r["hand"] == "forehand"]
        back = [r for r in results if r["hand"] == "backhand"]
        print("[observe-fast] 逐步过滤 候选初始位姿（世界系采样机械臂 base xyz+yaw）：")
        print(f"  ① 采样候选（{len(yaws)} yaw × {n_grid} 格点）             : {len(yaws) * n_grid}")
        print(f"  ② 正面过滤(bis_base.z≥0)                    : {len(yaws)} → {n_yaw_kept} yaw"
              f" → 候选 {n_yaw_kept * n_grid}")
        print(f"  ③ 端点在 ee 范围（{'竖焊缝:xy两端+较低端点z' if is_vertical else '横焊缝:xyz两端'}）      : {n_yaw_kept * n_grid} → {n_ep}")
        print(f"  ④ 焊缝中点 base-x > {x_min:.3f}m                 : {n_ep} → {n_xmin}")
        print(f"  ⑤ 场景 vs init_free 无交集                  : {n_xmin} → {n_xmin - n_free}（相交丢 {n_free}）")
        print(f"  ⑥ 轻去重(同朝向 + 2cm 同位)                 : {n_xmin - n_free} → {len(results)}（重复丢 {n_dedup}）")
        print(f"  ⇒ 合格 {len(results)}（正手 {len(fore)} / 反手 {len(back)}；yaw {n_yaw_kept} 种）")

        fore_c = [InitPoseCandidate.from_kejian2(d) for d in fore]
        back_c = [InitPoseCandidate.from_kejian2(d) for d in back]

        history = self.init_pose_candidates.get(seam_id, {"forehand": [], "backhand": []})
        # 按【机械臂 base 位姿差异】降序排列（差异大的排前面，xyz 权重高于 yaw）——替代随机排序
        yw = self.cfg.obs_fast_sort_yaw_weight
        history["forehand"].extend(self._sort_by_base_pose_diversity(fore_c, yaw_weight=yw))
        history["backhand"].extend(self._sort_by_base_pose_diversity(back_c, yaw_weight=yw))
        self.init_pose_candidates[seam_id] = history
        self.init_pose_candidates_length[seam_id] = {
            "forehand": len(history["forehand"]),
            "backhand": len(history["backhand"]),
        }
        # 逐步过滤快照：debug=True 才建（每步全局再抽样封顶 dbg_max；⑥ 合格步全存不受限），
        # 随 Scene.save 落 pkl，供 show_observe_init_poses_debug 单步可视化；否则置空（与父类字段兼容）。
        if DBG:
            def _trim(lst):
                return random.sample(lst, dbg_max) if len(lst) > dbg_max else lst
            self.init_pose_debug_steps[seam_id] = [
                snap_reps + _trim(snap_raw),   # ① 采样候选(未过滤)：每 yaw 代表 + 全格点抽样
                _trim(snap_front),             # ② 正面过滤后
                _trim(snap_ep),                # ③ 端点在 ee 范围后
                _trim(snap_xmin),              # ④ 焊缝中点 base-x 后
                _trim(snap_free),              # ⑤ init_free 无交集后（去重前）
                list(results),                 # ⑥ 去重后（合格），全存
            ]
            self.init_pose_debug_step_names[seam_id] = list(self._OBS_DEBUG_STEP_NAMES)
        else:
            self.init_pose_debug_steps[seam_id] = []
            self.init_pose_debug_step_names[seam_id] = []
        self.init_pose_prefilter_steps[seam_id] = {}
        return self.init_pose_candidates[seam_id]

    @staticmethod
    def _sort_by_base_pose_diversity(cands: List["InitPoseCandidate"],
                                     yaw_weight: float = 0.3) -> List["InitPoseCandidate"]:
        """把同一手别候选按【机械臂 base 位姿差异】独立分数降序排列（差异大的排前面）。

        与父类 Scene._sort_by_xyz_diversity（只看工件平移 t）不同：本类采样的是机械臂 base 在
        世界系的位姿(xyz+yaw)，而每个候选的 t 落在各自 base 系（R 不同），跨候选比 t 无物理意义。
        故先由 T_workpiece_in_base 还原每个候选的 base 世界位姿：
          · R = Rz(yaw)ᵀ ⇒ yaw = atan2(R[0,1], R[0,0])；
          · pos_world = -Rᵀ · t（机械臂 base 原点在世界系）。
        再算差异分：每个候选到本组其余候选的【加权平均距离】，加权 = xyz 欧氏距离(米) + yaw_weight
        × yaw 环形角距(rad, 归一到各自最大量级后合成，故 yaw_weight 直接是相对权重)；分越高=越离群/
        铺得开，排越前。**xyz 比 yaw 更重要**（yaw_weight<1）。≤1 个时原样返回；稳定排序（同分保序）。
        """
        n = len(cands)
        if n <= 1:
            return list(cands)
        pos = np.empty((n, 3), dtype=np.float64)   # 机械臂 base 原点（世界系）
        yaw = np.empty(n, dtype=np.float64)        # base 绕世界竖直轴 yaw（rad）
        for i, c in enumerate(cands):
            R = np.asarray(c.R, dtype=np.float64).reshape(3, 3)
            t = np.asarray(c.t, dtype=np.float64).reshape(3)
            pos[i] = -(R.T @ t)
            yaw[i] = np.arctan2(R[0, 1], R[0, 0])
        # 两两 xyz 欧氏距离
        d_xyz = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)          # (n,n) 米
        # 两两 yaw 环形角距 ∈ [0, π]
        dyaw = np.abs(yaw[:, None] - yaw[None, :])
        d_yaw = np.minimum(dyaw, 2.0 * np.pi - dyaw)                               # (n,n) rad
        # 各自按最大量级归一（尺度无关），再按权重合成——xyz 权重固定 1，yaw 权重更低
        d_xyz_n = d_xyz / (d_xyz.max() + 1e-12)
        d_yaw_n = d_yaw / (d_yaw.max() + 1e-12)
        D = d_xyz_n + float(yaw_weight) * d_yaw_n                                  # (n,n)
        score = D.sum(axis=1) / float(n - 1)
        order = sorted(range(n), key=lambda i: -float(score[i]))                   # 分数降序、稳定
        return [cands[i] for i in order]

