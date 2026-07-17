"""Step 10: 主循环编排（准备阶段 + ①~⑦）+ 冷启动 + 卡住处理。

见 docs/privileged-nbv.md §4.5 完整主循环、docs/step10-plan.md。

四个「世界」分工（详见 step10-plan.md）：
- h_truth：MESH 真值世界（含工件 mesh）——全知特权专家：P*（plan_on_truth）、候选 IK +
  check_state（工件避障）、NBV 打分。不执行 GT。
- h_expl：VOXEL 纯三态探索世界（无 mesh）——机械臂真正规划/执行的世界，GT 来自这里。
  UNKNOWN 在 sync 后当障碍 → 规划出的轨迹天然只走已确认 FREE。
- truth_scene：trimesh 真值网格——raycast 几何源（实拍 + NBV 假设性 raycast）。
- voxmap：三态体素图（UNKNOWN/FREE/OCCUPIED）——唯一持久记忆，reach_pt/B/可达/进展都查它。

一句话：机械臂只敢走「亲眼看过是空的」区域。每轮先问「现在能直接到目标吗？」(①)，不能就请
全知教练指一个「最值得看一眼」的位置(②③④)，走过去边走边拍(⑤⑥)，已知区像水面一样扩大直到
淹没目标(① 成功)。⑦ 防止原地打转无限循环。
"""
from __future__ import annotations

import os
import time

import numpy as np

# PROFILEMAIN=1 时在 generate_gt 主循环内按步累计耗时（sync/直达/P*/NBV/move/observe/empty_cache），
# 循环结束打印每步总耗时——用于定位主循环内的瓶颈步骤。
_PROFILEMAIN = os.environ.get("PROFILEMAIN") == "1"


# ---------------- 私有辅助 ----------------

def _observe(voxmap, fk_handle, q, camera_model, truth_scene, max_depth, pixel_stride: int = 16):
    """在构型 q 处实拍一次并写回 voxmap（更新占用的唯一入口）。

    fk_handle 仅用于相机 FK（camera_pose_from_config）——FK 与碰撞世界无关，h_truth/h_expl 任一皆可。
    返回 observe_and_update 的生效体素数 dict。
    """
    from gt_gen.sensor import camera_pose_from_config
    from gt_gen.mapping import observe_and_update

    cam_pose = camera_pose_from_config(fk_handle, q, camera_model)
    return observe_and_update(voxmap, cam_pose, camera_model, truth_scene, max_depth,
                              pixel_stride=pixel_stride)


def _move_to(h_expl, voxmap, cur_cfg, target_cfg, camera_model, truth_scene, max_depth,
             every_n=None, pixel_stride: int = 16, max_stomp_try=1):
    """在 h_expl(VOXEL) 上规划 cur_cfg→target_cfg，取插值轨迹；沿途每 every_n 个路点 _observe
    一次（边走边拍），终点构型再补拍一次。

    every_n=None 时取 h_expl.config.params.loop.observe_every_n（默认 10）。
    返回 (seg, seg_obs)：该段轨迹去掉首点后的 (t,dof) np（接进 GT）+ 等长 0/1 观测标记
    （该路点是否被 _observe 拍过）；规划失败返回 (None, None)。
    """
    from gt_gen import curobo_iface as ci

    if every_n is None:
        every_n = int(h_expl.config.params.get("loop", {}).get("observe_every_n", 10))
    every_n = max(1, int(every_n))

    # _debug_viz_voxmap(voxmap, h_expl, target_cfg, truth_scene, show_unknown=True, cur_cfg=cur_cfg, cfg_show=True)
    # _debug_viz_voxmap(voxmap, h_expl, target_cfg, truth_scene, show_unknown=True, cur_cfg=cur_cfg, cfg_show=False)
    cfg = h_expl.config
    if cfg.planner_backend == "stomp":
        # STOMP 后端：把 voxmap 的非 FREE 区域转 STOMP 世界（mesh 或 cuboid，见 voxel_world），关节目标规划 cur->target。
        from gt_gen import stomp_iface as si
        world, ck = si.world_from_voxmap_auto(cfg, voxmap)
        traj = si.plan_joint_single(cfg, cur_cfg=cur_cfg, target_cfg=target_cfg, world=world,
                                    checker_type=ck)
        if traj is None:
            for _ in range(max_stomp_try):
                traj = si.plan_joint_single(cfg, cur_cfg=cur_cfg, target_cfg=target_cfg, world=world,
                                    checker_type=ck)
                if traj is not None:
                    break

        if traj is None:
            print("[_move_to] STOMP plan_joint 失败:", ci.explain_endpoints(h_expl, cur_cfg, target_cfg))
            return None, None
    else:
        res = ci.plan_to_config(h_expl, cur_cfg, target_cfg, max_attempts=cfg.plan_max_attempts)
        if res is None or not bool(res.success.item()):
            # 诊断：起点/终点哪个在碰撞，还是中间连不上（区分三种成因）
            print("[_move_to] plan_to_config 失败:", ci.explain_endpoints(h_expl, cur_cfg, target_cfg))
            # cuRobo 实际避障的占据场（sync 后、含 inflate），看 target_cfg 整臂是否泡在障碍里
            return None, None
        traj = res.get_interpolated_plan().position.detach().cpu().numpy()

    # 沿途每 every_n 个路点拍一次（h_expl 做相机 FK，与 h_truth 同一套运动学）
    obs_flags = np.zeros(len(traj), dtype=np.int64)
    for i in range(0, len(traj), every_n):
        _observe(voxmap, h_expl, traj[i], camera_model, truth_scene, max_depth, pixel_stride)
        obs_flags[i] = 1
    _observe(voxmap, h_expl, traj[-1], camera_model, truth_scene, max_depth, pixel_stride)
    obs_flags[-1] = 1

    # _debug_viz_voxmap(voxmap, h_expl, traj[-1], truth_scene, show_unknown=True, transparent=False)  # 本段走完后的 voxmap 三态（含灰 UNKNOWN）
    return traj[1:], obs_flags[1:]                    # 去掉与上一段重复的首点


def _frontier_cells(voxmap) -> np.ndarray:
    """探索前沿：所有『自身 UNKNOWN 且 6-邻接含 FREE』的体素下标 (M,3)。"""
    from gt_gen.voxmap import UNKNOWN, FREE

    grid = voxmap.grid
    unknown = (grid == UNKNOWN)
    free = (grid == FREE)
    nb = np.zeros_like(free)
    nb[1:, :, :] |= free[:-1, :, :]
    nb[:-1, :, :] |= free[1:, :, :]
    nb[:, 1:, :] |= free[:, :-1, :]
    nb[:, :-1, :] |= free[:, 1:, :]
    nb[:, :, 1:] |= free[:, :, :-1]
    nb[:, :, :-1] |= free[:, :, 1:]
    return np.argwhere(unknown & nb).astype(np.int64)


def handle_stuck(h_truth, h_expl, voxmap, cur_cfg, camera_model, truth_scene, max_depth,
                 every_n=None):
    """「就近揭示」兜底（B 被遮死/够不着时）：不再非 B 不可，改为去揭开任意 frontier 未知。

    以 _frontier_cells(vm) 为目标，generate_candidates(h_truth) 找可达候选，选「假设性 raycast
    揭开 UNKNOWN 最多」的那个，_move_to 过去。先把已知空间整体摊大，常能间接绕开遮挡。
    返回 (progressed: bool, new_cfg, seg, seg_obs)；无可达候选/规划失败 → (False, cur_cfg, None, None)。
    """
    from gt_gen.candidates import generate_candidates
    from gt_gen.nbv import raycast_reveal
    from gt_gen.voxmap import UNKNOWN

    frontier = _frontier_cells(voxmap)
    if frontier.shape[0] == 0:
        return False, cur_cfg, None, None

    cands = generate_candidates(h_truth, voxmap, frontier, camera_model, cur_cfg)
    if not cands:
        return False, cur_cfg, None, None

    best, best_gain = None, 0
    for c in cands:
        reveal = raycast_reveal(voxmap, c.cam_pose, camera_model, truth_scene)
        if reveal.shape[0] == 0:
            continue
        gain = int((np.asarray(voxmap.get(reveal)) == UNKNOWN).sum())   # 能揭开多少未知
        if gain > best_gain:
            best, best_gain = c, gain
    if best is None or best_gain <= 0:
        return False, cur_cfg, None, None

    seg, seg_obs = _move_to(h_expl, voxmap, cur_cfg, best.config, camera_model, truth_scene,
                            max_depth, every_n=every_n)
    if seg is None:
        return False, cur_cfg, None, None
    return True, list(best.config), seg, seg_obs


def look_around(handle, voxmap, cur_cfg, camera_model, truth_scene, max_depth):
    """冷启动 / 兜底的小幅环视。v1：在当前构型补拍一次（保留函数位以备加强为按关节 ±dq 摆动多拍）。

    返回本次观测的生效体素数 dict。
    """
    return _observe(voxmap, handle, cur_cfg, camera_model, truth_scene, max_depth)


def _debug_viz_observe(voxmap, fk_handle, q, camera_model, truth_scene, max_depth, stage):
    """调试用：可视化 _observe 前/后 voxmap 三态——看清「在构型 q 处拍一次」把哪些 UNKNOWN
    翻成 FREE/OCCUPIED。默认不调用（调用点处注释掉），需要目视时手动取消注释。

    复用 scripts/verify_step8 的 open3d 工具（需显示器 + open3d）。画 FREE(蓝半透明)+OCCUPIED(红)
    +工件(灰)+整臂(绿)+相机视锥(紫,远面=max_depth)。stage∈{"before","after"}：
      before — 拍前（如冷启动时仅初始圆柱 FREE、OCC=0）；紫锥示意即将观测的方向/范围；
      after  — 拍后：视锥扫过处沿射线新增 FREE、命中工件表面处新增 OCCUPIED（红壳）。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import (_arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base,
                              _fov_frustum)
    from gt_gen.sensor import camera_pose_from_config
    from gt_gen.voxmap import FREE, OCCUPIED

    M = camera_pose_from_config(fk_handle, q, camera_model)      # 构型 q 处相机 4x4 位姿
    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(fk_handle, list(q)), "lit", None)]
    fc = voxmap.state_centers(FREE)
    n_free = int(fc.shape[0])
    if n_free:
        geoms.append(("free", _cells_mesh(voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.12]))
    oc = voxmap.state_centers(OCCUPIED)
    n_occ = int(oc.shape[0])
    if n_occ:
        om = _cells_mesh(voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))
    edges, cone = _fov_frustum(M, camera_model, max_depth)       # 相机视锥（紫，远面=max_depth）
    geoms.append(("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.12]))
    geoms.append(("fov_edges", edges, "line", None))
    geoms += _roi_and_base(voxmap)
    tag = "之前(仅初始FREE,OCC=0)" if stage == "before" else "之后(视锥扫过新增FREE+命中红壳OCC)"
    _draw(geoms, f"main_loop observe {tag}: FREE={n_free}格 OCC={n_occ}格 "
                 f"紫锥=相机视野(深{max_depth:.1f}m) 绿=整臂 灰=工件")


def _debug_viz_curobo(h_expl, voxmap, fk_handle, q, truth_scene, stage="", every_n_layers: int = 8):
    """调试用①【cuRobo 世界】：可视化 h_expl(voxel) 碰撞世界【实际判为占据】的体素——sync 之后
    cuRobo 把哪些格当障碍。目视核对占据有没有灌错位（历史 Y 轴帧错位 bug）。默认不调用，需手动取消注释。

    占据来自 curobo_occupied_centers(h_expl)（读回 ESDF feature > 阈值的格）。整块占据 ≈ 全 ROI 减
    FREE 圆柱(~80 万格)，直接画会卡死；故沿 z【每 every_n_layers 层取 1 层】水平切片(暗红)抽稀显示。
    叠加整臂(绿) + 工件(灰，纯参照，h_expl 里并无 mesh)。
    stage∈{"before","after",""}：
      before — sync 前（h_expl 默认全自由）→ 红切片应为空（占据=0）；
      after  — sync 后 → 各层红切片布满、仅 FREE 圆柱处留洞（占据≫0）。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base
    from gt_gen.collision_sync import curobo_occupied_centers

    vs = voxmap.voxel_size
    n_layers = max(1, int(every_n_layers))

    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(fk_handle, list(q)), "lit", None)]
    occ = curobo_occupied_centers(h_expl)                          # cuRobo 实判占据的体素中心
    n_occ = int(occ.shape[0])
    n_show = 0
    if n_occ:
        z_origin = float(voxmap.origin[2])
        layer = np.rint((occ[:, 2] - z_origin) / vs).astype(int)   # 各占据体素的 z 层索引
        slabs = occ[layer % n_layers == 0]                         # 每 n 层留 1 层
        n_show = int(slabs.shape[0])
        if n_show:
            sm = _cells_mesh(voxmap, slabs); sm.paint_uniform_color([0.65, 0.05, 0.05])
            geoms.append(("curobo_occ_slabs", sm, "lit", None))

    geoms += _roi_and_base(voxmap)
    tag = {"before": "(sync前,应全自由)", "after": "(sync后,各层布满仅圆柱留洞)"}.get(stage, "")
    _draw(geoms, f"main_loop cuRobo世界{tag}: 占据={n_occ}格(每{n_layers}层取1层显示{n_show}格) "
                 f"暗红=cuRobo占据切片 绿=整臂 灰=工件")


def _debug_viz_w1(w1, fk_handle, voxmap, cur_cfg, truth_scene, goal_pose=None, rnd=None):
    """调试用【STOMP 探索世界 _w1】：可视化 world_from_voxmap 产出的 marching-cubes mesh
    （= 非 FREE 区的「自由泡边界 + ROI 外壳」等值面，STOMP 实际拿来避障的那张网）+ 当前整臂@cur_cfg。
    目视核对：mesh 是否把已确认 FREE 区正确围出空腔、机械臂当前是否在腔内、面数是否爆炸。
    默认在步① _w1 算出后调用（每轮弹一次阻塞窗口；只想看首轮可在调用处加 if rnd == 0）。

    w1 内每个 Mesh 顶点已是 base 系绝对坐标、pose=单位 → 直接转 open3d，无需再变换。
    橙色半透明=_w1 mesh，绿=当前整臂，灰=工件，另叠 goal 坐标系 + ROI/base。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _work_mesh, _draw, _roi_and_base
    import open3d as o3d

    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm_cur", _arm_mesh(fk_handle, list(cur_cfg)), "lit", None)]   # 当前整臂(绿)
    n_faces = 0
    for i, m in enumerate(getattr(w1, "mesh", None) or []):
        tm = m.get_trimesh_mesh()
        v = np.asarray(tm.vertices, float)
        f = np.asarray(tm.faces, np.int32)
        n_faces += int(f.shape[0])
        o3m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                        o3d.utility.Vector3iVector(f))
        o3m.compute_vertex_normals()
        geoms.append((f"w1_mesh_{i}", o3m, "fill", [0.95, 0.45, 0.10, 0.35]))  # 探索障碍 mesh(橙,半透明)
    n_cub = 0
    for i, c in enumerate(getattr(w1, "cuboid", None) or []):                  # cuboid 版：合并后的大盒
        dx, dy, dz = (float(v) for v in c.dims)
        cx, cy, cz = (float(v) for v in c.pose[:3])                            # 轴对齐(quat=单位)，无需旋转
        bx = o3d.geometry.TriangleMesh.create_box(dx, dy, dz)
        bx.translate((cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0))
        bx.compute_vertex_normals()
        geoms.append((f"w1_cuboid_{i}", bx, "fill", [0.95, 0.45, 0.10, 0.35]))
        n_cub += 1
    if goal_pose is not None:
        gp = (goal_pose[0] if (isinstance(goal_pose, (tuple, list)) and len(goal_pose) == 2
                               and hasattr(goal_pose[0], "__len__")) else goal_pose[:3])
        gf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        gf.translate(np.asarray(gp, float)[:3])
        geoms.append(("goal", gf, "lit", None))
    geoms += _roi_and_base(voxmap)
    rtag = "" if rnd is None else f" R{rnd}"
    _draw(geoms, f"main_loop _w1{rtag}: 探索障碍 mesh面={n_faces}/cuboid={n_cub}（橙,半透明） 绿=当前整臂 灰=工件")


def _debug_viz_voxmap(voxmap, fk_handle, q, truth_scene, every_n_layers: int = 8,
                      show_unknown: bool = False, transparent: bool = True, cur_cfg=None,
                      cfg_show: bool = True):
    """调试用②【voxmap 世界】：可视化三态体素图本身（sync 的输入，不随 sync 改变）。默认不调用，需手动取消注释。

    画 FREE(蓝，整块) + OCCUPIED(红，整块) + 可选 UNKNOWN(灰，抽稀)；叠加目标整臂@q(绿) + 工件(灰)。
    cur_cfg 非 None 时再叠加【当前整臂@cur_cfg(橙)】，一眼对比「这一步从哪走到哪」。
    cfg_show：True(默认) → 画目标整臂@q(绿)与当前整臂@cur_cfg(橙)；False → 两条整臂都不画，只看体素三态。
    FREE/OCCUPIED 量级小直接整块画；UNKNOWN ≈ 全 ROI(~80 万格)，show_unknown=True 时才沿 z
    每 every_n_layers 层抽 1 层显示，避免卡死。
    transparent：True → 体素半透明填充("fill"，能透视内部/被遮挡的格)；False → 不透明实心("lit"，看外形更清楚)。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base
    from gt_gen.voxmap import FREE, OCCUPIED, UNKNOWN

    vs = voxmap.voxel_size
    n_layers = max(1, int(every_n_layers))

    def _cells(name, centers, rgb, alpha):
        """按 transparent 选透明填充("fill"+rgba) 或不透明实心("lit"+paint)。"""
        m = _cells_mesh(voxmap, centers)
        if transparent:
            return (name, m, "fill", list(rgb) + [alpha])
        m.paint_uniform_color(list(rgb))
        return (name, m, "lit", None)

    geoms = [("work", _work_mesh(truth_scene), "lit", None)]
    if cfg_show:
        at = _arm_mesh(fk_handle, list(q)); at.paint_uniform_color([0.10, 0.80, 0.20])
        geoms.append(("arm_target", at, "lit", None))               # 目标构型整臂(绿)
        if cur_cfg is not None:
            ac = _arm_mesh(fk_handle, list(cur_cfg)); ac.paint_uniform_color([1.0, 0.55, 0.0])
            geoms.append(("arm_cur", ac, "lit", None))              # 当前构型整臂(橙)
    fc = voxmap.state_centers(FREE)
    n_free = int(fc.shape[0])
    if n_free:
        geoms.append(_cells("vm_free", fc, [0.20, 0.45, 0.95], 0.12))
    oc = voxmap.state_centers(OCCUPIED)
    n_occ = int(oc.shape[0])
    if n_occ:
        geoms.append(_cells("vm_occ", oc, [0.92, 0.12, 0.12], 0.5))
    n_unk_show = 0
    if show_unknown:
        uc = voxmap.state_centers(UNKNOWN)
        if uc.shape[0]:
            z_origin = float(voxmap.origin[2])
            layer = np.rint((uc[:, 2] - z_origin) / vs).astype(int)
            slabs = uc[layer % n_layers == 0]                      # UNKNOWN 巨量 → 抽稀
            n_unk_show = int(slabs.shape[0])
            if n_unk_show:
                geoms.append(_cells("vm_unk", slabs, [0.55, 0.55, 0.55], 0.10))

    geoms += _roi_and_base(voxmap)
    unk_tag = f" 灰UNKNOWN(每{n_layers}层显示{n_unk_show}格)" if show_unknown else ""
    fill_tag = "半透明" if transparent else "不透明"
    arm_tag = ("绿=目标整臂 橙=当前整臂" if cur_cfg is not None else "绿=目标整臂") if cfg_show else "不画整臂"
    _draw(geoms, f"main_loop voxmap世界({fill_tag}): FREE={n_free}格(蓝) OCC={n_occ}格(红){unk_tag} {arm_tag} 灰=工件")


def _debug_viz_pstar(h_truth, voxmap, cur_cfg, P, truth_scene, goal_pose=None, rnd=None,
                     every_n: int = 20):
    """调试用【真值最优路 P*】：可视化 plan_on_truth 返回的 P*（步②）。被调用即弹窗(需显示器+open3d)。

    公共底图：工件(灰) + 当前整臂@cur_cfg(绿) + voxmap FREE(蓝半透明)/OCCUPIED(红) + ROI/base；
    叠加：P* 的 flange 原点折线(黑) + 【沿 P* 每 every_n 个路点的整臂碰撞球】(青→品红渐变，含起点/终点)
    + goal 位置(品红球)。一眼看出「真值上从当前构型到目标整条路怎么绕、整臂在每一段的姿态」——
    P*=None 时只画底图并在标题示意不可行。依赖 verify_step8 的 open3d 工具。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base, _lines, _ball
    from gt_gen.voxmap import FREE, OCCUPIED
    from gt_gen.candidates import flange_origin

    tag = f"R{rnd} " if rnd is not None else ""

    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(h_truth, list(cur_cfg)), "lit", None)]
    fc = voxmap.state_centers(FREE)
    n_free = int(fc.shape[0])
    if n_free:
        geoms.append(("free", _cells_mesh(voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.10]))
    oc = voxmap.state_centers(OCCUPIED)
    n_occ = int(oc.shape[0])
    if n_occ:
        om = _cells_mesh(voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))

    n_arms = 0
    if P is not None and len(P):
        step = max(1, len(P) // 60)
        fo = np.asarray([flange_origin(h_truth, list(P[i])) for i in range(0, len(P), step)])
        if fo.shape[0] >= 2:
            segs = [(fo[i], fo[i + 1]) for i in range(fo.shape[0] - 1)]
            geoms.append(("Pstar", _lines(segs, [0.1, 0.1, 0.1]), "line", None))   # P* flange 折线（黑）
        # 沿 P* 每 every_n 个路点画整臂（含起点与终点）；颜色青→品红线性渐变示意先后
        n = max(1, int(every_n))
        idxs = sorted(set(list(range(0, len(P), n)) + [len(P) - 1]))
        m = max(1, len(idxs) - 1)
        for j, i in enumerate(idxs):
            t = j / m
            col = [0.0 + 0.85 * t, 0.75 - 0.65 * t, 0.85]                            # 青(起)→品红(终)
            a = _arm_mesh(h_truth, list(P[i])); a.paint_uniform_color(col)
            geoms.append((f"P{i}", a, "lit", None))
        n_arms = len(idxs)
    if goal_pose is not None:
        geoms.append(("goal", _ball(np.asarray(goal_pose[0], float), 0.03, [0.85, 0.10, 0.85]), "lit", None))
    geoms += _roi_and_base(voxmap)

    n_tag = f"{len(P)}点(每{max(1, int(every_n))}步画臂×{n_arms})" if P is not None else "None(真值不可达)"
    _draw(geoms, f"{tag}main_loop P*({n_tag}): 黑=flange折线 青→品红=沿路整臂 绿=当前臂 灰=工件")


def _debug_viz_seg(h_truth, voxmap, cur_cfg, seg, truth_scene, goal_pose=None, rnd=None,
                   every_n: int = 20):
    """调试用【执行段轨迹 seg】：可视化 _move_to 规划出、即将接进 GT 的这一段插值轨迹。

    底图同 _debug_viz_pstar：工件(灰) + 当前整臂@cur_cfg(绿) + voxmap FREE(蓝半透明)/OCCUPIED(红)
    + ROI/base；叠加：seg 的 flange 原点折线(橙) + 沿 seg 每 every_n 个路点的整臂(青→品红渐变，含
    起点/终点) + goal 位置(品红球)。seg 为 (t,dof) 关节轨迹。依赖 verify_step8 的 open3d 工具
    （需显示器 + open3d）。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base, _lines, _ball
    from gt_gen.voxmap import FREE, OCCUPIED
    from gt_gen.candidates import flange_origin

    tag = f"R{rnd} " if rnd is not None else ""
    S = np.asarray(seg, dtype=float) if seg is not None else None

    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(h_truth, list(cur_cfg)), "lit", None)]
    fc = voxmap.state_centers(FREE)
    if int(fc.shape[0]):
        geoms.append(("free", _cells_mesh(voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.10]))
    oc = voxmap.state_centers(OCCUPIED)
    if int(oc.shape[0]):
        om = _cells_mesh(voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))

    n_arms = 0
    if S is not None and len(S):
        step = max(1, len(S) // 60)
        fo = np.asarray([flange_origin(h_truth, list(S[i])) for i in range(0, len(S), step)])
        if fo.shape[0] >= 2:
            segs = [(fo[i], fo[i + 1]) for i in range(fo.shape[0] - 1)]
            geoms.append(("seg", _lines(segs, [1.0, 0.55, 0.0]), "line", None))     # seg flange 折线（橙）
        # 沿 seg 每 every_n 个路点画整臂（含起点与终点）；颜色青→品红线性渐变示意先后
        n = max(1, int(every_n))
        idxs = sorted(set(list(range(0, len(S), n)) + [len(S) - 1]))
        m = max(1, len(idxs) - 1)
        for j, i in enumerate(idxs):
            t = j / m
            col = [0.0 + 0.85 * t, 0.75 - 0.65 * t, 0.85]                            # 青(起)→品红(终)
            a = _arm_mesh(h_truth, list(S[i]), link_name="xiaoyu_accessory_link"); a.paint_uniform_color(col)
            geoms.append((f"S{i}", a, "lit", None))
        n_arms = len(idxs)
    if goal_pose is not None:
        geoms.append(("goal", _ball(np.asarray(goal_pose[0], float), 0.03, [0.85, 0.10, 0.85]), "lit", None))
    geoms += _roi_and_base(voxmap)

    n_tag = f"{0 if S is None else len(S)}点(每{max(1, int(every_n))}步画臂×{n_arms})"
    _draw(geoms, f"{tag}main_loop seg({n_tag}): 橙=flange折线 青→品红=沿段整臂 绿=当前臂 灰=工件")


def _seg_dump_path(path=None):
    """执行段落盘文件路径：显式 path > 环境变量 GT_SEG_DUMP > 默认 configs/_seg_dump_isaacsim.pkl。"""
    import os
    if path:
        return path
    return os.environ.get("GT_SEG_DUMP") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "_seg_dump_isaacsim.pkl")


def _dump_seg_isaacsim(cur_cfg, seg, rnd=None, path=None, reset=False):
    """与 _debug_viz_seg 对应的【isaacsim 版】：不在本进程弹窗，而是把【执行段 seg】的关节轨迹落盘，
    交给独立干净进程用 isaacsim 回放这段移动——工件 + 障碍物 + 机械臂，**不含 voxmap**。
    回放脚本见 scripts/viz_seg_isaacsim.py。

    为何落盘而非直接画：show_scene_isaacsim 的 SimulationApp 必须在【未加载 warp】的干净进程里最先
    启动，不能与跑 generate_gt(warp/curobo) 的本进程同框（同 Scene.save/load 的动机）。故这里只存
    轨迹，可视化另起进程。

    落盘格式：pickle 一个 list，每元素 {"rnd": int|None, "positions": (T,dof) np.float64}。
    positions 在段首补回 cur_cfg（_move_to 返回的 seg 已去掉与上一段重复的首点），使拼接回放连续。

    参数：
      reset=True —— 清空/新建落盘（generate_gt 开跑时调一次，免得累加上一轮 run 的段）；此时 cur_cfg/seg 忽略。
      否则把本段 (rnd, [cur_cfg]+seg) 追加进列表；seg 为 None/空则跳过。
      path=None 时取 _seg_dump_path()（环境变量 GT_SEG_DUMP 或默认 configs 下）。
    """
    import os
    import pickle

    path = _seg_dump_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if reset:
        with open(path, "wb") as f:
            pickle.dump([], f)
        print(f"[seg-dump] 清空 → {path}")
        return path

    if seg is None:
        return path
    S = np.asarray(seg, dtype=np.float64)
    if S.ndim != 2 or len(S) == 0:
        return path
    positions = np.vstack([np.asarray(cur_cfg, dtype=np.float64)[None, :], S])   # 补回段首起点

    data = []
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            data = []
    data.append({"rnd": rnd, "positions": positions})
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[seg-dump] R{rnd} 追加段 {positions.shape[0]} 点 → {path}（累计 {len(data)} 段）")
    return path


def _debug_viz_nbv(h_truth, voxmap, cur_cfg, r, camera_model, truth_scene, max_depth, params,
                   rnd=None):
    """调试用③【一轮 NBV 结果】：可视化 best_next_view_using_oracle 返回的 r（参考 verify_step9 的
    raycast_reveal/score 窗口）。默认不调用（调用点处注释掉），诊断 step10 stuck 时手动取消注释。

    每轮弹一个窗口，按 r.status 画不同内容（公共底图：工件灰 + 当前整臂@cur_cfg绿 + voxmap FREE蓝半透明
    + OCCUPIED红 + P*末端轨迹白线 + reach_idx处球）：
      ok                     —— 选中视点整臂@r.cfg(青) + 相机帧/FOV视锥(紫,远面过T) + 视线→T(红) +
                                 目标T(品红球) + B 按【该视点假设性 reveal 是否覆盖】着色(覆盖=绿/没覆盖=橙)；
                                 标题写 gain/score/|B|/候选数。一眼看出「选中的视点到底揭不揭得开 B」。
      corridor_confirmed     —— B 空、走廊已确认：只画 P*/reach_idx，标题示意应能直接规划到目标。
      no_reachable_candidate —— 有 B 但无可达候选(常是 stuck 主因)：B 全画橙(没有候选能看它)，标题示意转兜底。
      scene_infeasible       —— P* 不存在：仅底图，标题示意场景不可行。

    一屏最多三个不同颜色的整臂，各是一个不同的关节构型：
      绿臂  cur_cfg          —— 当前整臂：机械臂这一轮实际所在的构型（底图）。
      黄臂  P[reach_idx]     —— reach_pt 构型：沿真值最优路 P* 走到「被 UNKNOWN 挡住、走不下去」
                               的那个路点的整臂（仅 P* 存在时画）。
      青臂  r.cfg            —— 选中的下一视点整臂：NBV 这一轮 argmax 挑出的候选构型（把相机移到这看 B），
                               仅 status==ok 时画。
    一句话：绿=现在在哪，黄=沿最优路卡在哪，青=下一步打算把相机移到哪。

    依赖 verify_step8 的 open3d 工具（需显示器 + open3d）。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import open3d as o3d
    from verify_step8 import (_arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base,
                              _ball, _lines, _fov_frustum)
    from gt_gen.voxmap import FREE, OCCUPIED
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.candidates import flange_origin
    from gt_gen import nbv as _nbv

    tag = f"R{rnd} " if rnd is not None else ""

    # —— 公共底图：工件 + 当前整臂 + voxmap 三态 ——
    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(h_truth, list(cur_cfg)), "lit", None)]
    fc = voxmap.state_centers(FREE)
    n_free = int(fc.shape[0])
    if n_free:
        geoms.append(("free", _cells_mesh(voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.10]))
    oc = voxmap.state_centers(OCCUPIED)
    n_occ = int(oc.shape[0])
    if n_occ:
        om = _cells_mesh(voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))

    # —— P* 末端轨迹（黑线，子采样的 flange 原点）+ reach_pt 处整臂碰撞球（黄） ——
    P = r.P_star
    if P is not None and len(P):
        step = max(1, len(P) // 40)
        fo = np.asarray([flange_origin(h_truth, list(P[i])) for i in range(0, len(P), step)])
        if fo.shape[0] >= 2:
            segs = [(fo[i], fo[i + 1]) for i in range(fo.shape[0] - 1)]
            geoms.append(("Pstar", _lines(segs, [0.1, 0.1, 0.1]), "line", None))
        ri = int(np.clip(r.reach_idx, 0, len(P) - 1))
        ra = _arm_mesh(h_truth, list(P[ri])); ra.paint_uniform_color([0.95, 0.85, 0.0])
        geoms.append(("reach_arm", ra, "lit", None))                  # reach_pt 构型整臂碰撞球（黄）

    # —— 重算 B（NBVResult 只给 n_B，可视化需体素本身）——
    B = np.empty((0, 3), dtype=np.int64)
    if P is not None and r.status in ("ok", "no_reachable_candidate"):
        k = int((params or {}).get("nbv", {}).get("k_lookahead", 6))
        B = compute_blocking_B(h_truth, voxmap, P, r.reach_idx, k)

    if r.status == "ok":
        # 选中视点假设性 reveal → B 是否被覆盖（绿=覆盖/橙=没覆盖），直接看「这一步揭不揭得开 B」
        Bw = voxmap.voxel_to_world(B) if B.shape[0] else np.empty((0, 3))
        reveal = _nbv.raycast_reveal(voxmap, r.cam_pose, camera_model, truth_scene, max_depth=max_depth)
        seen = (np.array([tuple(b) in set(map(tuple, reveal)) for b in B], bool)
                if B.shape[0] else np.zeros(0, bool))
        if B.shape[0] and (~seen).any():
            mm = _cells_mesh(voxmap, Bw[~seen]); mm.paint_uniform_color([1.0, 0.55, 0.0])
            geoms.append(("B_miss", mm, "lit", None))                  # B 没被看到 橙
        if B.shape[0] and seen.any():
            mm = _cells_mesh(voxmap, Bw[seen]); mm.paint_uniform_color([0.1, 0.85, 0.2])
            geoms.append(("B_seen", mm, "lit", None))                  # B 被看到 绿
        # 选中视点的整臂（青）+ 相机帧/FOV/视线/目标
        am = _arm_mesh(h_truth, list(r.cfg)); am.paint_uniform_color([0.10, 0.75, 0.80])
        geoms.append(("arm_next", am, "lit", None))
        eye = np.asarray(r.cam_pose)[:3, 3]
        T = np.asarray(r.target) if r.target is not None else eye
        depth = float(np.linalg.norm(T - eye)) or max_depth
        edges, cone = _fov_frustum(np.asarray(r.cam_pose), camera_model, depth)
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12); fr.transform(np.asarray(r.cam_pose))
        geoms += [("cam", fr, "lit", None),
                  ("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.15]),
                  ("fov_edges", edges, "line", None),
                  ("ray", _lines([(eye, T)], [0.9, 0.1, 0.1]), "line", None),
                  ("T", _ball(T, 0.04, [0.9, 0.1, 0.9]), "lit", None)]
        n_seen = int(seen.sum())
        title = (f"main_loop NBV {tag}status=ok: 选中视点(青臂) gain={r.gain:.0f} score={r.score:.2f} "
                 f"|B|={r.n_B} 候选={r.n_candidates} | reveal∩B 实测覆盖={n_seen}/{r.n_B}(绿)未覆盖(橙)")
    elif r.status == "no_reachable_candidate":
        if B.shape[0]:
            geoms.append(("B", _cells_mesh(voxmap, voxmap.voxel_to_world(B)), "fill", [1.0, 0.55, 0.0, 0.30]))  # B 全橙半透明：没有候选能看它
        title = (f"main_loop NBV {tag}status=no_reachable_candidate: |B|={r.n_B}(橙) 无可达候选 "
                 f"→ 转就近揭示兜底(常为 stuck 主因)")
    elif r.status == "corridor_confirmed":
        title = (f"main_loop NBV {tag}status=corridor_confirmed: B空,走廊已确认 reach_idx={r.reach_idx}"
                 f"/{len(P)-1 if P is not None else '?'} → 应能直接规划到目标")
    else:  # scene_infeasible
        title = f"main_loop NBV {tag}status={r.status}: P* 不存在 → 场景不可行"

    geoms += _roi_and_base(voxmap)
    _draw(geoms, title + f"  [FREE={n_free} OCC={n_occ} 绿=当前臂 黑线=P* 黄臂=reach_pt构型]")


def _debug_viz_candidates(h_truth, voxmap, cur_cfg, r, r_list, camera_model, truth_scene, max_depth,
                          params, rnd=None):
    """调试用④【逐候选 + 各自分数】：这一轮 NBV 的【每一个候选】各弹一窗、画全（参考 verify_step9
    的 score 窗口，一候选一窗）。默认不调用，需手动取消注释。

    直接复用 best_next_view_using_oracle 返回的 r_list（已按 score 降序的 NBVResult 列表，
    每项自带 cfg/cam_pose/target/gain/score），【不再重算 generate_candidates + score_candidate】——
    这样窗口里画的就是本轮 NBV 真正评估过的那批候选，避免重算时 IK 多种子随机抖动画出与实际
    决策不一致的候选。仅 B（橙阻塞集底图）仍按 r.P_star/r.reach_idx 重算（NBVResult 不带 B 体素、
    且这不算候选）；path_cost 用焊枪平移代价即时算出仅供显示。
    控制台先打印逐候选 gain/path_cost/score 表（★标 argmax=score 最高，即 r_list[0]）；随后
    【每个候选一窗】，每窗画该候选的整臂(青，真摆成看 B 的姿态) + 相机帧 + FOV视锥(远面过 T) +
    视线 p→T(红) + 目标 T(品红球)，底图含工件灰 + 当前整臂@cur_cfg绿 + FREE蓝半透明 + OCCUPIED红
    + B橙；标题写该候选 i/N、gain/path_cost/score、是否 ★argmax。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import open3d as o3d
    from verify_step8 import (_arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base,
                              _ball, _lines, _fov_frustum)
    from gt_gen.voxmap import FREE, OCCUPIED
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen import nbv as _nbv

    tag = f"R{rnd} " if rnd is not None else ""
    nbv_p = (params or h_truth.config.params).get("nbv", {})
    k = int(nbv_p.get("k_lookahead", 6))
    lam = float(nbv_p.get("lambda_cost", 0.0))

    # —— B（橙底图）仍按 P*/reach_idx 重算；候选直接用 r_list（不重算 generate_candidates/score）——
    P = r.P_star
    B = (compute_blocking_B(h_truth, voxmap, P, r.reach_idx, k)
         if P is not None and r.status in ("ok", "no_reachable_candidate")
         else np.empty((0, 3), dtype=np.int64))
    cands = list(r_list)                                               # 本轮 NBV 已评估过的候选（NBVResult，含 cfg/cam_pose/target/gain/score）
    rows = []                                                          # (gain, path_cost, score)
    for c in cands:
        pc = _nbv._gun_translation_cost(h_truth, cur_cfg, c.cfg)       # 仅显示用；非重算候选
        rows.append((c.gain, pc, c.score))

    print(f"\n== _debug_viz_candidates {tag}status={r.status} |B|={B.shape[0]} 候选={len(cands)} "
          f"lambda_cost={lam} ==")
    best_i = int(np.argmax([x[2] for x in rows])) if rows else -1
    for i, (g, pc, s) in enumerate(rows):
        print(f"    候选#{i}: gain={int(g):3d}  path_cost={pc:.3f}  score={s:.2f}"
              f"{' ★argmax' if i == best_i else ''}")

    # —— 公共底图（每个候选窗都含）——
    def base_geoms():
        g = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(h_truth, list(cur_cfg)), "lit", None)]
        fc = voxmap.state_centers(FREE)
        if fc.shape[0]:
            g.append(("free", _cells_mesh(voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.08]))
        oc = voxmap.state_centers(OCCUPIED)
        if oc.shape[0]:
            om = _cells_mesh(voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
            g.append(("occ", om, "lit", None))
        if B.shape[0]:
            bm = _cells_mesh(voxmap, voxmap.voxel_to_world(B)); bm.paint_uniform_color([1.0, 0.55, 0.0])
            g.append(("B", bm, "lit", None))                          # B 橙（候选都朝它看）
        g += _roi_and_base(voxmap)
        return g

    if not cands:
        _draw(base_geoms(), f"main_loop 候选打分 {tag}status={r.status}: |B|={B.shape[0]} 无候选可打分"
                            f"（B空=走廊已确认 / 有B无候选=转就近揭示兜底） 绿=当前臂 橙=B")
        return

    # —— 每个候选一窗：整臂(青) + 相机帧 + FOV + 视线→T ——
    N = len(cands)
    for i, c in enumerate(cands):
        g, pc, s = rows[i]
        geoms = base_geoms()
        am = _arm_mesh(h_truth, list(c.cfg)); am.paint_uniform_color([0.10, 0.75, 0.80])
        geoms.append(("arm_cand", am, "lit", None))                   # 该候选构型整臂（真摆成看 B）
        eye = np.asarray(c.cam_pose)[:3, 3]; T = np.asarray(c.target)
        depth = float(np.linalg.norm(T - eye)) or max_depth
        edges, cone = _fov_frustum(np.asarray(c.cam_pose), camera_model, depth)
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12); fr.transform(np.asarray(c.cam_pose))
        geoms += [("cam", fr, "lit", None),
                  ("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.15]),
                  ("fov_edges", edges, "line", None),
                  ("ray", _lines([(eye, T)], [0.9, 0.1, 0.1]), "line", None),
                  ("T", _ball(T, 0.04, [0.9, 0.1, 0.9]), "lit", None)]
        star = " ★argmax" if i == best_i else ""
        _draw(geoms, f"main_loop 候选 {tag}{i+1}/{N}{star}: gain={int(g)} path_cost={pc:.3f} "
                     f"score=gain−{lam}·cost={s:.2f} |B|={B.shape[0]} 青臂=该候选 橙=B 紫=FOV(远面过T)")



# ---------------- 顶层：完整主循环 ----------------

def generate_gt(h_truth, h_expl, voxmap, truth_scene, goal_pose,
                camera_model=None, params=None, h_truth_plan=None, p_star_init=None,
                world_plan=None, goal_cfg=None, start_cfg=None, max_stomp_try=1):
    """完整 ①~⑦ 主循环（goal 已含 standoff 后退）。

    入参：
      h_truth     : MESH 真值 handle（含工件 mesh）——P*/NBV/候选避障。
      h_expl      : VOXEL 探索 handle（纯三态，无 mesh）——①/⑤ 的实际无碰撞规划。
      voxmap      : 三态体素图，调用方已建好并罩好初始 FREE（圆柱法）。
      truth_scene : base 系 trimesh，raycast 几何源。
      goal_pose   : 末端 standoff 目标位姿 ((x,y,z),(qw,qx,qy,qz))。goal_cfg 给定且此参为 None
                    时，由 FK(goal_cfg) 求出（与关节目标自洽），仅供 NBV/curobo 后端用。
      goal_cfg    : 可选【关节空间目标】(长度=dof)。给定则 STOMP 后端的步①(直达)与步②(P*)改走
                    plan_joint_single——直接规划到该关节角，不再解 IK / 走位姿目标（见对话确认：
                    place_obstacles_to_gt2 用 compute_goal_pose 的 joints 变体当目标）。curobo 后端
                    忽略此参、仍用 goal_pose。
      camera_model: None → load_camera_model(h_truth.config)。
      params      : None → h_truth.config.params。
      h_truth_plan: 可选 MESH 真值 handle，仅用于步② 规划 P*——其【放置的障碍已外扩 buffer、
                    工件不变】（同 place_obstacles 的 world_inflated）。传入则 P* 与真实障碍留间隙，
                    机械臂沿 P* 探索时不会贴着障碍走；None → 退回用 h_truth（无 buffer，原行为）。
                    注意：碰撞/NBV/raycast 仍用 h_truth（真实尺寸），buffer 只影响 P* 的走向。
      p_star_init : 可选 (T,dof) 预规划轨迹（= --scene npz 里 place_obstacles 已成功规划好的绕行轨迹）。
                    传入则【第一轮(rnd==0)直接拿它当 P*，不再重新规划】——绕开「同一空间这里 plan 却
                    失败」的随机性/位姿差异，用全知阶段已验证可行的那条路起步。其起点须 = 实际起点
                    （start_cfg 给定则为它，否则 retract_config；place_obstacles 也从 retract 规划，故一致）。
                    仅第一轮用；之后臂已移动，照常重新规划。
      world_plan  : 可选 cuRobo WorldConfig（= h_plan 对应的 MESH 世界，工件+膨胀障碍）。仅当
                    cfg.planner_backend=='stomp' 时步② 需要它（STOMP 把 world 作参数直接传入）；
                    curobo 后端忽略此参数（用 h_plan 内部世界）。None 时 stomp 步② 无法规划 P*。
      start_cfg   : 可选起始关节角(长度=dof)。给定则主循环从它起步；None → 用 cfg.retract_config
                    （保持原行为）。供 place_obstacles_to_gt2 传入 scene.cur_cfg，免去覆盖 retract_config 的 hack。

    返回 (GT, status, info)：
      GT     : (T, dof) np.float64 关节角序列（含起点 retract）。
      status : reached / infeasible / stuck / max_rounds。
      info   : 诊断 dict（rounds / n_B 轨迹 / free 增长 / status_seq / P_len）。
    """
    import torch
    from gt_gen import curobo_iface as ci
    from gt_gen.collision_sync import sync_collision_world
    from gt_gen.nbv import best_next_view_using_oracle
    from gt_gen.sensor import load_camera_model
    from gt_gen.voxmap import FREE

    cfg = h_truth.config
    if camera_model is None:
        camera_model = load_camera_model(cfg)
    if params is None:
        params = cfg.params
    # goal_cfg（关节空间目标）：STOMP 步①/步② 直接规划到该关节角（plan_joint_single）；goal_pose
    # 缺省则由 FK(goal_cfg) 求出，供 NBV（plan_on_truth 兜底）/curobo 后端使用，与关节目标自洽。
    if goal_cfg is not None:
        goal_cfg = [float(v) for v in goal_cfg]
        if goal_pose is None:
            eep, eeq, _ = ci.fk(h_truth, goal_cfg)
            goal_pose = (list(map(float, eep)), list(map(float, eeq)))
    loop_p = params.get("loop", {})
    max_rounds = int(loop_p.get("max_rounds", 200))
    stuck_rounds = int(loop_p.get("stuck_rounds", 5))
    every_n = int(loop_p.get("observe_every_n", 10))
    max_depth = cfg.max_depth_m

    cur_cfg = [float(v) for v in start_cfg] if start_cfg is not None else list(cfg.retract_config)
    GT = [np.asarray(cur_cfg, dtype=np.float64)]
    OBS = [1]           # 是否被 _observe 拍过，与 GT 逐行对齐；起点紧接着下面就补拍一次
    GOAL_FLAG = [0]      # 是否是本段规划目标到达点（观测目标 goal_cfg/goal_pose），与 GT 逐行对齐
    # _dump_seg_isaacsim(None, None, reset=True)               # 清空执行段落盘（供 viz_seg_isaacsim.py 干净进程回放）
    metric = ci.free_pose_metric(h_truth, free_rot=(0,))     # 放开焊枪绕接近轴 roll
    h_plan = h_truth_plan if h_truth_plan is not None else h_truth  # 步② P* 规划用（带障碍 buffer / 退回 h_truth）

    # 准备阶段：冷启动补拍一次（首次更新占用）
    # _debug_viz_observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth, "before")  # 拍前
    _prof = {} if _PROFILEMAIN else None      # step -> 累计秒；PROFILEMAIN=1 时启用

    def _tick(key, t0):
        if _prof is not None:
            _prof[key] = _prof.get(key, 0.0) + (time.perf_counter() - t0)

    def _emit_prof():
        if _prof is None:
            return
        tot = sum(_prof.values())
        rounds = info.get("rounds", 0) or 0
        print(f"[PROFILEMAIN][generate_gt] 主循环分步耗时（{rounds} 轮，合计 {tot:.3f}s）：")
        for k, v in sorted(_prof.items(), key=lambda kv: -kv[1]):
            per = v / rounds if rounds else 0.0
            print(f"    {k:<16} {v:>8.3f}s  {100.0 * v / tot if tot else 0:>5.1f}%  (每轮 {per * 1000:>6.1f}ms)")

    _t = time.perf_counter()
    _observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth)
    _tick("observe", _t)
    # _debug_viz_observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth, "after")   # 拍后

    info = {"rounds": 0, "n_B": [], "free": [], "status_seq": [], "P_len": None}
    free_prev = voxmap.counts()[FREE]
    reach_prev = -1
    stale = 0
    status = "max_rounds"

    for rnd in range(max_rounds):
        _t = time.perf_counter()
        torch.cuda.empty_cache()
        _tick("empty_cache", _t)
        # _debug_viz_voxmap(voxmap, h_truth, cur_cfg, truth_scene, every_n_layers=4)                     # voxmap 三态（sync 输入，不随 sync 变）
        # _debug_viz_curobo(h_expl, voxmap, h_truth, cur_cfg, truth_scene, "before", every_n_layers=10)   # sync 前：cuRobo 占据应空
        _t = time.perf_counter()
        sync_collision_world(h_expl, voxmap)                 # 步0：最新「非 FREE」→ h_expl 障碍场
        _tick("sync_world", _t)
        # _debug_viz_curobo(h_expl, voxmap, h_truth, cur_cfg, truth_scene, "after", every_n_layers=10)    # sync 后：仅圆柱留洞

        # 步①：试在已确认自由区直接规划到 goal（h_expl，UNKNOWN 已当障碍）
        if cfg.planner_backend == "stomp":
            # STOMP：把当前 voxmap 非 FREE 区转 mesh；goal_cfg 给定→直接规划到目标关节角，否则规划到 goal 位姿。
            from gt_gen import stomp_iface as si
            _t = time.perf_counter()
            _w1, _ck1 = si.world_from_voxmap_auto(cfg, voxmap)
            _tick("step1_world", _t)
            # _debug_viz_w1(_w1, h_expl, voxmap, cur_cfg, truth_scene, goal_pose, rnd=rnd)  # 看 _w1（mesh/cuboid）+ 当前整臂（每轮弹窗；只看首轮改 if rnd==0）
            _t = time.perf_counter()
            if goal_cfg is not None:
                # from gt_gen.repro_plan_joint import dump_inputs; dump_inputs("plan_joint_case.pkl", cfg, _w1, cur_cfg, goal_cfg, _ck1)  # 落盘复现用
                seg = si.plan_joint_single(cfg, _w1, cur_cfg, goal_cfg, checker_type=_ck1)
                # from gt_gen.repro_plan_joint import dump_inputs
                # dump_inputs("plan_joint_case.pkl", cfg, _w1, cur_cfg, goal_cfg, _ck1)
            else:
                seg = si.plan_pose_single(cfg, _w1, cur_cfg, goal_pose, checker_type=_ck1)

            if seg is None:
                for _ in range(max_stomp_try):
                    seg = si.plan_joint_single(cfg, _w1, cur_cfg, goal_cfg, checker_type=_ck1)
                    if seg is not None:
                        break

            _tick("step1_direct", _t)
            reached_direct = seg is not None
        else:
            _t = time.perf_counter()
            res = ci.plan_to_pose(h_expl, cur_cfg, goal_pose,
                                  max_attempts=cfg.plan_max_attempts, pose_cost_metric=metric)
            _tick("step1_direct", _t)
            reached_direct = res is not None and bool(res.success.item())
            seg = (res.get_interpolated_plan().position.detach().cpu().numpy()
                   if reached_direct else None)
        if reached_direct:
            n_seg = len(seg) - 1
            GT.extend(seg[1:])
            OBS.extend([0] * n_seg)                     # 步①直达段全程未调 _observe
            if n_seg > 0:
                GOAL_FLAG.extend([0] * (n_seg - 1) + [1])   # 最后一行=真正到达 goal
            status = "reached"
            info["rounds"] = rnd + 1
            # _debug_viz_seg(h_truth, voxmap, cur_cfg, seg, truth_scene, goal_pose, rnd=rnd)  # 步① 直达目标段 seg 路径
            # _dump_seg_isaacsim(cur_cfg, seg[1:], rnd=rnd)    # 同段落盘（isaacsim 回放：工件+障碍+臂，无 voxmap）
            break

        # 步②：真值上的全知最优路 P*（挡住的只可能是 UNKNOWN）；h_plan 的障碍已含 buffer（若调用方传入）。
        _t = time.perf_counter()
        if rnd == 0 and p_star_init is not None:
            # 第一轮：直接用 --scene 里 place_obstacles 已成功规划好的绕行轨迹当 P*，不重新规划。
            # （同一 3D 空间 + 同起点 retract，但这里 plan 会随机失败/位姿略差；全知阶段那条已验证可行。）
            P = np.asarray(p_star_init, dtype=np.float64)
            print(f"[step② R0] 直接用预规划 P*（--scene 绕行轨迹）{P.shape[0]} 点，跳过重新规划")
        elif cfg.planner_backend == "stomp":
            # STOMP：在 h_plan 对应的膨胀 MESH 世界(world_plan)上规划 P*；single 已返回最优一条
            # （等价 multi 取首条，但更简）。world_plan 由调用方按 h_plan 的 WorldConfig 传入。
            from gt_gen import stomp_iface as si
            if world_plan is None:
                print(f"[step② P*失败 R{rnd}] backend=stomp 但未传 world_plan → 无法规划 P*")
                P = None
            else:
                # goal_cfg 给定→规划到目标关节角（plan_joint_single，无 IK），否则规划到 goal 位姿。
                if goal_cfg is not None:
                    # 步② P* 的 STOMP 迭代次数（覆盖 cfg.stomp_params['num_iterations']）；按需调此常量。
                    P = si.plan_joint_single(cfg, world_plan, cur_cfg, goal_cfg)
                else:
                    P = si.plan_pose_single(cfg, world_plan, cur_cfg, goal_pose)
                    # [1.2627240419387817, -2.024371862411499, 6.27759313583374, -0.5791741609573364, -1.5592968463897705, 3.2276997566223145]
                    # [-0.13571767508983612, -0.9203471541404724, 1.2579456567764282, -1.0674389600753784, -0.9313104748725891, -2.814171075820923]
                #_debug_viz_seg(h_truth, voxmap, cur_cfg, seg, truth_scene, goal_pose, rnd=rnd)
                #为了解决较难case, 多规划几遍可能就有解了
                if P is None:
                    for _ in range(max_stomp_try):
                        P = si.plan_joint_single(cfg, world_plan, cur_cfg, goal_cfg)
                        if P is not None:
                            break
                if P is None:
                    print(f"[step② P*失败 R{rnd}] STOMP 在 world_plan 上未找到到 goal 的合格轨迹")
                    # STOMP 规划不出 P* → 本场景不可行，直接失败退出 generate_gt（不再进 NBV/后续轮）
                    info["rounds"] = rnd + 1
                    status = "infeasible"
                    info["observe"] = np.asarray(OBS, dtype=np.int64)
                    info["goal"] = np.asarray(GOAL_FLAG, dtype=np.int64)
                    _tick("step2_pstar", _t)
                    _emit_prof()
                    return np.asarray(GT, dtype=np.float64), status, info
        else:
            # 改走 plan_to_pose_all（= place_obstacles.detour_exists 那条成功路径）：对多条 IK 分支逐个 plan，
            # 取第 0 条（IK 误差最小的成功解）。与 plan_to_pose 同核，但显式跑满 IK 多解、对随机失败更稳。
            _Ps = ci.plan_to_pose_all(h_plan, cur_cfg, goal_pose, max_attempts=cfg.plan_max_attempts,
                                      pose_cost_metric=metric, max_solutions=1, dedup_rad=0.0)
            P = _Ps[0] if _Ps else None
            if P is None:
                # 诊断 P* 失败到底卡在哪（IK 无解 / 起点碰撞 / 终点碰撞 / 中段连不上）。
                _ik = ci.solve_ik(h_plan, goal_pose, pose_cost_metric=metric, return_seeds=cfg.ik_return_seeds)
                _cands = ci.ik_configs(h_plan, _ik)
                if not _cands:
                    print(f"[step② P*失败 R{rnd}] IK 完全无解：goal 在(膨胀)真值世界不可达/被障碍包住 "
                          f"goal_pos={np.round(np.asarray(goal_pose[0], float), 3)}")
                else:
                    print(f"[step② P*失败 R{rnd}] IK 有 {len(_cands)} 解但 plan 全失败 → "
                          f"{ci.explain_endpoints(h_plan, cur_cfg, _cands[0][0])}")
        # _debug_viz_pstar(h_truth, voxmap, cur_cfg, P, truth_scene, goal_pose, rnd=rnd)  # 看真值最优路 P*（注释此行可关）
        _tick("step2_pstar", _t)
        # 步③④：一轮特权 NBV（P* → reach_pt/B → 候选 → 假设性 raycast 打分 → argmax）
        _t = time.perf_counter()
        r, r_list = best_next_view_using_oracle(h_truth, cur_cfg, voxmap, truth_scene, goal_pose,
                                        params=params, camera_model=camera_model,
                                        pose_cost_metric=metric, p_star=P)
        _tick("nbv", _t)
        # _debug_viz_candidates(h_truth, voxmap, cur_cfg, r, r_list, camera_model, truth_scene, max_depth, params, rnd=rnd)  # 每轮全部候选+分数
        # _debug_viz_nbv(h_truth, voxmap, cur_cfg, r, camera_model, truth_scene, max_depth, params, rnd=rnd)  # 每轮 NBV 结果
        info["status_seq"].append(r.status)
        info["n_B"].append(int(r.n_B))#r.n_B:本轮阻塞段B的体素个数
        if P is not None and info["P_len"] is None:
            info["P_len"] = int(len(P))

        # 步⑤：按 r.status 决定这一轮怎么走
        if r.status == "scene_infeasible":                   # 真值上 P* 都不存在
            status = "infeasible"
            info["rounds"] = rnd + 1
            break
        elif r.status == "ok":                               # 正常探索一步
            for cand_idx, cand in enumerate(r_list):                              # 按 score 降序逐个试，第一个能走通(seg 非 None)的就用
                _t = time.perf_counter()
                seg, seg_obs = _move_to(h_expl, voxmap, cur_cfg, cand.cfg, camera_model, truth_scene,
                                        max_depth, every_n=every_n, max_stomp_try=max_stomp_try)
                _tick("move", _t)
                #_debug_viz_seg(h_truth, voxmap, cur_cfg, seg, truth_scene, goal_pose, rnd=rnd)
                if seg is not None:
                    # _debug_viz_seg(h_truth, voxmap, cur_cfg, seg, truth_scene, goal_pose, rnd=rnd)  # 收尾段 seg 路径
                    # _dump_seg_isaacsim(cur_cfg, seg, rnd=rnd)    # 同段落盘（isaacsim 回放：工件+障碍+臂，无 voxmap）
                    GT.extend(seg)
                    OBS.extend(seg_obs.tolist())
                    GOAL_FLAG.extend([0] * len(seg))          # NBV 候选点，非 goal
                    cur_cfg = list(cand.cfg)
                    print(f'cand_idx: {cand_idx}')
                    break
                # [2.6993298530578613, -3.7917306423187256, 1.969797968864441, -1.7160824537277222, -1.9512810707092285, -3.3656280040740967]
                else:
                    print(f'cand {cand_idx} is invalid')
        elif r.status == "corridor_confirmed":               # B 空但 ① 没成 → 沿 P* 推进已确认段
            info["rounds"] = rnd + 1
            status = "infeasible"
            info["observe"] = np.asarray(OBS, dtype=np.int64)
            info["goal"] = np.asarray(GOAL_FLAG, dtype=np.int64)
            _emit_prof()
            return np.asarray(GT, dtype=np.float64), status, info
        elif r.status == "no_reachable_candidate":            # B 遮死/够不着 → 就近揭示兜底
            _t = time.perf_counter()
            prog, new_cfg, seg, seg_obs = handle_stuck(h_truth, h_expl, voxmap, cur_cfg,
                                              camera_model, truth_scene, max_depth, every_n=every_n)
            _tick("stuck", _t)
            if prog and seg is not None:
                GT.extend(seg)
                OBS.extend(seg_obs.tolist())
                GOAL_FLAG.extend([0] * len(seg))              # 就近揭示兜底点，非 goal
                cur_cfg = new_cfg

        # 步⑥：终点补拍（_move_to 沿途已拍；对未移动/兜底失败的轮次再确保当前构型有观测）
        _t = time.perf_counter()
        _observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth)
        _tick("observe", _t)
        OBS[-1] = 1

        # 步⑦：进展判定（free 或 reach 增长 = 有进展；连续 stuck_rounds 轮无进展 → 卡死）
        free_now = voxmap.counts()[FREE]
        reach_now = r.reach_idx
        info["free"].append(int(free_now))
        if free_now > free_prev or reach_now > reach_prev:
            stale = 0
        else:
            stale += 1
        free_prev = free_now
        reach_prev = reach_now
        if stale >= stuck_rounds:
            status = "stuck"
            info["rounds"] = rnd + 1
            break
    else:
        info["rounds"] = max_rounds

    info["observe"] = np.asarray(OBS, dtype=np.int64)
    info["goal"] = np.asarray(GOAL_FLAG, dtype=np.int64)
    _emit_prof()
    return np.asarray(GT, dtype=np.float64), status, info
