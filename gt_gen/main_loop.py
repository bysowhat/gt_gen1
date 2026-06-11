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

import numpy as np


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
             every_n=None, pixel_stride: int = 16):
    """在 h_expl(VOXEL) 上规划 cur_cfg→target_cfg，取插值轨迹；沿途每 every_n 个路点 _observe
    一次（边走边拍），终点构型再补拍一次。

    every_n=None 时取 h_expl.config.params.loop.observe_every_n（默认 10）。
    返回该段轨迹去掉首点后的 (t,dof) np（接进 GT）；规划失败返回 None。
    """
    from gt_gen import curobo_iface as ci

    if every_n is None:
        every_n = int(h_expl.config.params.get("loop", {}).get("observe_every_n", 10))
    every_n = max(1, int(every_n))

    # _debug_viz_voxmap(voxmap, h_expl, target_cfg, truth_scene, show_unknown=True)
    res = ci.plan_to_config(h_expl, cur_cfg, target_cfg, max_attempts=h_expl.config.plan_max_attempts)
    if res is None or not bool(res.success.item()):
        # 诊断：起点/终点哪个在碰撞，还是中间连不上（区分三种成因）
        print("[_move_to] plan_to_config 失败:", ci.explain_endpoints(h_expl, cur_cfg, target_cfg))
        # cuRobo 实际避障的占据场（sync 后、含 inflate=1），看 target_cfg 整臂是否泡在障碍里
        return None
    traj = res.get_interpolated_plan().position.detach().cpu().numpy()

    # 沿途每 every_n 个路点拍一次（h_expl 做相机 FK，与 h_truth 同一套运动学）
    for i in range(0, len(traj), every_n):
        _observe(voxmap, h_expl, traj[i], camera_model, truth_scene, max_depth, pixel_stride)
    _observe(voxmap, h_expl, traj[-1], camera_model, truth_scene, max_depth, pixel_stride)

    _debug_viz_voxmap(voxmap, h_expl, traj[-1], truth_scene, show_unknown=True, transparent=False)  # 本段走完后的 voxmap 三态（含灰 UNKNOWN）
    return traj[1:]                                   # 去掉与上一段重复的首点


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
    返回 (progressed: bool, new_cfg, seg)；无可达候选/规划失败 → (False, cur_cfg, None)。
    """
    from gt_gen.candidates import generate_candidates
    from gt_gen.nbv import raycast_reveal
    from gt_gen.voxmap import UNKNOWN

    frontier = _frontier_cells(voxmap)
    if frontier.shape[0] == 0:
        return False, cur_cfg, None

    cands = generate_candidates(h_truth, voxmap, frontier, camera_model, cur_cfg)
    if not cands:
        return False, cur_cfg, None

    best, best_gain = None, 0
    for c in cands:
        reveal = raycast_reveal(voxmap, c.cam_pose, camera_model, truth_scene)
        if reveal.shape[0] == 0:
            continue
        gain = int((np.asarray(voxmap.get(reveal)) == UNKNOWN).sum())   # 能揭开多少未知
        if gain > best_gain:
            best, best_gain = c, gain
    if best is None or best_gain <= 0:
        return False, cur_cfg, None

    seg = _move_to(h_expl, voxmap, cur_cfg, best.config, camera_model, truth_scene,
                   max_depth, every_n=every_n)
    if seg is None:
        return False, cur_cfg, None
    return True, list(best.config), seg


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


def _debug_viz_voxmap(voxmap, fk_handle, q, truth_scene, every_n_layers: int = 8,
                      show_unknown: bool = False, transparent: bool = True):
    """调试用②【voxmap 世界】：可视化三态体素图本身（sync 的输入，不随 sync 改变）。默认不调用，需手动取消注释。

    画 FREE(蓝，整块) + OCCUPIED(红，整块) + 可选 UNKNOWN(灰，抽稀)；叠加整臂(绿) + 工件(灰)。
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

    geoms = [("work", _work_mesh(truth_scene), "lit", None),
             ("arm", _arm_mesh(fk_handle, list(q)), "lit", None)]
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
    _draw(geoms, f"main_loop voxmap世界({fill_tag}): FREE={n_free}格(蓝) OCC={n_occ}格(红){unk_tag} 绿=整臂 灰=工件")


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


def _debug_viz_candidates(h_truth, voxmap, cur_cfg, r, camera_model, truth_scene, max_depth,
                          params, rnd=None):
    """调试用④【逐候选 + 各自分数】：这一轮 NBV 的【每一个候选】各弹一窗、画全（参考 verify_step9
    的 score 窗口，一候选一窗）。默认不调用，需手动取消注释。

    复用 r.P_star/r.reach_idx（避免重规划），重算 B + 候选 + 逐候选打分（NBVResult 只带最优解，
    候选全集需重算 generate_candidates + score_candidate，与 best_next_view_using_oracle 内部同逻辑）。
    控制台先打印逐候选 gain/path_cost/score 表（★标 argmax）；随后【每个候选一窗】，每窗画该候选的
    整臂(青，真摆成看 B 的姿态) + 相机帧 + FOV视锥(远面过 T) + 视线 p→T(红) + 目标 T(品红球)，
    底图含工件灰 + 当前整臂@cur_cfg绿 + FREE蓝半透明 + OCCUPIED红 + B橙；标题写该候选 i/N、
    gain/path_cost/score、是否 ★argmax。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import open3d as o3d
    from verify_step8 import (_arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base,
                              _ball, _lines, _fov_frustum)
    from gt_gen.voxmap import FREE, OCCUPIED
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.candidates import generate_candidates
    from gt_gen import nbv as _nbv

    tag = f"R{rnd} " if rnd is not None else ""
    nbv_p = (params or h_truth.config.params).get("nbv", {})
    k = int(nbv_p.get("k_lookahead", 6))
    lam = float(nbv_p.get("lambda_cost", 0.0))

    # —— 重算 B + 候选 + 打分 ——
    P = r.P_star
    B = (compute_blocking_B(h_truth, voxmap, P, r.reach_idx, k)
         if P is not None and r.status in ("ok", "no_reachable_candidate")
         else np.empty((0, 3), dtype=np.int64))
    cands = generate_candidates(h_truth, voxmap, B, camera_model, cur_cfg) if B.shape[0] else []
    rows = []                                                          # (gain, path_cost, score)
    for c in cands:
        g, s, _ = _nbv.score_candidate(voxmap, c, B, truth_scene, camera_model, cur_cfg, lambda_cost=lam)
        pc = float(np.linalg.norm(np.asarray(c.config, float) - np.asarray(cur_cfg, float)))
        rows.append((g, pc, s))

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
        am = _arm_mesh(h_truth, list(c.config)); am.paint_uniform_color([0.10, 0.75, 0.80])
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
                camera_model=None, params=None):
    """完整 ①~⑦ 主循环（goal 已含 standoff 后退）。

    入参：
      h_truth     : MESH 真值 handle（含工件 mesh）——P*/NBV/候选避障。
      h_expl      : VOXEL 探索 handle（纯三态，无 mesh）——①/⑤ 的实际无碰撞规划。
      voxmap      : 三态体素图，调用方已建好并罩好初始 FREE（圆柱法）。
      truth_scene : base 系 trimesh，raycast 几何源。
      goal_pose   : 末端 standoff 目标位姿 ((x,y,z),(qw,qx,qy,qz))。
      camera_model: None → load_camera_model(h_truth.config)。
      params      : None → h_truth.config.params。

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
    loop_p = params.get("loop", {})
    max_rounds = int(loop_p.get("max_rounds", 200))
    stuck_rounds = int(loop_p.get("stuck_rounds", 5))
    every_n = int(loop_p.get("observe_every_n", 10))
    max_depth = cfg.max_depth_m

    cur_cfg = list(cfg.retract_config)
    GT = [np.asarray(cur_cfg, dtype=np.float64)]
    metric = ci.free_pose_metric(h_truth, free_rot=(0,))     # 放开焊枪绕接近轴 roll

    # 准备阶段：冷启动补拍一次（首次更新占用）
    # _debug_viz_observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth, "before")  # 拍前
    _observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth)
    # _debug_viz_observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth, "after")   # 拍后

    info = {"rounds": 0, "n_B": [], "free": [], "status_seq": [], "P_len": None}
    free_prev = voxmap.counts()[FREE]
    reach_prev = -1
    stale = 0
    status = "max_rounds"

    for rnd in range(max_rounds):
        torch.cuda.empty_cache()
        # _debug_viz_voxmap(voxmap, h_truth, cur_cfg, truth_scene, every_n_layers=4)                     # voxmap 三态（sync 输入，不随 sync 变）
        # _debug_viz_curobo(h_expl, voxmap, h_truth, cur_cfg, truth_scene, "before", every_n_layers=10)   # sync 前：cuRobo 占据应空
        sync_collision_world(h_expl, voxmap)                 # 步0：最新「非 FREE」→ h_expl 障碍场
        # _debug_viz_curobo(h_expl, voxmap, h_truth, cur_cfg, truth_scene, "after", every_n_layers=10)    # sync 后：仅圆柱留洞

        # 步①：试在已确认自由区直接规划到 goal（h_expl，UNKNOWN 已当障碍）
        res = ci.plan_to_pose(h_expl, cur_cfg, goal_pose,
                              max_attempts=cfg.plan_max_attempts, pose_cost_metric=metric)
        if res is not None and bool(res.success.item()):
            seg = res.get_interpolated_plan().position.detach().cpu().numpy()
            GT.extend(seg[1:])
            status = "reached"
            info["rounds"] = rnd + 1
            break

        # 步②：真值上的全知最优路 P*（挡住的只可能是 UNKNOWN）
        P = ci.plan_on_truth(h_truth, cur_cfg, goal_pose,
                             max_attempts=cfg.plan_max_attempts, pose_cost_metric=metric)
        # 步③④：一轮特权 NBV（P* → reach_pt/B → 候选 → 假设性 raycast 打分 → argmax）
        r = best_next_view_using_oracle(h_truth, cur_cfg, voxmap, truth_scene, goal_pose,
                                        params=params, camera_model=camera_model,
                                        pose_cost_metric=metric, p_star=P)
        # _debug_viz_candidates(h_truth, voxmap, cur_cfg, r, camera_model, truth_scene, max_depth, params, rnd=rnd)  # 每轮全部候选+分数
        _debug_viz_nbv(h_truth, voxmap, cur_cfg, r, camera_model, truth_scene, max_depth, params, rnd=rnd)  # 每轮 NBV 结果
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
            seg = _move_to(h_expl, voxmap, cur_cfg, r.cfg, camera_model, truth_scene,
                           max_depth, every_n=every_n)
            if seg is not None:
                GT.extend(seg)
                cur_cfg = list(r.cfg)
        elif r.status == "corridor_confirmed":               # B 空但 ① 没成 → 沿 P* 推进已确认段
            reach_idx = r.reach_idx
            if reach_idx >= len(P) - 1:                       # 整条 P* 已落在 FREE → 直接收尾
                seg = _move_to(h_expl, voxmap, cur_cfg, list(P[-1]), camera_model, truth_scene,
                               max_depth, every_n=every_n)
                if seg is not None:
                    GT.extend(seg)
                    cur_cfg = list(P[-1])
                    status = "reached"
                    info["rounds"] = rnd + 1
                    break
            else:                                             # 沿 P* 往前挪一段，记一次进展
                seg = _move_to(h_expl, voxmap, cur_cfg, list(P[reach_idx]), camera_model,
                               truth_scene, max_depth, every_n=every_n)
                if seg is not None:
                    GT.extend(seg)
                    cur_cfg = list(P[reach_idx])
        elif r.status == "no_reachable_candidate":            # B 遮死/够不着 → 就近揭示兜底
            prog, new_cfg, seg = handle_stuck(h_truth, h_expl, voxmap, cur_cfg,
                                              camera_model, truth_scene, max_depth, every_n=every_n)
            if prog and seg is not None:
                GT.extend(seg)
                cur_cfg = new_cfg

        # 步⑥：终点补拍（_move_to 沿途已拍；对未移动/兜底失败的轮次再确保当前构型有观测）
        _observe(voxmap, h_truth, cur_cfg, camera_model, truth_scene, max_depth)

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

    return np.asarray(GT, dtype=np.float64), status, info
