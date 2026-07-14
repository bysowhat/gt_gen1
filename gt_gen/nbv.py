"""Step 9: 特权 NBV 打分与选择（oracle-path 版）。

见 privileged-nbv.md §4.5 ④（打分选择）、§8 顶层伪代码。

一轮 NBV：Step 7 给出阻塞段 B（卡住下一步、只因没看过的 UNKNOWN 体素），Step 8 给出一批
"看得见 B、又走得到"的候选构型；本步对每个候选【向真值场景做假设性 raycast】，算它能揭开
B 多少（gain），减去路径代价得 score，argmax 选出下一视点。

关键：这里的 raycast 是【假设性】的——"如果把相机移过去会看到什么"，仅用于比较候选、不改地图
（真正改图是主循环 ⑥）。真值 mesh 的遮挡天然处理：被挡在障碍后面的 B 不会被算作揭开。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NBVResult:
    """一轮 NBV 的结果。status：
      ok                     选出下一视点（cfg/cam_pose/gain/score 有效）
      scene_infeasible       真值上 P* 不存在 → 场景不可行
      corridor_confirmed     B 为空 → 走廊已确认，应能直接规划到目标
      no_reachable_candidate 有 B 但无可达候选 → 主循环转"就近揭示"兜底（§6）
    """
    status: str
    cfg: Optional[list] = None
    cam_pose: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    gain: float = 0.0
    score: float = 0.0
    reach_idx: int = -1
    n_B: int = 0
    n_candidates: int = 0
    P_star: Optional[np.ndarray] = None
    trans_cost: float = 0.0
    angle_cost: float = 0.0


# ---------------- 假设性 raycast 与打分 ----------------

def raycast_reveal(voxmap, camera_pose, camera_model, truth_scene,
                   max_depth: Optional[float] = None, pixel_stride: Optional[int] = None) -> np.ndarray:
    """假设性观测（warp 视锥体素雕刻）：从 camera_pose 向真值场景投射，返回该视点【将确定】的体素下标 (M,3)（不改地图）。

    = unique(FREE 体素 ∪ OCCUPIED 体素)，均已在 voxmap 界内。
    真值 mesh 的遮挡天然处理：被障碍挡住的体素归 UNKNOWN，不算"将确定"。
    pixel_stride 已废弃（旧 trimesh 形参），保留仅为兼容，忽略。

    多分辨率：reveal 与 B（reach_b.compute_blocking_B）须在【同一】索引空间做 gain=|reveal∩B| 集合
    运算，故都在 coarse 子网格上算（index_grid）——仅影响视点选择启发式，不碰 GT 闸门（见 index_grid）。
    """
    from gt_gen.sensor import carve_observe
    from gt_gen.voxmap import index_grid

    free_idx, occ_idx = carve_observe(index_grid(voxmap), camera_pose, camera_model,
                                      truth_scene, max_depth)
    chunks = [a for a in (free_idx, occ_idx) if a.shape[0]]
    if not chunks:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(chunks, axis=0), axis=0)


def _count_intersection(reveal, B) -> int:
    """|reveal ∩ B|：两组体素下标 (·,3) 的交集大小（B 通常较小，做成 set 查）。"""
    B = np.asarray(B)
    if reveal.shape[0] == 0 or B.shape[0] == 0:
        return 0
    Bset = set(map(tuple, B))
    return int(sum(1 for r in map(tuple, reveal) if r in Bset))


def _gun_translation_cost(handle, cur_cfg, cand_cfg) -> float:
    """焊枪(xiaoyu_accessory_link)平移代价：对该 link 的每个碰撞球，算 cur→cand 的球心平移量，
    按碰撞球半径加权求均值(m)。权重 w_i = r_i / r_max：最大的球权重为 1，最小的球权重为其半径
    与最大半径之比。半径大的球位移影响更大。

    取不到该 link 的球掩码时退回关节空间位移（保守不失效）。
    """
    from gt_gen.swept import fk_spheres_batch, link_sphere_mask

    mask = link_sphere_mask(handle, "xiaoyu_accessory_link")
    if mask is None or not bool(np.any(mask)):
        return float(np.linalg.norm(np.asarray(cand_cfg, float) - np.asarray(cur_cfg, float)))
    sph = fk_spheres_batch(handle, [list(cur_cfg), list(cand_cfg)])   # (2,S,4) xyz+r
    disp = np.linalg.norm(sph[1][mask, :3] - sph[0][mask, :3], axis=1)  # 每球平移量(m)
    radii = sph[0][mask, 3]                                             # 每球半径(m)
    r_max = float(radii.max())
    if r_max <= 0.0:
        return float(disp.mean())
    w = radii / r_max                                                  # 最大球=1，最小球=r_min/r_max
    return float((w * disp).sum() / w.sum())


def _gun_rotation_cost(handle, cur_cfg, cand_cfg) -> float:
    """焊枪(xiaoyu_accessory_link)角度代价：该 link 从 cur→cand 的空间朝向变化(rad)。

    取两构型下该 link 的四元数 q0,q1，返回两姿态间的测地角 θ = 2·arccos(|<q0,q1>|)，
    涵盖焊枪的完整朝向变化（含绕枪轴 roll）。

    取不到该 link 位姿时（未注册进 link_names）返回 0.0（保守不影响打分）。
    需 init_curobo 把 'xiaoyu_accessory_link' 加进 link_names（已加）。
    """
    import torch

    qt = torch.as_tensor(np.asarray([list(cur_cfg), list(cand_cfg)], dtype=np.float32),
                         device=handle.ta.device)
    st = handle.mg.kinematics.get_state(qt)
    lp = st.link_pose.get("xiaoyu_accessory_link") if hasattr(st.link_pose, "get") else None
    if lp is None:
        return 0.0
    q = lp.quaternion.detach().cpu().numpy()          # (2,4) wxyz
    dot = float(np.clip(abs(np.dot(q[0], q[1])), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))                # 两姿态测地角(rad)


def score_candidate(voxmap, cand, B, truth_scene, camera_model, cur_cfg, handle,
                    lambda_cost: float = 0.0, lambda_angle: float = 0.0,
                    max_depth: Optional[float] = None):
    """对一个候选打分：gain = |揭开 ∩ B|；score = gain − λ·trans_cost − λ_a·angle_cost。

    trans_cost = 焊枪(xiaoyu_accessory_link)碰撞球从 cur_cfg 到 cand.config 的平移量、按球半径加权的均值(m)。
    angle_cost = 焊枪 link 从 cur_cfg 到 cand.config 的空间朝向变化(rad，测地角)。
    返回 (gain, score, reveal)；reveal 为该候选假设性 raycast 将确定的体素下标 (M,3)。
    """
    reveal = raycast_reveal(voxmap, cand.cam_pose, camera_model, truth_scene, max_depth=max_depth)
    gain = _count_intersection(reveal, B)
    trans_cost = _gun_translation_cost(handle, cur_cfg, cand.config)
    angle_cost = _gun_rotation_cost(handle, cur_cfg, cand.config)
    score = float(gain) - float(lambda_cost) * trans_cost - float(lambda_angle) * angle_cost
    return gain, score, reveal, float(lambda_cost) * trans_cost, float(lambda_angle) * angle_cost



# ---------------- 调试可视化（open3d；默认关，置环境变量 NBV_VIZ=1 开启） ----------------

def _viz_helpers():
    """惰性载入 verify_step8 的 open3d 工具（需显示器 + open3d）。返回 (helpers模块字典)。"""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base, _lines
    return dict(arm=_arm_mesh, work=_work_mesh, cells=_cells_mesh, draw=_draw,
                roi=_roi_and_base, lines=_lines)


def _debug_viz_voxmap(handle, voxmap, P, reach_idx, truth_scene):
    """【line 116 后】可视化当前 voxmap 三态 + reach_pt 处整臂：
    FREE(蓝半透明) + OCCUPIED(红实心) + 工件(灰) + reach_idx 构型整臂碰撞球(绿) + ROI/base。
    参考 verify_step7.viz_voxmap_arm / main_loop._debug_viz_voxmap。被调用即弹窗(需显示器+open3d)；
    无显示器/服务器跑时把调用行注释掉即可（调用行本身就是开关）。
    """
    from gt_gen.voxmap import FREE, OCCUPIED
    h = _viz_helpers()
    q = list(P[int(np.clip(reach_idx, 0, len(P) - 1))])
    geoms = [("work", h["work"](truth_scene), "lit", None),
             ("arm", h["arm"](handle, q), "lit", None)]
    fc = voxmap.state_centers(FREE)
    if fc.shape[0]:
        geoms.append(("free", h["cells"](voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.12]))
    oc = voxmap.state_centers(OCCUPIED)
    if oc.shape[0]:
        om = h["cells"](voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))
    geoms += h["roi"](voxmap)
    h["draw"](geoms, f"nbv voxmap: FREE={int(fc.shape[0])}格(蓝) OCC={int(oc.shape[0])}格(红) "
                     f"绿=reach@{reach_idx} 灰=工件")


def _debug_viz_B(handle, voxmap, P, reach_idx, B, truth_scene):
    """【line 117 后】可视化阻塞段 B（橙不透明格）叠加在 voxmap 底图上：
    FREE(蓝半透明) + OCCUPIED(红) + 工件(灰) + reach_pt 整臂(绿) + B(橙) + ROI/base。
    参考 verify_step7.verify_compute_blocking_B 的 b_cells 橙色显示。被调用即弹窗(需显示器+open3d)；
    无显示器/服务器跑时把调用行注释掉即可（调用行本身就是开关）。
    """
    from gt_gen.voxmap import FREE, OCCUPIED
    h = _viz_helpers()
    q = list(P[int(np.clip(reach_idx, 0, len(P) - 1))])
    geoms = [("work", h["work"](truth_scene), "lit", None),
             ("arm", h["arm"](handle, q), "lit", None)]
    fc = voxmap.state_centers(FREE)
    if fc.shape[0]:
        geoms.append(("free", h["cells"](voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.12]))
    oc = voxmap.state_centers(OCCUPIED)
    if oc.shape[0]:
        om = h["cells"](voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))
    if B is not None and B.shape[0]:
        bm = h["cells"](voxmap, voxmap.voxel_to_world(B)); bm.paint_uniform_color([1.0, 0.55, 0.0])
        geoms.append(("B", bm, "lit", None))                   # B = 橙不透明（待观测的阻塞未知区）
    geoms += h["roi"](voxmap)
    h["draw"](geoms, f"nbv 阻塞段B(橙{int(B.shape[0]) if B is not None else 0}格) "
                     f"reach_idx={reach_idx} 绿=reach整臂 灰=工件")


def _debug_viz_all_candidates(handle, voxmap, cands, B, truth_scene, camera_model, cur_cfg):
    """【generate_candidates 之后】一窗看【全部候选】视点：叠加 voxmap 底图 + 阻塞集 B。
    工件(灰) + cur_cfg 整臂(绿) + FREE(蓝半透明) + OCCUPIED(红) + B(橙) + ROI/base；
    每个候选画：相机坐标系 + FOV 视锥(紫，远面过 T) + 眼→T 光轴射线(青) + 目标点 T(品红小球)。
    整臂只画起点 cur_cfg 一条（候选多、逐条画臂会糊成一团）。被调用即弹窗(需显示器+open3d)；
    无显示器/服务器跑时把调用行注释掉即可（调用行本身就是开关）。
    """
    import open3d as o3d
    from gt_gen.voxmap import FREE, OCCUPIED
    h = _viz_helpers()
    from verify_step8 import _fov_frustum, _ball                       # _viz_helpers 已把 scripts 加进 sys.path

    geoms = [("work", h["work"](truth_scene), "lit", None),
             ("cur_arm", h["arm"](handle, list(cur_cfg)), "lit", None)]
    fc = voxmap.state_centers(FREE)
    if fc.shape[0]:
        geoms.append(("free", h["cells"](voxmap, fc), "fill", [0.20, 0.45, 0.95, 0.10]))
    oc = voxmap.state_centers(OCCUPIED)
    if oc.shape[0]:
        om = h["cells"](voxmap, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))
    if B is not None and B.shape[0]:
        bm = h["cells"](voxmap, voxmap.voxel_to_world(B)); bm.paint_uniform_color([1.0, 0.55, 0.0])
        geoms.append(("B", bm, "lit", None))

    for i, c in enumerate(cands):
        eye = c.cam_pose[:3, 3]
        depth = float(np.linalg.norm(c.target - eye))
        edges, cone = _fov_frustum(c.cam_pose, camera_model, depth)
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10); fr.transform(c.cam_pose)
        geoms.append((f"cam{i}", fr, "lit", None))
        geoms.append((f"fov{i}", cone, "fill", [0.6, 0.2, 0.85, 0.10]))
        geoms.append((f"fov_e{i}", edges, "line", None))
        geoms.append((f"ray{i}", h["lines"]([(eye, c.target)], [0.10, 0.75, 0.85]), "line", None))
        geoms.append((f"T{i}", _ball(c.target, 0.03, [0.9, 0.1, 0.9]), "lit", None))

    geoms += h["roi"](voxmap)
    h["draw"](geoms, f"nbv 全部候选({len(cands)}个)视点: 相机系+FOV(紫)+光轴(青)+T(品红) | "
                     f"绿=cur整臂 橙=B 蓝=FREE 红=OCC")


# ---------------- 顶层：oracle-path NBV（§8 伪代码） ----------------

def best_next_view_using_oracle(handle, cur_cfg, voxmap, truth_scene, goal_pose,
                                params=None, camera_model=None, pose_cost_metric=None,
                                p_star=None) -> NBVResult:
    """一轮特权 NBV（oracle-path 版）：P* → reach_pt/B → 候选 → 假设性 raycast 打分 → argmax。

    handle 须用【真值 world】(含工件 mesh)初始化——P* 与假设性 raycast 都基于真值。
    goal_pose：末端位姿 (pos,quat_wxyz)。params/camera_model 为 None 时从 handle.config 取。
    p_star：可选，外部已算好的真值最优路 (T,dof)；传入则跳过本函数内部规划（主循环每轮算一次
        P* 注入即可，既省一次规划又避免 cuRobo 多种子的随机抖动）。None 时本函数自行规划。
    返回 (best, ranked)：best 为最优 NBVResult（见其 status）；ranked 为所有候选各自的
        NBVResult 列表、按 score 降序排列（status=="ok" 时非空，其余状态为 []，best==ranked[0]）。
    """
    from gt_gen import curobo_iface as ci
    from gt_gen.reach_b import compute_reach_pt, compute_blocking_B
    from gt_gen.candidates import generate_candidates
    from gt_gen.sensor import load_camera_model

    cfg = handle.config
    nbv = (params if params is not None else cfg.params).get("nbv", {})
    k = int(nbv.get("k_lookahead", 6))
    lam = float(nbv.get("lambda_cost", 0.0))
    lam_ang = float(nbv.get("lambda_angle", 0.0))
    if camera_model is None:
        camera_model = load_camera_model(cfg)
    if pose_cost_metric is None:
        pose_cost_metric = ci.free_pose_metric(handle, free_rot=(0,))

    # 1) 真值上的全知最优路径 P*（外部已算好则直接用）
    P = p_star
    if P is None:
        raise NotImplementedError()
        P = ci.plan_on_truth(handle, cur_cfg, goal_pose, max_attempts=cfg.plan_max_attempts,
                             pose_cost_metric=pose_cost_metric)
    if P is None:
        return NBVResult("scene_infeasible"), []

    # 2) 沿 P* 求 reach_pt，取前方 k 段的 UNKNOWN = 阻塞段 B
    reach_idx = compute_reach_pt(handle, voxmap, P)
    # _debug_viz_voxmap(handle, voxmap, P, reach_idx, truth_scene)        # 看 voxmap 三态 + reach 整臂（注释此行可关）
    B = compute_blocking_B(handle, voxmap, P, reach_idx, k)
    # _debug_viz_B(handle, voxmap, P, reach_idx, B, truth_scene)          # 看阻塞段 B（橙）（注释此行可关）
    if B.shape[0] == 0:
        return NBVResult("corridor_confirmed", reach_idx=reach_idx, P_star=P), []

    # 3) 朝 B 生成"自由区内可达"的候选
    cands = generate_candidates(handle, voxmap, B, camera_model, cur_cfg)
    # _debug_viz_all_candidates(handle, voxmap, cands, B, truth_scene, camera_model, cur_cfg)  # 看全部候选视点（去注释开窗）
    if not cands:
        return NBVResult("no_reachable_candidate", reach_idx=reach_idx, n_B=int(B.shape[0]), P_star=P), []

    # 4) 假设性 raycast 打分，按 score 降序排列所有候选
    scored = []                                                    # (score, gain, cand)
    for c in cands:
        gain, score, _, trans_cost, angle_cost = score_candidate(voxmap, c, B, truth_scene, camera_model, cur_cfg, handle,
                                          lambda_cost=lam, lambda_angle=lam_ang)
        scored.append((float(score), float(gain), c, trans_cost, angle_cost))
    scored.sort(key=lambda x: x[0], reverse=True)

    ranked = [NBVResult("ok", cfg=list(c.config), cam_pose=c.cam_pose, target=c.target,
                        gain=gain, trans_cost=trans_cost, angle_cost=angle_cost, score=score, reach_idx=reach_idx,
                        n_B=int(B.shape[0]), n_candidates=len(cands), P_star=P)
              for score, gain, c, trans_cost, angle_cost in scored]
    return ranked[0], ranked
