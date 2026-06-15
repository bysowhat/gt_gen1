"""沿「关键 link 扫掠走廊」自动放置障碍物（见 docs/障碍物位置.md）。

原则（文档）：障碍物的最佳位置不是焊缝点、也不是随机位置，而是机械臂关键 link
（Link2~6 + xiaoyu_accessory_link）沿默认（无障碍）轨迹的扫掠走廊。判定一个布置是否「合适」要满足：
  ① 默认路径会碰撞   ② 仍存在绕行解   ③ 绕行明显不同于默认路径。

两层设计（用户确定）：
- 底层 `place_in_corridor(...)`：参数驱动、确定性——输入 link / 障碍类型 / 尺寸缩放 / 角度 / 位置偏移，
  在该 link 的扫掠走廊里放 1 个障碍（M=1；签名预留将来 M>1）。
- 上层 `search_placement(...)`：用不同尺寸/角度/位置反复调底层；失败按原因自适应（太弱→增大/靠近、
  无解→缩小/挪远、绕行不明显→换位置），重试至 N 次后停止。
- `generate_scenes(...)`：每个关键 link 产 1 个场景（类型未用优先随机）。

M（max_per_scene）、N（max_attempts）等参数在 configs/default.yaml 的 obstacle_placement 段（单一来源）。

碰撞世界：MESH 检查器（工件 mesh + 障碍原语转 mesh），用 MotionGen.update_world 在「工件」「工件+障碍」
之间切换，复用同一 handle（避免每次 attempt 重新 warmup）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from gt_gen import obstacles as ob
from gt_gen import curobo_iface as ci


# ------------------------------------------------------------------ 默认轨迹
def plan_default_traj(handle, retract, goal_pose, metric, max_attempts) -> Optional[np.ndarray]:
    """在「只含工件 mesh」的世界里规划 retract→goal 的默认无障碍轨迹。返回插值 (T,dof)，失败 None。"""
    res = ci.plan_to_pose(handle, retract, goal_pose, max_attempts=max_attempts,
                          pose_cost_metric=metric)
    if res is None or not bool(res.success.item()):
        return None
    return res.get_interpolated_plan().position.detach().cpu().numpy()


# ------------------------------------------------------------------ per-link 扫掠球
def _build_link_model(cfg):
    """独立 CudaRobotModel（注册全部 collision_link_names 做 FK），返回 (model, ta, spheres_def, coll_links)。

    注：另起一份 yml（load_yaml 给新 dict），避免改到 init_curobo 用的 kin。
    """
    import gt_gen.compat  # noqa: F401  warp shim
    gt_gen.compat.apply_trimesh_shim()
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import load_yaml

    rd = load_yaml(cfg.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    coll_links = list(kin["collision_link_names"])
    spheres_def = kin["collision_spheres"]
    kin["link_names"] = coll_links
    ta = TensorDeviceType()
    model = CudaRobotModel(RobotConfig.from_dict(rd["robot_cfg"], ta).kinematics)
    return model, ta, spheres_def, coll_links


def compute_link_sweep(cfg, traj, links) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """沿默认轨迹批量 per-link FK，给出各 link 的扫掠球与 link 原点轨迹。

    返回 (per_wp, origin)：
      per_wp[link]  : (T, S_link, 4) 每个路点该 link 各碰撞球的 [x,y,z,r]（基座系）。
      origin[link]  : (T, 3) 每个路点该 link 坐标系原点位置（用于估路径切向 / 关联时间）。
    只返回 links 中、且在 yml collision_spheres 里有定义的 link。
    """
    import torch

    model, ta, spheres_def, coll_links = _build_link_model(cfg)
    traj = np.asarray(traj, np.float32)
    st = model.get_state(torch.as_tensor(traj, device=ta.device))

    def _R_batch(quat):  # (T,4) wxyz -> (T,3,3)
        q = quat.detach().cpu().numpy()
        return R.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_matrix()      # wxyz->xyzw

    per_wp: Dict[str, np.ndarray] = {}
    origin: Dict[str, np.ndarray] = {}
    for ln in links:
        if ln not in coll_links or ln not in spheres_def:
            continue
        pos = st.link_pose[ln].position.detach().cpu().numpy()        # (T,3)
        Rm = _R_batch(st.link_pose[ln].quaternion)                    # (T,3,3)
        origin[ln] = pos
        local = [(np.asarray(s["center"], float), float(s["radius"]))
                 for s in spheres_def[ln] if float(s["radius"]) > 1e-4]
        if not local:
            continue
        lc = np.array([c for c, _ in local])                          # (S,3)
        lr = np.array([r for _, r in local])                          # (S,)
        # 世界球心 = R @ local_center + pos，逐路点
        wc = np.einsum("tij,sj->tsi", Rm, lc) + pos[:, None, :]       # (T,S,3)
        sph = np.concatenate([wc, np.broadcast_to(lr[None, :, None], wc.shape[:2] + (1,))], axis=2)
        per_wp[ln] = sph                                              # (T,S,4)
    return per_wp, origin


# ------------------------------------------------------------------ 朝向工具
def _rpy_align_x_to(direction, roll_deg: float = 0.0) -> List[float]:
    """求把局部 +X 对到 direction 的整体旋转(+绕该轴 roll)，返回 xyz 欧拉角(度)，供 obstacles 用。

    多数障碍默认「法向/正面 +X，截面在 Y-Z」，故让 +X 对齐路径切向 → 障碍正面横挡走廊。
    """
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return [0.0, 0.0, float(roll_deg)]
    x = d / n
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(up, x); y /= np.linalg.norm(y) + 1e-12
    z = np.cross(x, y)
    R0 = np.column_stack([x, y, z])
    Rtot = R.from_rotvec(x * math.radians(roll_deg)) * R.from_matrix(R0)
    return [float(v) for v in Rtot.as_euler("xyz", degrees=True)]


def _inside_init_free(p, cfg) -> bool:
    """点是否落在初始引导 FREE 圆柱内（轴过 base 原点，文档不建议把障碍放这里）。"""
    r = math.hypot(float(p[0]), float(p[1]))
    return r < cfg.init_free_cyl_radius and 0.0 <= float(p[2]) <= cfg.init_free_cyl_height


# ------------------------------------------------------------------ 障碍尺寸（按走廊管半径）
def _shape_for(otype: str, span: float, tube_r: float) -> Tuple[dict, Tuple[float, float, float]]:
    """据局部走廊跨度 span / 管半径 tube_r 给该障碍类型的形状参数 + anchor 微调偏置(局部系，一般 0)。

    span：障碍应覆盖的横向尺寸（≈走廊直径×size_scale）。返回 (shape_kwargs, local_offset)。
    多数类型 anchor 即结构中心，offset=(0,0,0)；少数（l_bracket/u_channel/steps/gantry 以角/底为锚）
    给一点偏置让结构主体压在 anchor 上。
    """
    s = float(np.clip(span, 0.15, 1.2))
    rt = float(np.clip(tube_r, 0.04, 0.20))
    th = 0.03
    O = (0.0, 0.0, 0.0)
    table = {
        "plate":          (dict(length=s, width=s, thickness=th, tilt_deg=0.0), O),
        "l_bracket":      (dict(length=s, width=s, thickness=th), (-s / 2, 0.0, -s / 2)),
        "u_channel":      (dict(length=s, width=s, height=s * 0.7, thickness=th), (0.0, 0.0, -s * 0.35)),
        "open_box":       (dict(size=(s, s, s), wall=th, open_face="front"), O),
        "pipe":           (dict(length=s * 1.5, radius=rt, axis="y"), O),
        "parallel_pipes": (dict(n=3, length=s * 1.5, radius=rt, gap=s * 0.5, axis="y", stack="z"), O),
        "crossed_pipes":  (dict(length=s * 1.5, radius=rt, cross_deg=90.0), O),
        "box_beam":       (dict(length=s * 1.5, side=max(0.06, rt * 1.4), axis="y"), O),
        "rect_frame":     (dict(width=s, height=s, beam=max(0.06, rt)), O),
        "gantry":         (dict(span=s, height=s, post=max(0.06, rt), beam=max(0.08, rt)), (0.0, 0.0, -s / 2)),
        "braced_frame":   (dict(width=s, height=s, beam=max(0.06, rt), brace=max(0.05, rt * 0.8)), O),
        "tripod":         (dict(height=s, base_half=s * 0.5, rod=max(0.05, rt)), O),
        "steps":          (dict(n=3, rise=s * 0.33, run=s * 0.4, width=s), (-s * 0.6, 0.0, -s / 2)),
        "box_with_pipe":  (dict(size=(s, s, s), wall=th, open_face="front", pipe_radius=rt), O),
        "frame_with_brace": (dict(width=s, height=s, beam=max(0.06, rt), brace=max(0.05, rt * 0.8)), O),
    }
    return table.get(otype, (dict(length=s, width=s, thickness=th), O))


# ------------------------------------------------------------------ 底层放置（M=1）
def place_in_corridor(per_wp, origin, link, otype, *, size_scale, angle_deg, pos_frac,
                      jitter_vec, cfg, goal_pos):
    """在 link 的扫掠走廊里放 1 个障碍（参数驱动、确定性）。返回 (prims, anchor, meta)；被排除区否决则返回 None。

    - pos_frac∈[0,1] 映射到 obstacle_placement.pos_t_window 内的路点 → 取该路点该 link 的扫掠球；
    - anchor = 这些球心均值 + jitter_vec；估走廊管半径 tube_r 与跨度 span(=2·tube_r·size_scale)；
    - 朝向：把障碍正面(+X)对齐该处路径切向，叠加 angle_deg 绕切向自转；
    - 排除：anchor 落入 init_free 圆柱、或距 goal < goal_clearance → 否决（返回 None）。
    """
    op = cfg.obstacle_placement
    if link not in per_wp:
        return None
    sph = per_wp[link]                                   # (T,S,4)
    T = sph.shape[0]
    t_lo, t_hi = op["pos_t_window"]
    t = int(round((t_lo + pos_frac * (t_hi - t_lo)) * (T - 1)))
    t = int(np.clip(t, 1, T - 2))                        # 留前后一格估切向

    centers = sph[t, :, :3]                              # (S,3)
    radii = sph[t, :, 3]
    anchor = centers.mean(axis=0) + np.asarray(jitter_vec, float)

    # 排除区（文档「不建议位置」）
    if _inside_init_free(anchor, cfg):
        return None
    if float(np.linalg.norm(anchor - np.asarray(goal_pos, float))) < op["goal_clearance_m"]:
        return None

    # 走廊管半径：球心相对 anchor 的最大「球面外缘距」
    tube_r = float(np.max(np.linalg.norm(centers - anchor, axis=1) + radii)) if len(centers) else 0.1
    span = 2.0 * tube_r * float(size_scale)

    # 路径切向（link 原点的前后差分）
    op_org = origin[link]
    tangent = op_org[min(t + 1, T - 1)] - op_org[max(t - 1, 0)]
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
    rpy = _rpy_align_x_to(tangent, roll_deg=angle_deg)

    shape, local_off = _shape_for(otype, span, tube_r)
    # local_off 在 anchor 朝向系下偏置 anchor（让以角/底为锚的结构主体压在走廊上）
    anchor_eff = anchor + R.from_euler("xyz", rpy, degrees=True).apply(np.asarray(local_off, float))
    if _inside_init_free(anchor_eff, cfg):
        return None

    prims = ob.build(otype, anchor_eff.tolist(), anchor_rpy_deg=tuple(rpy), **shape)
    meta = dict(link=link, otype=otype, anchor=anchor_eff.tolist(), tangent=tangent.tolist(),
                tube_r=tube_r, span=span, t=t, size_scale=float(size_scale),
                angle_deg=float(angle_deg), pos_frac=float(pos_frac),
                jitter=list(map(float, jitter_vec)), shape=shape)
    return prims, anchor_eff, meta


# ------------------------------------------------------------------ 世界构建 + 切换
def build_world(workpiece_mesh, prims):
    """工件 mesh + 障碍原语 → 全 mesh 的 WorldConfig（MESH 检查器最稳）。"""
    import gt_gen.compat  # noqa: F401
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import WorldConfig

    obs_wc = ob.to_world_config(prims)                   # cuboid=/cylinder=
    obs_mesh = obs_wc.get_mesh_world(process=False).mesh # 原语转 mesh
    return WorldConfig(mesh=[workpiece_mesh] + list(obs_mesh))


# ------------------------------------------------------------------ 三条件验证
def path_collides(handle, traj, max_checks: int = 80) -> Tuple[bool, int]:
    """默认轨迹在当前(含障碍)世界里是否会碰撞。逐路点 check_state，≥1 不可行即碰撞。返回 (collides, n_bad)。"""
    traj = np.asarray(traj, float)
    n = len(traj)
    idx = np.unique(np.linspace(0, n - 1, min(max_checks, n)).astype(int))
    n_bad = 0
    for i in idx:
        feasible, _ = ci.check_state(handle, traj[i].tolist())
        if not feasible:
            n_bad += 1
    return n_bad > 0, n_bad


def detour_exists(handle, retract, goal_pose, metric, max_attempts):
    """当前(含障碍)世界里能否规划出绕行解。返回 (ok, traj2 或 None)。"""
    res = ci.plan_to_pose(handle, retract, goal_pose, max_attempts=max_attempts,
                          pose_cost_metric=metric)
    if res is None or not bool(res.success.item()):
        return False, None
    return True, res.get_interpolated_plan().position.detach().cpu().numpy()


def _resample(a, n):
    a = np.asarray(a, float)
    ts = np.linspace(0.0, 1.0, len(a))
    tn = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(tn, ts, a[:, j]) for j in range(a.shape[1])], axis=1)


def is_detour_different(traj_default, traj2, thresh) -> Tuple[bool, float]:
    """绕行 vs 默认是否「明显不同」：等长重采样后，逐路点关节最大偏差的峰值 ≥ thresh。返回 (ok, dist)。"""
    n = max(len(traj_default), len(traj2))
    a = _resample(traj_default, n)
    b = _resample(traj2, n)
    dist = float(np.max(np.abs(a - b)))
    return dist >= float(thresh), dist


def validate_scene(handle, traj_default, retract, goal_pose, metric, cfg) -> dict:
    """综合三条件。handle 世界须已切到「工件+障碍」。返回 dict（含 fail_reason / detour_traj）。"""
    op = cfg.obstacle_placement
    collides, n_bad = path_collides(handle, traj_default)
    if not collides:
        return dict(ok=False, fail_reason="too_weak", n_bad=0, dist=0.0, detour_traj=None)
    ok2, traj2 = detour_exists(handle, retract, goal_pose, metric, cfg.plan_max_attempts)
    if not ok2:
        return dict(ok=False, fail_reason="no_solution", n_bad=n_bad, dist=0.0, detour_traj=None)
    okd, dist = is_detour_different(traj_default, traj2, op["detour_min_joint_rad"])
    if not okd:
        return dict(ok=False, fail_reason="not_different", n_bad=n_bad, dist=dist, detour_traj=traj2)
    return dict(ok=True, fail_reason=None, n_bad=n_bad, dist=dist, detour_traj=traj2)


# ------------------------------------------------------------------ 上层重试
def search_placement(handle, workpiece_mesh, per_wp, origin, link, otype,
                     traj_default, retract, goal_pose, metric, cfg, rng,
                     n_attempts: Optional[int] = None) -> dict:
    """对 (link, otype) 反复试放，失败按原因自适应改尺寸/角度/位置，至多 N 次。返回 result dict。"""
    op = cfg.obstacle_placement
    N = int(op["max_attempts"] if n_attempts is None else n_attempts)
    s_lo, s_hi = op["size_scale_range"]
    ang = float(op["angle_jitter_deg"])
    pj = float(op["pos_jitter_m"])

    # 可变状态（随失败原因调整）
    size_scale = rng.uniform(s_lo, s_hi)
    last = None
    for attempt in range(N):
        angle_deg = rng.uniform(-ang, ang)
        pos_frac = rng.uniform(0.0, 1.0)
        jitter_vec = np.array([rng.uniform(-pj, pj) for _ in range(3)])

        placed = place_in_corridor(per_wp, origin, link, otype, size_scale=size_scale,
                                   angle_deg=angle_deg, pos_frac=pos_frac,
                                   jitter_vec=jitter_vec, cfg=cfg, goal_pos=goal_pose[0])
        if placed is None:                                # 落在排除区 → 换位置重采
            last = "excluded"
            continue
        prims, anchor, meta = placed
        world = build_world(workpiece_mesh, prims)
        handle.mg.update_world(world)
        v = validate_scene(handle, traj_default, retract, goal_pose, metric, cfg)
        last = v["fail_reason"]
        if v["ok"]:
            return dict(ok=True, link=link, otype=otype, attempt=attempt + 1, prims=prims,
                        anchor=list(map(float, anchor)), meta=meta,
                        detour_traj=v["detour_traj"], n_bad=v["n_bad"], dist=v["dist"])
        # 自适应：太弱→增大；无解→缩小；绕行不明显→略增大（换位置已每轮随机）
        if v["fail_reason"] == "too_weak":
            size_scale = min(s_hi * 1.5, size_scale * 1.25)
        elif v["fail_reason"] == "no_solution":
            size_scale = max(s_lo * 0.5, size_scale * 0.8)
        else:                                             # not_different
            size_scale = min(s_hi * 1.5, size_scale * 1.1)

    return dict(ok=False, link=link, otype=otype, attempt=N, last_reason=last,
                prims=None, anchor=None, meta=None, detour_traj=None)


def generate_scenes(handle, workpiece_mesh, per_wp, origin, traj_default,
                    retract, goal_pose, metric, cfg, rng) -> List[dict]:
    """每个关键 link 产 1 个场景：障碍类型从全部类型「未用优先随机」抽，调 search_placement。"""
    op = cfg.obstacle_placement
    links = list(op["key_links"])
    all_types = list(op["obstacle_types"]) or ob.list_obstacles()

    unused = list(all_types)
    scenes = []
    for link in links:
        if not unused:                                    # 类型用尽则从全集重抽
            unused = list(all_types)
        otype = unused.pop(rng.randrange(len(unused)))
        res = search_placement(handle, workpiece_mesh, per_wp, origin, link, otype,
                               traj_default, retract, goal_pose, metric, cfg, rng)
        scenes.append(res)
    return scenes
