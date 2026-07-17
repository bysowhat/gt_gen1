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
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from gt_gen import obstacles as ob
from gt_gen import curobo_iface as ci


# ------------------------------------------------------------------ 默认轨迹
def plan_default_traj(handle, retract, goal_pose, metric, max_attempts, *,
                      cfg=None, world=None, checker_type=None) -> Optional[np.ndarray]:
    """在「只含工件 mesh」的世界里规划 retract→goal 的默认无障碍轨迹。返回插值 (T,dof)，失败 None。

    后端由 cfg.planner_backend 决定（curobo|stomp，见 gt_gen/stomp_iface.py）：
    - curobo：MotionGen(graph+trajopt)，用 handle 内部世界，metric 放开 roll；
    - stomp ：在传入的 world（仅工件、不膨胀）上跑 STOMP，固定朝向（metric 忽略），buffer≈0.001。
    """
    backend = cfg.planner_backend if cfg is not None else "curobo"
    if backend == "stomp":
        from gt_gen import stomp_iface as si
        return si.plan_pose_single(cfg, world, retract, goal_pose, checker_type=checker_type)
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


def _prim_bounding_sphere(p) -> Tuple[np.ndarray, float]:
    """障碍 prim 的包围球 (球心 xyz, 半径)。球心取 prim.pose 平移；半径取保守外接半径。

    Box  : 半对角线 = ½·‖dims‖；Tube: 沿局部 Z 的圆柱外接球 = hypot(radius, ½·height)。
    包围球是保守外估 → 用它判 init_free 清空只会偏严（多推一点），不会漏判。
    """
    c = np.asarray(p.pose[:3], float)
    if isinstance(p, ob.Box):
        rb = 0.5 * float(np.linalg.norm(np.asarray(p.dims, float)))
    else:                                                # Tube
        rb = float(math.hypot(float(p.radius), 0.5 * float(p.height)))
    return c, rb


def _point_solid_dist2(prim, pts) -> np.ndarray:
    """一批世界点 pts(N,3) 到 prim【实体】的最短距离²（点在实体内部为 0）。

    精确判据(非包围球)：把世界点变换到 prim 局部系 loc = Rᵀ(p−c)，再按实体类型算「各方向外溢量」：
      · Box (OBB，dims=全边长)：半边长 he=dims/2，over_i=max(0,|loc_i|−he_i)，dist²=Σ over_i²；
      · Tube (轴沿局部 +Z，半径 r、半高 h/2)：径向 ρ=hypot(loc_x,loc_y)，
        d_r=max(0,ρ−r)、d_z=max(0,|loc_z|−h/2)，dist²=d_r²+d_z²。
    两者都是「点到正交乘积区域」的精确最短距离（径向⊥轴向、各轴互相⊥，分量可分别取）。
    """
    c = np.asarray(prim.pose[:3], float)
    qw, qx, qy, qz = prim.pose[3:7]                       # wxyz
    Rm = R.from_quat([qx, qy, qz, qw]).as_matrix()        # world←local
    loc = (np.asarray(pts, float) - c) @ Rm               # 世界点→局部系：Rᵀ(p−c)
    if isinstance(prim, ob.Box):
        he = 0.5 * np.asarray(prim.dims, float)           # 半边长
        over = np.maximum(np.abs(loc) - he, 0.0)          # 各轴外溢
        return np.sum(over * over, axis=1)
    # Tube：有限实心圆柱
    radial = np.hypot(loc[:, 0], loc[:, 1])
    d_r = np.maximum(radial - float(prim.radius), 0.0)
    d_z = np.maximum(np.abs(loc[:, 2]) - 0.5 * float(prim.height), 0.0)
    return d_r * d_r + d_z * d_z


def _overlaps_sweep(prims, sph_link) -> bool:
    """障碍是否与该 link 扫掠球【真有交集】（精确 点-实体 距离，非包围球近似）。

    sph_link: (T,S,4) 该 link 沿默认轨迹的全部扫掠球 [x,y,z,r]。把球心摊平成 (N,3)，对每个 prim 用
    _point_solid_dist2 求各球心到实体的最短距离²，存在某球 dist² ≤ r² 即「障碍嵌进扫掠管」→ True。
    """
    balls = np.asarray(sph_link, float).reshape(-1, 4)
    bc, br = balls[:, :3], balls[:, 3]                    # (N,3),(N,)
    for p in prims:
        d2 = _point_solid_dist2(p, bc)
        if np.any(d2 <= br * br):
            return True
    return False


def _clears_init_free(prims, cfg) -> bool:
    """整只障碍是否完全落在初始引导 FREE 圆柱之外（与 init 空间无交集）。

    逐 prim 用保守包围球 (c, rb) 判它与圆柱（轴过 base 原点 x=y=0、半径 cyl_r、z∈[0,cyl_h]）无交：
      · 整球在圆柱下方 (c_z+rb ≤ 0) 或上方 (c_z−rb ≥ cyl_h)，或
      · 整球在圆柱径向外 (hypot(c_x,c_y)−rb ≥ cyl_r)。
    任一 prim 与圆柱有交即返回 False。包围球是保守外估 → 判「无交」偏严（多排除一点），绝不漏判
    （即返回 True 时整只障碍一定真在圆柱外）。
    """
    cyl_r = float(cfg.init_free_cyl_radius)
    cyl_h = float(cfg.init_free_cyl_height)
    for p in prims:
        c, rb = _prim_bounding_sphere(p)
        if c[2] + rb <= 0.0 or c[2] - rb >= cyl_h:       # 整球在圆柱上/下方 → 无交
            continue
        if math.hypot(float(c[0]), float(c[1])) - rb >= cyl_r:   # 整球在径向外 → 无交
            continue
        return False                                     # 该 prim 与圆柱有交
    return True


def _signed_dist_cyl(pts, cfg) -> np.ndarray:
    """一批点 pts(...,3) 到 init_free 实心有限圆柱的【带符号距离】（外正、内负），闭式向量化。

    圆柱：轴过 base 原点 x=y=0，半径 R，z∈[0,H]。对每点令 ρ=hypot(x,y)、dr=ρ−R、dz=max(−z, z−H)：
      · 圆柱内 (dr≤0 且 dz≤0)            → sd = max(dr,dz)   （负，到最近壁/盖的距离）
      · 柱壁外、z 在带内 (dr>0, dz≤0)     → sd = dr
      · 柱盖上下、ρ 在内 (dr≤0, dz>0)     → sd = dz
      · 圆边角外 (dr>0, dz>0)             → sd = hypot(dr,dz)
    sd>0 即点在圆柱外，且其值 = 点到圆柱实体的最短距离（解析、无迭代）。
    """
    R_cyl = float(cfg.init_free_cyl_radius)
    H = float(cfg.init_free_cyl_height)
    p = np.asarray(pts, float)
    rho = np.hypot(p[..., 0], p[..., 1])
    dr = rho - R_cyl
    dz = np.maximum(-p[..., 2], p[..., 2] - H)
    out_r = dr > 0.0
    out_z = dz > 0.0
    return np.where(out_r & out_z, np.hypot(dr, dz),
                    np.where(out_r, dr,
                             np.where(out_z, dz, np.maximum(dr, dz))))


def _signed_dist_box(pts, cfg) -> np.ndarray:
    """一批点 pts(...,3) 到 init_free 轴对齐立方体(AABB)的【带符号距离】（外正、内负），闭式向量化。

    盒 [lo, hi]（base 系，来自 cfg.init_free_box_min_for_init / init_free_box_max_for_init）。逐轴令
    d_lo=lo−p（低于下界时>0）、d_hi=p−hi（高于上界时>0）、over=max(d_lo,d_hi,0)（各轴外溢量）：
      · 盒外（任一轴 over>0）→ sd = ‖over‖（点到 AABB 的精确最短距离）；
      · 盒内（所有 over=0）→ sd = max_i max(d_lo_i, d_hi_i)（负，到最近面的距离）。
    sd>0 即点在盒外，且其值 = 点到盒实体的最短距离（解析、无迭代；与 _signed_dist_cyl 对称）。
    """
    lo = np.asarray(cfg.init_free_box_min_for_init, float)
    hi = np.asarray(cfg.init_free_box_max_for_init, float)
    p = np.asarray(pts, float)
    d_lo = lo - p                                        # (...,3) >0 在下界外
    d_hi = p - hi                                        # (...,3) >0 在上界外
    over = np.maximum(np.maximum(d_lo, d_hi), 0.0)       # (...,3) 各轴外溢
    outside = np.any(over > 0.0, axis=-1)
    sd_out = np.linalg.norm(over, axis=-1)               # 盒外精确距离
    sd_in = np.max(np.maximum(d_lo, d_hi), axis=-1)      # 盒内（各轴均≤0）取最近面（负）
    return np.where(outside, sd_out, sd_in)


def _clears_init_free_box(prims, cfg) -> bool:
    """整只障碍是否完全落在初始引导 FREE 立方体(AABB)之外（与 init 空间无交集）。

    逐 prim 用保守包围球 (c, rb)：球心 c 到盒的带符号距离 sd（_signed_dist_box）≥ rb 才算「整球在盒外」。
    任一 prim 的 sd<rb（含 c 在盒内 sd<0）即包围球够到盒 → 返回 False。包围球是保守外估 → 判「无交」
    偏严（多排除一点），绝不漏判（返回 True 时整只障碍一定真在盒外；与 _clears_init_free 对称）。
    """
    for p in prims:
        c, rb = _prim_bounding_sphere(p)
        sd = float(_signed_dist_box(np.asarray(c, float).reshape(1, 3), cfg)[0])
        if sd < rb:                                      # 包围球够到盒（或球心在盒内）→ 可能相交
            return False
    return True


def _signed_dist_init_free(pts, cfg) -> np.ndarray:
    """点到 init_free 空间的带符号距离，按 cfg.init_free_method_for_init 分派 box / 圆柱（单一来源）。

    method 未配置（旧 demo）时退回圆柱，保持向后兼容。
    """
    method = getattr(cfg, "init_free_method_for_init", "cylinder")
    if method == "box":
        return _signed_dist_box(pts, cfg)
    return _signed_dist_cyl(pts, cfg)


def _clears_init_free_any(prims, cfg) -> bool:
    """整只障碍是否在 init_free 空间外，按 cfg.init_free_method_for_init 分派 box / 圆柱（单一来源）。"""
    method = getattr(cfg, "init_free_method_for_init", "cylinder")
    if method == "box":
        return _clears_init_free_box(prims, cfg)
    return _clears_init_free(prims, cfg)


# ------------------------------------------------------------------ 障碍尺寸（按走廊管半径）
def _shape_for(otype: str, span: float, tube_r: float, *, cfg, th: float
               ) -> Tuple[dict, Tuple[float, float, float]]:
    """据局部走廊跨度 span / 管半径 tube_r 给该障碍类型的形状参数 + anchor 微调偏置(局部系，一般 0)。

    入参
    ----
    otype   : 障碍类型名（obstacles.REGISTRY 的键）。
    span    : 障碍应覆盖的横向尺寸（米）。来自 place_in_corridor 的 2·tube_r·size_scale，即「走廊
              直径 × 尺寸缩放」——让障碍大致横穿整条走廊、挡住路径。下面记 clip 后的值为 s。
    tube_r  : 局部走廊「管半径」（米）——该路点处各碰撞球外缘到 anchor 的最大距离，描述走廊有多粗。
              主要用于定「细长杆/管/梁」类构件的截面半径或边长。下面记 clip 后的值为 rt。
    cfg     : 配置对象；从 cfg.obstacle_placement.obstacle_params[otype] 读该类型的 span/tube_r 夹紧
              区间（span_clip_m / tube_r_clip_m，按类型单独控制），不在代码里写死。
    th      : 板/壁厚（米）。由调用方（place_in_corridor，值在 search_placement 按
              thickness_range_m 随机采样）传入，使每次放置的薄板/盒壁厚度有多样性。

    返回
    ----
    (shape_kwargs, local_offset)：
      shape_kwargs —— 直接喂给 obstacles.build(otype, ..., **shape) 的该类型形状参数 dict；
      local_offset —— anchor 朝向系下对 anchor 的微调偏置（米）。多数类型 anchor 即结构几何中心，
                      offset=(0,0,0)；少数以「角/底/端」为锚的结构给一点偏置，把结构主体压到走廊上。
      （各类型 shape 参数的逐项含义见下方 table 每条上方的行内注释；值由 s / rt / 随机壁厚 th 推出。）

    为何对 s / rt 做 clip（夹紧到固定区间，区间值来自 default.yaml）
    ----
    span 与 tube_r 都是从「当前路点的扫掠球」量出来的，再乘上可在重试中被自适应放大/缩小的
    size_scale（too_weak 时 ×1.25、no_solution 时 ×0.8…），数值会在很大范围漂移，甚至退化：
      · span 可能因某 link 在该处球簇很小而趋近 0，或被放大到不合理；
      · tube_r 同理可能极小或极大。
    若不夹紧，会出两类问题：
      （下界）尺寸趋 0 → 障碍小到挡不住路径（永远 too_weak）、或薄到生成的 mesh 退化、碰撞球穿过去；
      （上界）尺寸过大 → 障碍塞满整个场景，挡死所有绕行（永远 no_solution）、还可能与工件/基座/地面
              相交，且大 mesh 拖慢碰撞检查。
    所以（区间从 cfg.obstacle_placement.obstacle_params[otype] 读，按类型单独调，不写死在代码）：
      s  = clip(span,   span_clip_m[0],   span_clip_m[1])   —— 障碍主跨度：下界保证大到能横挡走廊，
                                                              上界防吞掉整个工作空间而无解。
      rt = clip(tube_r, tube_r_clip_m[0], tube_r_clip_m[1]) —— 杆/管/梁截面半径：下界防细到数值上可
                                                              忽略、从碰撞球缝隙漏过；上界防单根管成大圆柱。
    夹紧后这两个值才是「物理上合理、可复现」的形状输入。th(板/壁厚)同理由 thickness_range_m 随机取，
    既有多样性又被限定在合理薄板范围。
    """
    op = cfg.obstacle_placement
    # 该类型的形状参数（离散选项/结构数/缩放系数(_mult)/下限(_min)/夹紧区间）：从
    # cfg.obstacle_placement.obstacle_params[otype] 读覆盖值，未列的键回退下方内置默认（单一来源在
    # default.yaml，代码只留兜底默认）。
    pp = dict((op.get("obstacle_params", {}) or {}).get(otype, {}) or {})
    g = lambda k, d: pp.get(k, d)
    # span/tube_r 夹紧区间【按类型】：span_clip_m 所有类型都用；tube_r_clip_m 仅杆/管/梁类用——
    # 不用 rt 的类型（plate/l_bracket/u_channel…）yaml 里不写它，此处回退默认即可（rt 对其为死值，无影响）。
    s_lo, s_hi = g("span_clip_m", [0.15, 1.2])
    rt_lo, rt_hi = g("tube_r_clip_m", [0.04, 0.20])
    s = float(np.clip(span, float(s_lo), float(s_hi)))
    rt = float(np.clip(tube_r, float(rt_lo), float(rt_hi)))
    th = float(th)
    O = (0.0, 0.0, 0.0)
    table = {
        # plate: length=高(Z向), width=宽(Y向), thickness=板厚, tilt_deg=绕Y倾角
        #   → 一块 s×s 的薄板，法向(+X)对齐切向，正面横挡走廊。
        "plate":          (dict(length=s, width=s, thickness=th, tilt_deg=g("tilt_deg", 0.0)), O),
        # l_bracket: length=两板边长, width=板宽, thickness=板厚；
        #   offset=(-s/2,0,-s/2) 把 └ 的角从结构角挪到走廊中心（默认锚在 L 的拐角）。
        "l_bracket":      (dict(length=s, width=s, thickness=th), (-s / 2, 0.0, -s / 2)),
        # u_channel: length=槽长(X), width=槽宽(Y,两侧板间距), height=侧板高(Z,=s×height_mult), thickness=壁厚；
        #   offset=(0,0,-s·0.35) 把开口槽底压到走廊（默认锚在底板）。
        "u_channel":      (dict(length=s, width=s, height=s * g("height_mult", 0.7), thickness=th), (0.0, 0.0, -s * 0.35)),
        # open_box: size=(sx,sy,sz) 盒外形, wall=壁厚, open_face=缺哪面（默认 "front"=缺 +X，开口迎切向）。
        "open_box":       (dict(size=(s, s, s), wall=th, open_face=g("open_face", "front")), O),
        # pipe: length=管长(=s×length_mult，够长跨过走廊), radius=管半径(=rt), axis=管轴(默认 "y" 横拦走廊)。
        "pipe":           (dict(length=s * g("length_mult", 1.5), radius=rt, axis=g("axis", "y")), O),
        # parallel_pipes: n=管数, length=管长(s×length_mult), radius=rt, gap=管间距(s×gap_mult), axis=管轴,
        #   stack=排开方向 → 一排平行管像护栏。
        "parallel_pipes": (dict(n=g("n", 3), length=s * g("length_mult", 1.5), radius=rt,
                                gap=s * g("gap_mult", 0.5), axis=g("axis", "y"), stack=g("stack", "z")), O),
        # crossed_pipes: length=管长(s×length_mult), radius=rt, cross_deg=两管夹角 → Y-Z 面内成 X 形交叉。
        "crossed_pipes":  (dict(length=s * g("length_mult", 1.5), radius=rt, cross_deg=g("cross_deg", 90.0)), O),
        # box_beam: length=梁长(s×length_mult), side=方截面边长(=max(side_min, rt×side_mult)), axis=梁轴。
        "box_beam":       (dict(length=s * g("length_mult", 1.5),
                                side=max(g("side_min", 0.06), rt * g("side_mult", 1.4)), axis=g("axis", "y")), O),
        # rect_frame: width=框宽(Y), height=框高(Z), beam=边框方梁截面(=max(beam_min, rt)) → 中间留孔的矩形框。
        "rect_frame":     (dict(width=s, height=s, beam=max(g("beam_min", 0.06), rt)), O),
        # gantry: span=两立柱间距, height=立柱高, post=立柱截面(=max(post_min,rt)), beam=横梁截面(=max(beam_min,rt))；
        #   offset=(0,0,-s/2) 把门架从「立柱底=锚」下移，使横梁/门洞罩住走廊（默认锚在地面平面）。
        "gantry":         (dict(span=s, height=s, post=max(g("post_min", 0.06), rt),
                                beam=max(g("beam_min", 0.08), rt)), (0.0, 0.0, -s / 2)),
        # braced_frame: width,height,beam 同 rect_frame, brace=对角斜撑截面(=max(brace_min, rt×brace_mult))
        #   → 框+一根斜梁破坏直穿。
        "braced_frame":   (dict(width=s, height=s, beam=max(g("beam_min", 0.06), rt),
                                brace=max(g("brace_min", 0.05), rt * g("brace_mult", 0.8))), O),
        # tripod: height=三角高, base_half=底边半宽(=s×base_half_mult), rod=杆半径(=max(rod_min,rt)) → 竖立三角框。
        "tripod":         (dict(height=s, base_half=s * g("base_half_mult", 0.5), rod=max(g("rod_min", 0.05), rt)), O),
        # steps: n=台阶数, rise=单级升高(s×rise_mult), run=单级进深(s×run_mult), width=台阶宽(Y)；
        #   offset=(-s·0.6,0,-s/2) 把楼梯主体从「第一级底角=锚」挪到走廊中心。
        "steps":          (dict(n=g("n", 3), rise=s * g("rise_mult", 0.33), run=s * g("run_mult", 0.4), width=s),
                           (-s * 0.6, 0.0, -s / 2)),
        # box_with_pipe: size,wall,open_face 同 open_box, pipe_radius=开口前横管半径(=rt) → 开口盒+挡管组合。
        "box_with_pipe":  (dict(size=(s, s, s), wall=th, open_face=g("open_face", "front"), pipe_radius=rt), O),
        # frame_with_brace: 同 braced_frame（width,height,beam,brace）——语义别名入口。
        "frame_with_brace": (dict(width=s, height=s, beam=max(g("beam_min", 0.06), rt),
                                  brace=max(g("brace_min", 0.05), rt * g("brace_mult", 0.8))), O),
    }
    # 其它/未知类型 → 回退成一块 plate(length=s, width=s, thickness=th)。
    return table.get(otype, (dict(length=s, width=s, thickness=th), O))


# ------------------------------------------------------------------ 底层放置（M=1，解析解）
def _bounding_radius_about(prims, ctr) -> float:
    """整只障碍以世界点 ctr 为心的外接球半径 = max_p(‖prim_心−ctr‖ + prim 外接半径)。闭式。"""
    ctr = np.asarray(ctr, float)
    rb = 0.0
    for p in prims:
        c, r = _prim_bounding_sphere(p)
        rb = max(rb, float(np.linalg.norm(c - ctr)) + r)
    return rb


# ------------------------------------------------------------------ coal / hppfcl 精确分离
_FCL_MOD = "unset"   # 惰性导入缓存："unset" 未试 / None 无库 / 模块对象


def _fcl():
    """导入公开碰撞库：coal(新名) 优先、hppfcl(旧名 hpp-fcl)回退；都没有返回 None（退回包围球路径）。

    本番 env_isaaclab 装的是 hppfcl 2.4.4，故实际走 hppfcl 分支。结果缓存，避免每次放置重复 import。
    """
    global _FCL_MOD
    if _FCL_MOD != "unset":
        return _FCL_MOD
    mod = None
    for name in ("coal", "hppfcl"):
        try:
            mod = __import__(name)
            break
        except Exception:
            mod = None
    _FCL_MOD = mod
    return mod


def _fcl_T(pose7, fcl):
    """[x,y,z,qw,qx,qy,qz](base 系) → fcl.Transform3f（旋转矩阵 + 平移）。"""
    c = np.asarray(pose7[:3], float)
    qw, qx, qy, qz = (float(v) for v in pose7[3:7])
    Rm = R.from_quat([qx, qy, qz, qw]).as_matrix()       # wxyz → 矩阵
    T = fcl.Transform3f()
    T.setRotation(Rm)
    T.setTranslation(c)
    return T


def _fcl_geom_for_prim(prim, fcl):
    """障碍 prim → (fcl 几何, Transform3f)。Box→Box(全边长 dims)；Tube→Cylinder(radius, height，轴沿局部 +Z)。"""
    if isinstance(prim, ob.Box):
        g = fcl.Box(np.asarray(prim.dims, float))
    else:                                                # Tube：轴沿局部 +Z，与 ob.build 的 Tube 约定一致
        g = fcl.Cylinder(float(prim.radius), float(prim.height))
    return g, _fcl_T(prim.pose, fcl)


def _fcl_geom_init_free(cfg, fcl):
    """init_free 空间 → (fcl 几何, Transform3f)，按 cfg.init_free_method_for_init 分派 box / 圆柱。"""
    method = getattr(cfg, "init_free_method_for_init", "cylinder")
    if method == "box":
        lo = np.asarray(cfg.init_free_box_min_for_init, float)
        hi = np.asarray(cfg.init_free_box_max_for_init, float)
        g = fcl.Box(hi - lo)                             # 全边长
        T = fcl.Transform3f()
        T.setTranslation(0.5 * (lo + hi))               # AABB 中心
        return g, T
    R_cyl = float(cfg.init_free_cyl_radius)
    H = float(cfg.init_free_cyl_height)
    g = fcl.Cylinder(R_cyl, H)                           # 轴沿 +Z、全高 H
    T = fcl.Transform3f()
    T.setTranslation(np.array([0.0, 0.0, H / 2.0]))     # 圆柱 z∈[0,H] → 中点 H/2
    return g, T


def _penetration(prims, init_free_geom, fcl) -> Tuple[float, Optional[np.ndarray]]:
    """障碍各 prim 与 init_free 的【最深贯入】(depth≥0, 单位外推法线 n̂)；全分离时 (0, None)。

    对每个 prim 调 fcl.distance(init_free, prim, signed)：d<0 即贯入，|d|=贯入深度，res.normal 为
    「从 init_free 指向 prim」的分离方向（即把 prim 推出 init_free 的方向）。取贯入最深者的 (depth, n̂)。
    init_free_geom = _fcl_geom_init_free(cfg, fcl) 预先构好（每次 push 迭代复用，省重复构造）。
    """
    fg, fT = init_free_geom
    req = fcl.DistanceRequest()
    req.enable_signed_distance = True
    best_depth = 0.0
    best_n = None
    for p in prims:
        pg, pT = _fcl_geom_for_prim(p, fcl)
        res = fcl.DistanceResult()
        d = float(fcl.distance(fg, fT, pg, pT, req, res))
        if d < -best_depth:                              # 更深的贯入（d 越负越深）
            best_depth = -d
            n = np.asarray(res.normal, float)
            nn = float(np.linalg.norm(n))
            best_n = (n / nn) if nn > 1e-9 else None
    return best_depth, best_n


def _clears_init_free_exact(prims, cfg, fcl, tol: float = 1e-4) -> bool:
    """fcl 精确判据：所有 prim 与 init_free 的签名距离 ≥ −tol（实体不贯入）→ 整只在 init_free 外。"""
    depth, _ = _penetration(prims, _fcl_geom_init_free(cfg, fcl), fcl)
    return depth <= tol


def _push_out_of_init_free(c_start, Rm, otype, shape, local_off, rpy, cfg, fcl,
                           *, eps: float = 2e-3, tol: float = 1e-4, max_iter: int = 12):
    """从 c_start 沿「最深贯入法线」迭代平移障碍，直到整只脱离 init_free（fcl 精确）。

    姿态 Rm 固定（含水平偏航），仅平移 body 中心 c。每轮：造原语 → 算最深贯入 (depth,n̂) →
    若 depth≤tol 则已清空、返回；否则 c += (depth+eps)·n̂（真实厚度外推，非包围球 R_b）。
    返回 (v, prims, anchor)：v=累计外推向量(c_end−c_start)；max_iter 内未清空 → None。
    """
    init_free_geom = _fcl_geom_init_free(cfg, fcl)
    c = np.asarray(c_start, float).copy()
    off = np.asarray(local_off, float)
    for _ in range(max_iter):
        anchor = c + Rm.apply(off)                       # body 中心在 c
        prims = ob.build(otype, anchor.tolist(), anchor_rpy_deg=rpy, **shape)
        depth, n = _penetration(prims, init_free_geom, fcl)
        if depth <= tol or n is None:                    # fcl 精确判定：已脱离 init_free
            return (c - np.asarray(c_start, float)), prims, anchor
        c = c + (depth + eps) * n                        # 沿法线最小外推
    return None                                          # 未能在 max_iter 内推出（该姿态放不下）


def _yaw_solve(c_k, Rm0, otype, shape, local_off, rpy0, cfg, fcl, sph, goal, gc,
               *, yaw_max_deg: float = 90.0):
    """绕竖直轴的水平偏航一维求解：在 [−yaw_max,+yaw_max] 找「外推量 ‖v‖ 最小、且外推后仍咬住扫掠管」的 ψ*。

    对每个 ψ：Rm(ψ)=Rz(ψ)·Rm0（绕世界竖直轴，过 c_k），push 出 init_free 得 v(ψ)、prims(ψ)；
    保交判据 _overlaps_sweep + goal 间距。粗扫(step≈15°)择优后在邻域细扫(step≈3°)精化。
    返回 (psi*, v*, prims*, anchor*, rpy*) 或 None（任何 ψ 都无法「出 init_free 且保交」）。
    """
    z = np.array([0.0, 0.0, 1.0])

    def _eval(psi):
        Rm = R.from_rotvec(z * math.radians(psi)) * Rm0  # 绕世界竖直轴左乘
        rpy = [float(v) for v in Rm.as_euler("xyz", degrees=True)]
        out = _push_out_of_init_free(c_k, Rm, otype, shape, local_off, rpy, cfg, fcl)
        if out is None:
            return None
        v, prims, anchor = out
        if not _overlaps_sweep(prims, sph):              # 外推后脱离了扫掠管 → 不可行
            return None
        if float(np.linalg.norm(anchor - goal)) < gc:    # 贴焊缝 → 不可行
            return None
        return dict(psi=float(psi), v=v, norm=float(np.linalg.norm(v)),
                    prims=prims, anchor=anchor, rpy=rpy)

    best = None
    coarse = np.arange(-yaw_max_deg, yaw_max_deg + 1e-6, 15.0)
    for psi in coarse:
        r = _eval(float(psi))
        if r is not None and (best is None or r["norm"] < best["norm"]):
            best = r
    if best is None:
        return None
    # 邻域细扫（±15°、step 3°）精化最优 ψ
    for psi in np.arange(best["psi"] - 15.0, best["psi"] + 15.0 + 1e-6, 3.0):
        if abs(psi) > yaw_max_deg + 1e-6:
            continue
        r = _eval(float(psi))
        if r is not None and r["norm"] < best["norm"]:
            best = r
    return best["psi"], best["v"], best["prims"], best["anchor"], best["rpy"]


def _place_shrink_fallback(c_k, sd_k, t_k, tube_r0, rpy, tangent, otype, *,
                           size_scale, rot_jitter_deg, thickness, cfg, sph, goal, gc, link):
    """无 coal/hppfcl 库时的保守回退：以 c_k 为心的包围球 R_b 收缩尺寸（旧算法，偏严）。

    保留原「同一 c_k、仅缩 span/tube_r/壁厚、最多 3 轮」逻辑，仅供缺库环境兜底（本番 env 有 hppfcl，不走此路）。
    """
    Rm = R.from_euler("xyz", rpy, degrees=True)
    margin = 1.0 - 1e-2
    scale = 1.0
    prims = None
    anchor_eff = c_k
    shape = {}
    for _ in range(3):
        span = 2.0 * tube_r0 * float(size_scale) * scale
        shape, local_off = _shape_for(otype, span, tube_r0 * scale, cfg=cfg, th=float(thickness) * scale)
        anchor_eff = c_k + Rm.apply(np.asarray(local_off, float))
        prims = ob.build(otype, anchor_eff.tolist(), anchor_rpy_deg=tuple(rpy), **shape)
        R_b = _bounding_radius_about(prims, c_k)
        if R_b <= sd_k * margin:
            break
        scale *= (sd_k * margin) / max(R_b, 1e-6)
    if prims is None or not _clears_init_free_any(prims, cfg) or not _overlaps_sweep(prims, sph):
        return None
    if float(np.linalg.norm(anchor_eff - goal)) < gc:
        return None
    meta = dict(link=link, otype=otype, anchor=anchor_eff.tolist(), tangent=tangent.tolist(),
                tube_r=float(tube_r0 * scale), span=float(2.0 * tube_r0 * size_scale * scale),
                t=int(t_k), size_scale=float(size_scale), shrink_scale=float(scale),
                sd_k=float(sd_k), rot_jitter_deg=[float(a) for a in rot_jitter_deg],
                thickness=float(thickness) * float(scale), shape=shape, init_free_push=0.0)
    return prims, anchor_eff, meta


def _debug_show_sweep(per_wp, link, prims=None, *, cfg=None, anchor=None,
                      goal_pos=None, workpiece_mesh=None, seam_line=None,
                      sphere_stride: int = 1, title: str = "sweep-debug") -> None:
    """【手动调用】Open3D 可视化单 link 扫掠空间：工件 + 焊缝 + 机械臂碰撞球 + 障碍物 + init_free。

    - 工件 workpiece_mesh（浅灰实体）：需带 .file_path 与 .pose（[x,y,z,qw,qx,qy,qz]，base 系）；
      按该 pose 变到 base 系画（与放置系一致）；None 则不画。
    - 焊缝 seam_line（红色折线 + 端点红球）：(N,3)【base 系】。注意 Scene._seam_frame 给的是【工件 mesh 系】，
      需先乘 T_workpiece_in_base 变到 base 再传入。None 则不画。
    - 机械臂碰撞球 / 扫掠空间：per_wp[link] 的 (T,S,4)=[x,y,z,r]（base 系），逐球按真实半径画线框，
      颜色沿轨迹时间蓝(起)→红(末)渐变，其并集即「扫掠走廊」。球太多时把 sphere_stride 调大抽稀路点。
    - 障碍物 prims（Box/Tube，base 系）：橙色实体；prims=None（如放置前调用）则不画。
    - init_free（青色线框）：按 cfg.init_free_method_for_init 画 box 或圆柱——障碍应完全在其外。
    - anchor 绿球（障碍几何中心 c_k）、goal_pos 黄球（焊缝中心，goal_clearance 参照）。
    窗口阻塞，关闭后继续；缺 open3d 或无显示时打印告警跳过，不影响主流程。
    """
    try:
        import open3d as o3d
    except Exception as e:                                   # 缺库 → 跳过
        print(f"[debug-o3d] open3d 不可用，跳过可视化：{e}")
        return

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

    # —— 工件 mesh（浅灰实体）：按 base 系 pose 变换 ——
    has_wp = False
    if workpiece_mesh is not None:
        try:
            wp = o3d.io.read_triangle_mesh(workpiece_mesh.file_path)
            if not wp.is_empty():
                wp.transform(_pose_to_T(workpiece_mesh.pose))
                wp.compute_vertex_normals()
                wp.paint_uniform_color([0.7, 0.7, 0.7])
                geoms.append(wp)
                has_wp = True
        except Exception as e:
            print(f"[debug-o3d] 工件 mesh 加载失败：{e}")

    # —— 焊缝折线（红色 LineSet + 端点红球，base 系）——
    has_seam = False
    if seam_line is not None:
        try:
            sl = np.asarray(seam_line, float).reshape(-1, 3)
            if len(sl) >= 2:
                lines = [[i, i + 1] for i in range(len(sl) - 1)]
                seg = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(sl),
                    o3d.utility.Vector2iVector(np.asarray(lines, int)))
                seg.paint_uniform_color([0.9, 0.05, 0.05])
                geoms.append(seg)
                for ep in (sl[0], sl[-1]):                   # 两端点红球，线太细时也能看清位置
                    es = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    es.translate(ep)
                    es.compute_vertex_normals()
                    es.paint_uniform_color([0.9, 0.05, 0.05])
                    geoms.append(es)
                has_seam = True
        except Exception as e:
            print(f"[debug-o3d] 焊缝折线可视化失败：{e}")

    # —— 机械臂碰撞球 / 扫掠空间：逐球线框（真实半径），颜色沿时间蓝→红 ——
    sph = None if per_wp is None else per_wp.get(link)
    n_ball = 0
    if sph is not None and len(sph):
        arr = np.asarray(sph, float)                        # (T,S,4)
        Tn = arr.shape[0]
        step = max(1, int(sphere_stride))
        for ti in range(0, Tn, step):
            frac = ti / max(Tn - 1, 1)
            col = [frac, 0.15, 1.0 - frac]                  # 蓝(起)→红(末)
            for c in arr[ti]:
                cx, cy, cz, r = (float(v) for v in c)
                if r <= 1e-4:
                    continue
                ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=6)
                ball.translate((cx, cy, cz))
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color(col)
                geoms.append(ls)
                n_ball += 1

    # —— 障碍物原语（橙色实体）：Box→create_box（角在原点需居中）；Tube→create_cylinder（已居中、轴 Z）——
    n_prim = 0
    for p in (prims or []):
        if isinstance(p, ob.Box):
            dx, dy, dz = (float(v) for v in p.dims)
            g = o3d.geometry.TriangleMesh.create_box(dx, dy, dz)
            g.translate((-dx / 2, -dy / 2, -dz / 2))
        else:                                               # Tube
            g = o3d.geometry.TriangleMesh.create_cylinder(float(p.radius), float(p.height))
        g.transform(_pose_to_T(p.pose))
        g.compute_vertex_normals()
        g.paint_uniform_color([0.95, 0.55, 0.15])
        geoms.append(g)
        n_prim += 1

    # —— init_free 引导空间：按 method 画 box 或圆柱（亮青线框 + 角点小球，在成片扫掠球线框中也醒目）——
    #   legacy draw_geometries 不支持半透明实体，故用「线框 + 角点」表达这块「空间」；障碍应完全在其外。
    has_free = False
    if cfg is not None:
        try:
            method = getattr(cfg, "init_free_method_for_init", "cylinder")
            corners = None
            if method == "box":
                lo = np.asarray(cfg.init_free_box_min_for_init, float)
                hi = np.asarray(cfg.init_free_box_max_for_init, float)
                d = hi - lo
                box = o3d.geometry.TriangleMesh.create_box(float(d[0]), float(d[1]), float(d[2]))
                box.translate(lo)                            # create_box 角在原点 → 平移到 lo
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(box)
                corners = np.array([[lo[0] if a else hi[0], lo[1] if b else hi[1], lo[2] if c else hi[2]]
                                    for a in (1, 0) for b in (1, 0) for c in (1, 0)], float)
            else:
                cyl_r = float(cfg.init_free_cyl_radius)
                cyl_h = float(cfg.init_free_cyl_height)
                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_r, height=cyl_h,
                                                                resolution=32)
                cyl.translate((0.0, 0.0, cyl_h / 2.0))       # 居中于原点 → 抬到 z∈[0,h]
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(cyl)
            ls.paint_uniform_color([0.0, 0.95, 0.95])        # 亮青
            geoms.append(ls)
            for cp in (corners if corners is not None else []):   # box 8 角点小球，锚定这块空间
                m = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                m.translate(cp)
                m.compute_vertex_normals()
                m.paint_uniform_color([0.0, 0.95, 0.95])
                geoms.append(m)
            has_free = True
        except Exception as e:
            print(f"[debug-o3d] init_free 可视化失败：{e}")

    # —— anchor（绿球）+ goal（黄球）——
    if anchor is not None:
        a = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        a.translate(np.asarray(anchor, float))
        a.compute_vertex_normals()
        a.paint_uniform_color([0.1, 0.8, 0.2])
        geoms.append(a)
    if goal_pos is not None:
        gsp = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        gsp.translate(np.asarray(goal_pos, float))
        gsp.compute_vertex_normals()
        gsp.paint_uniform_color([0.95, 0.9, 0.1])
        geoms.append(gsp)

    print(f"[debug-o3d]「{title}」link={link}：工件 {'有' if has_wp else '无'}、焊缝 {'有' if has_seam else '无'}、"
          f"init_free {'有' if has_free else '无'}、碰撞球 {n_ball} 个、障碍原语 {n_prim} 个（关闭窗口继续）")
    o3d.visualization.draw_geometries(geoms, window_name=title)


def place_in_corridor(per_wp, origin, link, otype, *, size_scale, rot_jitter_deg, pos_frac,
                      jitter_vec, thickness, cfg, goal_pos,
                      workpiece_mesh=None, retract=None, seam_line=None, debug_show: bool = False):
    """【解析解】把 1 个【原尺寸】障碍摆到「与 init_free 无交、与该 link 扫掠并集有交」的位姿。

    自由度 = xyz 平移 + 绕竖直轴的水平偏航 ψ（竖直姿态不变）。用 coal/hppfcl 精确贯入深度 + 接触法线
    做最小外推，取代旧「以 c_k 为心的包围球 R_b」过度外推→收缩尺寸的做法（障碍保持参数原尺寸）。
    详见 docs/障碍物类型1-place_in_corridor重构-coal求解.md。

    阶段：
      1. 选锚：时间窗内取「球面探出 init_free（sd(c_i)+r_i≥0）且离 goal 够远」的扫掠球，argmax sd → c_k；
      2. 朝向：+X 对齐 c_k 处切向后，叠加三轴随机旋转 rot_jitter_deg=(rx,ry,rz)（body 系，Rm0）；
      3. 造原尺寸障碍（不缩）：span=2·tube_r0·size_scale，_shape_for 内部按 clip 区间夹紧；
      4. coal 水平偏航求解：在 ±yaw_max 内找「外推 ‖v‖ 最小、且外推后仍咬住扫掠管」的 ψ*；
      5. 兜底断言 + 返回；无 coal/hppfcl 库时退回旧包围球收缩路径（_place_shrink_fallback）。

    位置由几何确定（argmax sd + 外推），pos_frac / jitter_vec 不参与定位（保留入参仅为签名兼容）。
    """
    op = cfg.obstacle_placement
    if link not in per_wp:
        return None
    sph = per_wp[link]                                   # (T,S,4)
    org = origin[link]                                   # (T,3)
    T = sph.shape[0]
    if T < 3:
        return None
    t_lo, t_hi = op["pos_t_window"]
    i0 = int(np.clip(round(t_lo * (T - 1)), 1, T - 2))
    i1 = int(np.clip(round(t_hi * (T - 1)), 1, T - 2))
    if i1 < i0:
        i0, i1 = i1, i0

    # —— 【手动可视化】放置前肉眼核对该 link 的机械臂碰撞球 / 扫掠走廊 / init_free（默认注释关）——
    # _debug_show_sweep(per_wp, link, prims=None, cfg=cfg, goal_pos=goal_pos,
    #                   workpiece_mesh=workpiece_mesh, seam_line=seam_line,
    #                   title=f"{link} sweep (放置前)")

    # —— 1. 选锚：窗口内「球面探出 init_free（sd+r≥0）且离 goal 够远」的扫掠球，argmax sd → c_k ——
    win = sph[i0:i1 + 1]                                 # (Tw,S,4)
    centers = win[..., :3]                               # (Tw,S,3)
    radii = win[..., 3]                                  # (Tw,S)
    sd = _signed_dist_init_free(centers, cfg)            # (Tw,S) 按 method 分派 box/圆柱
    goal = np.asarray(goal_pos, float)
    gc = float(op["goal_clearance_m"])
    far = np.linalg.norm(centers - goal, axis=-1) >= gc  # 离焊缝够远
    in_B = (sd + radii) >= 0.0                           # 球面探出 init_free（含半径，即属于 B）
    cand = in_B & far
    if not cand.any():
        return None                                      # 该段扫掠几乎全埋在 init_free 内 → 上层换 link/窗口
    sd_masked = np.where(cand, sd, -np.inf)
    tw, s_idx = np.unravel_index(int(np.argmax(sd_masked)), sd_masked.shape)
    t_k = i0 + int(tw)
    c_k = centers[tw, s_idx].astype(float).copy()        # 锚：B 候选中离 init_free 最深（外推最小）
    sd_k = float(sd[tw, s_idx])

    # —— 2. 局部走廊管半径（t_k 簇，相对 c_k）+ 切向 → 初始朝向 Rm0 ——
    ck_centers = sph[t_k, :, :3]
    ck_radii = sph[t_k, :, 3]
    tube_r0 = float(np.max(np.linalg.norm(ck_centers - c_k, axis=1) + ck_radii))
    tangent = org[min(t_k + 1, T - 1)] - org[max(t_k - 1, 0)]
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
    # +X 对齐切向为基础朝向，再叠加三轴随机旋转（body 系 Rz·Ry·Rx），仅为姿态多样性
    rpy_align = _rpy_align_x_to(tangent, roll_deg=0.0)   # +X 横挡走廊（不含滚转）
    Rm0 = R.from_euler("xyz", rpy_align, degrees=True) * R.from_euler("xyz", rot_jitter_deg, degrees=True)
    rpy0 = [float(v) for v in Rm0.as_euler("xyz", degrees=True)]

    # —— 3. 造【原尺寸】障碍形状（不再收缩；_shape_for 内部对 span/tube_r 做 clip）——
    span = 2.0 * tube_r0 * float(size_scale)
    shape, local_off = _shape_for(otype, span, tube_r0, cfg=cfg, th=float(thickness))

    # —— 4/5. coal 精确外推 + 水平偏航求解；无库退回旧包围球收缩路径 ——
    fcl = _fcl()
    if fcl is None:
        return _place_shrink_fallback(c_k, sd_k, t_k, tube_r0, rpy0, tangent, otype,
                                      size_scale=size_scale, rot_jitter_deg=rot_jitter_deg,
                                      thickness=thickness, cfg=cfg, sph=sph, goal=goal, gc=gc, link=link)

    yaw_max = float(op.get("yaw_search_max_deg", 90.0))
    solved = _yaw_solve(c_k, Rm0, otype, shape, local_off, rpy0, cfg, fcl, sph, goal, gc,
                        yaw_max_deg=yaw_max)
    if solved is None:
        return None                                      # 任何 ψ 都无法「原尺寸出 init_free 且仍咬住扫掠管」
    psi, v, prims, anchor_eff, rpy = solved

    # —— 兜底断言（fcl 精确 clear + 保交 + goal 间距；构造性应恒成立）——
    if not _clears_init_free_exact(prims, cfg, fcl) or not _overlaps_sweep(prims, sph):
        return None
    if float(np.linalg.norm(anchor_eff - goal)) < gc:
        return None

    meta = dict(link=link, otype=otype, anchor=anchor_eff.tolist(), tangent=tangent.tolist(),
                tube_r=float(tube_r0), span=float(span), t=int(t_k),
                size_scale=float(size_scale), sd_k=float(sd_k),
                rot_jitter_deg=[float(a) for a in rot_jitter_deg],
                thickness=float(thickness), shape=shape, yaw_deg=float(psi),
                push_vec=[float(x) for x in v], push_norm=float(np.linalg.norm(v)))
    # —— 【手动可视化】放置后连障碍一起看（默认注释关）——
    # _debug_show_sweep(per_wp, link, prims, cfg=cfg, anchor=anchor_eff, goal_pos=goal_pos,
    #                   workpiece_mesh=workpiece_mesh, seam_line=seam_line,
    #                   title=f"{link}/{otype} t={t_k}(放置后)")
    return prims, anchor_eff, meta


# ------------------------------------------------------------------ 世界构建 + 切换
def inflate_prims(prims, buffer_m: float):
    """把障碍原语整体膨胀一圈（buffer_m 米），返回新列表；buffer_m<=0 时原样返回。

    Box：每条边长 +2·buffer（左右各扩 buffer）；Tube：半径 +buffer、高 +2·buffer（两端各扩 buffer）。
    膨胀的是副本（dataclasses.replace），传入的 prims 不被修改（存盘/可视化仍用真实尺寸）。
    单一来源：build_world 的「绕行留间隙」与 viz_obstacle_buffer 的「膨胀前后对比」都走这里。"""
    if not buffer_m or buffer_m <= 0:
        return list(prims)
    import dataclasses
    out = []
    for p in prims:
        if isinstance(p, ob.Box):
            out.append(dataclasses.replace(p, dims=[d + 2.0 * buffer_m for d in p.dims]))
        elif isinstance(p, ob.Tube):
            out.append(dataclasses.replace(
                p, radius=p.radius + buffer_m, height=p.height + 2.0 * buffer_m))
        else:
            out.append(p)
    return out


def inflate_trimesh_obb(tm, buffer_m: float):
    """单块 trimesh 按【有向包围盒(OBB)】各维 +2·buffer_m 膨胀，重建为盒 trimesh；buffer_m<=0 原样返回。

    用于对【无规则原语 dims】的障碍 mesh（如类型2 遮挡板的多边形棱柱）做膨胀：plate/open_box 各壁
    这类盒形 mesh 的 OBB 即其自身、膨胀精确；triangle/trapezoid 近似成外接矩形盒（偏保守=更安全）。
    与 inflate_prims 互补、同为障碍膨胀的单一来源（compute_goal_pose 碰撞世界 + viz 对比脚本共用）。"""
    if not buffer_m or buffer_m <= 0:
        return tm
    import trimesh
    obb = tm.bounding_box_oriented
    ext = np.asarray(obb.primitive.extents, float) + 2.0 * float(buffer_m)
    T = np.asarray(obb.primitive.transform, float)
    return trimesh.creation.box(extents=ext.tolist(), transform=T)


def build_world(workpiece_mesh, prims, buffer_m: float = 0.0, show_buffer: bool = False):
    """工件 mesh + 障碍原语 → 全 mesh 的 WorldConfig（MESH 检查器最稳）。
    buffer_m>0 时只把障碍原语膨胀一圈（工件 mesh 不变），让绕行解与真实障碍留间隙。
    膨胀的是 prims 的副本，传入的 prims 不被修改（存盘/可视化仍用真实尺寸）。
    show_buffer=True 且 buffer_m>0 时，弹 Open3D 窗对比膨胀前后（默认关——search 时会被调
    很多次，无条件弹窗会反复阻塞；调试单个场景再开）。"""
    import gt_gen.compat  # noqa: F401
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import WorldConfig

    prims = inflate_prims(prims, buffer_m)

    obs_wc = ob.to_world_config(prims)                   # cuboid=/cylinder=
    obs_mesh = obs_wc.get_mesh_world(process=False).mesh # 原语转 mesh
    return WorldConfig(mesh=[workpiece_mesh] + list(obs_mesh))


# ------------------------------------------------------------------ 三条件验证
def path_collides(handle, traj) -> Tuple[bool, int]:
    """默认轨迹在当前(含障碍)世界里是否会碰撞。逐路点 check_state，≥1 不可行即碰撞。返回 (collides, n_bad)。"""
    traj = np.asarray(traj, float)
    n = len(traj)
    idx = np.unique(np.linspace(0, n - 1, n).astype(int))
    n_bad = 0
    for i in idx:
        feasible, _ = ci.check_state(handle, traj[i].tolist())
        if not feasible:
            n_bad += 1
    return n_bad > 0, n_bad


def detour_exists(handle, retract, goal_pose, metric, max_attempts,
                  max_solutions: int = 1, dedup_rad: float = 0.3, *,
                  cfg=None, world=None, checker_type=None):
    """当前(含障碍)世界里能否规划出绕行解。返回 (ok, trajs)：trajs 是 list[ndarray]。

    后端由 cfg.planner_backend 决定（两后端用的是【同一个 world_inflated】：障碍膨胀、工件不膨胀）：
    - curobo：世界已由 validate_scene 的 update_world 灌进 handle，这里用 handle 规划，
              按 IK 误差升序收集多条互不相同(关节偏差峰值 ≥ dedup_rad)的绕行解；
    - stomp ：把 world 作【参数】直接传入 STOMP，固定朝向、buffer≈0.001；plan_to_pose_multi 返回
              所有合格候选(无碰撞+在限位，按 state_cost 升序、终点同一 IK 解)，本函数再按 dedup_rad
              贪心去重、收满 max_solutions 即停（与 curobo 分支同语义）。
    """
    backend = cfg.planner_backend if cfg is not None else "curobo"
    if backend == "stomp":
        from gt_gen import stomp_iface as si
        from gt_gen.curobo_iface import _traj_max_joint_dist
        op = cfg.obstacle_placement
        cands = si.plan_pose_multi(
            cfg, world, retract, goal_pose, checker_type=checker_type,
            ik_position_threshold=float(op["detour_ik_position_threshold"]),
            ik_rotation_threshold=float(op["detour_ik_rotation_threshold"]),
            showik=False)   # 调试写死：True 时弹 Open3D 画 IK 退回解整臂碰撞球+工件/障碍
        kept = []
        for traj in cands:                                    # 已按 state_cost 升序
            if max_solutions and len(kept) >= max_solutions:
                break
            if all(_traj_max_joint_dist(traj, k) >= dedup_rad for k in kept):
                kept.append(traj)
        return (len(kept) > 0), kept
    trajs = ci.plan_to_pose_all(handle, retract, goal_pose, max_attempts=max_attempts,
                                pose_cost_metric=metric, max_solutions=max_solutions,
                                dedup_rad=dedup_rad)
    return (len(trajs) > 0), trajs


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


def validate_scene(handle, traj_default, retract, goal_pose, metric, cfg,
                   world_real, world_inflated) -> dict:
    """综合三条件。①碰撞判定用真实尺寸世界 world_real，②③绕行规划用膨胀世界 world_inflated
    （障碍外扩一圈→绕行解与真实障碍留 buffer）。两世界外部 build_world 时分别构好传入。
    返回 dict（含 fail_reason / detour_trajs）。"""
    """
    validate_scene 是放置验证的核心裁判——判断一个已经摆好障碍的场景是否"合适"。它在内部切换世界：
    ①用真实尺寸世界 world_real 判碰撞、②③用膨胀世界 world_inflated 判绕行（障碍外扩 buffer，工件不变），
    然后顺序检验 docs/障碍物位置.md
    的三条件，任一条不满足就提前返回失败（并附上失败原因，供上层 search_placement 自适应调参重试）：

    ① 默认路径必须被挡住（path_collides）
    逐路点对默认无障碍轨迹做 check_state,只要有 ≥1 个点在含障碍世界里不可行就算碰撞。
    - 不碰 → fail_reason="too_weak"（障碍太弱/没挡住，白放）。

    ② 必须仍有绕行解（detour_exists）
    在含障碍世界里重新规划 retract→goal，能规划成功才行。按 detour_max_solutions 收集多条
    【互不相同】的绕行解（供存多份 GT 候选）。
    - 一条都规划不出 → fail_reason="no_solution"（障碍太强/把路堵死了，无解）。

    ③ 绕行必须明显不同于默认（is_detour_different）
    每条绕行与默认等长重采样后，逐路点关节最大偏差的峰值要 ≥ detour_min_joint_rad（默认 0.3 rad）；
    只保留满足此条的绕行解。
    - 无一条明显不同 → fail_reason="not_different"（绕了等于没绕，障碍没造成实质性扰动）。

    三条全过 → ok=True，detour_trajs 是【按 IK 误差升序、互不相同且都明显异于默认】的多条绕行轨迹，
    第 0 条即默认会优先取的那条；dist=这些绕行里关节偏差峰值的最大值。

    返回的 dict 形如：
    {ok, fail_reason ∈ {None, too_weak, no_solution, not_different}, n_bad, dist, detour_trajs}
    其中 detour_trajs 是 list[ndarray]（失败时为 []）。
    """
    op = cfg.obstacle_placement
    thresh = float(op["detour_min_joint_rad"])
    # ① 真实尺寸：默认路径是否真的撞到（buffer 不参与"是否碰撞"的判定）
    handle.mg.update_world(world_real)
    collides, n_bad = path_collides(handle, traj_default)
    if not collides:
        return dict(ok=False, fail_reason="too_weak", n_bad=0, dist=0.0, detour_trajs=[])
    # ②③ 膨胀尺寸：绕行解与真实障碍留 buffer 间隙；收集多条互不相同的绕行解
    if cfg.planner_backend != "stomp":
        handle.mg.update_world(world_inflated)            # cuRobo：世界经 handle 内部状态传入
    max_sol = int(op.get("detour_max_solutions", 1))
    ok2, trajs = detour_exists(handle, retract, goal_pose, metric,
                               int(op["detour_max_attempts"]),
                               max_solutions=max_sol, dedup_rad=thresh,
                               cfg=cfg, world=world_inflated)   # STOMP：世界作参数直接传入
    if not ok2:
        return dict(ok=False, fail_reason="no_solution", n_bad=n_bad, dist=0.0, detour_trajs=[])
    # ③ 只保留「明显不同于默认」的绕行解
    diff, best_dist = [], 0.0
    for t in trajs:
        okd, d = is_detour_different(traj_default, t, thresh)
        best_dist = max(best_dist, d)
        if okd:
            diff.append(t)
    if not diff:
        return dict(ok=False, fail_reason="not_different", n_bad=n_bad, dist=best_dist, detour_trajs=[])
    return dict(ok=True, fail_reason=None, n_bad=n_bad, dist=best_dist, detour_trajs=diff)


# ------------------------------------------------------------------ Open3D 调试可视化
def _pose_to_T(pose7) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz](base 系) → 4x4 齐次变换矩阵。"""
    import open3d as o3d
    T = np.eye(4)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
        np.asarray([pose7[3], pose7[4], pose7[5], pose7[6]], float))   # o3d 取 wxyz
    T[:3, 3] = np.asarray(pose7[:3], float)
    return T


def _debug_show_scene(workpiece_mesh, prims, per_wp, link, anchor,
                      *, cfg=None, retract=None, title: str = "scene") -> None:
    """用 Open3D 弹窗可视化当前待验证场景（工件 + 障碍 + 该 link 扫掠走廊 + anchor）。

    仅作肉眼核对用：调用前世界已摆好障碍。窗口阻塞，关闭后继续。
    传入 cfg+retract 时，额外画出机械臂在【初始位姿(retract)】下的整臂碰撞球（红色线框）。
    依赖 open3d；缺库或无显示时打印告警后跳过，不影响主流程。
    """
    try:
        import open3d as o3d
    except Exception as e:                                   # 缺库 → 跳过
        print(f"[debug-o3d] open3d 不可用，跳过可视化：{e}")
        return

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

    # 工件 mesh（浅灰）
    try:
        wp = o3d.io.read_triangle_mesh(workpiece_mesh.file_path)
        if not wp.is_empty():
            wp.transform(_pose_to_T(workpiece_mesh.pose))
            wp.compute_vertex_normals()
            wp.paint_uniform_color([0.7, 0.7, 0.7])
            geoms.append(wp)
    except Exception as e:
        print(f"[debug-o3d] 工件 mesh 加载失败：{e}")

    # 障碍原语（橙色）：Box→create_box（角在原点，需平移居中）；Tube→create_cylinder（已居中、轴 Z）
    for p in prims:
        if isinstance(p, ob.Box):
            dx, dy, dz = (float(v) for v in p.dims)
            g = o3d.geometry.TriangleMesh.create_box(dx, dy, dz)
            g.translate((-dx / 2, -dy / 2, -dz / 2))         # 居中到局部原点
        else:                                                # Tube
            g = o3d.geometry.TriangleMesh.create_cylinder(float(p.radius), float(p.height))
        g.transform(_pose_to_T(p.pose))
        g.compute_vertex_normals()
        g.paint_uniform_color([0.95, 0.55, 0.15])
        geoms.append(g)

    # 该 link 扫掠走廊球心（蓝色点云）+ anchor（绿色球）
    sph = per_wp.get(link)
    if sph is not None and len(sph):
        pts = np.asarray(sph, float).reshape(-1, 4)[:, :3]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.paint_uniform_color([0.1, 0.3, 0.9])
        geoms.append(pc)
    a = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    a.translate(np.asarray(anchor, float))
    a.compute_vertex_normals()
    a.paint_uniform_color([0.1, 0.8, 0.2])
    geoms.append(a)

    # 机械臂初始位姿(retract)下的整臂碰撞球（红色线框，按真实半径）
    n_init = 0
    if cfg is not None and retract is not None:
        try:
            init_wp, _ = compute_link_sweep(cfg, [list(retract)], cfg.collision_link_names)
            for ln, s in init_wp.items():
                for c in np.asarray(s, float)[0]:            # (S,4) 取唯一路点
                    cx, cy, cz, r = (float(v) for v in c)
                    if r <= 1e-4:
                        continue
                    ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
                    ball.translate((cx, cy, cz))
                    ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                    ls.paint_uniform_color([0.85, 0.1, 0.1])
                    geoms.append(ls)
                    n_init += 1
        except Exception as e:
            print(f"[debug-o3d] 初始碰撞球计算失败：{e}")

    # init_free 引导圆柱（青色线框）：轴过 base 原点(x=y=0)、半径 cyl_r、z∈[0, cyl_h]。
    # 障碍整只应在此圆柱外（push-to-clear 的目标）；线框不遮挡，便于看障碍是否贴外缘。
    if cfg is not None:
        try:
            cyl_r = float(cfg.init_free_cyl_radius)
            cyl_h = float(cfg.init_free_cyl_height)
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_r, height=cyl_h,
                                                            resolution=32)
            cyl.translate((0.0, 0.0, cyl_h / 2.0))           # create_cylinder 居中于原点 → 抬到 z∈[0,h]
            cyl_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cyl)
            cyl_ls.paint_uniform_color([0.0, 0.75, 0.75])
            geoms.append(cyl_ls)
        except Exception as e:
            print(f"[debug-o3d] init_free 圆柱可视化失败：{e}")

    print(f"[debug-o3d] 显示场景「{title}」(关闭窗口继续)；"
          f"障碍原语 {len(prims)} 个，走廊球 {0 if sph is None else len(sph)} 路点，"
          f"初始碰撞球 {n_init} 个。")
    o3d.visualization.draw_geometries(geoms, window_name=title)


# ------------------------------------------------------------------ 上层重试
def search_placement(handle, workpiece_mesh, per_wp, origin, link, otype,
                     traj_default, retract, goal_pose, metric, cfg, rng,
                     n_attempts: Optional[int] = None, debug_show: bool = False) -> dict:
    """对 (link, otype) 反复试放，失败按原因自适应改尺寸/角度/位置，至多 N 次。返回 result dict。"""
    op = cfg.obstacle_placement
    N = int(op["max_attempts"] if n_attempts is None else n_attempts)
    # size_scale_range 已改为按类型放 obstacle_params[otype]（无全局键）；此旧 demo 路径按 otype 取，缺省回退默认。
    s_lo, s_hi = ((op.get("obstacle_params", {}) or {}).get(otype, {}) or {}).get("size_scale_range", [0.6, 1.6])
    ang = float(op["angle_jitter_deg"])
    pj = float(op["pos_jitter_m"])
    th_lo, th_hi = op["thickness_range_m"]

    # 可变状态（随失败原因调整）
    size_scale = rng.uniform(s_lo, s_hi)
    last = None
    for attempt in range(N):
        rot_jitter = (rng.uniform(-ang, ang), rng.uniform(-ang, ang), rng.uniform(-ang, ang))
        pos_frac = rng.uniform(0.0, 1.0)
        jitter_vec = np.array([rng.uniform(-pj, pj) for _ in range(3)])
        thickness = rng.uniform(float(th_lo), float(th_hi))

        placed = place_in_corridor(per_wp, origin, link, otype, size_scale=size_scale,
                                   rot_jitter_deg=rot_jitter, pos_frac=pos_frac,
                                   jitter_vec=jitter_vec, thickness=thickness,
                                   cfg=cfg, goal_pos=goal_pose[0],
                                   workpiece_mesh=workpiece_mesh, retract=retract,
                                   debug_show=debug_show)
        if placed is None:                                # 落在排除区 → 换位置重采
            last = "excluded"
            continue
        prims, anchor, meta = placed
        buf = float(op.get("obstacle_buffer_m", 0.0))
        world_real = build_world(workpiece_mesh, prims, buffer_m=0.0)
        world_inflated = build_world(workpiece_mesh, prims, buffer_m=buf)
        # _debug_show_scene(workpiece_mesh, prims, per_wp, link, anchor,
        #                   cfg=cfg, retract=retract,
        #                   title=f"{link}/{otype} attempt{attempt + 1}")
        v = validate_scene(handle, traj_default, retract, goal_pose, metric, cfg,
                           world_real, world_inflated)
        last = v["fail_reason"]
        if v["ok"]:
            return dict(ok=True, link=link, otype=otype, attempt=attempt + 1, prims=prims,
                        anchor=list(map(float, anchor)), meta=meta,
                        detour_trajs=v["detour_trajs"], n_bad=v["n_bad"], dist=v["dist"])
        # 自适应：太弱→增大；无解→缩小；绕行不明显→略增大（换位置已每轮随机）
        if v["fail_reason"] == "too_weak":
            size_scale = min(s_hi * 1.5, size_scale * 1.25)
        elif v["fail_reason"] == "no_solution":
            size_scale = max(s_lo * 0.5, size_scale * 0.8)
        else:                                             # not_different
            size_scale = min(s_hi * 1.5, size_scale * 1.1)

    return dict(ok=False, link=link, otype=otype, attempt=N, last_reason=last,
                prims=None, anchor=None, meta=None, detour_trajs=[])


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
