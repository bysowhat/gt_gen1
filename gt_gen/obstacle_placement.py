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
    cfg     : 配置对象；从 cfg.obstacle_placement 读 span/tube_r 的夹紧区间（span_clip_m /
              tube_r_clip_m），不在代码里写死。
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
    所以（区间从 cfg.obstacle_placement 读，便于逐机器/逐场景调，不写死在代码）：
      s  = clip(span,   span_clip_m[0],   span_clip_m[1])   —— 障碍主跨度：下界保证大到能横挡走廊，
                                                              上界防吞掉整个工作空间而无解。
      rt = clip(tube_r, tube_r_clip_m[0], tube_r_clip_m[1]) —— 杆/管/梁截面半径：下界防细到数值上可
                                                              忽略、从碰撞球缝隙漏过；上界防单根管成大圆柱。
    夹紧后这两个值才是「物理上合理、可复现」的形状输入。th(板/壁厚)同理由 thickness_range_m 随机取，
    既有多样性又被限定在合理薄板范围。
    """
    op = cfg.obstacle_placement
    s_lo, s_hi = op["span_clip_m"]
    rt_lo, rt_hi = op["tube_r_clip_m"]
    s = float(np.clip(span, float(s_lo), float(s_hi)))
    rt = float(np.clip(tube_r, float(rt_lo), float(rt_hi)))
    th = float(th)
    O = (0.0, 0.0, 0.0)
    table = {
        # plate: length=高(Z向), width=宽(Y向), thickness=板厚, tilt_deg=绕Y倾角
        #   → 一块 s×s 的薄板，法向(+X)对齐切向，正面横挡走廊。
        "plate":          (dict(length=s, width=s, thickness=th, tilt_deg=0.0), O),
        # l_bracket: length=两板边长, width=板宽, thickness=板厚；
        #   offset=(-s/2,0,-s/2) 把 └ 的角从结构角挪到走廊中心（默认锚在 L 的拐角）。
        "l_bracket":      (dict(length=s, width=s, thickness=th), (-s / 2, 0.0, -s / 2)),
        # u_channel: length=槽长(X), width=槽宽(Y,两侧板间距), height=侧板高(Z), thickness=壁厚；
        #   offset=(0,0,-s·0.35) 把开口槽底压到走廊（默认锚在底板）。
        "u_channel":      (dict(length=s, width=s, height=s * 0.7, thickness=th), (0.0, 0.0, -s * 0.35)),
        # open_box: size=(sx,sy,sz) 盒外形, wall=壁厚, open_face="front"=缺 +X 面（开口迎着切向）。
        "open_box":       (dict(size=(s, s, s), wall=th, open_face="front"), O),
        # pipe: length=管长(取 s·1.5 让管足够长跨过走廊), radius=管半径(=rt), axis="y"=管轴沿 Y
        #   （切向对齐后即垂直于路径，横拦走廊）。
        "pipe":           (dict(length=s * 1.5, radius=rt, axis="y"), O),
        # parallel_pipes: n=管数(3), length=管长(s·1.5), radius=rt, gap=管间距(s·0.5), axis="y"管轴,
        #   stack="z" 沿 Z 排开 → 一排平行管像护栏。
        "parallel_pipes": (dict(n=3, length=s * 1.5, radius=rt, gap=s * 0.5, axis="y", stack="z"), O),
        # crossed_pipes: length=管长(s·1.5), radius=rt, cross_deg=90 → 两管在 Y-Z 面内成 X 形交叉。
        "crossed_pipes":  (dict(length=s * 1.5, radius=rt, cross_deg=90.0), O),
        # box_beam: length=梁长(s·1.5), side=方截面边长(≥0.06, 取 rt·1.4), axis="y" 梁沿 Y 横拦。
        "box_beam":       (dict(length=s * 1.5, side=max(0.06, rt * 1.4), axis="y"), O),
        # rect_frame: width=框宽(Y), height=框高(Z), beam=边框方梁截面(≥0.06,取 rt) → 中间留孔的矩形框。
        "rect_frame":     (dict(width=s, height=s, beam=max(0.06, rt)), O),
        # gantry: span=两立柱间距, height=立柱高, post=立柱截面(≥0.06,取 rt), beam=横梁截面(≥0.08,取 rt)；
        #   offset=(0,0,-s/2) 把门架从「立柱底=锚」下移，使横梁/门洞罩住走廊（默认锚在地面平面）。
        "gantry":         (dict(span=s, height=s, post=max(0.06, rt), beam=max(0.08, rt)), (0.0, 0.0, -s / 2)),
        # braced_frame: width,height,beam 同 rect_frame, brace=对角斜撑截面(≥0.05,取 rt·0.8)
        #   → 框+一根斜梁破坏直穿。
        "braced_frame":   (dict(width=s, height=s, beam=max(0.06, rt), brace=max(0.05, rt * 0.8)), O),
        # tripod: height=三角高, base_half=底边半宽, rod=杆半径(≥0.05,取 rt) → 三根杆组成的竖立三角框。
        "tripod":         (dict(height=s, base_half=s * 0.5, rod=max(0.05, rt)), O),
        # steps: n=台阶数(3), rise=单级升高(s·0.33), run=单级进深(s·0.4), width=台阶宽(Y)；
        #   offset=(-s·0.6,0,-s/2) 把楼梯主体从「第一级底角=锚」挪到走廊中心。
        "steps":          (dict(n=3, rise=s * 0.33, run=s * 0.4, width=s), (-s * 0.6, 0.0, -s / 2)),
        # box_with_pipe: size,wall,open_face 同 open_box, pipe_radius=开口前横管半径(=rt) → 开口盒+挡管组合。
        "box_with_pipe":  (dict(size=(s, s, s), wall=th, open_face="front", pipe_radius=rt), O),
        # frame_with_brace: 同 braced_frame（width,height,beam,brace）——语义别名入口。
        "frame_with_brace": (dict(width=s, height=s, beam=max(0.06, rt), brace=max(0.05, rt * 0.8)), O),
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


def place_in_corridor(per_wp, origin, link, otype, *, size_scale, angle_deg, pos_frac,
                      jitter_vec, thickness, cfg, goal_pos,
                      workpiece_mesh=None, retract=None, debug_show: bool = False):
    """【解析解】直接算出与 init 圆柱无交、与扫掠并集有交的障碍 pose（无任何「放→测→换」试探）。

    算法（见对话确认）：
      1. 时间窗 [i0,i1] 内，对该 link 所有扫掠球心闭式算到 init 圆柱的带符号距离 sd（_signed_dist_cyl），
         剔除离 goal < goal_clearance 的；取 sd 最大的球心 c_k —— 离 init「最深的外部扫掠点」，给障碍
         最大尺寸余量。c_k 即障碍几何中心 → 障碍实体含 (c_k,r_k) 的球心 → 与扫掠并集必相交（构造性保证）。
      2. 朝向：对齐 c_k 处局部切向（origin 前后差分）+ angle_deg 自转。尺寸 desired=2·tube_r·size_scale。
      3. 自动缩到放下：若障碍以 c_k 为心的外接半径 R_b > sd(c_k)，按比例缩 span/tube_r/壁厚（同一 c_k，
         仅缩尺寸、不动位置，最多 3 轮收敛；_shape_for 的 clip 下界撑住时无法再缩）→ 保证整只在圆柱外。
      4. 兜底断言 _clears_init_free + _overlaps_sweep；通过则返回 (prims, anchor_eff, meta)，否则 None
         （仅当尺寸 clip 下界 > sd(c_k)，即该 link/类型在最深点都塞不进 init 外余量——上层换 link/类型）。

    注：位置由几何唯一确定（argmax sd），故 pos_frac / jitter_vec 不参与定位（保留入参仅为签名兼容）。
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

    # —— 1. 闭式 argmax sd：窗口内离 init 圆柱最深、且离 goal 够远的扫掠球心 c_k ——
    win = sph[i0:i1 + 1]                                 # (Tw,S,4)
    centers = win[..., :3]                               # (Tw,S,3)
    sd = _signed_dist_cyl(centers, cfg)                  # (Tw,S)
    goal = np.asarray(goal_pos, float)
    gc = float(op["goal_clearance_m"])
    far = np.linalg.norm(centers - goal, axis=-1) >= gc  # 离焊缝够远
    sd_m = np.where(far, sd, -np.inf)
    if not np.isfinite(sd_m).any() or float(np.max(sd_m)) <= 0.0:
        return None                                      # 窗口内无「圆柱外且离 goal 够远」的扫掠点
    tw, s_idx = np.unravel_index(int(np.argmax(sd_m)), sd_m.shape)
    t_k = i0 + int(tw)
    c_k = centers[tw, s_idx].astype(float).copy()
    sd_k = float(sd[tw, s_idx])

    # —— 2. 局部走廊管半径（t_k 簇，相对 c_k）+ 切向 → 朝向 ——
    ck_centers = sph[t_k, :, :3]
    ck_radii = sph[t_k, :, 3]
    tube_r0 = float(np.max(np.linalg.norm(ck_centers - c_k, axis=1) + ck_radii))
    tangent = org[min(t_k + 1, T - 1)] - org[max(t_k - 1, 0)]
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
    rpy = _rpy_align_x_to(tangent, roll_deg=angle_deg)
    Rm = R.from_euler("xyz", rpy, degrees=True)

    # —— 3. 闭式造障碍 + 自动缩到放下（同一 c_k，仅缩尺寸）——
    margin = 1.0 - 1e-2                                  # 留 1% 余量，避免贴壁数值误差
    scale = 1.0
    prims = None
    anchor_eff = c_k
    shape = {}
    for _ in range(3):
        span = 2.0 * tube_r0 * float(size_scale) * scale
        shape, local_off = _shape_for(otype, span, tube_r0 * scale, cfg=cfg, th=float(thickness) * scale)
        anchor_eff = c_k + Rm.apply(np.asarray(local_off, float))   # body 中心落在 c_k
        prims = ob.build(otype, anchor_eff.tolist(), anchor_rpy_deg=tuple(rpy), **shape)
        R_b = _bounding_radius_about(prims, c_k)
        if R_b <= sd_k * margin:
            break
        scale *= (sd_k * margin) / max(R_b, 1e-6)        # 闭式缩放因子（线性几何一轮到位，floor 时多收敛一两轮）

    # —— 4. 兜底断言（构造性保证下应恒成立；clip 下界撑住放不下时 → None 由上层换 link/类型）——
    if prims is None or not _clears_init_free(prims, cfg) or not _overlaps_sweep(prims, sph):
        return None
    if float(np.linalg.norm(anchor_eff - goal)) < gc:    # body 偏置后再核一次 goal 间距
        return None

    meta = dict(link=link, otype=otype, anchor=anchor_eff.tolist(), tangent=tangent.tolist(),
                tube_r=float(tube_r0 * scale), span=float(2.0 * tube_r0 * size_scale * scale), t=int(t_k),
                size_scale=float(size_scale), shrink_scale=float(scale), sd_k=float(sd_k),
                angle_deg=float(angle_deg), thickness=float(thickness) * float(scale),
                shape=shape, init_free_push=0.0)
    # _debug_show_scene(workpiece_mesh, prims, per_wp, link, anchor_eff,
    #                     cfg=cfg, retract=retract, title=f"{link}/{otype} t={t_k}(解析解)")
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
    s_lo, s_hi = op["size_scale_range"]
    ang = float(op["angle_jitter_deg"])
    pj = float(op["pos_jitter_m"])
    th_lo, th_hi = op["thickness_range_m"]

    # 可变状态（随失败原因调整）
    size_scale = rng.uniform(s_lo, s_hi)
    last = None
    for attempt in range(N):
        angle_deg = rng.uniform(-ang, ang)
        pos_frac = rng.uniform(0.0, 1.0)
        jitter_vec = np.array([rng.uniform(-pj, pj) for _ in range(3)])
        thickness = rng.uniform(float(th_lo), float(th_hi))

        placed = place_in_corridor(per_wp, origin, link, otype, size_scale=size_scale,
                                   angle_deg=angle_deg, pos_frac=pos_frac,
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
