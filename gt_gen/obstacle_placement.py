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


def _init_free_push_dir(prims, cfg, tangent) -> Optional[np.ndarray]:
    """外推方向：与 init_free 圆柱 z 段 [0,cyl_h] 重叠的各 prim 包围球心的水平质心的「外向径向」单位向量。

    · 无 prim 与圆柱 z 段重叠 → 返回 None（整只障碍在圆柱上/下方，无需外推）；
    · 质心几乎压在轴线上(hypot≈0) → 退化为「路径切向的水平垂线」（保证仍横挡走廊地往外推）。
    """
    cyl_h = float(cfg.init_free_cyl_height)
    pts = []
    for p in prims:
        c, rb = _prim_bounding_sphere(p)
        if c[2] + rb < 0.0 or c[2] - rb > cyl_h:         # z 向无重叠 → 与圆柱无交
            continue
        pts.append(c[:2])
    if not pts:
        return None
    ctr = np.mean(np.asarray(pts, float), axis=0)
    nrm = float(np.linalg.norm(ctr))
    if nrm > 1e-6:
        return np.array([ctr[0] / nrm, ctr[1] / nrm, 0.0])
    d = np.array([float(tangent[1]), -float(tangent[0]), 0.0])   # 切向的水平垂线
    nn = float(np.linalg.norm(d))
    return d / nn if nn > 1e-6 else np.array([1.0, 0.0, 0.0])


def _init_free_clear_distance(prims, cfg, u_xy) -> float:
    """沿水平单位方向 u 把整只障碍刚好推出 init_free 圆柱所需的【精确】平移距离 d≥0（一次解析求解）。

    平移只在 xy 内 → 各 prim 的 z 不变 → 与圆柱 z 段 [0,cyl_h] 的重叠集恒定，故可一次算准、无需迭代。
    对每个与圆柱 z 段重叠的 prim 包围球 (球心 c, 半径 rb)，令 R=cyl_r+rb，要求平移后水平距离 ≥ R：
        |c_xy + d·u|² ≥ R²  ⟺  d² + 2(c_xy·u)·d + (|c_xy|² − R²) ≥ 0
    记 b=c_xy·u、e=|c_xy|²−R²、判别式 disc=b²−e：
      · disc ≤ 0 → 沿 u 该 prim 对任意 d 都不侵入（含已在外且不会再进入），门限 0；
      · disc > 0 → 取较大根 d_hi = −b+√disc 为该 prim 的清空门限（d≥d_hi 即落在清空区）。
    d = max(0, max_i d_hi)。因 d≥每个 prim 的 d_hi，所有 prim 同时清空——一次到位、零迭代。
    """
    cyl_r = float(cfg.init_free_cyl_radius)
    cyl_h = float(cfg.init_free_cyl_height)
    u = np.asarray(u_xy, float)[:2]
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        return 0.0
    u = u / nu
    d_req = 0.0
    for p in prims:
        c, rb = _prim_bounding_sphere(p)
        if c[2] + rb < 0.0 or c[2] - rb > cyl_h:         # z 向无重叠 → 与圆柱无交
            continue
        cxy = c[:2]
        Rr = cyl_r + rb
        b = float(cxy @ u)
        e = float(cxy @ cxy) - Rr * Rr
        disc = b * b - e
        if disc <= 0.0:                                  # 沿 u 永不侵入
            continue
        d_hi = -b + math.sqrt(disc)
        if d_hi > d_req:
            d_req = d_hi
    return d_req


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


# ------------------------------------------------------------------ 底层放置（M=1）
def place_in_corridor(per_wp, origin, link, otype, *, size_scale, angle_deg, pos_frac,
                      jitter_vec, thickness, cfg, goal_pos,
                      workpiece_mesh=None, retract=None, debug_show: bool = False):
    """在 link 的扫掠走廊里放 1 个障碍（参数驱动、确定性）。返回 (prims, anchor, meta)；被排除区否决则返回 None。

    per_wp:
            per_wp: Dict[str, np.ndarray]
            # per_wp[link] : shape (T, S_link, 4)

            - key：link 名（"Link2"/"Link3"/.../"xiaoyu_accessory_link"，只含在 yml collision_spheres 里有定义的）；
            - value：(T, S, 4) 的数组
            - T = 默认轨迹路点数（如日志里的 67）；
            - S = 该 link 的碰撞球个数（如 Link2/Link3 各 12 个、accessory 41 个，见运行日志 {'Link2': (67, 12, 4), ...}）；
            - 最后一维 4 = [x, y, z, r]，即基座系下该球的球心坐标 + 半径。
    
    origin:
            origin: Dict[str, np.ndarray]
            # origin[link] : shape (T, 3)

            - key：同 per_wp，是各关键 link 名；
            - value：(T, 3) —— 该 link 坐标系原点在基座系下、沿默认轨迹每个路点的位置轨迹（T = 路点数，如 67）。

            注意区别：
            - per_wp[link] 是该 link 上所有碰撞球的球心+半径 (T,S,4) → 描述「管子的粗细/范围」；
            - origin[link] 只是该 link 原点这一个点的轨迹 (T,3) → 描述「管子的走向/中心线」。

  
    - pos_frac∈[0,1] 映射到 obstacle_placement.pos_t_window 内的路点 → 取该路点该 link 的扫掠球；
    - anchor = 这些球心均值 + jitter_vec；估走廊管半径 tube_r 与跨度 span(=2·tube_r·size_scale)；
    - 朝向：把障碍正面(+X)对齐该处路径切向，叠加 angle_deg 绕切向自转；
    - init_free：不再「anchor 落入就否决重抽」，而是按整只障碍的真实几何（逐 prim 包围球）解析地
      沿径向把障碍整体外推到刚好清空圆柱（保证障碍全部在 init 外）。仅当走廊中心本身就在圆柱内、
      或外推到挡不住走廊时才返回 None。
    - 排除：走廊中心落入 init_free 圆柱、或外推后距 goal < goal_clearance → 否决（返回 None）。
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

    # 走廊中心点本身就落在 init_free 内 → 该处走廊穿过初始空间，无法「既挡住走廊又整只在 init 外」，
    # 放弃此路点（上层换 pos_frac 即可；纯几何判断、无 cuRobo 调用，不浪费验证配额）。
    if _inside_init_free(anchor, cfg):
        return None
    if len(centers) == 0:
        return None

    # 走廊管半径：球心相对 anchor 的最大「球面外缘距」
    tube_r = float(np.max(np.linalg.norm(centers - anchor, axis=1) + radii))
    span = 2.0 * tube_r * float(size_scale)

    # 路径切向（link 原点的前后差分） 为什么要求路径切向: 核心目的：让障碍「横挡」走廊，而不是顺着走廊摆。
    op_org = origin[link]
    tangent = op_org[min(t + 1, T - 1)] - op_org[max(t - 1, 0)]
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
    rpy = _rpy_align_x_to(tangent, roll_deg=angle_deg)

    shape, local_off = _shape_for(otype, span, tube_r, cfg=cfg, th=thickness)
    # local_off 在 anchor 朝向系下偏置 anchor（让以角/底为锚的结构主体压在走廊上）
    anchor_eff = anchor + R.from_euler("xyz", rpy, degrees=True).apply(np.asarray(local_off, float))

    # —— 保证整只障碍（所有 prim 的真实几何，不只 anchor）都在 init_free 圆柱之外 ——
    # init_free 是已知解析圆柱、外推只在 xy（不改 z）→ 各 prim 与圆柱 z 段的重叠集恒定，
    # 故【一次解析求解】即可：选定外推方向 u 后，对每个 prim 解二次式取较大根、取最大值得到精确清空距离，
    # 一步推到位，不再「随机摆→撞 init→重抽」、也不迭代。
    prims = ob.build(otype, anchor_eff.tolist(), anchor_rpy_deg=tuple(rpy), **shape)
    # 外推前的「初始」摆放可视化（与 search_placement 里那次同款；用于肉眼比对外推前/后）
    # _debug_show_scene(workpiece_mesh, prims, per_wp, link, anchor_eff,
    #                     cfg=cfg, retract=retract,
    #                     title=f"{link}/{otype} 初始(未外推)")

    u = _init_free_push_dir(prims, cfg, tangent)             # None=无 prim 与圆柱 z 段重叠，无需外推
    push_total = _init_free_clear_distance(prims, cfg, u) if u is not None else 0.0
    if push_total > 1e-6:
        if push_total > tube_r + 0.5 * span:                 # 推太远→已离开走廊、挡不住路径 → 放弃此处
            return None
        anchor_eff = anchor_eff + u * (push_total + 1e-2)    # 一次推到位（+1mm 余量）
        prims = ob.build(otype, anchor_eff.tolist(), anchor_rpy_deg=tuple(rpy), **shape)
        # _debug_show_scene(workpiece_mesh, prims, per_wp, link, anchor_eff,
        #                     cfg=cfg, retract=retract,
        #                     title=f"{link}/{otype} 外推后(已清空)")

    # 外推后再校验与 goal 的间距（不要紧贴焊缝）
    if float(np.linalg.norm(anchor_eff - np.asarray(goal_pos, float))) < op["goal_clearance_m"]:
        return None

    meta = dict(link=link, otype=otype, anchor=anchor_eff.tolist(), tangent=tangent.tolist(),
                tube_r=tube_r, span=span, t=t, size_scale=float(size_scale),
                angle_deg=float(angle_deg), pos_frac=float(pos_frac),
                jitter=list(map(float, jitter_vec)), thickness=float(thickness),
                shape=shape, init_free_push=float(push_total))
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
    """
    validate_scene 是放置验证的核心裁判——判断一个已经摆好障碍的场景是否"合适"。它假设传入的 handle 世界已经切到「工件 + 障碍」状态，然后顺序检验 docs/障碍物位置.md
    的三条件，任一条不满足就提前返回失败（并附上失败原因，供上层 search_placement 自适应调参重试）：

    ① 默认路径必须被挡住（path_collides）
    逐路点对默认无障碍轨迹做 check_state,只要有 ≥1 个点在含障碍世界里不可行就算碰撞。
    - 不碰 → fail_reason="too_weak"（障碍太弱/没挡住，白放）。

    ② 必须仍有绕行解（detour_exists）
    在含障碍世界里重新 plan_to_pose(retract→goal)，能规划成功才行。
    - 规划不出 → fail_reason="no_solution"（障碍太强/把路堵死了，无解）。

    ③ 绕行必须明显不同于默认（is_detour_different）
    两条轨迹等长重采样后，逐路点关节最大偏差的峰值要 ≥ detour_min_joint_rad（默认 0.3 rad）。
    - 偏差太小 → fail_reason="not_different"（绕了等于没绕，障碍没造成实质性扰动）。

    三条全过 → ok=True，返回里带上 detour_traj(绕行轨迹)、n_bad(碰撞点数)、dist(关节偏差峰值)。

    返回的 dict 形如：
    {ok, fail_reason ∈ {None, too_weak, no_solution, not_different}, n_bad, dist, detour_traj}
    """
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
                     n_attempts: Optional[int] = None) -> dict:
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
                                   debug_show=True)
        if placed is None:                                # 落在排除区 → 换位置重采
            last = "excluded"
            continue
        prims, anchor, meta = placed
        world = build_world(workpiece_mesh, prims)
        handle.mg.update_world(world)
        # _debug_show_scene(workpiece_mesh, prims, per_wp, link, anchor,
        #                     cfg=cfg, retract=retract,
        #                     title=f"{link}/{otype} attempt{attempt + 1}")
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
