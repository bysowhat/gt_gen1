"""独立脚本：参数化【水平直角(角)焊缝】+【焊枪(末端工具)碰撞球】open3d 可视化。

【第一步 / 当前逻辑】——先确认焊枪位姿是否正确：
不做 IK、不做规划、不画整臂。把「焊枪要求位姿」直接摆好，只画：
  · 焊枪(xiaoyu_accessory_link)在该位姿下的碰撞球（青色线框）+ tip 坐标系三轴；
  · 这条直角焊缝（红色圆柱=焊缝线 + 地板/立墙两面片 + bisector 蓝线 + 端点小球）+ 直角工件两面。

焊枪要求位姿（末端 xiaoyu_tip_link 的位姿）：
  · 位置 = 焊缝中点(mid)（可选沿 bisector 外移 standoff）；
  · 朝向 = 显式确定滚转的基准 R0，再绕 tip 局部 +y、+z 倾斜 (rot_y_deg, rot_z_deg)：
    R0: tip +X = -bisector（指向内角）、tip +Z = 焊缝线方向、tip +Y = +Z×+X（两面对称面内）。
    rot_y 绕 +Y = 沿焊缝前后倾（行走角）；rot_z 绕 +Z(=焊缝线) = 两面间左右摆（工作角，
    rot_z=0 落在角平分面【正中】对称）。R = R0 @ Ry(rot_y) @ Rz(rot_z)。
    注：tip_link 相对焊枪本体有固定安装角(≈54°)，喷嘴物理轴在 X-Z 面偏 +Z 约 36°（纯行走角
    偏移），rot_y=0 时喷嘴不与 bisector 共线——需 rot_y≈+36° 把喷嘴转到正对角平分线。

焊枪碰撞球来自 cuRobo 机器人 yml 的 collision_spheres['xiaoyu_accessory_link']（在 accessory 系，
= flange 系）。tip_link 相对 flange 是固定关节（URDF 读出），据此把 accessory 系的碰撞球变到
「以 tip 位姿为准」的 base 系——tip_link 原点恰落在焊枪尖（焊缝中点），焊枪体沿 tip 局部 -X 向后延伸。

【下一步（已实现，--solve-ik 开启）】：对该 tip 目标位姿反解机械臂关节角(cuRobo IKSolver)，
  取位置误差最小的解，FK 验证并叠加整臂碰撞球(灰色)。默认关闭以保留快速确认位姿模式。

焊缝几何约定与 scripts/plan_init_pose.py:load_welds 一致：
  p0_world/p1_world/mid_world/bisector_world/boundary_dirs(2,3)，bisector=unit(d1+d2)。

基准姿态（旋转/平移前，local 系）：
  焊缝线沿 +X（p0=(-L/2,0,0) → p1=(+L/2,0,0)，mid=原点）；
  面A(地板) 水平、法向 +Z、边界方向 d1=+Y；面B(立墙) 竖直、法向 +Y、边界方向 d2=+Z；
  bisector = unit(+Y+ +Z) = (0,1,1)/√2 —— 45° 斜上开口。
摆位：p_base = Rz(yaw) @ p_local + xyz（绕世界 +Z 水平旋转后平移）。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_right_angle_seam.py \
        --yaw-deg 30 --xyz 0.8 0.0 0.2 --seam-len 0.3 --rot-y-deg 0 --rot-z-deg 0
无显示器自检：加 --headless（不开窗，仅打印焊缝坐标 + tip 目标位姿 + 焊枪碰撞球数 + VIZ_DONE）。
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/ 供 import plan_init_pose


# ---------------------------------------------------------------- 几何
def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def make_right_angle_seam(seam_len: float, face_size: float) -> dict:
    """基准姿态(local 系)的水平直角焊缝：地面 + 立墙、开口朝斜上 45°。

    返回 dict（均 local 系）：p0/p1/mid、d1/d2(两面边界方向)、bisector、
    faceA/faceB（各 (4,3) 矩形面片四角，用于画出直角两面）。
    """
    L = float(seam_len)
    s = float(face_size)
    hx = L / 2.0
    p0 = np.array([-hx, 0.0, 0.0], float)       # 焊缝线沿 +X
    p1 = np.array([+hx, 0.0, 0.0], float)
    mid = 0.5 * (p0 + p1)

    d1 = np.array([0.0, 1.0, 0.0], float)       # 地板面边界方向 +Y
    d2 = np.array([0.0, 0.0, 1.0], float)       # 立墙面边界方向 +Z
    bisector = _unit(d1 + d2)                   # (0,1,1)/√2，45° 斜上开口

    # 面A(地板)：x∈[-hx,hx], y∈[0,s], z=0（法向 +Z）
    faceA = np.array([[-hx, 0.0, 0.0], [hx, 0.0, 0.0], [hx, s, 0.0], [-hx, s, 0.0]], float)
    # 面B(立墙)：x∈[-hx,hx], y=0, z∈[0,s]（法向 +Y）
    faceB = np.array([[-hx, 0.0, 0.0], [hx, 0.0, 0.0], [hx, 0.0, s], [-hx, 0.0, s]], float)

    return dict(p0=p0, p1=p1, mid=mid, d1=d1, d2=d2, bisector=bisector,
                faceA=faceA, faceB=faceB)


def seam_transform(yaw_deg: float, xyz) -> np.ndarray:
    """摆位变换 4x4：先绕世界 +Z 轴转 yaw_deg（水平旋转），再平移到 xyz。"""
    a = np.radians(float(yaw_deg))
    ca, sa = np.cos(a), np.sin(a)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], float)
    T = np.eye(4)
    T[:3, :3] = Rz
    T[:3, 3] = np.asarray(xyz, float)
    return T


def transformed_seam(seam_local: dict, T: np.ndarray) -> dict:
    """把 local 焊缝几何经 T 变到 base 系。点乘 R+t，方向向量只乘 R。"""
    R, t = T[:3, :3], T[:3, 3]

    def _pt(P):
        return np.asarray(P, float) @ R.T + t

    def _dir(v):
        return R @ np.asarray(v, float)

    return dict(
        p0=_pt(seam_local["p0"]), p1=_pt(seam_local["p1"]), mid=_pt(seam_local["mid"]),
        d1=_dir(seam_local["d1"]), d2=_dir(seam_local["d2"]),
        bisector=_dir(seam_local["bisector"]),
        faceA=_pt(seam_local["faceA"]), faceB=_pt(seam_local["faceB"]),
    )


# ---------------------------------------------------------------- 焊枪 tip 目标位姿
def tip_pose_from_seam(mid_base, bisector_base, seam_dir_base, rot_y_deg: float, rot_z_deg: float,
                       standoff_m: float = 0.0):
    """由焊缝中点(位置) + 角平分线 + 焊缝线方向(朝向基准) + rot_y/rot_z 倾斜，构 tip 目标位姿 (R,p)。

    基准朝向 R0（rot_y=rot_z=0）由三根轴【显式确定滚转】——不用最小旋转，避免绕 -bisector
    的滚转任意化导致 rot_z 摆动平面歪斜：
      tip +X = -bisector（指向直角内侧，即对准角平分线反方向）；
      tip +Z = 焊缝线方向 seam_dir（行走方向）；
      tip +Y = +Z × +X（在两面对称面内、⟂ 焊缝线）。
    再【绕 tip 局部 +y、+z 轴】各倾斜 (β,γ)：R = R0 @ Ry(β) @ Rz(γ)。
      · rot_y 绕 +Y = 沿焊缝线前后倾（行走/前进角）；
      · rot_z 绕 +Z(=焊缝线) = 在两个面之间左右摆（工作角），rot_z=0 正好落在角平分面【正中】对称。
    位置 = mid + standoff·bisector（standoff_m=0 时正好落在焊缝中点）。

    注：对准的是 tip 坐标轴（非喷嘴实体轴）——tip_link 相对焊枪本体有固定安装角(≈54°)，
    喷嘴物理轴在 tip 系里落在 X-Z 平面内、偏 +Z 约 36°（纯行走角偏移），需靠 rot_y≈+36° 校正。

    返回 (R(3,3), p(3,))：tip_link 在 base 系的姿态与原点。
    """
    from scipy.spatial.transform import Rotation as Rsp

    mid = np.asarray(mid_base, float)
    bis = _unit(bisector_base)
    x_axis = -bis                                    # tip +X → -bisector（指向内角）
    # tip +Z 对齐焊缝线（先对 -bisector 正交化，去掉与 x_axis 的分量，防数值不正交）
    z_raw = _unit(seam_dir_base)
    z_axis = _unit(z_raw - np.dot(z_raw, x_axis) * x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))         # 右手系：Y = Z × X
    z_axis = np.cross(x_axis, y_axis)                # 回正交化，保证 R0 正交右手
    R0 = np.column_stack([x_axis, y_axis, z_axis])   # 列 = tip 三轴在 base 系
    Ry = Rsp.from_euler("y", float(rot_y_deg), degrees=True).as_matrix()
    Rz = Rsp.from_euler("z", float(rot_z_deg), degrees=True).as_matrix()
    R = R0 @ Ry @ Rz
    p = mid + float(standoff_m) * bis
    return R, p


# ---------------------------------------------------------------- 焊枪碰撞球（从 yml + URDF 固定变换）
def _rpy_to_R(rpy) -> np.ndarray:
    """URDF rpy(弧度) → 旋转矩阵（URDF 约定：外旋 xyz = Rz(yaw)·Ry(pitch)·Rx(roll)）。"""
    from scipy.spatial.transform import Rotation as Rsp
    return Rsp.from_euler("xyz", [float(rpy[0]), float(rpy[1]), float(rpy[2])]).as_matrix()


def _urdf_joint_origin(urdf_root, child_link: str):
    """在 URDF 里找 child==child_link 的关节，返回其相对 parent 的 (R,t,parent_link)。"""
    for j in urdf_root.findall("joint"):
        child = j.find("child")
        if child is None or child.get("link") != child_link:
            continue
        parent = j.find("parent").get("link")
        origin = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        return _rpy_to_R(rpy), np.asarray(xyz, float), parent
    raise KeyError(f"URDF 里找不到 child={child_link} 的关节")


def load_torch_model(cfg):
    """读焊枪碰撞球(accessory 系) + tip_link↔accessory 固定变换。

    返回 dict：
      local_spheres : list[(center(3,), radius)]，accessory_link 系（半径>1e-4 才保留）；
      R_tip_acc(3,3)/t_tip_acc(3,) : accessory 系点 p_a → tip 系点 = R_tip_acc @ p_a + t_tip_acc。

    accessory_link 与 tip_link 均为 flange 的固定子；由 URDF 两关节 origin 求
    T_tip_acc = inv(T_flange_tip) · T_flange_acc。
    """
    kin = cfg.robot_cfg["robot_cfg"]["kinematics"]
    ee_link = kin["ee_link"]                                  # xiaoyu_tip_link
    spheres_def = kin["collision_spheres"]["xiaoyu_accessory_link"]
    local = [(np.asarray(s["center"], float), float(s["radius"]))
             for s in spheres_def if float(s["radius"]) > 1e-4]

    urdf_root = ET.parse(kin["urdf_path"]).getroot()
    R_ft, t_ft, parent_tip = _urdf_joint_origin(urdf_root, ee_link)              # flange→tip
    R_fa, t_fa, parent_acc = _urdf_joint_origin(urdf_root, "xiaoyu_accessory_link")  # flange→acc
    if parent_tip != parent_acc:
        print(f"[torch] 警告：tip 的父({parent_tip}) 与 accessory 的父({parent_acc}) 不同，"
              f"固定变换假设可能不成立")
    # T_tip_acc = inv(T_flange_tip) @ T_flange_acc
    #   p_tip = R_ft^T (p_flange - t_ft);  p_flange = R_fa p_a + t_fa
    R_tip_acc = R_ft.T @ R_fa
    t_tip_acc = R_ft.T @ (t_fa - t_ft)
    # 焊枪喷嘴物理轴 = accessory +Z（那段直球沿 +Z、指向焊枪尖/指向工件方向），在 tip 系的方向。
    nozzle_tip = R_tip_acc @ np.array([0.0, 0.0, 1.0])
    nozzle_tip = nozzle_tip / (np.linalg.norm(nozzle_tip) + 1e-12)
    return dict(local_spheres=local, R_tip_acc=R_tip_acc, t_tip_acc=t_tip_acc,
                nozzle_tip=nozzle_tip, ee_link=ee_link)


def torch_spheres_world(torch_model: dict, R_tip: np.ndarray, p_tip: np.ndarray):
    """把焊枪碰撞球（accessory 系）按 tip 目标位姿 (R_tip,p_tip) 变到 base 系。

    accessory 点 p_a → tip 系：p_t = R_tip_acc @ p_a + t_tip_acc；
    tip 系 → base：p_w = R_tip @ p_t + p_tip。
    返回 list[(cx,cy,cz,r)]。
    """
    R_ta, t_ta = torch_model["R_tip_acc"], torch_model["t_tip_acc"]
    R_tip = np.asarray(R_tip, float)
    p_tip = np.asarray(p_tip, float)
    out = []
    for c, r in torch_model["local_spheres"]:
        p_t = R_ta @ c + t_ta                # accessory → tip
        p_w = R_tip @ p_t + p_tip            # tip → base
        out.append((float(p_w[0]), float(p_w[1]), float(p_w[2]), float(r)))
    return out


# ---------------------------------------------------------------- 反解关节角 (IK) + 整臂碰撞球
def _matrix_to_quat_wxyz(R):
    """3x3 旋转矩阵 → 四元数 wxyz（cuRobo 约定）。"""
    from scipy.spatial.transform import Rotation as Rsp
    q = Rsp.from_matrix(np.asarray(R, float)).as_quat()   # scipy: xyzw
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def solve_arm_ik(cfg, R_tip, p_tip, return_seeds: int = 20):
    """对 tip 目标位姿 (R_tip, p_tip)（base 系）反解 UR12e 关节角（cuRobo 多种子 IKSolver）。

    返回 dict：
      ok           : 是否有成功解；
      q            : 最优解关节角 list（位置误差最小）；
      ee_pos/ee_quat : 该解 FK 出的末端(xiaoyu_tip_link)位姿，用于验证；
      spheres      : 整臂碰撞球 (S,4) xyz+r（base 系），供可视化；
      pos_err_mm   : 位置误差(mm)；rot_err_deg : 朝向误差(度)；
      n_success    : 过阈值解个数；handle : cuRobo handle（复用）。
    """
    from gt_gen import compat  # noqa: F401  warp shim 须在 import curobo 前
    from gt_gen import curobo_iface as ci
    import torch

    quat = _matrix_to_quat_wxyz(R_tip)
    print("[ik] 构建 cuRobo handle（首次 warmup 略慢）…")
    handle = ci.init_curobo(cfg)
    res = ci.solve_ik(handle, (list(map(float, p_tip)), quat), return_seeds=return_seeds)
    sols = ci.ik_configs(handle, res)
    if not sols:
        return dict(ok=False, handle=handle, n_success=0)

    q, pos_err = sols[0]                                    # 位置误差最小的解
    qt = torch.tensor([q], dtype=torch.float32, device="cuda")
    st = handle.mg.kinematics.get_state(qt)
    ee_pos = st.ee_position[0].detach().cpu().numpy()
    ee_quat = st.ee_quaternion[0].detach().cpu().numpy()
    sph = st.link_spheres_tensor[0].detach().cpu().numpy()  # (S,4) base 系

    # 朝向误差：FK 出的旋转 vs 目标 R_tip
    from scipy.spatial.transform import Rotation as Rsp
    R_fk = Rsp.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_matrix()
    cos_ang = (np.trace(np.asarray(R_tip, float).T @ R_fk) - 1.0) / 2.0
    rot_err_deg = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    return dict(ok=True, q=list(map(float, q)), ee_pos=ee_pos, ee_quat=ee_quat,
                spheres=sph, pos_err_mm=float(pos_err) * 1000.0, rot_err_deg=rot_err_deg,
                n_success=len(sols), handle=handle)


def arm_geoms(spheres, color=(0.6, 0.6, 0.6)):
    """整臂碰撞球 → open3d 实心球 mesh 列表（灰色，半径<=1e-4 或非有限的跳过）。"""
    import open3d as o3d
    geoms = []
    for s in np.asarray(spheres, float):
        r = float(s[3])
        if r <= 1e-4 or not np.isfinite(s[:3]).all():
            continue
        b = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
        b.translate(tuple(float(v) for v in s[:3]))
        b.compute_vertex_normals()
        b.paint_uniform_color(list(color))
        geoms.append(b)
    return geoms


# ---------------------------------------------------------------- 批量扫描 (sweep)：多位姿反解 + 存盘
def _grid_inc(lo, hi, step):
    """闭区间 [lo,hi] 按 step 均匀采样（含端点）。step<=0 或 hi<=lo → 只取 lo。"""
    lo, hi, step = float(lo), float(hi), abs(float(step))
    if step <= 1e-9 or hi <= lo:
        return np.array([lo], float)
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + np.arange(n) * step


def sample_positions(ee_xy_range, ee_z_range, step):
    """在 base 系采样焊缝中点位置：x,y∈[-xy_hi,xy_hi]、z∈ee_z_range 打网格(间隔 step)，
    只保留 xy 平面到原点径向距离 ∈[xy_lo,xy_hi] 的点（环形，语义同 ee_xy_range_m）。返回 (P,3)。"""
    xy_lo, xy_hi = float(ee_xy_range[0]), float(ee_xy_range[1])
    z_lo, z_hi = float(ee_z_range[0]), float(ee_z_range[1])
    xs = _grid_inc(-xy_hi, xy_hi, step)
    ys = _grid_inc(-xy_hi, xy_hi, step)
    zs = _grid_inc(z_lo, z_hi, step)
    pts = []
    for z in zs:
        for x in xs:
            for y in ys:
                r = float(np.hypot(x, y))
                if xy_lo - 1e-9 <= r <= xy_hi + 1e-9:
                    pts.append([float(x), float(y), float(z)])
    return np.asarray(pts, float).reshape(-1, 3)


def build_sweep_poses(seam_local, positions, yaws, rys, rzs, standoff_m):
    """笛卡尔积 位置×yaw×rot_y×rot_z → 一批 tip 目标位姿。

    关键提速：基准朝向 R0 只依赖 yaw、绕局部轴的 Ry/Rz 只依赖 (rot_y,rot_z)，故
    R_tip=R0@Ry@Rz 只随 (yaw,rot_y,rot_z) 变、与位置无关——先对每个 yaw 预算 R0/bisector/
    seam_dir，再对每个 (rot_y,rot_z) 预算 R_tip 与 quat，最后套所有位置只改平移 p_tip。

    返回并行数组：p_tip(N,3) 、quat_wxyz(N,4)、xyz(N,3 焊缝中点)、yaw(N,)、ry(N,)、rz(N,)。
    """
    from scipy.spatial.transform import Rotation as Rsp
    seam_dir_local = _unit(seam_local["p1"] - seam_local["p0"])
    p_list, q_list, xyz_list, yaw_list, ry_list, rz_list = [], [], [], [], [], []
    for yaw in yaws:
        Rz_yaw = seam_transform(yaw, [0.0, 0.0, 0.0])[:3, :3]     # 只取旋转，作用于方向向量
        bis = Rz_yaw @ seam_local["bisector"]
        sdir = Rz_yaw @ seam_dir_local
        bis_u = _unit(bis)
        # 每个 (ry,rz) 预算 R_tip、quat（与位置无关）
        pre = []
        for ry in rys:
            for rz in rzs:
                R_tip, _ = tip_pose_from_seam([0.0, 0.0, 0.0], bis_u, sdir,
                                              float(ry), float(rz), standoff_m=0.0)
                quat = _matrix_to_quat_wxyz(R_tip)
                pre.append((float(ry), float(rz), quat))
        # 套所有位置：p_tip = 焊缝中点 + standoff·bisector
        for pos in positions:
            pos = np.asarray(pos, float)
            p_tip = pos + float(standoff_m) * bis_u
            for ry, rz, quat in pre:
                p_list.append(p_tip.tolist())
                q_list.append(quat)
                xyz_list.append(pos.tolist())
                yaw_list.append(float(yaw))
                ry_list.append(ry)
                rz_list.append(rz)
    return (np.asarray(p_list, float).reshape(-1, 3),
            np.asarray(q_list, float).reshape(-1, 4),
            np.asarray(xyz_list, float).reshape(-1, 3),
            np.asarray(yaw_list, float), np.asarray(ry_list, float), np.asarray(rz_list, float))


def solve_ik_batch(handle, pos_arr, quat_arr, return_seeds: int, chunk: int):
    """对一批 tip 目标位姿批量反解（cuRobo IKSolver.solve_batch，按 chunk 分块控显存）。

    返回 (ok(N,) bool, best_q(N,dof), best_err(N,))：每个目标取「有成功解且位置误差最小」的解；
    无解处 ok=False、best_q=nan、best_err=inf。
    """
    from curobo.types.math import Pose
    ta = handle.ta
    N = int(pos_arr.shape[0])
    dof = len(handle.joint_names)
    ok = np.zeros(N, bool)
    best_q = np.full((N, dof), np.nan, float)
    best_err = np.full(N, np.inf, float)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        m = e - s
        pose = Pose(position=ta.to_device(pos_arr[s:e].tolist()).view(-1, 3),
                    quaternion=ta.to_device(quat_arr[s:e].tolist()).view(-1, 4))
        res = handle.ik.solve_batch(pose, return_seeds=return_seeds)
        succ = res.success.detach().cpu().numpy().reshape(m, return_seeds)
        sol = res.solution.detach().cpu().numpy().reshape(m, return_seeds, dof)
        err = res.position_error.detach().cpu().numpy().reshape(m, return_seeds)
        for i in range(m):
            mask = succ[i]
            if bool(mask.any()):
                j = int(np.argmin(np.where(mask, err[i], np.inf)))
                ok[s + i] = True
                best_q[s + i] = sol[i, j]
                best_err[s + i] = float(err[i, j])
        print(f"[sweep] IK 进度 {e}/{N}  累计有解 {int(ok[:e].sum())}")
    return ok, best_q, best_err


def run_sweep(cfg, args):
    """批量扫描：位置(环形)×yaw×rot_y×rot_z 组合所有焊枪位姿，批量反解，有解者存 .pt。"""
    from gt_gen import compat  # noqa: F401  warp shim 须在 import curobo 前
    from gt_gen import curobo_iface as ci
    import torch

    seam_local = make_right_angle_seam(args.seam_len, args.face_size)
    positions = sample_positions(args.ee_xy_range, args.ee_z_range, args.xyz_step)
    yaws = [float(v) for v in args.yaw_deg]
    rys = _grid_inc(args.rot_y_range[0], args.rot_y_range[1], args.rot_y_step)
    rzs = _grid_inc(args.rot_z_range[0], args.rot_z_range[1], args.rot_z_step)
    standoff_m = args.standoff_cm / 100.0

    print("=" * 60)
    print(f"[sweep] 位置点 {positions.shape[0]} 个  "
          f"(xy径向∈{list(args.ee_xy_range)}  z∈{list(args.ee_z_range)}  step={args.xyz_step}m)")
    print(f"[sweep] yaw {yaws}  rot_y {np.round(rys, 1).tolist()}  rot_z {np.round(rzs, 1).tolist()}")

    p_arr, q_arr, xyz_arr, yaw_arr, ry_arr, rz_arr = build_sweep_poses(
        seam_local, positions, yaws, rys, rzs, standoff_m)
    N = p_arr.shape[0]
    print(f"[sweep] 焊枪位姿总数 N = {positions.shape[0]}×{len(yaws)}×{len(rys)}×{len(rzs)} = {N}")

    print("[sweep] 构建 cuRobo handle（首次 warmup 略慢）…")
    handle = ci.init_curobo(cfg)
    ok, best_q, best_err = solve_ik_batch(handle, p_arr, q_arr,
                                          return_seeds=args.ik_seeds, chunk=args.sweep_chunk)

    idx = np.where(ok)[0]
    print("=" * 60)
    print(f"[sweep] 完成：有解 {idx.size}/{N}（{100.0 * idx.size / max(N, 1):.1f}%）")

    data = dict(
        joint_names=list(handle.joint_names),
        xyz=torch.tensor(xyz_arr[idx], dtype=torch.float32),          # 焊缝中点(base 系)
        p_tip=torch.tensor(p_arr[idx], dtype=torch.float32),          # tip 落点(=中点+standoff·bisector)
        quat_wxyz=torch.tensor(q_arr[idx], dtype=torch.float32),      # tip 目标朝向
        yaw_deg=torch.tensor(yaw_arr[idx], dtype=torch.float32),
        rot_y_deg=torch.tensor(ry_arr[idx], dtype=torch.float32),
        rot_z_deg=torch.tensor(rz_arr[idx], dtype=torch.float32),
        q=torch.tensor(best_q[idx], dtype=torch.float32),             # 关节角(rad)
        pos_err_mm=torch.tensor(best_err[idx] * 1000.0, dtype=torch.float32),
        meta=dict(
            ee_xy_range_m=list(map(float, args.ee_xy_range)),
            ee_z_range_m=list(map(float, args.ee_z_range)),
            xyz_step_m=float(args.xyz_step),
            yaw_deg=yaws,
            rot_y_range=list(map(float, args.rot_y_range)), rot_y_step=float(args.rot_y_step),
            rot_z_range=list(map(float, args.rot_z_range)), rot_z_step=float(args.rot_z_step),
            standoff_cm=float(args.standoff_cm), seam_len=float(args.seam_len),
            n_total=int(N), n_solved=int(idx.size), ik_seeds=int(args.ik_seeds),
        ),
    )
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), out_path)
    torch.save(data, out_path)
    print(f"[sweep] 已保存 {idx.size} 个有解位姿 → {out_path}")
    print("VIZ_DONE")


def _quat_wxyz_to_matrix(q):
    """四元数 wxyz(cuRobo 约定) → 3x3 旋转矩阵。"""
    from scipy.spatial.transform import Rotation as Rsp
    q = np.asarray(q, float)
    return Rsp.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy 收 xyzw


def _fk_arm_spheres(handle, q):
    """对关节角 q 做 FK，返回整臂碰撞球 (S,4) xyz+r（base 系）。"""
    import torch
    qt = torch.tensor([list(map(float, q))], dtype=torch.float32, device="cuda")
    st = handle.mg.kinematics.get_state(qt)
    return st.link_spheres_tensor[0].detach().cpu().numpy()


def range_frame_geoms(xy_range, z_range, color=(1.0, 0.85, 0.0)):
    """采样空间线框：xy 径向∈[xy_lo,xy_hi]、z∈[z_lo,z_hi] 的【环形柱体】。

    画内/外两个半径各在 z_lo、z_hi 高度的圆（4 圈）+ 竖直连线，勾出 ee_xy_range/
    ee_z_range 规定的采样区域（金黄线框）。返回 [o3d.LineSet]。"""
    import open3d as o3d
    r_lo, r_hi = float(xy_range[0]), float(xy_range[1])
    z_lo, z_hi = float(z_range[0]), float(z_range[1])
    K = 72                                       # 每圈采样点数（够圆滑）
    pts, lines = [], []

    def add_circle(r, z):
        base = len(pts)
        for k in range(K):
            t = 2.0 * np.pi * k / K
            pts.append([r * np.cos(t), r * np.sin(t), z])
        for k in range(K):
            lines.append([base + k, base + (k + 1) % K])
        return base

    circ = {(r, z): add_circle(r, z) for r in (r_lo, r_hi) for z in (z_lo, z_hi)}
    step = max(1, K // 8)                         # 每 45° 一根竖直连线
    for r in (r_lo, r_hi):
        b_lo, b_hi = circ[(r, z_lo)], circ[(r, z_hi)]
        for k in range(0, K, step):
            lines.append([b_lo + k, b_hi + k])

    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.asarray(pts, float)),
        o3d.utility.Vector2iVector(np.asarray(lines, int)))
    ls.paint_uniform_color(list(color))
    return [ls]


def run_viz_sweep(cfg, args):
    """逐个可视化 .pt 中保存的【所有有解机械臂位姿】：一次一个，按 C 看下一个。

    每个位姿画：整臂碰撞球(灰，由 q 做 FK) + 焊枪碰撞球(青，按 tip 目标位姿) +
    该焊缝(红线/两面/bisector) + tip 坐标系三轴。base 坐标系常驻。
    """
    from gt_gen import compat  # noqa: F401  warp shim 须在 import curobo 前
    from gt_gen import curobo_iface as ci
    import open3d as o3d
    import torch

    in_path = args.out
    if not os.path.isabs(in_path):
        in_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), in_path)
    if not os.path.isfile(in_path):
        print(f"[viz-sweep] ✗ 找不到文件：{in_path}（先跑 --sweep 生成，或用 --out 指定）")
        return
    data = torch.load(in_path, map_location="cpu", weights_only=False)

    q_all = data["q"].numpy()
    xyz_all = data["xyz"].numpy()
    ptip_all = data["p_tip"].numpy()
    quat_all = data["quat_wxyz"].numpy()
    yaw_all = data["yaw_deg"].numpy()
    ry_all = data["rot_y_deg"].numpy()
    rz_all = data["rot_z_deg"].numpy()
    err_all = data["pos_err_mm"].numpy()
    N = q_all.shape[0]
    seam_len = float(data.get("meta", {}).get("seam_len", args.seam_len))
    xy_range = data.get("meta", {}).get("ee_xy_range_m", args.ee_xy_range)   # 采样空间(线框)
    z_range = data.get("meta", {}).get("ee_z_range_m", args.ee_z_range)
    print(f"[viz-sweep] 载入 {N} 个有解位姿 ← {in_path}")
    if N == 0:
        print("[viz-sweep] 无位姿可显示")
        return

    print("[viz-sweep] 构建 cuRobo handle 做 FK（首次 warmup 略慢）…")
    handle = ci.init_curobo(cfg)
    torch_model = load_torch_model(cfg)
    seam_local = make_right_angle_seam(seam_len, args.face_size)

    def build_geoms(i):
        R_tip = _quat_wxyz_to_matrix(quat_all[i])
        p_tip = ptip_all[i]
        seam = transformed_seam(seam_local, seam_transform(float(yaw_all[i]), xyz_all[i]))
        spheres = torch_spheres_world(torch_model, R_tip, p_tip)
        nozzle_base = R_tip @ torch_model["nozzle_tip"]
        geoms = seam_geoms(seam, seam_len)
        geoms += torch_geoms(spheres, R_tip, p_tip, nozzle_base=nozzle_base)
        geoms += arm_geoms(_fk_arm_spheres(handle, q_all[i]))    # 整臂碰撞球(灰)
        return geoms

    def info(i):
        return (f"[viz-sweep] {i + 1}/{N}  yaw={float(yaw_all[i]):+.0f}° "
                f"rot_y={float(ry_all[i]):+.0f}° rot_z={float(rz_all[i]):+.0f}°  "
                f"焊缝中点={np.round(xyz_all[i], 3).tolist()}  "
                f"pos_err={float(err_all[i]):.3f}mm")

    state = {"i": 0}
    # 常驻几何：base 坐标系 + 采样空间线框(环形柱体：xy 径向∈ee_xy_range、z∈ee_z_range)
    persist = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
    persist += range_frame_geoms(xy_range, z_range)
    print(f"[viz-sweep] 采样框：xy径向∈{list(map(float, xy_range))}m  z∈{list(map(float, z_range))}m（金黄线框）")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"sweep 位姿 1/{N}  （按 C 下一个 / N 上一个 / Q 退出）")
    for g in persist:
        vis.add_geometry(g)
    for g in build_geoms(0):
        vis.add_geometry(g)
    print(info(0))

    def _show(i, vis):
        vis.clear_geometries()
        for g in persist:
            vis.add_geometry(g, reset_bounding_box=False)   # 常驻：base 系 + 采样框
        for g in build_geoms(i):
            vis.add_geometry(g, reset_bounding_box=False)
        vis.update_renderer()
        print(info(i))

    def next_cb(vis):
        if state["i"] < N - 1:
            state["i"] += 1
            _show(state["i"], vis)
        else:
            print(f"[viz-sweep] 已是最后一个（{N}/{N}），按 Q 退出")
        return False

    def prev_cb(vis):
        if state["i"] > 0:
            state["i"] -= 1
            _show(state["i"], vis)
        else:
            print("[viz-sweep] 已是第一个（1）")
        return False

    vis.register_key_callback(ord("C"), next_cb)      # C：下一个
    vis.register_key_callback(ord("N"), prev_cb)      # N：上一个
    print(f"[viz-sweep] open3d 开窗：灰=整臂碰撞球  青=焊枪  红线=焊缝  蓝线=bisector  "
          f"三轴=tip 位姿  金黄线框=采样空间(ee_xy_range×ee_z_range)。按 C 看下一个、N 看上一个、Q/关窗退出。")
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------- 焊枪 open3d 几何
def torch_geoms(spheres, R_tip, p_tip, nozzle_base=None, axis_len: float = 0.15,
                color=(0.1, 0.6, 0.6)):
    """焊枪碰撞球(青色线框) + tip 坐标系三轴(x=红 y=绿 z=蓝) + tip 原点小球(洋红)
    + 喷嘴物理轴(洋红线，从 tip 沿喷嘴指向；应与 bisector 蓝线共线反向)。"""
    import open3d as o3d
    geoms = []
    for cx, cy, cz, r in spheres:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
        ball.translate((cx, cy, cz))
        ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
        ls.paint_uniform_color(list(color))
        geoms.append(ls)

    # tip 坐标系三轴
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len, origin=(0.0, 0.0, 0.0))
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_tip, float)
    T[:3, 3] = np.asarray(p_tip, float)
    frame.transform(T)
    geoms.append(frame)

    # 喷嘴物理轴：洋红线（tip 原点沿喷嘴方向画 1.3*axis_len）
    if nozzle_base is not None:
        p = np.asarray(p_tip, float)
        tip2 = p + _unit(nozzle_base) * axis_len * 1.3
        line = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector([p.tolist(), tip2.tolist()]),
            o3d.utility.Vector2iVector([[0, 1]]))
        line.paint_uniform_color([1.0, 0.0, 1.0])
        geoms.append(line)

    # tip 原点小球（洋红实心）
    mk = o3d.geometry.TriangleMesh.create_sphere(radius=0.008, resolution=12)
    mk.translate(tuple(float(v) for v in p_tip))
    mk.compute_vertex_normals()
    mk.paint_uniform_color([1.0, 0.0, 1.0])
    geoms.append(mk)
    return geoms


# ---------------------------------------------------------------- 焊缝（open3d）
def _cylinder_between(p0, p1, radius, color):
    """一根圆柱 mesh 连接 p0→p1（默认沿 +z、中心原点 → 旋到 seam 方向再平移到中点）。"""
    import open3d as o3d
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L < 1e-9:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=L, resolution=16)
    d_hat = seg / L
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d_hat)
    sN = float(np.linalg.norm(v))
    c = float(np.dot(z, d_hat))
    if sN < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (sN * sN))
    cyl.rotate(R, center=np.zeros(3))
    cyl.translate((p0 + p1) / 2.0)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(list(color))
    return cyl


def _quad_mesh(quad, color):
    """(4,3) 矩形四角 → 两三角面片 open3d mesh。"""
    import open3d as o3d
    verts = np.asarray(quad, float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces))
    m.compute_vertex_normals()
    m.paint_uniform_color(list(color))
    return m


def seam_geoms(seam: dict, seam_len: float):
    """base 系焊缝 → open3d 几何：红圆柱焊缝线 + 地板/立墙面片 + bisector 蓝线 + 端点小球。"""
    import open3d as o3d
    geoms = []

    # 焊缝红线（圆柱，半径随长度自适应、至少 1mm）
    radius = max(0.001, float(seam_len) * 0.02)
    cyl = _cylinder_between(seam["p0"], seam["p1"], radius, [1.0, 0.0, 0.0])
    if cyl is not None:
        geoms.append(cyl)

    # 两面片：地板(浅蓝) / 立墙(浅橙)，标出直角
    geoms.append(_quad_mesh(seam["faceA"], [0.35, 0.55, 0.90]))   # 地板 A
    geoms.append(_quad_mesh(seam["faceB"], [0.95, 0.65, 0.30]))   # 立墙 B

    # bisector 蓝线（从 mid 沿开口方向 0.15m）
    mid = np.asarray(seam["mid"], float)
    tip = mid + 0.15 * _unit(seam["bisector"])
    bis = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector([mid, tip]),
        o3d.utility.Vector2iVector([[0, 1]]))
    bis.paint_uniform_color([0.0, 0.2, 1.0])
    geoms.append(bis)

    # 端点小球
    for p in (seam["p0"], seam["p1"]):
        b = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.006, radius * 1.5), resolution=8)
        b.translate(tuple(float(v) for v in p))
        b.compute_vertex_normals()
        b.paint_uniform_color([0.7, 0.0, 0.0])
        geoms.append(b)

    return geoms


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="参数化水平直角焊缝 + 焊枪(末端工具)碰撞球 open3d 可视化（第一步：确认焊枪位姿）")
    ap.add_argument("--yaw-deg", type=float, nargs="+", default=[0.0], metavar="DEG",
                    help="焊缝绕世界 +Z 轴水平旋转角度(度)；可多值，sweep 模式遍历，单位姿模式取第一个")
    ap.add_argument("--xyz", type=float, nargs=3, default=[0.8, 0.0, 0.2],
                    metavar=("X", "Y", "Z"), help="焊缝中心(local 原点)在 base 系落点(米)")
    ap.add_argument("--seam-len", type=float, default=0.3, help="焊缝线长度(米)")
    ap.add_argument("--face-size", type=float, default=0.15,
                    help="画出的地板/立墙面片边长(米，仅可视化，不影响焊缝线)")
    ap.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    # —— 焊枪 tip 目标位姿朝向（单值，绕 tip 局部 +y/+z 轴倾斜）——
    ap.add_argument("--rot-y-deg", type=float, default=0.0,
                    help="绕 tip 局部 +y 轴倾斜(度)")
    ap.add_argument("--rot-z-deg", type=float, default=0.0,
                    help="绕 tip 局部 +z 轴倾斜(度)")
    ap.add_argument("--standoff-cm", type=float, default=0.0,
                    help="tip 位置沿 bisector 外移 standoff(cm)；0=正好落在焊缝中点")
    ap.add_argument("--solve-ik", action="store_true",
                    help="对 tip 目标位姿反解机械臂关节角(cuRobo)，并叠加整臂碰撞球(灰色)")
    ap.add_argument("--ik-seeds", type=int, default=20,
                    help="IK 每问题返回的候选解个数(取位置误差最小者)")
    # —— 批量扫描 (sweep)：位置(环形)×yaw×rot_y×rot_z 组合多位姿，逐个反解，有解者存 .pt ——
    ap.add_argument("--sweep", action="store_true",
                    help="批量模式：按下列范围采样多焊枪位姿、批量反解、有解者存 --out(.pt)，不开窗")
    ap.add_argument("--ee-xy-range", type=float, nargs=2, default=[0.4, 2.0], metavar=("LO", "HI"),
                    help="[sweep] tip 在 base 系 xy 平面到原点的径向距离范围(米)，语义同 ee_xy_range_m")
    ap.add_argument("--ee-z-range", type=float, nargs=2, default=[-0.1, 0.1], metavar=("LO", "HI"),
                    help="[sweep] tip 的 z 范围(米)，语义同 ee_z_range_m")
    ap.add_argument("--xyz-step", type=float, default=0.1, help="[sweep] 位置采样间隔(米)")
    ap.add_argument("--rot-y-range", type=float, nargs=2, default=[-30.0, 30.0], metavar=("LO", "HI"),
                    help="[sweep] rot_y 扫描范围(度，含端点)")
    ap.add_argument("--rot-y-step", type=float, default=15.0, help="[sweep] rot_y 采样间隔(度)")
    ap.add_argument("--rot-z-range", type=float, nargs=2, default=[-30.0, 30.0], metavar=("LO", "HI"),
                    help="[sweep] rot_z 扫描范围(度，含端点)")
    ap.add_argument("--rot-z-step", type=float, default=15.0, help="[sweep] rot_z 采样间隔(度)")
    ap.add_argument("--sweep-chunk", type=int, default=200,
                    help="[sweep] IK 批量分块大小(控显存)；显存紧就调小")
    ap.add_argument("--out", default="configs/sweep_ik_solutions.pt",
                    help="[sweep] 有解位姿+关节角保存路径(.pt，相对项目根)；--viz-sweep 时作为读取源")
    ap.add_argument("--viz-sweep", action="store_true",
                    help="可视化模式：读取 --out(.pt) 中所有有解机械臂位姿，逐个显示，按 C 下一个/N 上一个")
    ap.add_argument("--headless", action="store_true",
                    help="无显示器自检：不开窗，仅打印焊缝坐标 + tip 目标位姿 + 焊枪碰撞球数 + VIZ_DONE")
    args = ap.parse_args()

    from gt_gen.config import load_config
    cfg = load_config(args.config)

    # 批量扫描模式：采样多位姿、批量反解、存盘，不走单位姿/开窗逻辑
    if args.sweep:
        run_sweep(cfg, args)
        return

    # 可视化模式：逐个显示 .pt 中所有有解机械臂位姿（按 C 下一个）
    if args.viz_sweep:
        run_viz_sweep(cfg, args)
        return

    yaw0 = float(args.yaw_deg[0])                          # 单位姿模式取第一个 yaw
    # 造焊缝 + 摆位
    seam_local = make_right_angle_seam(args.seam_len, args.face_size)
    T = seam_transform(yaw0, args.xyz)
    seam = transformed_seam(seam_local, T)

    # 打印 base 系焊缝信息（口径同 load_welds）
    print("=" * 60)
    print(f"[seam] 直角焊缝(base 系)  yaw={yaw0}°  xyz={list(args.xyz)}")
    print(f"       p0_world     = {np.round(seam['p0'], 4).tolist()}")
    print(f"       p1_world     = {np.round(seam['p1'], 4).tolist()}")
    print(f"       mid_world    = {np.round(seam['mid'], 4).tolist()}")
    print(f"       bisector     = {np.round(seam['bisector'], 4).tolist()}")
    print(f"       boundary_dirs= d1={np.round(seam['d1'], 4).tolist()} "
          f"d2={np.round(seam['d2'], 4).tolist()}")
    print("=" * 60)

    # 焊枪 tip 目标位姿：+X→-bisector、+Z→焊缝线，再绕局部 y/z 倾斜；位置=中点(+standoff)
    torch_model = load_torch_model(cfg)
    seam_dir = _unit(seam["p1"] - seam["p0"])             # 焊缝线方向（tip +Z 基准）
    R_tip, p_tip = tip_pose_from_seam(seam["mid"], seam["bisector"], seam_dir,
                                      args.rot_y_deg, args.rot_z_deg,
                                      standoff_m=args.standoff_cm / 100.0)
    spheres = torch_spheres_world(torch_model, R_tip, p_tip)
    nozzle_base = R_tip @ torch_model["nozzle_tip"]        # 喷嘴轴在 base 系（rot_y=0 时未必=-bisector）
    print(f"[torch] tip 目标位姿  rot_y={args.rot_y_deg:+.0f}° rot_z={args.rot_z_deg:+.0f}° "
          f"standoff={args.standoff_cm}cm")
    print(f"        tip 原点(=焊枪尖) = {np.round(p_tip, 4).tolist()}")
    print(f"        tip +X(对准-bisector) = {np.round(R_tip[:, 0], 4).tolist()}  "
          f"(-bisector={np.round(-_unit(seam['bisector']), 4).tolist()})")
    print(f"        tip +Z(对准焊缝线)   = {np.round(R_tip[:, 2], 4).tolist()}  "
          f"(seam_dir={np.round(seam_dir, 4).tolist()})")
    print(f"        tip +Y(对称面内)     = {np.round(R_tip[:, 1], 4).tolist()}")
    print(f"        喷嘴物理轴 = {np.round(nozzle_base, 4).tolist()}  "
          f"(与 -bisector 夹角 {np.degrees(np.arccos(np.clip(np.dot(_unit(nozzle_base), -_unit(seam['bisector'])), -1, 1))):.1f}°)")
    print(f"[torch] 焊枪(xiaoyu_accessory_link) 碰撞球 {len(spheres)} 个")

    # 第二步：反解关节角 + 整臂碰撞球
    ik = None
    if args.solve_ik:
        ik = solve_arm_ik(cfg, R_tip, p_tip, return_seeds=args.ik_seeds)
        if ik["ok"]:
            print(f"[ik] 反解成功：过阈值解 {ik['n_success']} 个，取位置误差最小者")
            print(f"     关节角(rad) = {np.round(ik['q'], 6).tolist()}")
            print(f"     位置误差 = {ik['pos_err_mm']:.3f} mm   朝向误差 = {ik['rot_err_deg']:.3f}°")
            print(f"     FK ee_pos  = {np.round(ik['ee_pos'], 4).tolist()}  "
                  f"(目标 tip 原点 = {np.round(p_tip, 4).tolist()})")
            print(f"     整臂碰撞球 {int(np.sum(ik['spheres'][:, 3] > 1e-4))} 个(有效)")
        else:
            print(f"[ik] ✗ 反解失败：该 tip 目标位姿无过阈值解（可能不可达/自碰撞）。"
                  f"可调 --xyz / --standoff-cm / rot_y / rot_z 或增大 --ik-seeds 再试")

    if args.headless:
        print("VIZ_DONE")
        return

    import open3d as o3d
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
    geoms += seam_geoms(seam, args.seam_len)
    geoms += torch_geoms(spheres, R_tip, p_tip, nozzle_base=nozzle_base)
    if ik is not None and ik["ok"]:
        geoms += arm_geoms(ik["spheres"])                  # 整臂碰撞球(灰)

    arm_hint = "+ 整臂碰撞球(灰) " if (ik is not None and ik["ok"]) else ""
    print(f"[viz] open3d 开窗：base 坐标系 + 直角焊缝（红线=焊缝，蓝浅=地板，橙=立墙，蓝线=bisector）"
          f"+ 焊枪碰撞球(青色线框) {arm_hint}+ tip 坐标系三轴(x=红 y=绿 z=蓝)。关闭窗口结束。")
    o3d.visualization.draw_geometries(
        geoms, window_name=f"直角焊缝 + 焊枪位姿 yaw={yaw0}° "
                           f"rot_y={args.rot_y_deg}° rot_z={args.rot_z_deg}°")


if __name__ == "__main__":
    main()
