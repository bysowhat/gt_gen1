#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从新机械臂 xacro 源码包（urdf_12e_260626/xiaoyu_description）离线生成 cuRobo 用的
单体 robot_description.urdf + ur12e_full.yml，所有碰撞球用 curobo.geom.sphere_fit 在 CPU 拟合
（不依赖 ROS/xacro，也不启动 Isaac/GPU）。

本次装配（与用户确认）：
  arm      = ur12e
  base     = a1_ground_rail_platform   （fixed 安装 z=0.26，无导轨平移关节 → 仍 6 自由度）
  accessory= No012_TRM651              （gas_shield=shield605_2033）
  cable    = No001_12ebarket           （link2_bracket@Link2, link3_bracket@Link3）

关节名沿用 ros2_controllers.yaml 的 xiaoyu_arm_joint1..6（kinematics.yaml 里的 jointN 只是查表 key）。
ee_link = xiaoyu_tip_link，其 flange->tip 变换(TCP)默认用 No012/shield605_2033 的 theoretical_tip
（--tcp params 可切换成 config/ur12e/params.yaml 里的 tcp_offset），生成后会把两种 ee 落点都打印出来。

用法：
  /home/a/miniforge3/envs/env_isaaclab/bin/python scripts/build_robot_from_xacro_pkg.py
  # 可选： --tcp {theoretical|params}  --out-dir <dir>  --print-only
"""
import argparse
import os

import numpy as np
import yaml

# ---------- 路径常量 ----------
PKG = "/home/a/Datas/curobo/example_new_robot/urdf_12e_260626/xiaoyu_description"
DEFAULT_OUT_DIR = "/home/a/Datas/curobo/example_new_robot/urdf_12e_260626"
ARM = "ur12e"
BASE = "a1_ground_rail_platform"
ACCESSORY = "No012_TRM651"
GAS_SHIELD = "shield605_2033"
CABLE_DIR = "others"            # No001_12ebarket 实际放在 xiaoyu_accessories/others/ 下
CABLE = "No001_12ebarket"

CFG = os.path.join(PKG, "xiaoyu_model_description", "config", ARM)
ARM_MESH = "xiaoyu_model_description/meshes/%s" % ARM          # 相对 PKG（asset_root）
BASE_DIR = "xiaoyu_base/%s/meshes" % BASE
ACC_DIR = "xiaoyu_accessories/%s/meshes" % ACCESSORY
CABLE_MESH = "xiaoyu_accessories/others/meshes/%s" % CABLE


def _load(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ---------- 位姿/变换工具 ----------
def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                       # URDF 约定 R = Rz(yaw) Ry(pitch) Rx(roll)


def T(xyz, rpy=(0, 0, 0)):
    m = np.eye(4)
    m[:3, :3] = rpy_to_R(rpy)
    m[:3, 3] = [float(v) for v in xyz]
    return m


def R_to_quat_wxyz(R):
    """旋转矩阵 -> (w,x,y,z)。"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ---------- 读源数据 ----------
params = _load(os.path.join(CFG, "params.yaml"))["/**"]["ros__parameters"]
kin = _load(os.path.join(CFG, "kinematics.yaml"))["kinematics"]
jlim = _load(os.path.join(CFG, "joint_limits.yaml"))["joint_limits"]
inertias = _load(os.path.join(CFG, "inertias.yaml"))["inertias"]
ros2c = _load(os.path.join(CFG, "ros2_controllers.yaml"))
JOINT_NAMES = list(ros2c["xiaoyu_arm_controller"]["ros__parameters"]["joints"])  # xiaoyu_arm_joint1..6

yurdf = params["yaml_urdf"]                       # 关节 origin（直接给 x/y/z/roll/pitch/yaw 数组）
tcp_offset = params["persistent"]["tcp_offset"]   # params 里的 flange->tip（备选）

acc_cfg = _load(os.path.join(PKG, "xiaoyu_accessories", ACCESSORY,
                              "global_config", "%s.yaml" % ACCESSORY))["xiaoyu_robot_arm"]["ros__parameters"]
THEO_TIP = [float(v) for v in acc_cfg[GAS_SHIELD]["theoretical_tip"]]   # flange->tcp（真实焊接 TCP）
GAS_LEN = float(acc_cfg[GAS_SHIELD]["gas_shield_length"])
NOZZLE_OUT = float(acc_cfg[GAS_SHIELD]["conductive_nozzle_stick_out"])
STICK_OUT = float(acc_cfg["stick_out_length"])
CAM_INSTALL = acc_cfg["mech"]["camera_install"]              # dict xyz/rpy
WELDER_CABLE = acc_cfg["welder_cable"]                       # dict xyz/rpy
GAS_MOUNT = acc_cfg["gas_shield_mount_surface"]              # dict xyz/rpy


def _xyz(d):
    return [float(v) for v in str(d["xyz"]).split()] if isinstance(d["xyz"], str) else [float(v) for v in d["xyz"]]


def _rpy(d):
    return [float(v) for v in str(d["rpy"]).split()] if isinstance(d["rpy"], str) else [float(v) for v in d["rpy"]]


def _s2f(s):
    """'0.0 0.0 1.57' -> [0.0,0.0,1.57]（kinematics.yaml 的 origin 是空格分隔字符串）。"""
    if isinstance(s, (list, tuple)):
        return [float(v) for v in s]
    return [float(v) for v in str(s).split()]


# ---------- 构建运动学树 ----------
def build_tree(tcp_mode):
    """返回 (joints, collisions)。
    joints: [dict(name,parent,child,type,xyz,rpy,axis_or_None)]
    collisions: {link_name: [(abs_mesh_path, T_origin_in_link)]}
    tcp_mode: 'theoretical' 用 No012 theoretical_tip 当 flange->tip；'params' 用 params.yaml tcp_offset。
    """
    base_o = kin["base_link"]["origin"]
    flange_o = kin["flange"]["origin"]

    if tcp_mode == "theoretical":
        tcp_xyz = THEO_TIP[:3]
        tcp_rpy = THEO_TIP[3:]
    else:
        tcp_xyz = [float(tcp_offset["x"]), float(tcp_offset["y"]), float(tcp_offset["z"])]
        tcp_rpy = [float(tcp_offset["roll"]), float(tcp_offset["pitch"]), float(tcp_offset["yaw"])]

    # 焊枪 mesh 放置在 flange 原生坐标：xiaoyu_accessory_joint origin = tcp_offset - theoretical_tip
    # （tcp 取 theoretical 时为 0；取 params 时为两者之差，保持与 xacro 一致）
    acc_off = [tcp_xyz[i] - THEO_TIP[i] for i in range(3)]

    joints = []
    j = joints.append
    # 底座（fixed，z=0.26，无平移关节）
    j(dict(name="xiaoyu_base_joint_magnetic_ext", parent="base_link", child="xiaoyu_base_link",
           type="fixed", xyz=[0.0, 0.0, 0.260], rpy=[0, 0, 0], axis=None))
    # 臂基座
    j(dict(name="parent_joint", parent="xiaoyu_base_link", child="xiaoyu_arm_base_link",
           type="fixed", xyz=_s2f(base_o["xyz"]), rpy=_s2f(base_o["rpy"]), axis=None))
    # 6 个旋转关节（名字用 ros2_controllers 的 xiaoyu_arm_jointN）
    parents = ["xiaoyu_arm_base_link", "Link1", "Link2", "Link3", "Link4", "Link5"]
    for i in range(6):
        j(dict(name=JOINT_NAMES[i], parent=parents[i], child="Link%d" % (i + 1),
               type="revolute",
               xyz=[float(yurdf["x"][i]), float(yurdf["y"][i]), float(yurdf["z"][i])],
               rpy=[float(yurdf["roll"][i]), float(yurdf["pitch"][i]), float(yurdf["yaw"][i])],
               axis=_s2f(kin["joint%d" % (i + 1)]["axis"]),
               limit=jlim["joint%d" % (i + 1)]))
    # flange
    j(dict(name="xiaoyu_flange", parent="Link6", child="xiaoyu_flange_link",
           type="fixed", xyz=_s2f(flange_o["xyz"]), rpy=_s2f(flange_o["rpy"]), axis=None))
    # ee = xiaoyu_tip_link（TCP）
    j(dict(name="xiaoyu_accessory_end_joint", parent="xiaoyu_flange_link", child="xiaoyu_tip_link",
           type="fixed", xyz=tcp_xyz, rpy=tcp_rpy, axis=None))
    # 焊枪本体
    j(dict(name="xiaoyu_accessory_joint", parent="xiaoyu_flange_link", child="xiaoyu_accessory_link",
           type="fixed", xyz=acc_off, rpy=[0, 0, 0], axis=None))
    j(dict(name="camera_cover_j", parent="xiaoyu_accessory_link", child="camera_cover",
           type="fixed", xyz=_xyz(CAM_INSTALL), rpy=_rpy(CAM_INSTALL), axis=None))
    j(dict(name="welder_cover_j", parent="xiaoyu_accessory_link", child="welder_cover",
           type="fixed", xyz=[0, 0, 0], rpy=[0, 0, 0], axis=None))
    j(dict(name="welder_jacket_head_j", parent="xiaoyu_tip_link", child="welder_jacket_head",
           type="fixed", xyz=[-(STICK_OUT + GAS_LEN + NOZZLE_OUT), 0, 0], rpy=[0, 0, 0], axis=None))
    # 线缆支架
    j(dict(name="link2_bracket_joint", parent="Link2", child="link2_bracket", type="fixed",
           xyz=[-0.3733722, 0.0, 0.176], rpy=[-1.5707963, 0.7853982, -1.5707963], axis=None))
    j(dict(name="link3_bracket_joint", parent="Link3", child="link3_bracket", type="fixed",
           xyz=[-0.3025, 0.0, 0.0388], rpy=[0, 1.5707963, 0], axis=None))
    # 焊丝（尖端球单独处理，不入碰撞 link 本体）
    j(dict(name="weld_wire_joint", parent="xiaoyu_tip_link", child="weld_wire",
           type="fixed", xyz=[0, 0, 0], rpy=[0, 0, 0], axis=None))

    # 各 link 的碰撞 mesh（路径相对 PKG；T 为 mesh 在该 link 局部系的 origin）
    AB = os.path.join                                       # 缩写
    col = {
        "xiaoyu_base_link": [
            ("%s/collision/%s.STL" % (BASE_DIR, BASE), T([0, 0, 0])),
            ("%s/collision/cylinder_base.STL" % BASE_DIR, T([0, 0, 0])),
            ("%s/collision/positioning_pen.STL" % BASE_DIR,
             T([0.0, 0.0357993, -0.1760207], [0.0, 0.5235988, 1.5707963])),
        ],
        "xiaoyu_arm_base_link": [("%s/collision/base_link.STL" % ARM_MESH, T([0, 0, 0]))],
        "Link1": [("%s/collision/Link1.STL" % ARM_MESH, T([0, 0, 0]))],
        "Link2": [("%s/collision/Link2.STL" % ARM_MESH, T([0, 0, 0])),
                  ("%s/link2_bracket.STL" % CABLE_MESH, "JOINT:link2_bracket_joint")],
        "Link3": [("%s/collision/Link3.STL" % ARM_MESH, T([0, 0, 0])),
                  ("%s/link3_bracket.STL" % CABLE_MESH, "JOINT:link3_bracket_joint")],
        "Link4": [("%s/collision/Link4.STL" % ARM_MESH, T([0, 0, 0]))],
        "Link5": [("%s/collision/Link5.STL" % ARM_MESH, T([0, 0, 0]))],
        "Link6": [("%s/collision/Link6.STL" % ARM_MESH, T([0, 0, 0]))],
        "xiaoyu_accessory_link": [
            ("%s/collision/xiaoyu_accessory_link.STL" % ACC_DIR, T([0, 0, 0])),
            ("%s/collision/welder_camera.STL" % ACC_DIR, T(_xyz(CAM_INSTALL), _rpy(CAM_INSTALL))),
            ("%s/collision/welder_cable.STL" % ACC_DIR, T(_xyz(WELDER_CABLE), _rpy(WELDER_CABLE))),
            ("%s/collision/%s/welder_jacket.STL" % (ACC_DIR, GAS_SHIELD),
             T(_xyz(GAS_MOUNT), _rpy(GAS_MOUNT))),
        ],
        "camera_cover": [("%s/collision/camera_cover.STL" % ACC_DIR, T([0, 0, 0]))],
        "welder_cover": [("%s/collision/goose_neck.STL" % ACC_DIR, T([0, 0, 0]))],
        "welder_jacket_head": [("%s/collision/%s/welder_jacket_head.STL" % (ACC_DIR, GAS_SHIELD),
                                T([0, 0, 0]))],
    }
    # 把 "JOINT:<name>" 占位换成该 joint 的 origin 变换
    jmap = {jj["name"]: jj for jj in joints}
    for ln, items in col.items():
        new = []
        for path, t in items:
            if isinstance(t, str) and t.startswith("JOINT:"):
                jj = jmap[t.split(":", 1)[1]]
                t = T(jj["xyz"], jj["rpy"])
            new.append((path, t))
        col[ln] = new
    return joints, col


# 碰撞锚 link（= 旧 yml 的 collision_link_names）。其余固连子件按父链折叠进最近的锚。
ANCHORS = ["xiaoyu_base_link", "xiaoyu_arm_base_link",
           "Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "xiaoyu_accessory_link"]


def nearest_anchor(link, joints):
    """沿父关节上溯，返回 (anchor, T_anchor_link)。"""
    child2j = {jj["child"]: jj for jj in joints}
    cur = link
    Tt = np.eye(4)                       # T_cur_link
    while cur not in ANCHORS:
        if cur not in child2j:
            return None, None            # 到 base_link 都没遇到锚
        jj = child2j[cur]
        Tt = T(jj["xyz"], jj["rpy"]) @ Tt
        cur = jj["parent"]
    return cur, Tt


def R_to_rpy(R):
    """旋转矩阵 -> URDF rpy（Rz(yaw)Ry(pitch)Rx(roll) 的逆解）。"""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                   # 万向锁
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return [roll, pitch, yaw]


# ---------- 生成 URDF ----------
def gen_urdf(joints, col):
    links = set(["base_link"])
    for jj in joints:
        links.add(jj["parent"]); links.add(jj["child"])

    def fmt(v):
        return " ".join("%.10g" % float(x) for x in v)

    out = ['<?xml version="1.0"?>', '<robot name="ur12e">']
    for ln in sorted(links):
        out.append('  <link name="%s">' % ln)
        # inertial：臂 link 用真实值，其余给极小默认（避免某些解析器报错）
        inr = inertias.get(ln)
        if inr:
            io = inr["origin"]
            i = inr["inertia"]
            out.append('    <inertial><origin xyz="%s" rpy="%s"/><mass value="%g"/>'
                       '<inertia ixx="%g" ixy="%g" ixz="%g" iyy="%g" iyz="%g" izz="%g"/></inertial>'
                       % (fmt(_s2f(io["xyz"])), fmt(_s2f(io["rpy"])), inr["mass"],
                          i["xx"], i["xy"], i["xz"], i["yy"], i["yz"], i["zz"]))
        else:
            out.append('    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>'
                       '<inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>')
        for path, t in col.get(ln, []):
            xyz = fmt(t[:3, 3]); rpy = fmt(R_to_rpy(t[:3, :3]))
            abspath = os.path.join(PKG, path)          # 写绝对路径：对任意查看器与 cuRobo 都可解析
            for tag in ("visual", "collision"):
                out.append('    <%s><origin xyz="%s" rpy="%s"/><geometry>'
                           '<mesh filename="%s" scale="1 1 1"/></geometry></%s>'
                           % (tag, xyz, rpy, abspath, tag))
        out.append('  </link>')
    for jj in joints:
        out.append('  <joint name="%s" type="%s">' % (jj["name"], jj["type"]))
        out.append('    <parent link="%s"/><child link="%s"/>' % (jj["parent"], jj["child"]))
        out.append('    <origin xyz="%s" rpy="%s"/>' % (fmt(jj["xyz"]), fmt(jj["rpy"])))
        if jj["type"] == "revolute":
            out.append('    <axis xyz="%s"/>' % fmt(jj["axis"]))
            lm = jj["limit"]
            out.append('    <limit lower="%g" upper="%g" velocity="%g" effort="%g"/>'
                       % (lm["lower"], lm["upper"], lm["velocity"], lm["effort"]))
        out.append('  </joint>')
    out.append('</robot>')
    return "\n".join(out)


# ---------- 拟合碰撞球（中轴球心 + 到真实表面的内切半径 + 贪心覆盖）----------
# 关键：球心取填实体素的中轴（EDT 深处），但【半径取该点到真实 mesh 表面的距离】
#       （mesh.nearest.on_surface，无符号；候选点已在填实体素内部 → 该距离即内切半径）。
#       不再用 edt*pitch —— 那是到【体素边界】的距离，粗 pitch 下比真实表面大半~一个体素，
#       球会鼓出 mesh（旧版 base 半径 0.07，本法误成 0.19）。再乘 SHRINK 留安全余量，
#       保证球⊆mesh、绝不溢出表面（代价：覆盖率略降，符合需求）。
N_SPHERES = {                       # 每锚 link 的最大球数（够覆盖即停）
    "xiaoyu_base_link": 150, "xiaoyu_arm_base_link": 12,
    "Link1": 16, "Link2": 24, "Link3": 24, "Link4": 16, "Link5": 16, "Link6": 16,
    "xiaoyu_accessory_link": 40,
}
PITCH = {                           # 每锚 link 的体素边长（米）：越小越细、球越多
    "xiaoyu_base_link": 0.020, "xiaoyu_accessory_link": 0.012,
}
DEFAULT_PITCH = 0.016               # 臂连杆默认体素边长（细一些，补偿球变小后的覆盖）
SHRINK = 0.98                       # 半径安全收缩系数：半径=到最近表面距离时球恰好内切(相切不穿)，
                                    #   0.98 仅留极小数值余量，绝不溢出又不至于到处留大空隙
OVERLAP = 0.65                      # 贪心覆盖判据：候选落入已选球 OVERLAP*r 内即算覆盖（越小越密、重叠越多）

# 细长杆补球：主拟合把整锚 mesh 合并后贪心选球，细长/突出杆件常被主体大球挤掉而盖不全。
# 这里【单独】对指定子 mesh 用「轴向不重叠球链」拟合（rod_chain_spheres）：沿连杆主轴从一端码到
# 另一端，球心取轴向薄片质心(跟随弯曲)，半径=质心到该子 mesh 表面距离(内切→不溢出)，相邻球相切不重叠。
# 追加到所属锚 link，不改已有球。targets = (urdf_link, mesh_idx, anchor, min_r, max_n)。
# mesh_idx 与 viz_robot_spheres_isaacsim.py 里 <link>_<idx> 命名一致（= URDF 内该 link 第 idx 个 visual）。
EXTRA_THIN = [
    ("welder_cover", 0, "xiaoyu_accessory_link", 0.005, 40),       # goose_neck 鹅颈（弯杆，30×199mm）
    ("xiaoyu_accessory_link", 3, "xiaoyu_accessory_link", 0.005, 30),  # welder_jacket 喷嘴套（30×78mm）
    ("xiaoyu_accessory_link", 1, "xiaoyu_accessory_link", 0.010, 20),  # welder_camera（48×123mm）
    ("Link3", 1, "Link3", 0.010, 40),                              # link3_bracket 线缆支架（70×446mm）
    ("Link2", 1, "Link2", 0.010, 40),                              # link2_bracket 线缆支架（70×334mm）
    ("xiaoyu_base_link", 2, "xiaoyu_base_link", 0.012, 30),        # positioning_pen 定位笔（90×201mm）
]


def medial_spheres(mesh, n, pitch):
    """中轴球拟合：球心=中轴体素，半径=到真实表面距离×SHRINK（保证⊆mesh）。
    返回 (centers[K,3], radii[K])。"""
    from scipy.ndimage import distance_transform_edt
    vg = mesh.voxelized(pitch)
    try:
        vg = vg.fill()                          # 填实内部（mesh 须近似 watertight）
    except Exception:
        pass
    mat = vg.matrix.astype(bool)
    if not mat.any():
        return np.zeros((0, 3)), np.zeros(0)
    edt = distance_transform_edt(mat)           # 体素内部到外部距离（体素单位）
    idx = np.argwhere(mat)
    pts = vg.indices_to_points(idx.astype(float))
    edt_r = edt[mat]
    keep = edt_r >= 1.0                          # 至少离体素边界 1 层，去掉贴边碎球、省查询
    if not keep.any():
        keep = edt_r >= edt_r.max() * 0.5
    pts = pts[keep]
    # 半径 = 候选点到真实 mesh 表面的距离（点在内部 → 即内切半径），不依赖 watertight
    try:
        _, dist, _ = mesh.nearest.on_surface(pts)
    except Exception:
        dist = edt_r[keep] * pitch              # 退化：用体素半径
    rad = np.asarray(dist) * SHRINK
    good = rad > pitch * 0.4                      # 太小的球不要（防碎屑）
    pts, rad = pts[good], rad[good]
    if len(pts) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    order = np.argsort(-rad)                      # 半径降序
    covered = np.zeros(len(pts), bool)
    cc, cr = [], []
    for k in order:
        if covered[k]:
            continue
        c, r = pts[k], rad[k]
        cc.append(c); cr.append(r)
        covered |= np.linalg.norm(pts - c, axis=1) <= r * OVERLAP
        if len(cc) >= n:
            break
    return np.array(cc), np.array(cr)


def rod_chain_spheres(mesh, min_r, max_n):
    """沿连杆轴向码放【不重叠】内切球链（跟随弯曲）。返回 (centers[K,3], radii[K])。
      ① 密集采样表面点(+顶点)，与顶点数/是否封闭无关，盒体/薄板/管件都密；
      ② PCA 主轴；③ 沿主轴细步前进，每步取该【轴向薄片】采样点【质心】当球心
         （= 截面中心，落在中轴；弯杆逐片质心也跟随弯曲）；
      ④ 半径 = 球心到 mesh 表面距离 × SHRINK（真实内切，绝不上浮 → 必不溢出）；
         若 mesh 封闭且球心落在体外(L 形拐角)则跳过；仅当与上一颗球【球心距 ≥ 两半径和】才落球
         → 相邻必不重叠（不算覆盖率）。
    min_r: 半径阈值（米，截面太细就跳过）；max_n: 最多球数。"""
    np.random.seed(0)                                    # 采样确定化（可复现）
    try:
        S = np.asarray(mesh.sample(8000))
    except Exception:
        S = np.asarray(mesh.vertices)
    Pts = np.vstack([S, np.asarray(mesh.vertices, float)])
    c0 = Pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((Pts - c0).T))
    axis = vecs[:, -1]
    axis = axis / (np.linalg.norm(axis) or 1.0)
    proj = (Pts - c0) @ axis
    smin, smax = float(proj.min()), float(proj.max())
    span = smax - smin
    watertight = bool(mesh.is_watertight)
    step = max(min_r * 0.4, span * 0.01)                 # 细步搜索增量（不是球间距）
    cc, cr = [], []
    prev_c, prev_r = None, None
    s = smin
    guard = 0
    while s <= smax and len(cc) < max_n and guard < 20000:
        guard += 1
        s += step
        half = max(prev_r or min_r, span * 0.015)        # 薄片半宽
        sel = np.abs(proj - s) <= half
        if int(sel.sum()) < 5:
            continue
        center = Pts[sel].mean(0)                         # 截面质心 → 中轴点（跟随弯曲）
        if watertight and not bool(mesh.contains(center[None])[0]):
            continue                                     # 球心落在体外（L 形拐角）→ 跳过
        r = float(mesh.nearest.on_surface(center[None])[1][0]) * SHRINK
        if r < min_r:
            continue                                     # 截面太细处不放球（不撑大 → 不溢出）
        if prev_c is not None and np.linalg.norm(center - prev_c) < prev_r + r:
            continue                                     # 会与上一颗重叠 → 再往前找
        cc.append(center); cr.append(r)
        prev_c, prev_r = center, r
    return np.array(cc), np.array(cr)


def _coverage(mesh, c, r, tol=0.005):
    if len(c) == 0:
        return 0.0
    pts = mesh.sample(4000)
    d = np.linalg.norm(pts[:, None, :] - np.array(c)[None, :, :], axis=2) - np.array(r)[None, :]
    return float((d.min(axis=1) <= tol).mean())


def fit_all_spheres(joints, col):
    import trimesh

    # 把每个有 mesh 的 link 折叠到最近锚
    groups = {a: [] for a in ANCHORS}
    for ln, items in col.items():
        a, T_a_ln = nearest_anchor(ln, joints)
        if a is None:
            print("  [warn] %s 找不到锚，跳过" % ln); continue
        for path, t in items:
            groups[a].append((path, T_a_ln @ t))

    spheres = {}
    for a in ANCHORS:
        meshes = []
        for path, t in groups[a]:
            m = trimesh.load(os.path.join(PKG, path), force="mesh")
            m.apply_transform(t)
            meshes.append(m)
        merged = trimesh.util.concatenate(meshes)
        pts, rad = medial_spheres(merged, N_SPHERES.get(a, 16), PITCH.get(a, DEFAULT_PITCH))
        lst = [{"center": [round(float(c), 6) for c in pts[i]], "radius": round(float(rad[i]), 6)}
               for i in range(len(pts))]
        spheres[a] = lst
        cov = _coverage(merged, pts, rad)
        print("  %-22s mesh=%d -> %3d 球 (r %.3f~%.3f) 覆盖率=%.1f%%"
              % (a, len(meshes), len(lst),
                 min(s["radius"] for s in lst), max(s["radius"] for s in lst), cov * 100))
    return spheres


def fit_extra_thin(joints, col, spheres):
    """对 EXTRA_THIN 指定的细长杆子 mesh 用「轴向不重叠球链」拟合，【追加】到所属锚 link。
    不改动 spheres 里已有的球（只在末尾 append）。"""
    import trimesh
    for urdf_link, idx, anchor, min_r, max_n in EXTRA_THIN:
        path, t_in_link = col[urdf_link][idx]
        a, T_a_ln = nearest_anchor(urdf_link, joints)     # 子 mesh 所在 link 折叠到锚的变换
        if a != anchor:
            print("  [extra][warn] %s 的锚是 %s，与配置 %s 不符，跳过" % (urdf_link, a, anchor))
            continue
        m = trimesh.load(os.path.join(PKG, path), force="mesh")
        m.apply_transform(T_a_ln @ t_in_link)             # 变到锚局部系（与已有球同系）
        pts, rad = rod_chain_spheres(m, min_r, max_n)     # 轴向相切球链，内切→不溢出
        add = [{"center": [round(float(c), 6) for c in pts[i]], "radius": round(float(rad[i]), 6)}
               for i in range(len(pts))]
        spheres[anchor] = spheres.get(anchor, []) + add
        tag = "%s_%d" % (urdf_link, idx)
        if add:
            print("  [extra] %-26s +%2d 球链 (r %.3f~%.3f) -> %s"
                  % (tag, len(add), min(s["radius"] for s in add),
                     max(s["radius"] for s in add), anchor))
        else:
            print("  [extra] %-26s +0 球（mesh 太薄/顶点过少）" % tag)
    return spheres


def weld_wire_tip_spheres(tcp_mode):
    """焊丝 5 颗尖端球（r=0.003），换算到 xiaoyu_accessory_link 局部系。
    weld_wire 挂 xiaoyu_tip_link 原点、沿 -X 的 2cm 圆柱（头在原点）。"""
    if tcp_mode == "theoretical":
        tcp_xyz, tcp_rpy = THEO_TIP[:3], THEO_TIP[3:]
    else:
        tcp_xyz = [float(tcp_offset["x"]), float(tcp_offset["y"]), float(tcp_offset["z"])]
        tcp_rpy = [float(tcp_offset["roll"]), float(tcp_offset["pitch"]), float(tcp_offset["yaw"])]
    acc_off = [tcp_xyz[i] - THEO_TIP[i] for i in range(3)]
    T_fl_acc = T(acc_off, [0, 0, 0])
    T_fl_tip = T(tcp_xyz, tcp_rpy)
    T_acc_tip = np.linalg.inv(T_fl_acc) @ T_fl_tip
    centers = []
    for k in range(5):
        p_tip = np.array([-(0.002 + 0.004 * k), 0.0, 0.0, 1.0])   # 沿 -X 等距 5 点
        p_acc = (T_acc_tip @ p_tip)[:3]
        centers.append([round(float(v), 6) for v in p_acc])
    return centers


# ---------- 写 ur12e_full.yml ----------
SELF_COLL_IGNORE = {
    # 仅忽略运动链【相邻】link 对（与旧 ur12e_full.yml 一致）。球已内切不再假自碰
    # （retract 实测 0 重叠），故不再额外忽略非相邻对，以免掩盖其它构型的真实自碰。
    "xiaoyu_arm_base_link": ["xiaoyu_base_link", "Link1"],
    "Link1": ["Link2"], "Link2": ["Link3"], "Link3": ["Link4"],
    "Link4": ["Link5"], "Link5": ["Link6"], "Link6": ["xiaoyu_accessory_link"],
}
SELF_COLL_BUFFER = {a: 0.001 for a in ANCHORS}
# retract 沿用旧 ur12e_full.yml（臂未变，关节角与命名无关）
RETRACT = [1.5707824230194092, -2.0071660480894984, 1.3613484541522425,
           -0.9599629205516357, -1.570770565663473, 0.0]


def gen_yml(spheres, urdf_path, tip_centers):
    spheres = dict(spheres)
    # 焊丝尖端 5 球并入 xiaoyu_accessory_link（kejian2 tip_spheres 据此排除膨胀）
    spheres["xiaoyu_accessory_link"] = spheres.get("xiaoyu_accessory_link", []) + \
        [{"center": c, "radius": 0.003} for c in tip_centers]
    robot_cfg = {
        "robot_cfg": {"kinematics": {
            "usd_path": "", "usd_robot_root": "/ur12e", "isaac_usd_path": "",
            "usd_flip_joints": {}, "usd_flip_joint_limits": [],
            "urdf_path": urdf_path, "asset_root_path": PKG,
            "base_link": "base_link", "ee_link": "xiaoyu_tip_link",
            "link_names": None, "lock_joints": None, "extra_links": None,
            "collision_link_names": list(ANCHORS),
            "collision_spheres": spheres,
            "collision_sphere_buffer": 0.002, "extra_collision_spheres": {},
            "self_collision_ignore": SELF_COLL_IGNORE,
            "self_collision_buffer": SELF_COLL_BUFFER,
            "use_global_cumul": True, "mesh_link_names": None, "external_asset_path": None,
            "cspace": {
                "joint_names": list(JOINT_NAMES), "retract_config": RETRACT,
                "null_space_weight": [1.0] * 6, "cspace_distance_weight": [1.0] * 6,
                "max_jerk": 500.0, "max_acceleration": 15.0,
            },
        }}
    }
    return robot_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcp", choices=["theoretical", "params"], default="theoretical")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--print-only", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    joints, col = build_tree(args.tcp)

    # 打印两种 TCP 的 ee 落点（flange 系 + base 系需 FK，这里给 flange->tip 平移供核对）
    print("[TCP] theoretical_tip(flange->tip) =", [round(v, 4) for v in THEO_TIP])
    print("[TCP] params.tcp_offset            = xyz",
          [round(float(tcp_offset[k]), 4) for k in ("x", "y", "z")],
          "rpy", [round(float(tcp_offset[k]), 4) for k in ("roll", "pitch", "yaw")])
    print("[TCP] 本次采用:", args.tcp)

    urdf = gen_urdf(joints, col)
    urdf_path = os.path.join(args.out_dir, "robot_description.urdf")
    yml_path = os.path.join(args.out_dir, "ur12e_full.yml")

    print("[spheres] 拟合中（CPU）...")
    spheres = fit_all_spheres(joints, col)
    print("[spheres] 细长杆补球（追加，不改已有）...")
    spheres = fit_extra_thin(joints, col, spheres)
    tip_centers = weld_wire_tip_spheres(args.tcp)
    print("[tip] 焊丝尖端球(accessory 系) =", tip_centers)

    robot_cfg = gen_yml(spheres, urdf_path, tip_centers)

    if args.print_only:
        print(urdf[:800], "\n...(urdf 截断)")
        return
    with open(urdf_path, "w") as f:
        f.write(urdf)
    with open(yml_path, "w") as f:
        yaml.safe_dump(robot_cfg, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print("[done] urdf ->", urdf_path)
    print("[done] yml  ->", yml_path)
    print("[next] 改 configs/default.yaml: robot.cfg_path 指向上面 yml；"
          "plan_init_pose_kejian2.tip_spheres.centers 用上面 tip 值。")


if __name__ == "__main__":
    main()
