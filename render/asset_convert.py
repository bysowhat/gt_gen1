"""资产转换助手：URDF→USD（机械臂）、OBJ→USD（工件），以及定位 Link6 子路径。

这些函数依赖 isaaclab.sim.converters 与 pxr，必须在 AppLauncher 启动 app 之后调用
（否则 import pxr 失败）。转换带缓存：force_usd_conversion=False，命中已存在的 USD 即跳过。
"""
import os
import re

# 缓存目录：render/_assets_cache/
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_assets_cache")


def _ascii_safe(name: str) -> str:
    """把任意文件名压成 ASCII 安全名，用作缓存 USD 文件名（规避中文 '柱' 等）。

    USD 标识符不能以数字开头，故只保留 [0-9A-Za-z_]，并在数字开头时加前缀。
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_")
    if not safe:
        safe = "asset"
    if safe[0].isdigit():
        safe = "m_" + safe
    return safe


def _parse_obj(obj_path: str):
    """读 OBJ 的顶点与面，返回 (points, face_vertex_counts, face_vertex_indices)。

    只取几何（v/f），忽略法线/UV/材质（工件无材质，渲染时另投影库材质）。支持多边形面、
    ``v``、``v/vt``、``v//vn``、``v/vt/vn`` 格式与负索引（相对当前顶点数）。
    """
    points = []
    face_counts = []
    face_indices = []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                points.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idxs = []
                for tok in line.split()[1:]:
                    vi = int(tok.split("/")[0])
                    idxs.append(len(points) + vi if vi < 0 else vi - 1)  # → 0-based
                if len(idxs) >= 3:
                    face_counts.append(len(idxs))
                    face_indices.extend(idxs)
    return points, face_counts, face_indices


def _write_obj_usd(obj_path: str, usd_path: str) -> None:
    """直接用 pxr 把 OBJ 几何写成 USD（绕开 IsaacLab MeshConverter / omni 转换器）。

    omni.kit.asset_converter 在本工件上会「返回成功但产出空几何」，导致 MeshConverter 在
    ``geom_prim.GetChildren()`` 处崩（Accessed invalid null prim）。工件是纯视觉体、无 UV/
    材质，这里自建 ``/workpiece``(Xform) + ``/workpiece/mesh``(Mesh)，结构、坐标系（Z-up，
    米）、prim 路径全可控，且与 va_simulation 的 ``{prim}/mesh`` 约定一致。
    """
    from pxr import Usd, UsdGeom, Vt, Gf

    points, face_counts, face_indices = _parse_obj(obj_path)
    if not points or not face_counts:
        raise RuntimeError(f"OBJ 解析为空（无顶点或无面）：{obj_path}")

    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/workpiece")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/workpiece/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)  # 多边形网格，不做细分平滑
    xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
    mesh.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs)),
    ]))
    stage.GetRootLayer().Save()


def convert_robot_urdf(urdf_path: str, usd_dir: str | None = None,
                       force: bool = False) -> str:
    """URDF→USD（固定底座、合并 fixed joint、位置驱动）。返回 USD 路径。"""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    if usd_dir is None:
        usd_dir = os.path.join(_CACHE_DIR, "ur12e")
    os.makedirs(usd_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name="ur12e.usd",
        force_usd_conversion=force,
        make_instanceable=False,        # 便于按 prim 名定位 Link6
        fix_base=True,                  # 机械臂底座固定
        merge_fixed_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1.0e5, damping=1.0e3,
            ),
        ),
    )
    conv = UrdfConverter(cfg)
    print(f"[asset_convert] robot USD: {conv.usd_path}")
    return conv.usd_path


def convert_workpiece_obj(obj_path: str, usd_dir: str | None = None,
                          force: bool = False) -> str:
    """OBJ→USD（纯视觉，渲染用，不加物理/碰撞）。返回 USD 路径。

    直接用 pxr 写几何（见 _write_obj_usd），不走 IsaacLab MeshConverter——后者底层的
    omni 转换器在本工件上会产出空几何而崩溃。带缓存：USD 已存在且非 force 即跳过。
    """
    if usd_dir is None:
        usd_dir = os.path.join(_CACHE_DIR, "workpiece")
    os.makedirs(usd_dir, exist_ok=True)

    usd_path = os.path.join(usd_dir, f"{_ascii_safe(obj_path)}.usd")
    if force and os.path.exists(usd_path):
        os.remove(usd_path)
    if not os.path.exists(usd_path):
        _write_obj_usd(obj_path, usd_path)
    print(f"[asset_convert] workpiece USD: {usd_path}")
    return usd_path


def find_link_subpath(usd_path: str, link_name: str) -> str:
    """在转换后的 USD 里找到名为 link_name 的 prim，返回它相对 defaultPrim 的子路径。

    spawn 时 UsdFileCfg 会把 USD 的 defaultPrim 内容引用到 <prim_path> 下，故
    Link6 实际路径 = <prim_path>/<本函数返回值>。不同 URDF 链层级不同，必须实测。
    """
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetDefaultPrim()
    root_path = default_prim.GetPath() if default_prim else None

    found = None
    for prim in stage.Traverse():
        if prim.GetName() == link_name:
            found = prim.GetPath()
            break
    if found is None:
        raise RuntimeError(f"在 {usd_path} 未找到名为 {link_name} 的 prim")

    found_str = found.pathString
    if root_path is not None:
        root_str = root_path.pathString
        if found_str.startswith(root_str + "/"):
            return found_str[len(root_str) + 1:]
        if found_str == root_str:
            return ""
    # 兜底：去掉最前面的根段（/World 之类不会出现在 USD 内部，这里多为 /ur12e/...）
    return found_str.lstrip("/").split("/", 1)[-1]
