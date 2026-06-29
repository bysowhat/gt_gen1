"""资产转换助手：URDF→USD（机械臂）、OBJ→USD（工件），以及定位 Link6 子路径。

这些函数依赖 isaaclab.sim.converters 与 pxr，必须在 AppLauncher 启动 app 之后调用
（否则 import pxr 失败）。转换带缓存：force_usd_conversion=False，命中已存在的 USD 即跳过。
"""
import os
import re

# 缓存目录：render/_assets_cache/
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_assets_cache")


def _ascii_safe(name: str) -> str:
    """把任意文件名压成 ASCII 安全名（规避中文如 '柱' 引发的 USD 路径问题）。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_")
    return safe or "asset"


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
    """OBJ→USD（纯视觉，渲染用，不加物理/碰撞）。返回 USD 路径。"""
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    if usd_dir is None:
        usd_dir = os.path.join(_CACHE_DIR, "workpiece")
    os.makedirs(usd_dir, exist_ok=True)

    cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=usd_dir,
        usd_file_name=f"{_ascii_safe(obj_path)}.usd",
        force_usd_conversion=force,
        make_instanceable=False,
        mass_props=None,          # 静态视觉体，无需质量/碰撞
        rigid_props=None,
        collision_props=None,
    )
    conv = MeshConverter(cfg)
    print(f"[asset_convert] workpiece USD: {conv.usd_path}")
    return conv.usd_path


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
