"""环境兼容 shim。

根因：本机 warp 1.13.0 把 torch 互操作函数移到了顶层（wp.device_from_torch 等），
但当前 curobo dev 检出仍调用旧命名空间 wp.torch.device_from_torch（仅 1 处，
curobo/geom/sdf/world_mesh.py）。这里把 wp.torch 指回顶层函数，避免改动 curobo/warp
安装本身。

用法：在 import curobo 之前 `import gt_gen.compat`（或 from gt_gen import compat）。
注意：这是绕过 curobo↔warp 版本不匹配的权宜之计；若要根治应统一两者版本。
"""
from __future__ import annotations

import sys
import types


def apply_warp_torch_shim() -> bool:
    try:
        import warp as wp
    except Exception:
        return False
    if hasattr(wp, "torch") and getattr(wp.torch, "device_from_torch", None):
        return True  # 已可用
    mod = types.ModuleType("warp.torch")
    for name in (
        "device_from_torch", "device_to_torch",
        "dtype_from_torch", "dtype_to_torch",
        "from_torch", "to_torch",
        "stream_from_torch", "stream_to_torch",
    ):
        if hasattr(wp, name):
            setattr(mod, name, getattr(wp, name))
    wp.torch = mod
    sys.modules["warp.torch"] = mod
    return hasattr(mod, "device_from_torch")


# 导入即生效
applied = apply_warp_torch_shim()


def apply_trimesh_shim() -> bool:
    """curobo geom/types.py 的 `import trimesh` 被注释掉了，但 get_trimesh_mesh 仍直接用
    `trimesh.load(...)` → NameError。这里把 trimesh 注入 curobo.geom.types 命名空间。

    须在 import curobo 之后调用（本函数会 import curobo.geom.types）。
    """
    try:
        import trimesh
        import trimesh.scene  # noqa: F401
        import curobo.geom.types as ct
        if not getattr(ct, "trimesh", None):
            ct.trimesh = trimesh
        return True
    except Exception:
        return False
