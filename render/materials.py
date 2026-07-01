"""工件材质扫描 + 绑定助手（移植自 va_simulation 的 MaterialsHelper2 / MdlFileCfg 流程）。

材质库 .mdl 在远程服务器：/kpfs_dataset/dataset/baiyu/haoyue/Materials/{Base,vMaterials_2}
（本地不存在则 scan_materials 返回 []，render_seam 会跳过加材质、保持灰色默认外观，
 这样本地自检照常能跑）。

关键：工件 OBJ 无 UV/法线，绝大多数库材质默认按 UV 采样纹理。bind_material_to_prim
对 OmniPBR 系材质开启 project_uvw（世界/物体空间投影），无 UV 也能显示纹理；不支持该
参数的材质（部分 vMaterials_2）会忽略它，退化为基础色。

依赖 isaaclab.sim 与 pxr，必须在 AppLauncher 启动 app 之后调用。
"""
import os
import random

# ======================= 材质库根目录（远程）=======================
MATERIALS_ROOT = "/kpfs_dataset/dataset/baiyu/haoyue/Materials"

# 扫描的类目（与 va_simulation helper.py 的 MaterialsHelper2 保持一致）
BASE_TYPES = [
    "Architecture", "CSV_Mapping_Files", "Carpet", "Emissives", "Masonry",
    "Metals", "Miscellaneous", "Natural", "Plastics", "Stone", "Styles",
    "Templates", "Textiles", "Wall_Board", "Wood",
]
VMATER2_TYPES = [
    "Carpet", "Ceramic", "Composite", "Concrete", "Fabric", "Gems", "Glass",
    "Leather", "Liquids", "Masonry", "Metal", "Other", "Paint", "Paper",
    "Plaster", "Plastic", "Stone", "Wood",
]

# 透明/半透明材质黑名单（与 va_simulation 一致；透明材质会让工件“看不见”，剔除）
BLACKLIST = [
    "Glass_Colored", "Glass_Smudged", "Glazing_Clear", "Lolite", "PET_Clear",
    "Plastic_Acrylic", "Plastic_Clear", "Polycarbonate_Clear", "Polyethylene_Clear",
    "Polymethylmethacrylate_Clear", "Polypropylene_Clear", "Polystyrene_Clear",
    "Silk_Georgette", "Silk_Plain_Chiffon", "Turquoise", "Water", "Zircon",
    "Glass_Clear", "Glass_Optical", "Glass_Dirty", "Glass_Fritted",
    "Glass_Glazing_Spandrel", "Glass_Glazing_Tinted", "Glazing_Float", "Mirror",
    "Diamond", "Emerald", "Ruby", "Sapphire", "Aquamarine", "Topaz",
    "Alexandrite", "Amethyst", "Ametrine", "Citrine", "Garnet",
    "Iolite", "Jade", "Morganite", "Onyx", "Pearl", "Peridot",
    "Tanzanite", "Tourmaline",
    "Plastic_Thick_Translucent", "Plastic_Standardized_Surface_Finish",
    "Polycarbonate_Spectral_Clear", "Polycarbonate_Cloudy", "Polycarbonate_Opaque",
    "Polyethylene_Cloudy", "Polyethylene_Opaque",
    "Polypropylene_Cloudy", "Polypropylene_Opaque",
    "Light_",  # 自发光色温材质族（Light_3000K/7000K…），会把工件照得过曝/发光，整族剔除
]

# ======================= 投影参数（针对无 UV 工件）=======================
PROJECT_UVW = True        # OmniPBR：True=世界/物体空间投影，无 UV 也能出纹理
WORLD_OR_OBJECT = True    # True=物体空间（纹理锁在工件上，随位姿走）；False=世界空间
TEXTURE_SCALE = 1.0       # 投影纹理平铺尺度（米/平铺），12m 长柱用 1.0 约 12 个平铺


def _in_blacklist(name):
    return any(b in name for b in BLACKLIST)


def scan_materials(root=MATERIALS_ROOT):
    """扫 Base + vMaterials_2 指定类目下的 .mdl（过滤透明黑名单）。

    返回 [(mdl_path, name), ...]；root 不存在（如本地）返回 []。
    """
    if not os.path.isdir(root):
        print(f"[materials] 材质库不存在，跳过加材质：{root}")
        return []

    out = []
    for sub_root, types in ((os.path.join(root, "Base"), BASE_TYPES),
                            (os.path.join(root, "vMaterials_2"), VMATER2_TYPES)):
        for t in types:
            d = os.path.join(sub_root, t)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.endswith(".mdl"):
                    continue
                name = fn[:-4]
                if _in_blacklist(name):
                    continue
                out.append((os.path.join(d, fn), name))
    print(f"[materials] 扫描到 {len(out)} 个可用材质（root={root}）")
    return out


def make_picker(materials, seed=0):
    """返回一个 pick(idx)->(mdl_path,name) 闭包：按 idx 确定性随机取材质（可重复）。

    用全局 pose 序号 idx 作为采样下标，保证每个 pose 的材质既随机又可复现。
    materials 为空时 pick 返回 None。
    """
    if not materials:
        return lambda idx: None
    rng = random.Random(seed)
    # 预生成一个充分长的随机序列，按 idx 取，保证确定性且与 idx 一一对应
    order = [rng.randrange(len(materials)) for _ in range(max(1, len(materials)))]

    def pick(idx):
        return materials[order[idx % len(order)]]

    return pick


def _set_shader_projection(mat_prim_path, project_uvw, world_or_object, texture_scale):
    """在材质的 shader prim 上写 OmniPBR 投影参数（无 UV 也能出纹理）。

    对非 OmniPBR 系材质，这些 input 不被引用、无害。找不到 shader 直接返回。
    """
    import omni.usd
    from pxr import Usd, UsdShade, Sdf, Gf

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(mat_prim_path)
    if not root or not root.IsValid():
        return False
    shader = None
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            break
    if shader is None:
        return False
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(bool(project_uvw))
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(bool(world_or_object))
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(float(texture_scale), float(texture_scale)))
    return True


def bind_material_to_prim(prim_path, mdl_path, mat_prim_path,
                          project_uvw=PROJECT_UVW, world_or_object=WORLD_OR_OBJECT,
                          texture_scale=TEXTURE_SCALE):
    """生成 MDL 材质 prim、开启世界/物体空间投影，并绑定到 prim_path（向下继承）。

    与 va_simulation 同流程：MdlFileCfg(mdl).func(mat_prim) -> bind_visual_material。
    """
    import isaaclab.sim as sim_utils
    import omni.usd

    # 同一进程渲多条 seam 时共用 stage，材质 prim 名按 (env, seam内pose序号) 命名会跨 seam
    # 重名（每条 seam 的 gidx 都从 0 起）。spawn_from_mdl_file 遇到已存在的 prim 会抛
    # ValueError，故先删掉残留的同名 prim，使绑定幂等。
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(mat_prim_path).IsValid():
        stage.RemovePrim(mat_prim_path)

    cfg = sim_utils.MdlFileCfg(mdl_path=mdl_path)
    cfg.func(mat_prim_path, cfg)
    _set_shader_projection(mat_prim_path, project_uvw, world_or_object, texture_scale)
    sim_utils.bind_visual_material(prim_path, mat_prim_path)
