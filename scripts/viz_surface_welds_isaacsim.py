"""【独立脚本】在 Isaac Sim 里可视化 find_surface_welds.py 找到的所有焊缝。

打开原始 USD 场景，把 welds.json 里每条焊缝叠加画出来：
    · 焊缝       p0→p1        —— 红色圆柱（半径 5cm，长度=焊缝长度）
    · bisector   中点→外部     —— 绿色线 + 端点小球（角平分线，指向张开的外部空间）
    · boundary_dirs 两侧面方向 —— （可选，--show-boundary）黄/青短线，从中点沿两侧面出去

坐标系：welds.json 是世界系(米)，本脚本直接 open 同一个 USD 场景作为 stage，
故焊缝世界坐标与场景几何 1:1 对齐（前提 metersPerUnit=1，Isaac warehouse 即如此）。

相机：默认为每条焊缝建一台 UsdGeom.Camera（/World/weld_cams/cam_wXXXXX），
放在 bisector 正方向、距焊缝 --cam-dist(默认0.5m) 处，看回焊缝中点（视线=bisector
反方向）。GUI 里从视口相机下拉框即可切到任一焊缝视角。--no-cam 关闭。

用法（有显示器，弹 GUI）：
    conda run -n env_isaaclab --no-capture-output python scripts/viz_surface_welds_isaacsim.py \
        --usd <场景.usd> --welds /tmp/welds.json
无显示器自检（不弹窗，只跑通建 prim）：加 --headless。

拍照（--save <目录>）：为每条焊缝沿 bisector 的反方向、距焊缝 0.5m 处放一台相机，
看向焊缝中点，拍 1 张 RGB 存成 weld_{i}.png。批量拍照建议配 --headless。
"""
import argparse
import json
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Isaac Sim 可视化 USD 场景表面焊缝")
    ap.add_argument("--usd", required=True, help="原始 USD 场景路径（与 find_surface_welds 同一个）")
    ap.add_argument("--welds", required=True, help="find_surface_welds.py 输出的 welds.json")
    ap.add_argument("--line-width", type=float, default=0.01, help="焊缝线宽(米)")
    ap.add_argument("--seam-radius", type=float, default=0.05,
                    help="焊缝圆柱半径(米)，默认 0.05=5cm")
    ap.add_argument("--bisector-len", type=float, default=0.06,
                    help="bisector 箭头长度(米)；0=不画")
    ap.add_argument("--show-boundary", action="store_true",
                    help="额外画两侧 boundary_dir 短线(黄/青)")
    ap.add_argument("--max-welds", type=int, default=0, help="最多画多少条(0=全部)，控性能")
    ap.add_argument("--cam-dist", type=float, default=0.5,
                    help="为每条焊缝建相机的距离(米)，相机沿 bisector 正方向、看回中点，默认 0.5")
    ap.add_argument("--cam-focal", type=float, default=18.0,
                    help="焊缝相机焦距(mm)，越小视角越广，默认 18")
    ap.add_argument("--no-cam", action="store_true", help="不为焊缝创建相机 prim")
    ap.add_argument("--save", type=str, default="",
                    help="若指定目录，则为每条焊缝沿 bisector 反方向、距焊缝 --save-dist 米处"
                         "拍 1 张 RGB 存到该目录(weld_{i}.png)")
    ap.add_argument("--save-dist", type=float, default=0.5, help="拍照距离(米)，默认 0.5")
    ap.add_argument("--save-res", type=int, nargs=2, default=(1280, 720),
                    metavar=("W", "H"), help="拍照分辨率，默认 1280 720")
    return ap


# ---- AppLauncher 必须在导入其余 isaaclab/pxr 之前启动 ----
from isaaclab.app import AppLauncher  # noqa: E402

_parser = parse_args()
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
# 拍照需要 replicator/渲染扩展；isaaclab 只有 enable_cameras=True 才会加载它们
if args_cli.save:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ======================= 启动后再导入 =======================
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402


def add_polyline(prim_path, points, width, color):
    """一条 linear BasisCurves 折线。"""
    stage = omni.usd.get_context().get_stage()
    c = UsdGeom.BasisCurves.Define(stage, prim_path)
    c.CreateTypeAttr("linear")
    c.CreateCurveVertexCountsAttr([len(points)])
    c.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    c.CreateWidthsAttr([float(width)] * len(points))
    c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def add_cylinder(prim_path, p0, p1, radius, color):
    """用一个圆柱表示 p0→p1 的焊缝：半径 radius，长度=|p1-p0|，颜色 color。

    UsdGeom.Cylinder 默认沿 +Z 轴、以原点为中心。这里把它旋转到 p0→p1 方向，
    再平移到中点。"""
    stage = omni.usd.get_context().get_stage()
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return
    mid = 0.5 * (p0 + p1)

    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateHeightAttr(length)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    # 让 extent 正确，避免包围盒异常
    r = float(radius)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -length / 2.0),
                          Gf.Vec3f(r, r, length / 2.0)])

    # +Z 旋转到焊缝方向
    z = Gf.Vec3d(0.0, 0.0, 1.0)
    d = Gf.Vec3d(float(axis[0]), float(axis[1]), float(axis[2])).GetNormalized()
    rot = Gf.Rotation(z, d)

    xf = UsdGeom.Xformable(cyl)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(rot.GetQuat()))


def add_camera(prim_path, cam_pos, look_at, focal=18.0, up=(0.0, 0.0, 1.0)):
    """在 cam_pos 建一台 UsdGeom.Camera，本地 -Z 轴看向 look_at。

    Isaac warehouse 为 Z-up，故默认上向量 (0,0,1)。相机放在 bisector 正方向、
    看回焊缝中点 => 视线方向 = bisector 反方向。GUI 里可从相机下拉框切到此视角。"""
    stage = omni.usd.get_context().get_stage()
    eye = np.asarray(cam_pos, float)
    tgt = np.asarray(look_at, float)
    if float(np.linalg.norm(tgt - eye)) < 1e-9:
        return
    up = np.asarray(up, float)
    if abs(float(np.dot((tgt - eye) / np.linalg.norm(tgt - eye), up / np.linalg.norm(up)))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])            # 视线与上向量近平行，换一个

    cam = UsdGeom.Camera.Define(stage, prim_path)
    cam.CreateFocalLengthAttr(float(focal))       # 焦距越小视角越广，18mm≈74° 便于框住焊缝
    # 近裁剪面默认 1.0 会把 0.5m 处的焊缝裁掉，这里放到 1cm
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1.0e6))

    # SetLookAt 给的是 world->eye 视图矩阵，相机 prim 的 local->world 是它的逆
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
        Gf.Vec3d(float(tgt[0]), float(tgt[1]), float(tgt[2])),
        Gf.Vec3d(float(up[0]), float(up[1]), float(up[2])))
    world = view.GetInverse()

    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(world)


def add_sphere(prim_path, center, radius, color):
    stage = omni.usd.get_context().get_stage()
    s = UsdGeom.Sphere.Define(stage, prim_path)
    s.CreateRadiusAttr(float(radius))
    s.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(s)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))


def _save_rgb(rgb, path):
    """把 annotator 的 rgb 数据(HxWx4/3 uint8)存成 PNG。"""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)


def main():
    welds = json.load(open(args_cli.welds, encoding="utf-8"))
    if args_cli.max_welds > 0:
        welds = welds[:args_cli.max_welds]
    print(f"[viz] 读到 {len(welds)} 条焊缝")

    # ---- 打开原始场景 stage ----
    usd_path = str(Path(args_cli.usd).expanduser())
    ok = omni.usd.get_context().open_stage(usd_path)
    if not ok:
        raise RuntimeError(f"打开 USD 失败: {usd_path}")
    for _ in range(20):                       # 等场景加载
        simulation_app.update()
    print(f"[viz] 已打开场景 {usd_path}")

    lw = args_cli.line_width
    blen = args_cli.bisector_len
    tip_r = max(lw * 0.9, 0.006)

    shots = []          # 待拍照的焊缝: (i, cam_pos, look_at)
    for i, w in enumerate(welds):
        p0 = np.asarray(w["corrected_p0"], float)
        p1 = np.asarray(w["corrected_p1"], float)
        base = f"/World/welds/w{i:05d}"
        # 焊缝（红色圆柱：半径 seam_radius，长度=焊缝长度）
        add_cylinder(f"{base}/seam", p0, p1, args_cli.seam_radius, (1.0, 0.05, 0.05))

        mid = 0.5 * (p0 + p1)
        # bisector（绿 + 端点球）
        if blen > 0 and w.get("bisector"):
            b = np.asarray(w["bisector"], float)
            tip = mid + b * blen
            add_polyline(f"{base}/bisector", [mid, tip], lw * 0.6, (0.1, 1.0, 0.1))
            add_sphere(f"{base}/bisector_tip", tip, tip_r, (0.1, 1.0, 0.1))
        # 每条焊缝一台相机：bisector 正方向 cam_dist 处、看回中点（视线=bisector 反方向）
        if not args_cli.no_cam and w.get("bisector"):
            b = np.asarray(w["bisector"], float)
            nb = float(np.linalg.norm(b))
            if nb > 1e-9:
                cam_pos = mid + (b / nb) * args_cli.cam_dist
                add_camera(f"/World/weld_cams/cam_w{i:05d}", cam_pos, mid,
                           focal=args_cli.cam_focal)
        # 两侧 boundary_dir（黄/青）
        if args_cli.show_boundary and w.get("boundary_dirs"):
            cols = [(1.0, 1.0, 0.1), (0.1, 1.0, 1.0)]
            for k, d in enumerate(w["boundary_dirs"][:2]):
                d = np.asarray(d, float)
                add_polyline(f"{base}/bdir{k}", [mid, mid + d * blen * 0.7],
                             lw * 0.5, cols[k])

        # 记录拍照位姿：相机在 bisector 正方向 dist 处，看向中点 => 拍摄方向=bisector 反方向
        if args_cli.save and w.get("bisector"):
            b = np.asarray(w["bisector"], float)
            nb = float(np.linalg.norm(b))
            if nb > 1e-9:
                cam_pos = mid + (b / nb) * args_cli.save_dist
                shots.append((i, cam_pos, mid))

    print(f"[viz] 焊缝已建（红=焊缝圆柱 绿=bisector"
          f"{' 黄/青=boundary_dir' if args_cli.show_boundary else ''}"
          f"{'' if args_cli.no_cam else '，每条焊缝一台相机 /World/weld_cams/cam_wXXXXX'}）")

    # ---- 拍照 ----
    if args_cli.save:
        _capture_shots(shots)

    if args_cli.save and getattr(args_cli, "headless", False):
        return          # headless 批量拍照完直接退出，不进 GUI 循环

    print("[viz] 场景就绪，GUI 交互中（关闭窗口或 Ctrl+C 退出）...")
    while simulation_app.is_running():
        simulation_app.update()


def _capture_shots(shots):
    """用 replicator 相机逐条焊缝拍 RGB。"""
    try:
        import omni.replicator.core as rep
    except ModuleNotFoundError:
        # 兜底：动态启用扩展再导入（不同 Isaac 版本 enable_extension 路径不同）
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except ModuleNotFoundError:
            from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        simulation_app.update()
        import omni.replicator.core as rep

    out_dir = str(Path(args_cli.save).expanduser())
    os.makedirs(out_dir, exist_ok=True)
    if not shots:
        print("[viz] 没有可拍照的焊缝（缺 bisector），跳过")
        return

    res = (int(args_cli.save_res[0]), int(args_cli.save_res[1]))
    rep.orchestrator.set_capture_on_play(False)
    cam = rep.create.camera()
    rp = rep.create.render_product(cam, res)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])

    # 预热：首帧要编译 RTX 着色器，很慢（几十秒）。先跑一次 step 让它把编译吃掉。
    print("[viz] 预热渲染器（首帧编译着色器，可能数十秒）...")
    for _ in range(5):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=4)
    print("[viz] 渲染器就绪，开始逐条拍照")

    print(f"[viz] 开始拍照 {len(shots)} 张 -> {out_dir} (分辨率 {res[0]}x{res[1]})")
    for n, (i, cam_pos, look_at) in enumerate(shots):
        with cam:
            rep.modify.pose(
                position=(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
                look_at=(float(look_at[0]), float(look_at[1]), float(look_at[2])))
        simulation_app.update()             # 让新位姿生效
        rep.orchestrator.step(rt_subframes=4)
        rgb = annot.get_data()
        path = os.path.join(out_dir, f"weld_{i:05d}.png")
        _save_rgb(rgb, path)
        print(f"[viz]   已拍 {n + 1}/{len(shots)}  -> {os.path.basename(path)}")
    print(f"[viz] 拍照完成，共 {len(shots)} 张，存于 {out_dir}")


if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    main()
    simulation_app.close()
