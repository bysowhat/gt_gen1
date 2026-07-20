"""对新 Scene pkl 轨迹做【关键帧稀疏采样】，另存到指定输出目录（不覆盖原 pkl）。

消费 scripts/demo_scene.py 产出的 Scene pkl（Scene.load + summarize_trajectories），
对每条成功轨迹（status=="reached"）的 positions 做关键帧下采样，得 (L,8) 数组
（列 = [q1..q6, observe, goal]，见方案 D5），写进 Scene.sampled_trajectories
（与 self.trajectories 平行、1:1 对齐），再 Scene.save 到 --out-dir 下【保留原文件名】。

FK 全部取自 config（robot.cfg_path 的 curobo 运动学 + sensor.camera 外参，见
traj_sampler/geometry_sampling/kinematics.CameraFKConfig / 方案 D6），无写死值。
需 curobo + GPU（env_isaaclab 环境）。

用法（单文件）：
    conda run -n env_isaaclab python scripts/traj_downsample.py \
        --pkl /media/a/新加卷/tempt/5/xxx_type2_seam0.pkl --out-dir /media/a/新加卷/tempt/5_ds
用法（整目录）：
    conda run -n env_isaaclab python scripts/traj_downsample.py \
        --in-dir /media/a/新加卷/tempt/5 --out-dir /media/a/新加卷/tempt/5_ds
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TS = os.path.join(_ROOT, "traj_sampler")
for _p in (_ROOT, _TS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_CFG = os.path.join(_ROOT, "configs", "default.yaml")


def _load_sampling_params(config_path):
    """从 configs/default.yaml 的 traj_downsample 段读取采样参数（单一真源）。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("traj_downsample", {}) or {}
    return dict(
        w_trans=float(d.get("w_trans", 1.0)),
        w_rot=float(d.get("w_rot", 0.1)),
        D_target=float(d.get("D_target", 0.1)),
        node_step=int(d.get("node_step", 50)),
    )


def downsample_pkl(pkl_path, out_dir, sp, fk=None, device="cuda", verbose=True):
    """对单个 Scene pkl 采样 → 写 sampled_trajectories → 另存到 out_dir/<原文件名>。

    fk 可传入已构建的 CameraFKConfig 复用（整目录批处理时避免重复初始化 curobo）；
    为 None 时用 scene.cfg 现建一个。返回输出路径。
    """
    from gt_gen.scene import Scene
    from geometry_sampling import CameraFKConfig, sample_keyframes_single

    scene = Scene.load(pkl_path)
    summary = scene.summarize_trajectories(verbose=verbose)

    if fk is None:
        fk = CameraFKConfig(scene.cfg, device=device)

    n_traj = n_sampled = 0
    sampled = {}
    for sid, per_key in scene.trajectories.items():
        sampled.setdefault(sid, {})
        for key, entries in per_key.items():          # key = (hand, index)
            out_list = []
            for entry in entries:
                n_traj += 1
                if entry.get("status") != "reached":
                    out_list.append(None)             # 非成功：占位 None，保持下标对齐
                    continue
                positions = entry["positions"]                 # (T,6)
                observe = entry["observe"]                     # (T,1)
                goal = entry["goal"]                           # (T,1)
                cam_poses = fk.forward_cam_pose(positions)     # (T,4,4)，FK 全取自 config
                actions = sample_keyframes_single(
                    cam_poses, positions, observe, goal,
                    w_trans=sp["w_trans"], w_rot=sp["w_rot"],
                    D_target=sp["D_target"], node_step=sp["node_step"])
                out_list.append(actions)                       # (L,8)
                n_sampled += 1
            sampled[sid][key] = out_list
    scene.sampled_trajectories = sampled

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(pkl_path))
    scene.save(out_path)                              # 另存（不覆盖原 pkl，D8）
    if verbose:
        print(f"[traj_downsample] {os.path.basename(pkl_path)}："
              f"采样 {n_sampled}/{n_traj} 条 → {out_path}")
    return out_path, fk


def main():
    ap = argparse.ArgumentParser(description="Scene pkl 轨迹关键帧下采样（另存到 --out-dir，不覆盖原文件）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkl", help="单个 Scene pkl")
    g.add_argument("--in-dir", help="输入目录（处理其下所有 .pkl）")
    ap.add_argument("--out-dir", required=True, help="输出目录（另存、保留原文件名、不覆盖原 pkl）")
    ap.add_argument("--config", default=_DEFAULT_CFG,
                    help="采样参数来源 yaml（默认 configs/default.yaml，读其 traj_downsample 段）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sp = _load_sampling_params(args.config)

    if args.pkl:
        downsample_pkl(args.pkl, args.out_dir, sp, device=args.device)
        return

    pkls = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".pkl"))
    print(f"[traj_downsample] {args.in_dir} 下 {len(pkls)} 个 pkl 待采样")
    fk = None
    for i, fname in enumerate(pkls):
        pkl_path = os.path.join(args.in_dir, fname)
        print(f"[traj_downsample] ({i + 1}/{len(pkls)}) {fname}")
        # 同一 config → CameraFKConfig 可跨文件复用（省去重复初始化 curobo）
        _, fk = downsample_pkl(pkl_path, args.out_dir, sp, fk=fk, device=args.device)


if __name__ == "__main__":
    main()
