"""关键帧下采样调度器：扫描 --in-dir 下的 Scene pkl，逐个跑 scripts/traj_downsample.py，
另存到 --out-dir（保留原文件名、不覆盖原 pkl）。

替代老流程：不再调用 down_sampling.py、不产老 output/<工件>/traj/seam_N.pkl 格式（方案 D1）。
数据量不大时用 --jobs 1 单进程顺序即可；--jobs>1 时按 GPU 轮转并行（每文件一个子进程）。

用法：
    conda run -n env_isaaclab python traj_sampler/main_controller_sp.py \
        --in-dir /media/a/新加卷/tempt/5 --out-dir /media/a/新加卷/tempt/5_ds
"""
import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_TASK = os.path.join(_ROOT, "scripts", "traj_downsample.py")


def run(in_dir, out_dir, config, num_gpus, per_gpu_jobs):
    pkls = sorted(f for f in os.listdir(in_dir) if f.endswith(".pkl"))
    print(f"共找到 {len(pkls)} 个 Scene pkl，用于关键帧下采样")
    os.makedirs(out_dir, exist_ok=True)

    max_parallel = max(1, num_gpus) * max(1, per_gpu_jobs)
    queue = pkls.copy()
    running = []                       # (proc, gpu_id, fname)
    gpu_load = [0 for _ in range(max(1, num_gpus))]

    def start(fname, gpu_id):
        env = os.environ.copy()
        if num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [sys.executable, _TASK,
               "--pkl", os.path.join(in_dir, fname),
               "--out-dir", out_dir,
               "--config", config,
               "--device", "cuda"]
        print(f"[GPU {gpu_id}] === 开始下采样 {fname} ===")
        return subprocess.Popen(cmd, env=env)

    while queue or running:
        while queue and len(running) < max_parallel:
            avail = [i for i, l in enumerate(gpu_load) if l < max(1, per_gpu_jobs)]
            if not avail:
                break
            fname = queue.pop(0)
            gpu_id = avail[0]
            running.append((start(fname, gpu_id), gpu_id, fname))
            gpu_load[gpu_id] += 1

        for proc, gpu_id, fname in running[:]:
            ret = proc.poll()
            if ret is None:
                continue
            running.remove((proc, gpu_id, fname))
            gpu_load[gpu_id] -= 1
            if ret != 0:
                print(f"[GPU{gpu_id}] !!! 下采样失败: {fname}")
            else:
                print(f"[GPU{gpu_id}] <<< 完成: {fname}")
        time.sleep(0.05)

    print(f"[main] 全部完成 → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scene pkl 关键帧下采样调度器")
    ap.add_argument("--in-dir", required=True, help="输入 Scene pkl 目录")
    ap.add_argument("--out-dir", required=True, help="输出目录（另存、不覆盖原 pkl）")
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "default.yaml"),
                    help="采样参数来源 yaml（默认 configs/default.yaml，读其 traj_downsample 段）")
    ap.add_argument("--num-gpus", type=int, default=1, help="可用 GPU 数（0=不设 CUDA_VISIBLE_DEVICES）")
    ap.add_argument("--per-gpu-jobs", type=int, default=1, help="每 GPU 并发子进程数")
    args = ap.parse_args()
    run(args.in_dir, args.out_dir, args.config, args.num_gpus, args.per_gpu_jobs)
