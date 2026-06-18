import os
import argparse
import subprocess
import time
import shutil
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

task_pose = os.path.join(dir_path, "optimize_pose.py")
task_traj = os.path.join(dir_path, "optimize_traj.py")


class GTGenerator:
    def __init__(self, input_root: str, output_root: str, done_folder_root: str, sub_folder_pose: str, sub_folder_traj: str, num_gpus: int = 2, per_gpu_jobs: int = 2):
        self.input_root = input_root
        self.output_root = output_root
        self.done_folder_root = done_folder_root
        self.sub_folder_pose = sub_folder_pose
        self.sub_folder_traj = sub_folder_traj

        self.NUM_GPUS = num_gpus
        self.PER_GPU_JOBS = per_gpu_jobs
        self.MAX_PARALLEL = self.NUM_GPUS * self.PER_GPU_JOBS

    def start_mps_for_gpu(self, gpu_id):
        pipe_dir = f"/tmp/nvidia-mps-pipe{gpu_id}"
        log_dir = f"/tmp/nvidia-mps-log{gpu_id}"
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir

        subprocess.run(["sudo", "nvidia-cuda-mps-control", "-d"], env=env)
        print(f"[GPU {gpu_id}] MPS started")

    def stop_mps_for_gpu(self, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run("echo quit | sudo nvidia-cuda-mps-control", shell=True, env=env)
        print(f"[GPU {gpu_id}] MPS stopped")

    def run(self):
        os.makedirs(self.output_root, exist_ok=True)

        # ==== Step 1: 找所有有效 folder ====
        all_folders = []
        for folder_name in sorted(os.listdir(self.input_root)):
            input_folder = os.path.join(self.input_root, folder_name)
            if not os.path.isdir(input_folder):
                continue
            has_pkl = False
            has_usd = False
            for fname in os.listdir(input_folder):
                if fname.endswith(".pkl"):
                    has_pkl = True
                elif fname.endswith(".usd"):
                    has_usd = True
                if has_pkl and has_usd:
                    break
            if not has_pkl:
                print(f"跳过空文件夹 {input_folder}")
                continue
            if not has_usd:
                print(f"文件夹不完整 {input_folder}")
                continue
            all_folders.append(folder_name)

        print(f"共找到 {len(all_folders)} 个有效文件夹")

        # ==== Step 2: 启动 MPS ====
        for gpu_id in range(self.NUM_GPUS):
            self.start_mps_for_gpu(gpu_id)
            time.sleep(0.2)  # 确保 MPS 启动完成

        # ==== Step 3: 启动 pose 任务 ====
        def start_pose(folder_name, gpu_id):
            input_folder = os.path.join(self.input_root, folder_name)
            output_folder = os.path.join(self.output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-pipe{gpu_id}"
            env["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-mps-log{gpu_id}"

            # 生成观测位姿
            print(f"[GPU {gpu_id}] === 开始生成观测位姿 {folder_name} ===")
            proc = subprocess.Popen([
                "/workspace/isaaclab/_isaac_sim/python.sh", task_pose, "--headless",
                "--input_folder", input_folder,
                "--output_folder", output_folder,
                "--sub_folder_pose", self.sub_folder_pose,
                "--sub_folder_traj", self.sub_folder_traj,
                "--device", "cuda"
            ], env=env)

            return proc

        # ==== Step 4: 启动 traj 任务 ====
        def start_traj(folder_name, gpu_id):
            output_folder = os.path.join(self.output_root, folder_name)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-pipe{gpu_id}"
            env["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-mps-log{gpu_id}"

            # 生成轨迹
            print(f"[GPU {gpu_id}] === 开始生成轨迹 {folder_name} ===")
            proc = subprocess.Popen([
                "/workspace/isaaclab/_isaac_sim/python.sh", task_traj, "--headless",
                "--output_folder", output_folder,
                "--sub_folder_pose", self.sub_folder_pose,
                "--sub_folder_traj", self.sub_folder_traj,
                "--device", "cuda"
            ], env=env)

            return proc

        # ==== Step 5: 主调度循环 ====
        running_jobs = []  # (process, gpu_id, stage, folder_name)
        folder_queue = all_folders.copy()
        gpu_load = [0] * self.NUM_GPUS
        
        while folder_queue or running_jobs:
            
            # 启动新任务
            while folder_queue:
                # 找一个空闲 GPU
                available_gpus = [i for i, load in enumerate(gpu_load) if load < self.PER_GPU_JOBS]
                if not available_gpus:
                    break   # 没有空闲 GPU，等待

                folder_name = folder_queue.pop(0)
                gpu_id = available_gpus[0]

                proc = start_pose(folder_name, gpu_id)
                running_jobs.append((proc, gpu_id, "pose", folder_name))
                gpu_load[gpu_id] += 1

            # 检查任务完成
            for proc, gpu_id, stage, folder_name in running_jobs[:]:
                ret = proc.poll()
                if ret is None:
                    continue
                
                # 移除已结束的任务
                running_jobs.remove((proc, gpu_id, stage, folder_name))
                gpu_load[gpu_id] -= 1   # 释放 GPU

                done_file = os.path.join(self.output_root, folder_name, ".done")
                if ret != 0 and not os.path.exists(done_file):
                    print(f"[GPU{gpu_id}] !!! {stage} 失败(ret={ret}): {folder_name}")
                    continue
                if os.path.exists(done_file):
                    os.remove(done_file)
                print(f"[GPU{gpu_id}] <<< 完成 {stage.upper()}: {folder_name}")

                if stage == "pose":
                    # pose 完成 → 启动 traj
                    p2 = start_traj(folder_name, gpu_id)
                    running_jobs.append((p2, gpu_id, "traj", folder_name))
                    gpu_load[gpu_id] += 1
                elif stage == "traj":
                    # traj 完成 → 移动文件夹
                    src_folder = os.path.join(self.input_root, folder_name)
                    dst_folder = os.path.join(self.done_folder_root, folder_name)
                    try:
                        shutil.move(src_folder, dst_folder)
                        print(f"已移动 {folder_name} 到 {self.done_folder_root}")
                    except Exception as e:
                        print(f"移动 {folder_name} 出错: {e}")

            time.sleep(0.3)

        # ==== Step 6: 停止 MPS ====
        for gpu_id in range(self.NUM_GPUS):
            self.stop_mps_for_gpu(gpu_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--done_folder_root", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--per_gpu_jobs", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.done_folder_root, exist_ok=True)

    sub_folder_pose = "pose"
    sub_folder_traj = "traj"
    generator = GTGenerator(args.input_root, args.output_root, args.done_folder_root, sub_folder_pose, sub_folder_traj, args.num_gpus, args.per_gpu_jobs)
    generator.run()
