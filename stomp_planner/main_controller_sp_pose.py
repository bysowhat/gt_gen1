import os
import signal
import argparse
import subprocess
import time
import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
task_pose = os.path.join(dir_path, "optimize_pose.py")

SUB_FOLDER_POSE = "pose"
SUB_FOLDER_TRAJ = "traj"
POSE_DONE_MARKER = "pose_done"
BLACKLIST_POSE = "blacklist_pose.txt"
CURRENT_SEAM_MARKER_POSE = ".current_seam_pose"


class PoseGenerator:
    def __init__(self, input_root: str, output_root: str, num_gpus: int = 2, per_gpu_jobs: int = 2,
                 logger_folder: str = None, pose_timeout_min: int = 3, reference_root: str = None):
        self.input_root = input_root
        self.output_root = output_root
        self.NUM_GPUS = num_gpus
        self.PER_GPU_JOBS = per_gpu_jobs
        self.MAX_PARALLEL = self.NUM_GPUS * self.PER_GPU_JOBS
        self.POSE_TIMEOUT = pose_timeout_min * 60

        self.logger_folder = logger_folder
        self.reference_root = reference_root
        if logger_folder:
            os.makedirs(logger_folder, exist_ok=True)
        self.available_slots = list(range(self.MAX_PARALLEL))
        self._slot_loggers = {}

    def _cleanup_shm(self):
        import glob
        for path in glob.glob('/dev/shm/carb-RStringInternals-*'):
            pid_str = path.rsplit('-', 1)[-1]
            try:
                pid = int(pid_str)
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                for f in [path, f'/dev/shm/sem.carb-RStringInternals-{pid_str}']:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            except PermissionError:
                pass
        for f in ['/dev/shm/sem.carbonite-sharedmemory', '/dev/shm/carbonite-sharedmemory']:
            try:
                os.remove(f)
            except Exception:
                pass

    def _kill_proc(self, proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _record_stuck_seam(self, folder_name):
        """读取 .current_seam_pose 标记，把卡住的 seam 名追加到 input_root/<folder>/blacklist_pose.txt"""
        output_folder = os.path.join(self.output_root, folder_name)
        marker_file = os.path.join(output_folder, CURRENT_SEAM_MARKER_POSE)
        if not os.path.exists(marker_file):
            return None
        try:
            with open(marker_file, "r") as f:
                stuck_seam = f.read().strip()
        except Exception:
            return None
        if not stuck_seam:
            return None

        bl_dir = os.path.join(self.input_root, folder_name)
        bl_file = os.path.join(bl_dir, BLACKLIST_POSE)
        try:
            os.makedirs(bl_dir, exist_ok=True)
            existing = set()
            if os.path.exists(bl_file):
                with open(bl_file, "r") as f:
                    existing = {line.strip() for line in f if line.strip()}
            if stuck_seam not in existing:
                with open(bl_file, "a") as f:
                    f.write(stuck_seam + "\n")
        except Exception:
            pass
        try:
            os.remove(marker_file)
        except Exception:
            pass
        return stuck_seam

    def _slot_logger(self, slot_id: int) -> logging.Logger:
        if slot_id not in self._slot_loggers:
            log_dir = self.logger_folder if self.logger_folder else self.output_root
            os.makedirs(log_dir, exist_ok=True)
            lg = logging.getLogger(f"parallel_{slot_id}")
            lg.setLevel(logging.INFO)
            if not lg.handlers:
                fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
                sh = logging.StreamHandler()
                sh.setFormatter(fmt)
                lg.addHandler(sh)
                fh = logging.FileHandler(os.path.join(log_dir, f"parallel_{slot_id}.log"))
                fh.setFormatter(fmt)
                lg.addHandler(fh)
            self._slot_loggers[slot_id] = lg
        return self._slot_loggers[slot_id]

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

        # ==== Step 2: 预检查 ====
        folder_queue = []
        for folder_name in all_folders:
            output_folder = os.path.join(self.output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            pose_done_marker = os.path.join(output_folder, POSE_DONE_MARKER)

            if os.path.exists(pose_done_marker):
                print(f"已完成 pose（标记存在）: {folder_name}")
                continue

            pose_subfolder = os.path.join(output_folder, SUB_FOLDER_POSE)
            if os.path.exists(pose_subfolder):
                open(pose_done_marker, "w").close()
                print(f"pose 子文件夹已存在，直接生成 pose_done: {folder_name}")
                continue

            folder_queue.append(folder_name)

        print(f"需要执行 pose 任务: {len(folder_queue)} 个")
        if not folder_queue:
            print("所有 pose 任务已完成")
            return

        # ==== Step 3: 启动 pose 任务 ====
        def start_pose(folder_name, gpu_id, slot_id):
            input_folder = os.path.join(self.input_root, folder_name)
            output_folder = os.path.join(self.output_root, folder_name)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self._slot_logger(slot_id).info(f"[GPU {gpu_id}] === 开始生成观测位姿 {folder_name} ===")
            self._cleanup_shm()
            cmd = [
                "/workspace/isaaclab/_isaac_sim/python.sh", task_pose, "--headless",
                "--input_folder", input_folder,
                "--output_folder", output_folder,
                "--sub_folder_pose", SUB_FOLDER_POSE,
                "--sub_folder_traj", SUB_FOLDER_TRAJ,
                "--device", "cuda",
                "--slot_id", str(slot_id)
            ]
            if self.logger_folder:
                cmd += ["--logger_folder", self.logger_folder]
            if self.reference_root:
                ref_folder = os.path.join(self.reference_root, folder_name, SUB_FOLDER_POSE)
                if os.path.isdir(ref_folder):
                    cmd += ["--reference_folder", ref_folder]
            proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
            return proc

        # ==== Step 4: 主调度循环 ====
        # running_jobs: (proc, gpu_id, folder_name, slot_id, start_time)
        running_jobs = []
        gpu_load = [0] * self.NUM_GPUS

        while folder_queue or running_jobs:

            # 启动新任务
            while folder_queue and self.available_slots:
                available_gpus = [i for i, load in enumerate(gpu_load) if load < self.PER_GPU_JOBS]
                if not available_gpus:
                    break
                folder_name = folder_queue.pop(0)
                gpu_id = available_gpus[0]
                slot_id = self.available_slots.pop(0)
                proc = start_pose(folder_name, gpu_id, slot_id)
                running_jobs.append((proc, gpu_id, folder_name, slot_id, time.time()))
                gpu_load[gpu_id] += 1

            # 检查超时 & 任务完成
            for item in running_jobs[:]:
                proc, gpu_id, folder_name, slot_id, start_time = item
                # per-seam 超时：基于 .current_seam_pose marker 的 mtime
                marker_file = os.path.join(self.output_root, folder_name, CURRENT_SEAM_MARKER_POSE)
                if os.path.exists(marker_file):
                    elapsed = time.time() - os.path.getmtime(marker_file)
                else:
                    # marker 尚未建立（启动初始化期），退回到子进程启动时间
                    elapsed = time.time() - start_time

                if elapsed > self.POSE_TIMEOUT:
                    stuck = self._record_stuck_seam(folder_name)
                    if stuck:
                        self._slot_logger(slot_id).error(
                            f"[GPU{gpu_id}] !!! pose 超时({elapsed/60:.1f}分钟): {folder_name}，"
                            f"卡住 seam={stuck} 已加入黑名单，强制终止"
                        )
                    else:
                        self._slot_logger(slot_id).error(
                            f"[GPU{gpu_id}] !!! pose 超时({elapsed/60:.1f}分钟): {folder_name}，"
                            f"未识别到卡住 seam（marker 缺失），强制终止"
                        )
                    self._kill_proc(proc)
                    running_jobs.remove(item)
                    gpu_load[gpu_id] -= 1
                    self.available_slots.append(slot_id)
                    continue

                ret = proc.poll()
                if ret is None:
                    continue

                running_jobs.remove(item)
                gpu_load[gpu_id] -= 1

                output_folder = os.path.join(self.output_root, folder_name)
                done_file = os.path.join(output_folder, ".done")
                if ret != 0 and not os.path.exists(done_file):
                    self._slot_logger(slot_id).error(f"[GPU{gpu_id}] !!! pose 失败(ret={ret}): {folder_name}")
                    self.available_slots.append(slot_id)
                    continue
                if os.path.exists(done_file):
                    os.remove(done_file)

                pose_subfolder = os.path.join(output_folder, SUB_FOLDER_POSE)
                pose_done_marker = os.path.join(output_folder, POSE_DONE_MARKER)

                if os.path.exists(pose_subfolder):
                    open(pose_done_marker, "w").close()
                    self._slot_logger(slot_id).info(f"[GPU{gpu_id}] <<< 完成 POSE，生成 pose_done: {folder_name}")
                else:
                    self._slot_logger(slot_id).info(f"[GPU{gpu_id}] <<< POSE 完成但无结果（无观测位姿解）: {folder_name}")

                self.available_slots.append(slot_id)

            time.sleep(0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--per_gpu_jobs", type=int, default=2)
    parser.add_argument("--logger_folder", type=str, default=None)
    parser.add_argument("--pose_timeout_min", type=int, default=3)
    parser.add_argument("--reference_root", type=str, default=None)
    args = parser.parse_args()

    generator = PoseGenerator(args.input_root, args.output_root, args.num_gpus, args.per_gpu_jobs,
                              args.logger_folder, args.pose_timeout_min, args.reference_root)
    generator.run()
