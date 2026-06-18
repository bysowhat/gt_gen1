import os
import signal
import argparse
import subprocess
import time
import shutil
import logging


dir_path = os.path.dirname(os.path.realpath(__file__))

task_pose = os.path.join(dir_path, "optimize_pose.py")
task_traj = os.path.join(dir_path, "optimize_traj.py")

BLACKLIST_POSE = "blacklist_pose.txt"
BLACKLIST_TRAJ = "blacklist_traj.txt"
CURRENT_SEAM_MARKER_POSE = ".current_seam_pose"
CURRENT_SEAM_MARKER_TRAJ = ".current_seam_traj"


class GTGenerator:
    def __init__(self, input_root: str, output_root: str, done_folder_root: str, sub_folder_pose: str, sub_folder_traj: str, num_gpus: int = 2, per_gpu_jobs: int = 2, logger_folder: str = None, pose_timeout_min: int = 3, traj_timeout_min: int = 3):
        self.input_root = input_root
        self.output_root = output_root
        self.done_folder_root = done_folder_root
        self.sub_folder_pose = sub_folder_pose
        self.sub_folder_traj = sub_folder_traj
        self.logger_folder = logger_folder

        self.NUM_GPUS = num_gpus
        self.PER_GPU_JOBS = per_gpu_jobs
        self.MAX_PARALLEL = self.NUM_GPUS * self.PER_GPU_JOBS
        self.POSE_TIMEOUT = pose_timeout_min * 60
        self.TRAJ_TIMEOUT = traj_timeout_min * 60

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

    def _record_stuck_seam(self, stage, folder_name):
        """读取 .current_seam_{stage} 标记，把卡住的 seam 名追加到对应黑名单。
        pose 黑名单写到 input_root/<folder>/blacklist_pose.txt
        traj 黑名单写到 output_root/<folder>/blacklist_traj.txt
        返回卡住的 seam 名（如果识别到）。"""
        output_folder = os.path.join(self.output_root, folder_name)
        marker_name = CURRENT_SEAM_MARKER_POSE if stage == "pose" else CURRENT_SEAM_MARKER_TRAJ
        marker_file = os.path.join(output_folder, marker_name)
        if not os.path.exists(marker_file):
            return None
        try:
            with open(marker_file, "r") as f:
                stuck_seam = f.read().strip()
        except Exception:
            return None
        if not stuck_seam:
            return None

        if stage == "pose":
            bl_dir = os.path.join(self.input_root, folder_name)
            bl_file = os.path.join(bl_dir, BLACKLIST_POSE)
        else:
            bl_dir = output_folder
            bl_file = os.path.join(bl_dir, BLACKLIST_TRAJ)
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
        # 删除标记，避免下次误读
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

        def start_pose(folder_name, gpu_id, slot_id):
            input_folder = os.path.join(self.input_root, folder_name)
            output_folder = os.path.join(self.output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self._slot_logger(slot_id).info(f"[GPU {gpu_id}] === 开始生成观测位姿 {folder_name} ===")
            self._cleanup_shm()
            cmd = [
                "/workspace/isaaclab/_isaac_sim/python.sh", task_pose, "--headless",
                "--input_folder", input_folder,
                "--output_folder", output_folder,
                "--sub_folder_pose", self.sub_folder_pose,
                "--sub_folder_traj", self.sub_folder_traj,
                "--device", "cuda",
                "--slot_id", str(slot_id)
            ]
            if self.logger_folder:
                cmd += ["--logger_folder", self.logger_folder]
            proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
            return proc

        def start_traj(folder_name, gpu_id, slot_id):
            output_folder = os.path.join(self.output_root, folder_name)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self._slot_logger(slot_id).info(f"[GPU {gpu_id}] === 开始生成轨迹 {folder_name} ===")
            self._cleanup_shm()
            cmd = [
                "/workspace/isaaclab/_isaac_sim/python.sh", task_traj, "--headless",
                "--output_folder", output_folder,
                "--sub_folder_pose", self.sub_folder_pose,
                "--sub_folder_traj", self.sub_folder_traj,
                "--device", "cuda",
                "--slot_id", str(slot_id)
            ]
            if self.logger_folder:
                cmd += ["--logger_folder", self.logger_folder]
            proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
            return proc

        # running_jobs: (process, gpu_id, stage, folder_name, slot_id, start_time)
        running_jobs = []
        folder_queue = all_folders.copy()
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
                running_jobs.append((proc, gpu_id, "pose", folder_name, slot_id, time.time()))
                gpu_load[gpu_id] += 1

            # 检查超时 & 任务完成
            for item in running_jobs[:]:
                proc, gpu_id, stage, folder_name, slot_id, start_time = item
                # per-seam 超时：基于 .current_seam_{stage} marker 的 mtime
                marker_name = CURRENT_SEAM_MARKER_POSE if stage == "pose" else CURRENT_SEAM_MARKER_TRAJ
                marker_file = os.path.join(self.output_root, folder_name, marker_name)
                if os.path.exists(marker_file):
                    elapsed = time.time() - os.path.getmtime(marker_file)
                else:
                    # marker 尚未建立（启动初始化期），退回到子进程启动时间
                    elapsed = time.time() - start_time
                timeout = self.POSE_TIMEOUT if stage == "pose" else self.TRAJ_TIMEOUT

                # 超时：记录卡住 seam 到黑名单，强制终止，释放槽位
                if elapsed > timeout:
                    stuck = self._record_stuck_seam(stage, folder_name)
                    if stuck:
                        self._slot_logger(slot_id).error(
                            f"[GPU{gpu_id}] !!! {stage} 超时({elapsed/60:.1f}分钟): {folder_name}，"
                            f"卡住 seam={stuck} 已加入黑名单，强制终止"
                        )
                    else:
                        self._slot_logger(slot_id).error(
                            f"[GPU{gpu_id}] !!! {stage} 超时({elapsed/60:.1f}分钟): {folder_name}，"
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

                done_file = os.path.join(self.output_root, folder_name, ".done")
                if ret != 0 and not os.path.exists(done_file):
                    self._slot_logger(slot_id).error(f"[GPU{gpu_id}] !!! {stage} 失败(ret={ret}): {folder_name}")
                    self.available_slots.append(slot_id)
                    continue
                if os.path.exists(done_file):
                    os.remove(done_file)
                self._slot_logger(slot_id).info(f"[GPU{gpu_id}] <<< 完成 {stage.upper()}: {folder_name}")

                if stage == "pose":
                    p2 = start_traj(folder_name, gpu_id, slot_id)
                    running_jobs.append((p2, gpu_id, "traj", folder_name, slot_id, time.time()))
                    gpu_load[gpu_id] += 1
                elif stage == "traj":
                    src_folder = os.path.join(self.input_root, folder_name)
                    dst_folder = os.path.join(self.done_folder_root, folder_name)
                    try:
                        shutil.move(src_folder, dst_folder)
                        self._slot_logger(slot_id).info(f"已移动 {folder_name} 到 {self.done_folder_root}")
                    except Exception as e:
                        self._slot_logger(slot_id).error(f"移动 {folder_name} 出错: {e}")
                    self.available_slots.append(slot_id)

            time.sleep(0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--done_folder_root", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--per_gpu_jobs", type=int, default=2)
    parser.add_argument("--logger_folder", type=str, default=None)
    parser.add_argument("--pose_timeout_min", type=int, default=3)
    parser.add_argument("--traj_timeout_min", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.done_folder_root, exist_ok=True)
    sub_folder_pose = "pose"
    sub_folder_traj = "traj"
    generator = GTGenerator(args.input_root, args.output_root, args.done_folder_root,
                            sub_folder_pose, sub_folder_traj, args.num_gpus, args.per_gpu_jobs,
                            args.logger_folder, args.pose_timeout_min, args.traj_timeout_min)
    generator.run()
