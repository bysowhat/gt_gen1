import os
import signal
import argparse
import subprocess
import time
import shutil
import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
task_traj = os.path.join(dir_path, "optimize_traj.py")

SUB_FOLDER_POSE = "pose"
SUB_FOLDER_TRAJ = "traj"
POSE_DONE_MARKER = "pose_done"
TRAJ_DONE_MARKER = "traj_done"
BLACKLIST_TRAJ = "blacklist_traj.txt"
CURRENT_SEAM_MARKER_TRAJ = ".current_seam_traj"


class TrajGenerator:
    def __init__(self, input_root: str, output_root: str, done_folder_root: str,
                 num_gpus: int = 2, per_gpu_jobs: int = 2,
                 logger_folder: str = None, traj_timeout_min: int = 3):
        self.input_root = input_root
        self.output_root = output_root
        self.done_folder_root = done_folder_root
        self.NUM_GPUS = num_gpus
        self.PER_GPU_JOBS = per_gpu_jobs
        self.MAX_PARALLEL = self.NUM_GPUS * self.PER_GPU_JOBS
        self.TRAJ_TIMEOUT = traj_timeout_min * 60

        self.logger_folder = logger_folder
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
        """读取 .current_seam_traj 标记，把卡住的 seam 名追加到 output_root/<folder>/blacklist_traj.txt"""
        output_folder = os.path.join(self.output_root, folder_name)
        marker_file = os.path.join(output_folder, CURRENT_SEAM_MARKER_TRAJ)
        if not os.path.exists(marker_file):
            return None
        try:
            with open(marker_file, "r") as f:
                stuck_seam = f.read().strip()
        except Exception:
            return None
        if not stuck_seam:
            return None

        bl_file = os.path.join(output_folder, BLACKLIST_TRAJ)
        try:
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

    def try_finalize(self, folder_name, slot_id=None):
        """pose_done 和 traj_done 均存在时，将 input 子文件夹移动到 done_folder_root。"""
        output_folder = os.path.join(self.output_root, folder_name)
        if not os.path.exists(os.path.join(output_folder, POSE_DONE_MARKER)):
            return
        if not os.path.exists(os.path.join(output_folder, TRAJ_DONE_MARKER)):
            return
        src_folder = os.path.join(self.input_root, folder_name)
        if not os.path.exists(src_folder):
            msg = f"输入文件夹已不存在，跳过移动: {folder_name}"
            if slot_id is not None:
                self._slot_logger(slot_id).info(msg)
            else:
                print(msg)
            return
        dst_folder = os.path.join(self.done_folder_root, folder_name)
        try:
            shutil.move(src_folder, dst_folder)
            msg = f"已移动 {folder_name} 到 {self.done_folder_root}"
            if slot_id is not None:
                self._slot_logger(slot_id).info(msg)
            else:
                print(msg)
        except Exception as e:
            msg = f"移动 {folder_name} 出错: {e}"
            if slot_id is not None:
                self._slot_logger(slot_id).error(msg)
            else:
                print(msg)

    def run(self):
        # os.makedirs(self.done_folder_root, exist_ok=True)

        # ==== Step 1: 找所有含 pose_done 的 output 子文件夹 ====
        all_folders = []
        for folder_name in sorted(os.listdir(self.output_root)):
            output_folder = os.path.join(self.output_root, folder_name)
            if not os.path.isdir(output_folder):
                continue
            if not os.path.exists(os.path.join(output_folder, POSE_DONE_MARKER)):
                continue
            all_folders.append(folder_name)

        print(f"共找到 {len(all_folders)} 个含 pose_done 的文件夹")

        # ==== Step 2: 预检查，已完成的直接标记并尝试移动 ====
        folder_queue = []
        for folder_name in all_folders:
            output_folder = os.path.join(self.output_root, folder_name)
            traj_done_marker = os.path.join(output_folder, TRAJ_DONE_MARKER)

            if os.path.exists(traj_done_marker):
                print(f"已完成 traj（标记存在）: {folder_name}")
                # self.try_finalize(folder_name)
                continue

            traj_subfolder = os.path.join(output_folder, SUB_FOLDER_TRAJ)
            if os.path.exists(traj_subfolder):
                pose_subfolder = os.path.join(output_folder, SUB_FOLDER_POSE)
                pose_seams = {f[:-4] for f in os.listdir(pose_subfolder) if f.endswith(".pkl")} \
                             if os.path.exists(pose_subfolder) else set()
                bl_file = os.path.join(output_folder, BLACKLIST_TRAJ)
                blacklisted = set()
                if os.path.exists(bl_file):
                    with open(bl_file, "r") as f:
                        blacklisted = {line.strip() for line in f if line.strip()}
                expected = pose_seams - blacklisted
                traj_seams = {f[:-4] for f in os.listdir(traj_subfolder) if f.endswith(".pkl")}
                if traj_seams >= expected:
                    open(traj_done_marker, "w").close()
                    print(f"traj 已完整（{len(traj_seams)}/{len(pose_seams)} seam），生成 traj_done: {folder_name}")
                    # self.try_finalize(folder_name)
                    continue
                print(f"traj 未完整（{len(traj_seams)}/{len(pose_seams)} seam，黑名单 {len(blacklisted)} 条），加入队列: {folder_name}")

            folder_queue.append(folder_name)

        print(f"需要执行 traj 任务: {len(folder_queue)} 个")

        if not folder_queue:
            print("所有 traj 任务已完成")
            return

        # ==== Step 3: 定义 traj 任务启动函数 ====
        def start_traj(folder_name, gpu_id, slot_id):
            output_folder = os.path.join(self.output_root, folder_name)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self._slot_logger(slot_id).info(f"[GPU {gpu_id}] === 开始生成轨迹 {folder_name} ===")
            self._cleanup_shm()
            cmd = [
                "/workspace/isaaclab/_isaac_sim/python.sh", task_traj, "--headless",
                "--output_folder", output_folder,
                "--sub_folder_pose", SUB_FOLDER_POSE,
                "--sub_folder_traj", SUB_FOLDER_TRAJ,
                "--device", "cuda",
                "--slot_id", str(slot_id)
            ]
            if self.logger_folder:
                cmd += ["--logger_folder", self.logger_folder]
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
                proc = start_traj(folder_name, gpu_id, slot_id)
                running_jobs.append((proc, gpu_id, folder_name, slot_id, time.time()))
                gpu_load[gpu_id] += 1

            # 检查超时 & 任务完成
            for item in running_jobs[:]:
                proc, gpu_id, folder_name, slot_id, start_time = item
                # per-seam 超时：基于 .current_seam_traj marker 的 mtime
                marker_file = os.path.join(self.output_root, folder_name, CURRENT_SEAM_MARKER_TRAJ)
                if os.path.exists(marker_file):
                    elapsed = time.time() - os.path.getmtime(marker_file)
                else:
                    # marker 尚未建立（启动初始化期），退回到子进程启动时间
                    elapsed = time.time() - start_time

                if elapsed > self.TRAJ_TIMEOUT:
                    stuck = self._record_stuck_seam(folder_name)
                    if stuck:
                        self._slot_logger(slot_id).error(
                            f"[GPU{gpu_id}] !!! traj 超时({elapsed/60:.1f}分钟): {folder_name}，"
                            f"卡住 seam={stuck} 已加入黑名单，强制终止"
                        )
                    else:
                        self._slot_logger(slot_id).error(
                            f"[GPU{gpu_id}] !!! traj 超时({elapsed/60:.1f}分钟): {folder_name}，"
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
                    self._slot_logger(slot_id).error(f"[GPU{gpu_id}] !!! traj 失败(ret={ret}): {folder_name}")
                    self.available_slots.append(slot_id)
                    continue
                if os.path.exists(done_file):
                    os.remove(done_file)

                traj_subfolder = os.path.join(output_folder, SUB_FOLDER_TRAJ)
                traj_done_marker = os.path.join(output_folder, TRAJ_DONE_MARKER)

                if os.path.exists(traj_subfolder):
                    open(traj_done_marker, "w").close()
                    self._slot_logger(slot_id).info(f"[GPU{gpu_id}] <<< 完成 TRAJ，生成 traj_done: {folder_name}")
                    # self.try_finalize(folder_name, slot_id)
                else:
                    self._slot_logger(slot_id).info(f"[GPU{gpu_id}] <<< TRAJ 完成但无结果（output 已被清理）: {folder_name}")

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
    parser.add_argument("--traj_timeout_min", type=int, default=3)
    args = parser.parse_args()

    # os.makedirs(args.done_folder_root, exist_ok=True)

    generator = TrajGenerator(args.input_root, args.output_root, args.done_folder_root,
                              args.num_gpus, args.per_gpu_jobs,
                              args.logger_folder, args.traj_timeout_min)
    generator.run()
