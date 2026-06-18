import argparse
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--output_folder', type = str)
parser.add_argument('--sub_folder_pose', type = str)
parser.add_argument('--sub_folder_traj', type = str)
parser.add_argument('--logger_folder', type = str, default=None)
parser.add_argument('--slot_id', type = int, default=0)
# parser.add_argument('--device', type = str)
AppLauncher.add_app_launcher_args(parser)
args_cli=parser.parse_args()
app_launcher=AppLauncher(args_cli)
simulation_app=app_launcher.app
device = args_cli.device

import os
import signal
import shutil
import torch
import logging

import pickle
import json

import itertools

from config_traj import ConfigurationTraj as Configuration
from stomp_traj import StompTraj as Stomp
from scene_traj import SceneTraj as Scene

BLACKLIST_FILE = "blacklist_traj.txt"
CURRENT_SEAM_MARKER = ".current_seam_traj"


class OptimizeTraj:
    def __init__(self, output_folder, sub_folder_pose, sub_folder_traj, device, logger_folder=None, slot_id=0):
        self.device = device
        self.config = Configuration()
        self.output_folder = output_folder
        self.sub_folder_pose = sub_folder_pose
        self.sub_folder_traj = sub_folder_traj

        # 设置 logger：同时写到 stdout 和 logger_folder/parallel_{slot_id}.log
        log_dir = logger_folder if logger_folder else output_folder
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"parallel_{slot_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)
            fh = logging.FileHandler(os.path.join(log_dir, f"parallel_{slot_id}.log"))
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)


    def _load_blacklist(self):
        bl_file = os.path.join(self.output_folder, BLACKLIST_FILE)
        if not os.path.exists(bl_file):
            return set()
        with open(bl_file, "r") as f:
            entries = {line.strip() for line in f if line.strip()}
        if entries:
            self.logger.info(f"读取 traj 黑名单 {len(entries)} 条: {sorted(entries)}")
        return entries


    def run(self,):
        input_folder = os.path.join(self.output_folder, self.sub_folder_pose)
        # 检测是否存在pose sub-folder,不存在则直接删除整个output folder
        if not os.path.exists(input_folder):
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
                self.logger.warning(f"{self.output_folder} 已被删除")
            else:
                self.logger.warning(f"{self.output_folder} 本身不存在")
            return
        out_folder = os.path.join(self.output_folder, self.sub_folder_traj)
        # 提前创建输出文件夹，支持断点续跑
        os.makedirs(out_folder, exist_ok=True)

        # 读取黑名单
        blacklist = self._load_blacklist()

        # 读取该子文件夹下所有文件
        path_dict_path = os.path.join(self.output_folder, f"path.json")
        with open(path_dict_path, 'rb') as f:
            path_dict = json.load(f)
        self.config.usd_path = path_dict["usd_path"]
        self.config.pc_path = path_dict["pc_path"]
        data = {}

        for fname in sorted(os.listdir(input_folder)):
            fpath = os.path.join(input_folder, fname)
            if not fname.endswith(".pkl"):
                continue
            seam_name = fname[:-4]
            if seam_name in blacklist:
                self.logger.info(f"seam {seam_name} 在黑名单，跳过")
                continue
            with open(fpath, 'rb') as f:
                data[seam_name] = pickle.load(f)

        scene = Scene(self.config, device=self.device)

        # 调用处理函数（每完成一条 seam 立即保存，处理前写当前 seam 标记）
        self.process(data, scene, out_folder)


    def process(self, data_dict: dict[str, torch.Tensor], scene: Scene, out_folder: str):
        """
        运行轨迹优化，每完成一条 seam 立即写出 PKL（断点续跑）
        处理每条 seam 之前写入 .current_seam_traj 标记，便于 controller 在超时时识别卡住的 seam
        """
        marker_file = os.path.join(self.output_folder, CURRENT_SEAM_MARKER)
        # 进入处理阶段先 touch marker，避免旧 mtime 让 controller 误判超时
        try:
            with open(marker_file, "w") as f:
                f.write("")
        except Exception:
            pass

        for seam_name, seam_data in data_dict.items():
            out_file = os.path.join(out_folder, f"{seam_name}.pkl")
            if os.path.exists(out_file):
                # 跳过已完成的，但 touch marker 更新 mtime（避免被误判超时）
                try:
                    with open(marker_file, "w") as f:
                        f.write("")
                except Exception:
                    pass
                self.logger.info(f"seam {seam_name} 已存在 {out_file}，跳过")
                continue

            # 写当前正在处理的 seam 标记（controller 超时时读取该标记加入黑名单）
            try:
                with open(marker_file, "w") as f:
                    f.write(seam_name)
            except Exception:
                pass

            self.logger.info(f"开始处理 seam: {seam_name}，共 {len(seam_data)} 条数据")
            output_list = []
            # retrive data for settings
            for i, data in enumerate(seam_data):
                self.logger.info(f"  [{seam_name}] 第 {i+1}/{len(seam_data)} 条轨迹规划开始")
                robot_pose = torch.as_tensor(data["robot_pose"], dtype=torch.float, device=self.device)        # (7)
                fixed_pts = torch.as_tensor(data["joint"], dtype=torch.float, device=self.device)        # (N, M, 7)
                self.config.horizontal = int(data["horizontal"])        # (1,)
                seam_median = torch.as_tensor(data["seam_median"], dtype=torch.float, device=self.device)     # (3,)
                robot_pose_original = torch.as_tensor(data["robot_pose_original"], dtype=torch.float, device=self.device)        # (7)
                piece_pose_original = torch.as_tensor(data["piece_pose_original"], dtype=torch.float, device=self.device)        # (7)
                seam_line = torch.as_tensor(data["seam_line"], dtype=torch.float, device=self.device)
                seam_limits = torch.as_tensor(data["seam_limits"], dtype=torch.float, device=self.device)

                # 生成所有 M! 种排列
                N, M, D = fixed_pts.shape
                perms = list(itertools.permutations(range(M)))   # [(0,1,2), (0,2,1), ...]
                num_perms = len(perms)  # M!
                # 根据排列重新索引
                fixed_pts = torch.stack([fixed_pts[:, perm, :] for perm in perms], dim=1)  # (N, M!, M, D)
                # generate entire fixed points of trajs
                first = scene.initial_joint_pos[:N].unsqueeze(-2).unsqueeze(-2).expand(-1, num_perms, -1, -1)
                fixed_pts = torch.cat((first, fixed_pts), dim=-2)
                # select out the shortest trajs
                dist = torch.sum(torch.norm((fixed_pts[:, :, 1:, :] - fixed_pts[:, :, :-1, :]), dim=-1), dim=-1)    # (N, M!)
                min_idx = torch.argmin(dist, dim=-1)    # (N,)
                batch_idx = torch.arange(N, device=self.device)
                fixed_pts = fixed_pts[batch_idx, min_idx]   # (N, M, D)

                num_batch = self.config.num_batch
                mul = (num_batch + N - 1) // N
                fixed_pts = fixed_pts.repeat(mul, 1, 1)[:num_batch]

                # generate trajectories
                scene.reset(robot_pose, seam_median)
                stomp = Stomp(config, scene, self.device)

                trajectpry_list = []
                idx_3d_list = []

                # print("fixed_pts shape:", fixed_pts.shape)

                for j in range(fixed_pts.shape[1]-1):
                    path = fixed_pts[:, j:j+2].clone()
                    has_vision = self.config.has_vision
                    trajectory, _ = stomp.solve(path, has_vision)
                    if j == 0:
                        trajectpry_list.append(trajectory.clone())
                    else:
                        trajectory = trajectory[..., 1:].clone()
                        trajectpry_list.append(trajectory.clone())
                    idx_3d_part = torch.zeros((trajectory.shape[-1],), dtype=torch.int, device=self.device)
                    idx_3d_part[-1] = 1
                    idx_3d_list.append(idx_3d_part.clone())

                trajectory = torch.cat(trajectpry_list, dim=-1).to(self.device)
                idx_3d = torch.cat(idx_3d_list, dim=-1).to(self.device)

                # save outputs
                output_single = {}
                output_single["robot_pose"] = robot_pose.clone()        # (7,)
                output_single["trajectory"] = trajectory.clone()        # (N', D, M)
                output_single["idx_3d"] = idx_3d.clone()                # (M,)
                output_single["piece_pose_original"] = piece_pose_original.clone()
                output_single["robot_pose_original"] = robot_pose_original.clone()
                output_single["seam_line"] = seam_line.clone()
                output_single["seam_limits"] = seam_limits.clone()
                output_list.append(output_single.copy())
                self.logger.info(f"  [{seam_name}] 第 {i+1}/{len(seam_data)} 条轨迹规划完成")

            if output_list:
                with open(out_file, "wb") as f:
                    pickle.dump(output_list, f)
                self.logger.info(f"保存 {out_file}（{len(output_list)}/{len(seam_data)} 条有效轨迹）")
            else:
                self.logger.warning(f"seam {seam_name} 无有效轨迹，未保存")

        # 全部 seam 处理完毕，清空 marker 内容并更新 mtime（不删除，避免与 controller 超时检测竞争）
        try:
            with open(marker_file, "w") as f:
                f.write("")
        except Exception:
            pass


if __name__ == "__main__":
    device = "cuda"
    config = Configuration()

    opt = OptimizeTraj(args_cli.output_folder, args_cli.sub_folder_pose,
                       args_cli.sub_folder_traj, device, args_cli.logger_folder, args_cli.slot_id)
    opt.run()
    # 写 sentinel 文件表示任务正常完成
    open(os.path.join(args_cli.output_folder, ".done"), "w").close()
    os.kill(os.getpid(), signal.SIGKILL)
