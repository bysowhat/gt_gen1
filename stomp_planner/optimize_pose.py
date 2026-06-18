# import os
import signal
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--input_folder', type = str)
parser.add_argument('--output_folder', type = str)
parser.add_argument('--sub_folder_pose', type = str)
parser.add_argument('--sub_folder_traj', type = str)
parser.add_argument('--logger_folder', type = str, default=None)
parser.add_argument('--slot_id', type = int, default=0)
parser.add_argument('--reference_folder', type = str, default=None)
# parser.add_argument('--device', type = str)
AppLauncher.add_app_launcher_args(parser)
args_cli=parser.parse_args()
app_launcher=AppLauncher(args_cli)
simulation_app=app_launcher.app
device = args_cli.device


import torch
import os
import pickle
import json
import logging

from config_pose import ConfigurationPose as Configuration
from optimizer_pose import OptimizerPose as Optimizer
from scene_pose import ScenePose as Scene

BLACKLIST_FILE = "blacklist_pose.txt"
CURRENT_SEAM_MARKER = ".current_seam_pose"


class OptimizePose:
    def __init__(self, input_folder, output_folder, sub_folder_pose, device, logger_folder=None, slot_id=0, reference_folder=None):
        self.device = device
        self.config = Configuration()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.sub_folder_pose = sub_folder_pose
        self.reference_folder = reference_folder

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
        bl_file = os.path.join(self.input_folder, BLACKLIST_FILE)
        if not os.path.exists(bl_file):
            return set()
        with open(bl_file, "r") as f:
            entries = {line.strip() for line in f if line.strip()}
        if entries:
            self.logger.info(f"读取 pose 黑名单 {len(entries)} 条: {sorted(entries)}")
        return entries


    def run(self,):
        out_sub_folder = os.path.join(self.output_folder, self.sub_folder_pose)
        # 提前创建输出文件夹，支持断点续跑
        os.makedirs(out_sub_folder, exist_ok=True)

        # 读取黑名单
        blacklist = self._load_blacklist()

        # 读取该子文件夹下所有文件
        data = {}
        for fname in sorted(os.listdir(self.input_folder)):
            fpath = os.path.join(self.input_folder, fname)
            if not fname.endswith(".pkl"):
                if fname.endswith(".usd"):
                    self.config.usd_path = fpath
                if fname.endswith(".ply"):
                    self.config.pc_path = fpath
                continue
            seam_name = fname[:-4]
            if seam_name in blacklist:
                self.logger.info(f"seam {seam_name} 在黑名单，跳过")
                continue
            with open(fpath, 'rb') as f:
                data[seam_name] = pickle.load(f)

        # 若指定了 reference_folder，只保留其中存在有效位姿的 seam
        if self.reference_folder and os.path.isdir(self.reference_folder):
            valid_seams = {f[:-4] for f in os.listdir(self.reference_folder) if f.endswith('.pkl')}
            before = len(data)
            data = {k: v for k, v in data.items() if k in valid_seams}
            self.logger.info(f"reference 过滤：{before} 条 seam → {len(data)} 条有效 seam")

        # 提前保存 path.json（traj 阶段读取），即使后续处理被中断也能保留路径信息
        path_file = os.path.join(self.output_folder, f"path.json")
        path_dict = {}
        path_dict["usd_path"] = self.config.usd_path
        path_dict["pc_path"] = self.config.pc_path
        with open(path_file, "w") as f:
            json.dump(path_dict, f, indent=4)

        scene = Scene(self.config, device=self.device)

        # 调用处理函数（每完成一条 seam 立即保存，处理前写当前 seam 标记）
        self.process(data, scene, out_sub_folder)


    def process(self, data_dict: dict[str, torch.Tensor], scene: Scene, out_sub_folder: str):
        """
        运行位姿优化，每完成一条 seam 立即写出 PKL（断点续跑）
        处理每条 seam 之前写入 .current_seam_pose 标记，便于 controller 在超时时识别卡住的 seam
        """
        marker_file = os.path.join(self.output_folder, CURRENT_SEAM_MARKER)
        # 进入处理阶段先 touch marker，避免旧 mtime 让 controller 误判超时
        try:
            with open(marker_file, "w") as f:
                f.write("")
        except Exception:
            pass

        for seam_name, seam_data in data_dict.items():
            out_file = os.path.join(out_sub_folder, f"{seam_name}.pkl")
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

            self.logger.info(f"开始处理 seam: {seam_name}")
            # retrive data for settings
            robot_pose = torch.as_tensor(seam_data["robot_pose"], dtype=torch.float, device=self.device)        # (M, 7)
            piece_pose = torch.as_tensor(seam_data["piece_pose"], dtype=torch.float, device=self.device)        # (M, 7)
            seam_line = torch.as_tensor(seam_data["seam_line"], dtype=torch.float, device=self.device)          # (N, 3)
            seam_tangent = torch.as_tensor(seam_data["seam_tangent"], dtype=torch.float, device=self.device)    # (N, 3)
            seam_limits = torch.as_tensor(seam_data["seam_limits"], dtype=torch.float, device=self.device)      # (N, 2, 3)
            horizontal = torch.as_tensor(seam_data["horizontal"], dtype=torch.int, device=self.device)        # (M,)
            median_idx = seam_line.shape[0] // 2
            seam_median = seam_line[median_idx]       # (3,)

            output_list = []
            total = robot_pose.shape[0]
            for i in range(total):
                self.logger.info(f"  [{seam_name}] 第 {i+1}/{total} 个位姿优化开始")
                # reset scene
                robot_pose_rel = scene.reset(robot_pose[i], int(horizontal[i]), piece_pose[i])
                optimizer = Optimizer(cfg=self.config, scene=scene, device=self.device)
                optimizer.resetSeamData(seam_line, seam_tangent, seam_limits)

                # solve optimization problem
                cam_pose_save, joint_save, start_pts_save, end_pts_save = optimizer.solve()
                if cam_pose_save is not None:
                    output_single = {}
                    output_single["robot_pose"] = robot_pose_rel.clone()
                    output_single["cam_pose"] = cam_pose_save.clone()
                    output_single["joint"] = joint_save.clone()
                    output_single["start_pts"] = start_pts_save.clone()
                    output_single["end_pts"] = end_pts_save.clone()
                    output_single["horizontal"] = int(horizontal[i])
                    output_single["seam_median"] = seam_median.clone()
                    output_single["piece_pose_original"] = piece_pose[i].clone()
                    output_single["robot_pose_original"] = robot_pose[i].clone()
                    output_single["seam_line"] = seam_line.clone()
                    output_single["seam_limits"] = seam_limits.clone()
                    output_list.append(output_single.copy())
                    self.logger.info(f"  [{seam_name}] 第 {i+1}/{total} 个位姿优化成功")
                else:
                    self.logger.info(f"  [{seam_name}] 第 {i+1}/{total} 个位姿优化无解")

            if output_list:
                with open(out_file, "wb") as f:
                    pickle.dump(output_list, f)
                self.logger.info(f"保存 {out_file}（{len(output_list)}/{total} 个有效位姿）")
            else:
                self.logger.warning(f"seam {seam_name} 无有效位姿，未保存")

        # 全部 seam 处理完毕，清空 marker 内容并更新 mtime（不删除，避免与 controller 超时检测竞争）
        try:
            with open(marker_file, "w") as f:
                f.write("")
        except Exception:
            pass


if __name__ == "__main__":
    config = Configuration()
    opt = OptimizePose(args_cli.input_folder, args_cli.output_folder, args_cli.sub_folder_pose,
                       device, args_cli.logger_folder, args_cli.slot_id, args_cli.reference_folder)
    opt.run()
    # 写 sentinel 文件表示任务正常完成
    open(os.path.join(args_cli.output_folder, ".done"), "w").close()
    os.kill(os.getpid(), signal.SIGKILL)
