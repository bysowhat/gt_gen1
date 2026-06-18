import os
import subprocess
import time

import torch

import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

task_pose = os.path.join(dir_path, "optimize_pose.py")
task_traj = os.path.join(dir_path, "optimize_traj.py")

class GTGenerator:
    def __init__(self, input_root: str, output_root: str, sub_folder_pose: str, sub_folder_traj: str):
        self.input_root = input_root
        self.output_root = output_root

        self.sub_folder_pose = sub_folder_pose
        self.sub_folder_traj = sub_folder_traj
    
    def run(self,):
        os.makedirs(self.output_root, exist_ok=True)

        # 遍历输入路径下的所有子文件夹
        for folder_name in sorted(os.listdir(self.input_root)):
            input_folder = os.path.join(self.input_root, folder_name)

            # 读取该子文件夹下所有文件,检查完整性
            data = {}
            has_usd = False
            for fname in sorted(os.listdir(input_folder)):
                fpath = os.path.join(input_folder, fname)
                if not fname.endswith(".pkl"):
                    if fname.endswith(".usd"):
                        has_usd = True
                    continue
                else:
                    with open(fpath, 'rb') as f:  # 以二进制方式读取
                        data[fname[:-4]] = pickle.load(f)
            
            if not data:
                print(f"跳过空文件夹 {input_folder}")
                continue
            if not has_usd:
                print(f"文件夹不完整 {input_folder}")
                continue

            output_folder = os.path.join(self.output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(0)
            # 生成观测位姿
            print(f"=== 开始生成观测位姿 {input_folder} ===")
            subprocess.run(["python", task_pose, "--headless", "--input_folder", input_folder, 
                            "--output_folder", output_folder, "--sub_folder_pose", self.sub_folder_pose, 
                            "--sub_folder_traj", self.sub_folder_traj, "--device", "cuda:0"], check=True, env=env)
            print(f"=== 结束生成观测位姿 {input_folder} ===")

            # 生成轨迹
            print(f"=== 开始关节轨迹 {output_folder} ===")
            subprocess.run(["python", task_traj, "--headless", "--output_folder", output_folder, 
                            "--sub_folder_pose", self.sub_folder_pose, "--sub_folder_traj", self.sub_folder_traj, 
                            "--device", "cuda:0"], check=True, env=env)
            print(f"=== 结束关节轨迹 {output_folder} ===")


if __name__ == "__main__":
    input_root = "/home/kejian/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/gt_overall/gt_data/input"
    output_root = "/home/kejian/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/gt_overall/gt_data/output"
    sub_folder_pose = "pose"
    sub_folder_traj = "traj"
    generator = GTGenerator(input_root, output_root, sub_folder_pose, sub_folder_traj)
    start_time = time.time()
    generator.run()
    end_time = time.time()
    print("duration:", end_time - start_time)