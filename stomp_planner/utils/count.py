import pickle
import json
import torch
import os
import shutil

def count_pkl_files(root_path):
    total_folder = 0
    total_pkl = 0
    # 遍历 root_path 下的所有目录和子目录
    for folder_name in sorted(os.listdir(root_path)):
        total_folder += 1
        input_folder = os.path.join(root_path, folder_name)
        # 统计当前目录下 .pkl 文件数量
        for fname in sorted(os.listdir(input_folder)):
            if fname.endswith(".pkl"):
                total_pkl += 1
    return total_folder, total_pkl

def count_pkl_files_output(root_path):
    total_folder = 0
    total_pkl = 0
    # 遍历 root_path 下的所有目录和子目录
    for folder_name in sorted(os.listdir(root_path)):
        total_folder += 1
        input_folder = os.path.join(root_path, folder_name, "pose")
        # 统计当前目录下 .pkl 文件数量
        if os.path.exists(input_folder):
            for fname in sorted(os.listdir(input_folder)):
                if fname.endswith(".pkl"):
                    total_pkl += 1
    return total_folder, total_pkl

# # 示例使用
# root_dir = "/DATA/baiyu/20251125/input"
# total_folder, total_pkl = count_pkl_files(root_dir)
# print("input:")
# print(f"总共 folder 数量: {total_folder}")
# print(f"总共 .pkl 文件数量: {total_pkl}")

root_dir = "/kpfs_dataset/dataset/base_data/data_2/completed"
total_folder, total_pkl = count_pkl_files(root_dir)
print("completed:")
print(f"总共 folder 数量: {total_folder}")
print(f"总共 .pkl 文件数量: {total_pkl}")

root_dir = "/kpfs_dataset/dataset/base_data/data_2/output"
total_folder, total_pkl = count_pkl_files_output(root_dir)
print("output:")
print(f"总共 folder 数量: {total_folder}")
print(f"总共 .pkl 文件数量: {total_pkl}")