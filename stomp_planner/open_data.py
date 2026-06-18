import pickle
import json
import torch
import os
import shutil

with open("/kpfs_dataset/dataset/base_data/data_2/output_25/132K_1cNxif0017fZ4tCJSrDJas_part/pose/seam_1.pkl", "rb") as f:
    data = pickle.load(f)
print(type(data[0]))
print(data[0].keys())
print(data[0]["cam_pose"].shape)
# print(data[0]["seam_line"].shape)
# print(data["seam_tangent"].shape)
# print(data[0]["seam_limits"].shape)
# print(data[0]["robot_pose"])
# print(data["horizontal"])

# with open("/kpfs_dataset/dataset/base_data/data_2/output/Part_41_1Zt5HE000BIZ4sDpKpE3as_part/traj/seam_0.pkl", "rb") as f:
#     data = pickle.load(f)
# print(type(data))
# print(data[0].keys())
# print(data[0]["trajectory"].shape)
# print(data[0]["trajectory"][0])
# # print(data["seam_tangent"].shape)
# # print(data["seam_limits"].shape)
# # print(data["robot_pose"].shape)
# # print(data["piece_pose"].shape)
# # print(data["horizontal"])

# with open("/home/kejian/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/gt_overall/" \
# "gt_data/output/钢柱_1dlcPQ000ABJ4tD30rD3at_part/traj/seam_1.pkl", "rb") as f:
#     data = pickle.load(f)
# print(type(data))
# print(data[0].keys())
# # print(data["robot_pose"].shape)
# # print(data[0].keys())
# # print(data[0]["robot_pose"])
# # print(data[0]["trajectory"].shape)
# # print(data[0]["idx_3d"])

# x = torch.tensor([0, 0, 0])
# y = torch.tensor([0, 1, 0])
# z = torch.tensor([1, 1, 1])
# print(not x.bool().any())
# print(not y.bool().any())
# print(not z.bool().any())