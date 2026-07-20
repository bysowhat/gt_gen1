import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--folder', type = str)
parser.add_argument('--config_path', type = str)
parser.add_argument('--device', type = str)
args = parser.parse_args()

import os

import pickle
import yaml
import torch

from geometry_sampling import get_segmented_keyframes, build_action_matrix, UR12e_t

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")

def run(folder, config_path, device):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        w_trans = config['se3_action']['w_trans']
        w_rot = config['se3_action']['w_rot']
        D_target = config['sampling']['D_target']
        res_ratio = config['sampling']['res_ratio']

    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith(".pkl"):
            file = os.path.join(folder, file_name)
            with open(file, "rb") as f:
                data_list = pickle.load(f)   # list
        else:
            continue
            
        data_list_new = []
        for data in data_list:
            traj = torch.as_tensor(data["trajectory"], 
                                dtype=torch.float, device=device).permute(0, 2, 1)     # (B, L, D)
            B, L, D =traj.shape
            idx_3d = torch.as_tensor(data["idx_3d"], 
                                    dtype=torch.int, device=device).unsqueeze(0).expand(B, -1)    # (B, L)

            # calculate Transform Matrix for joints
            fk = UR12e_t(num_envs=B*L, device=device)
            T_all = fk.forward_cam_pose(traj.reshape(-1, D)).reshape(B, L, 4, 4)      # (B, L, 4, 4)

            # get action matrix
            M_action = build_action_matrix(T_all, w_trans, w_rot)

            # retrieve action results of downsampling
            actions, valid = get_segmented_keyframes(M_action, traj, idx_3d, nodes=None, D_target=D_target, res_ratio=res_ratio)
            
            data["actions"] = actions.clone()
            data["valid"] = valid.clone()
            data_list_new.append(data.copy())

        with open(file, "wb") as f:
            pickle.dump(data_list_new, f)


if __name__ == "__main__":
    # folder = "/home/kejian/Downloads/pieces/data/selected_folders/3L_1cBPgF0009QJ4tCJGoDZSm_part/traj"
    # config_path = "/home/kejian/va/traj_sampler/configs/geometry_sampling.yaml"
    # device = "cuda"
    folder = args.folder
    config_path = args.config_path
    device = args.device
    run(folder, config_path, device)