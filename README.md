生成轨迹
/workspace/isaaclab/_isaac_sim/python.sh -m pip install libigl==2.6.2
cd /kpfs_dataset_ssd/dataset/baiyu/code/gt_gen_hanfeng
FILELIST=/kpfs_dataset_ssd/dataset/render_baiyu/filenames.txt TIMEOUT=7200 scripts/bash/v1/run_obstacle_type1.sh
FILELIST=/kpfs_dataset_ssd/dataset/render_baiyu/filenames.txt TIMEOUT=1800 scripts/bash/v1/run_obstacle_type2.sh
 ./scripts/bash/v1/run_obstacle_type3.sh 0 /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type3/usd/warehouse.usdz /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type3/jsons/warehouse.json
 
下采样轨迹
PY=/workspace/isaaclab/_isaac_sim/python.sh scripts/bash/v1/traj_downsample_batch.sh /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type2.bk /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type2.ds
PY=/workspace/isaaclab/_isaac_sim/python.sh scripts/bash/v1/traj_downsample_batch.sh /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type1.bk /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type1.ds
渲染
bash scripts/bash/v1/render_trajectory_scheduler.sh