#!/bin/bash
set -eu

# 服务器上 /isaac-sim 自带 python 未装 curobo（dev 检出，预编译 .so 已就位）。
# 把 curobo 的 src 加进 PYTHONPATH 即可 import（无需 nvcc 重编）。
export PYTHONPATH="/kpfs_dataset/dataset/baiyu/code/render/curobo/src${PYTHONPATH:+:$PYTHONPATH}"

# 等价于 .vscode/launch.json 的 "place_obstacles" 配置（本地 conda env_isaaclab 跑）。
# 沿「关键 link 扫掠走廊」自动放障碍 + 三条件验证，每个成功场景存一个 npz 到 OUT_DIR。
# M（每场景障碍数）、key_links、各阈值在 configs/default.yaml 的 obstacle_placement 段。
# 下列变量可用环境变量覆盖，缺省值与 launch.json 一致：
#   SEAM=... OUT_DIR=... SEED=0 OFFSET_CM=10 LINKS=... TYPES=... MAX_ATTEMPTS=20 ./scripts/bash/place_obstacles.sh

# 切到项目根目录（本脚本在 scripts/bash/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SEAM="${SEAM:-/kpfs_dataset/dataset/baiyu/dataset_v2/debug/segment_output_sub/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl}"
OFFSET_CM="${OFFSET_CM:-10}"
OUT_DIR="${OUT_DIR:-/tmp/placed_obstacles}"
SEED="${SEED:-0}"
LINKS="${LINKS:-xiaoyu_accessory_link}"
TYPES="${TYPES:-plate}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"

/isaac-sim/python.sh -u scripts/place_obstacles.py \
    --seam "$SEAM" \
    --offset_cm "$OFFSET_CM" \
    --out_dir "$OUT_DIR" \
    --seed "$SEED" \
    --links "$LINKS" \
    --types "$TYPES" \
    --max_attempts "$MAX_ATTEMPTS"
