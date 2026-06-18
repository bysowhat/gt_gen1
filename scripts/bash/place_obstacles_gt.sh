#!/bin/bash
set -eu

# 服务器上 /isaac-sim 自带 python 未装 curobo（dev 检出，预编译 .so 已就位）。
# 把 curobo 的 src 加进 PYTHONPATH 即可 import（无需 nvcc 重编）。
export PYTHONPATH="/kpfs_dataset/dataset/baiyu/code/render/curobo/src${PYTHONPATH:+:$PYTHONPATH}"

# 切到项目根目录（本脚本在 scripts/bash/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENE="${SCENE:-/tmp/placed_obstacles/scene_00_xiaoyu_accessory_link_plate.npz}"
OUT="${OUT:-/tmp/placed_obstacles_active/scene_00_xiaoyu_accessory_link_plate_gt.npz}"

/isaac-sim/python.sh -u scripts/place_obstacles_to_gt.py \
    --scene "$SCENE" \
    --out "$OUT"
