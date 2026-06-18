#!/bin/bash
set -eu

# 服务器上 /isaac-sim 自带 python 未装 curobo（dev 检出，预编译 .so 已就位）。
# 把 curobo 的 src 加进 PYTHONPATH 即可 import（无需 nvcc 重编）。
export PYTHONPATH="/kpfs_dataset/dataset/baiyu/code/render/curobo/src${PYTHONPATH:+:$PYTHONPATH}"

# curobo 的 isaac_sim 示例目录（含 helper.add_robot_to_scene）；viz 脚本按此 env 覆盖本地默认路径。
export CUROBO_ISAAC="${CUROBO_ISAAC:-/kpfs_dataset/dataset/baiyu/code/render/curobo/examples/isaac_sim}"

# 等价于 .vscode/launch.json 的 "viz_placed_obstacle_isaacsim" 配置。
# 回放 place_obstacles.py 产出的一个场景（npz）：工件 USD + 机器人 + 放置的障碍物，
# 播默认轨迹(WHICH=default，应撞障碍) 或 绕行轨迹(WHICH=detour，应绕开)。
# 下列变量可用环境变量覆盖，缺省值与 launch.json 一致：
#   OUT_DIR=... INDEX=0 WHICH=detour FPS=30 DETOUR_INDEX=0 ./scripts/bash/viz_placed_obstacle_isaacsim.sh
#   HEADLESS=1 ./scripts/bash/viz_placed_obstacle_isaacsim.sh   # 无显示器自检（spawn+跑几帧+VIZ_PLACED_DONE）
#   SCENE=/tmp/placed_obstacles/scene_00_Link2_plate.npz ./...  # 直接指定场景文件（忽略 OUT_DIR/INDEX）

# 切到项目根目录（本脚本在 scripts/bash/ 下）
# ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "$ROOT"

OUT_DIR="${OUT_DIR:-/tmp/placed_obstacles}"
INDEX="${INDEX:-0}"
WHICH="${WHICH:-detour}"
FPS="${FPS:-30}"
DETOUR_INDEX="${DETOUR_INDEX:-0}"

# 选场景：给了 SCENE 就直接指定文件（忽略 OUT_DIR/INDEX）；否则用 目录+序号。
if [ -n "${SCENE:-}" ]; then
    SCENE_ARGS=(--scene "$SCENE")
else
    SCENE_ARGS=(--out_dir "$OUT_DIR" --index "$INDEX")
fi

# HEADLESS 非空则加 --headless（store_true）。
HEADLESS_ARGS=()
if [ -n "${HEADLESS:-}" ]; then
    HEADLESS_ARGS=(--headless)
fi

/isaac-sim/python.sh -u scripts/viz_placed_obstacle_isaacsim.py \
    "${SCENE_ARGS[@]}" \
    --which "$WHICH" \
    --fps "$FPS" \
    "${HEADLESS_ARGS[@]}"
