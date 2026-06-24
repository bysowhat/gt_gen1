#!/usr/bin/env bash
# 批量跑 compute_goal_poses2.py：对 SEAM_DIR 下所有 seam_*.pkl 逐条求观测位姿，结果存到 OUT_DIR。
#
# 用法：
#   bash scripts/bash/run_compute_goal_poses2.sh [SEAM_DIR] [OUT_DIR]
#   PY=/path/to/python bash scripts/bash/run_compute_goal_poses2.sh   # 覆盖 python 解释器
#
# 注意：--save 是单文件路径，故按 seam 文件名分别落盘（seam_16.pkl -> OUT_DIR/seam_16.pkl），避免互相覆盖。
#       --obj 显式传入（取 seam 同目录的 *_part.obj），不依赖脚本内部默认。

set -uo pipefail   # 故意不加 -e：单条 seam 失败要继续跑下一条，不中断整批

SEAM_DIR="${1:-/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part}"
OUT_DIR="${2:-/tmp/goal_poses2}"
PY="${PY:-/home/a/miniforge3/envs/env_isaaclab/bin/python}"

# 定位 compute_goal_poses2.py：本脚本在 scripts/bash/ 下，目标脚本在上一级 scripts/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(cd "$SCRIPT_DIR/.." && pwd)/compute_goal_poses2.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "[run] 找不到 compute_goal_poses2.py: $SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$SEAM_DIR" ]]; then
    echo "[run] 输入目录不存在: $SEAM_DIR" >&2
    exit 1
fi

# 解析工件 obj：取 SEAM_DIR 下第一个 *_part.obj（与 compute_goal_poses2.py 的默认逻辑一致）
shopt -s nullglob
objs=("$SEAM_DIR"/*_part_watertight.obj)
seams=("$SEAM_DIR"/seam_*.pkl)
shopt -u nullglob

if [[ ${#objs[@]} -eq 0 ]]; then
    echo "[run] 在 $SEAM_DIR 下未找到 *_part_watertight.obj" >&2
    exit 1
fi
OBJ="${objs[0]}"

total=${#seams[@]}
if [[ "$total" -eq 0 ]]; then
    echo "[run] 在 $SEAM_DIR 下未找到 seam_*.pkl" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "[run] seam 目录 : $SEAM_DIR"
echo "[run] 工件 obj : $OBJ"
echo "[run] 输出目录 : $OUT_DIR"
echo "[run] python   : $PY"
echo "[run] 共 $total 条 seam"

ok=0
fail=0
i=0
for pkl in "${seams[@]}"; do
    i=$((i + 1))
    name="$(basename "$pkl" .pkl)"
    out="$OUT_DIR/$name.pkl"
    echo ""
    echo "========== [$i/$total] $name =========="
    if "$PY" "$SCRIPT" --seam-pkl "$pkl" --obj "$OBJ" --save "$out"; then
        ok=$((ok + 1))
        echo "[run] [$i/$total] $name 完成 -> $out"
    else
        fail=$((fail + 1))
        echo "[run] [$i/$total] $name 失败（已跳过）" >&2
    fi
done

echo ""
echo "[run] 全部结束：成功 $ok / 失败 $fail（共 $total），输出目录 $OUT_DIR"
[[ "$fail" -gt 0 ]] && exit 1
exit 0
