#!/usr/bin/env bash
# 批量跑 compute_goal_poses.py（Isaac 版）：对 SEAM_DIR 下所有 seam_*.pkl 逐条求观测位姿，结果存到 OUT_DIR。
#
# 用法：
#   bash scripts/bash/run_compute_goal_poses.sh [SEAM_DIR] [OUT_DIR]
#   PY=/path/to/python bash scripts/bash/run_compute_goal_poses.sh   # 覆盖 python 解释器
#
# 注意：--save 是单文件路径，故按 seam 文件名分别落盘（seam_16.pkl -> OUT_DIR/seam_16.pkl），避免互相覆盖。
#       --usd 显式传入（取 seam 同目录的 *_part.usd），不依赖脚本内部默认；Isaac 需要 --headless。
#
# 关键问题：compute_goal_poses.py 调 simulation_app.close() 后 Isaac Sim 进程常无法正常退出而卡死，
#           导致整批停在第一条 seam。这里不改原代码，改在 shell 侧处理：
#           每条 seam 用 setsid 起独立进程组，一旦日志出现完成标记（已保存 / 无解）即视为实处理完毕，
#           强杀该进程组进入下一条；另设单条超时上限兜底真正的崩溃/卡死。

set -uo pipefail   # 故意不加 -e：单条 seam 失败要继续跑下一条，不中断整批

SEAM_DIR="${1:-/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part}"
OUT_DIR="${2:-/tmp/goal_poses}"
PY="${PY:-/home/a/miniforge3/envs/env_isaaclab/bin/python}"
PER_SEAM_TIMEOUT="${PER_SEAM_TIMEOUT:-900}"   # 单条 seam 最长等待秒数（兜底，默认 15 分钟）

# 定位 compute_goal_poses.py：本脚本在 scripts/bash/ 下，目标脚本在上一级 scripts/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(cd "$SCRIPT_DIR/.." && pwd)/compute_goal_poses.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "[run] 找不到 compute_goal_poses.py: $SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$SEAM_DIR" ]]; then
    echo "[run] 输入目录不存在: $SEAM_DIR" >&2
    exit 1
fi

# 解析工件 usd：取 SEAM_DIR 下第一个 *_part.usd（与 compute_goal_poses.py 的默认逻辑一致）
shopt -s nullglob
usds=("$SEAM_DIR"/*_part.usd)
seams=("$SEAM_DIR"/seam_*.pkl)
shopt -u nullglob

if [[ ${#usds[@]} -eq 0 ]]; then
    echo "[run] 在 $SEAM_DIR 下未找到 *_part.usd" >&2
    exit 1
fi
USD="${usds[0]}"

total=${#seams[@]}
if [[ "$total" -eq 0 ]]; then
    echo "[run] 在 $SEAM_DIR 下未找到 seam_*.pkl" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[run] seam 目录 : $SEAM_DIR"
echo "[run] 工件 usd : $USD"
echo "[run] 输出目录 : $OUT_DIR"
echo "[run] 日志目录 : $LOG_DIR"
echo "[run] python   : $PY"
echo "[run] 单条超时 : ${PER_SEAM_TIMEOUT}s"
echo "[run] 共 $total 条 seam"

# 完成标记（任一出现即认为 python 实处理已结束，可强杀卡死的 Isaac 进程）
DONE_RE='已保存 goal poses|没有任何 robot_pose 求得观测位姿解'

ok=0      # 求得并保存了观测位姿
none=0    # 正常跑完但无解（未落盘）
fail=0    # 既无完成标记 -> 真正崩溃/超时
i=0
for pkl in "${seams[@]}"; do
    i=$((i + 1))
    name="$(basename "$pkl" .pkl)"
    out="$OUT_DIR/$name.pkl"
    log="$LOG_DIR/$name.log"
    echo ""
    echo "========== [$i/$total] $name =========="

    # setsid 起独立进程组；日志写文件，同时 tail 到终端（tail 随 python 退出而退出）
    # 关键：python -u + PYTHONUNBUFFERED=1 强制无缓冲。否则 stdout 重定向到文件时是块缓冲，
    #       完成标记（print）会一直攒在缓冲里不落盘，而进程卡在 close() 永不退出 → 缓冲永不冲刷
    #       → grep 永远看不到标记 → 检测失效，只能空等到超时。无缓冲后标记即时写入日志。
    : >"$log"
    setsid env PYTHONUNBUFFERED=1 "$PY" -u "$SCRIPT" --headless --seam-pkl "$pkl" --usd "$USD" --save "$out" >"$log" 2>&1 &
    pid=$!
    tail -n +1 -f --pid="$pid" "$log" 2>/dev/null &
    tail_pid=$!

    waited=0
    done_marker=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -qE "$DONE_RE" "$log" 2>/dev/null; then
            done_marker=1
            sleep 1   # 留点时间冲刷最后几行日志
            kill -KILL -- "-$pid" 2>/dev/null   # 进程组 id == setsid 子进程 pid
            break
        fi
        if [[ $waited -ge $PER_SEAM_TIMEOUT ]]; then
            echo "[run] [$i/$total] $name 超过 ${PER_SEAM_TIMEOUT}s，强制终止" >&2
            kill -KILL -- "-$pid" 2>/dev/null
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    wait "$pid" 2>/dev/null
    kill "$tail_pid" 2>/dev/null
    wait "$tail_pid" 2>/dev/null

    # 判定结果：以日志标记为准（因 close() 卡死被强杀，退出码不可靠）
    if grep -q '已保存 goal poses' "$log" 2>/dev/null; then
        ok=$((ok + 1))
        echo "[run] [$i/$total] $name 完成 -> $out"
    elif [[ $done_marker -eq 1 ]]; then
        none=$((none + 1))
        echo "[run] [$i/$total] $name 跑完但无观测位姿解（未落盘）"
    else
        fail=$((fail + 1))
        echo "[run] [$i/$total] $name 失败/超时（详见 $log）" >&2
    fi
done

echo ""
echo "[run] 全部结束：保存 $ok / 无解 $none / 失败 $fail（共 $total），输出目录 $OUT_DIR"
[[ "$fail" -gt 0 ]] && exit 1
exit 0
