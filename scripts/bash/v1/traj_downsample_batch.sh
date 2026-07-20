#!/bin/bash
# 关键帧下采样批处理入口：对【输入目录】里的 Scene pkl 去重后，多 GPU 动态调度跑
# scripts/traj_downsample.py，结果另存到【输出目录】（保留原文件名、不覆盖原 pkl）。
#
# 去重：per-seam pkl 是整场景累积快照——同一前缀里 <...>_seam47.pkl 已含 seam0..46 的
#       全部信息，故每组只保留 seam 号最大的那个，其余（子集）全丢弃。不匹配
#       *_seam<N>.pkl 的文件（如 *_all.pkl）各自成组、原样保留。
#
# 调度：每张卡起 PER_GPU_JOBS 个后台 worker，从共享队列 flock 原子出队——谁先跑完谁
#       领下一个，直到队列空（比静态 idx%N 均分更抗轨迹数不均）。
#
# 用法：
#   scripts/bash/v1/traj_downsample_batch.sh <输入目录> <输出目录>
#
# 本地（默认 conda 环境）：
#   scripts/bash/v1/traj_downsample_batch.sh /media/a/新加卷/tempt/6/tempt /media/a/新加卷/tempt/6/ds
# 服务器（环境参考 run_obstacle_type1.sh，覆盖 PY / NUM_GPUS）：
#   PY=/workspace/isaaclab/_isaac_sim/python.sh NUM_GPUS=8 \
#       scripts/bash/v1/traj_downsample_batch.sh <in> <out>
# 后台长跑（断开 ssh 不中断）：
#   nohup scripts/bash/v1/traj_downsample_batch.sh <in> <out> > ds.out 2>&1 &
#
# 可用环境变量覆盖：
#   PY=...              python 解释器（默认 "conda run -n env_isaaclab python"）
#   NUM_GPUS=8          GPU 张数（留空=nvidia-smi 自动探测）
#   PER_GPU_JOBS=1      每张卡并发子进程数（默认 1）
#   CONFIG=...          采样参数 yaml（默认 configs/default.yaml，读 traj_downsample 段）
#   FORCE=0             1=输出目录已存在同名 pkl 也重跑（默认跳过 → 断点续跑）
#   CUROBO_SRC=...      curobo src 目录（存在才加进 PYTHONPATH；默认服务器路径）
#
# 故意不用 set -e：单文件崩溃不能中断整个队列，靠捕获退出码处理。
set -u

IN_DIR="${1:?用法: traj_downsample_batch.sh <输入目录> <输出目录>}"
OUT_DIR="${2:?用法: traj_downsample_batch.sh <输入目录> <输出目录>}"

# 切到项目根（本脚本在 scripts/bash/v1/ 下，上三级）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-conda run -n env_isaaclab python}"
CONFIG="${CONFIG:-$ROOT/configs/default.yaml}"
PER_GPU_JOBS="${PER_GPU_JOBS:-1}"
FORCE="${FORCE:-0}"

# curobo dev 检出加进 PYTHONPATH（存在才加：服务器有、本地 conda 环境自带则跳过，无害）
CUROBO_SRC="${CUROBO_SRC:-/kpfs_dataset/dataset/baiyu/code/render/curobo/src}"
[ -d "$CUROBO_SRC" ] && export PYTHONPATH="$CUROBO_SRC${PYTHONPATH:+:$PYTHONPATH}"

# GPU 张数：未显式给则自动探测
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
if ! [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
    echo "[ds] 探测不到 GPU（NUM_GPUS=$NUM_GPUS），退出"; exit 1
fi

[ -d "$IN_DIR" ] || { echo "[ds] 输入目录不存在：$IN_DIR"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ds] 找不到 config：$CONFIG"; exit 1; }

LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ==== 去重：同前缀只留 seam 号最大者，非 seam 文件原样保留 ====
mapfile -t KEEP < <(
    for p in "$IN_DIR"/*.pkl; do
        [ -e "$p" ] || continue
        printf '%s\n' "$p"
    done | awk '
    {
        path = $0
        n = split(path, a, "/"); base = a[n]
        if (match(base, /_seam[0-9]+\.pkl$/)) {
            grp = substr(base, 1, RSTART - 1)              # _seam 之前的前缀
            num = substr(base, RSTART + 5, RLENGTH - 9) + 0 # _seam(5) 与 .pkl(4) 之间的数字
            if (!(grp in best) || num > bnum[grp]) { best[grp] = path; bnum[grp] = num }
        } else {
            best[base] = path                              # 非 _seamN 文件：各自成组、必留
        }
    }
    END { for (g in best) print best[g] }
    ' | sort
)

TOTAL_PKL="$(find "$IN_DIR" -maxdepth 1 -name '*.pkl' | wc -l)"
NKEEP=${#KEEP[@]}
if [ "$NKEEP" -eq 0 ]; then
    echo "[ds] 输入目录无 .pkl：$IN_DIR"; exit 1
fi

echo "=========================================================="
echo "关键帧下采样批处理"
echo "  IN_DIR   : $IN_DIR（共 $TOTAL_PKL 个 pkl，去重后保留 $NKEEP 个）"
echo "  OUT_DIR  : $OUT_DIR"
echo "  PY       : $PY"
echo "  CONFIG   : $CONFIG"
echo "  NUM_GPUS=$NUM_GPUS  PER_GPU_JOBS=$PER_GPU_JOBS  FORCE=$FORCE"
echo "=========================================================="

# ==== 共享队列 + flock 原子出队 ====
QUEUE="$LOG_DIR/.queue.$$"
LOCK="$LOG_DIR/.queue.$$.lock"
RESULTS="$LOG_DIR/.results.$$"
printf '%s\n' "${KEEP[@]}" > "$QUEUE"
: > "$RESULTS"
trap 'rm -f "$QUEUE" "$LOCK" "$RESULTS"' EXIT

pop() {                       # 原子取队首一行并从队列删除，打到 stdout（空=队列空）
    (
        flock 9
        line="$(head -n1 "$QUEUE")"
        [ -n "$line" ] && sed -i '1d' "$QUEUE"
        printf '%s' "$line"
    ) 9>"$LOCK"
}

worker() {
    local gpu="$1" f base out log rc status t0
    while true; do
        f="$(pop)"
        [ -z "$f" ] && break
        base="$(basename "$f")"
        out="$OUT_DIR/$base"
        if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
            printf '[gpu%s] skip 已存在 %s\n' "$gpu" "$base"
            printf 'SKIP %s\n' "$base" >> "$RESULTS"
            continue
        fi
        log="$LOG_DIR/${base%.pkl}.log"
        printf '[gpu%s] start %s\n' "$gpu" "$base"
        t0=$SECONDS
        CUDA_VISIBLE_DEVICES="$gpu" $PY -u scripts/traj_downsample.py \
            --pkl "$f" --out-dir "$OUT_DIR" --config "$CONFIG" --device cuda \
            > "$log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            status="OK"; printf 'OK %s\n' "$base" >> "$RESULTS"
        else
            status="FAIL(rc=$rc)"; printf 'FAIL %s\n' "$base" >> "$RESULTS"
        fi
        printf '[gpu%s] %-12s %s (%ds)\n' "$gpu" "$status" "$base" "$((SECONDS - t0))"
    done
    printf '[gpu%s] 队列空，worker 退出\n' "$gpu"
}

# 每张卡 PER_GPU_JOBS 个 worker（worker w → gpu = w %% NUM_GPUS）
NWORK=$((NUM_GPUS * PER_GPU_JOBS))
for ((w = 0; w < NWORK; w++)); do
    worker "$((w % NUM_GPUS))" &
done
wait

NOK=$(grep -c '^OK '   "$RESULTS" 2>/dev/null || echo 0)
NSK=$(grep -c '^SKIP ' "$RESULTS" 2>/dev/null || echo 0)
NFA=$(grep -c '^FAIL ' "$RESULTS" 2>/dev/null || echo 0)

echo "=========================================================="
echo "完成：OK=$NOK  SKIP=$NSK  FAIL=$NFA  (共 $NKEEP)"
[ "$NFA" -gt 0 ] && { echo "失败文件（详见 $LOG_DIR/<名>.log）："; grep '^FAIL ' "$RESULTS" | sed 's/^FAIL /  /'; }
echo "结果：$OUT_DIR/   日志：$LOG_DIR/"
echo "TRAJ_DOWNSAMPLE_BATCH_DONE"
echo "=========================================================="
