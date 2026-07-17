#!/bin/bash
# 公共核心：把 filenames.txt 里的工件按 GPU 轮转均分，每张卡起 1 个后台 worker，
# 串行跑分到它的工件，调 scripts/demo_scene.py --task $TASK 计算该工件【所有焊缝】的轨迹。
# 由 run_obstacle_type1.sh / run_obstacle_type2.sh 设好 TASK + OUT_ROOT 后调用。
#
# 用法（一般经 wrapper 调，不直接跑）：
#   TASK=type1 OUT_ROOT=/kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type1 \
#       scripts/bash/v1/_run_obstacle.sh
#
# 可用环境变量覆盖：
#   FILELIST=...        工件清单（每行 obj<TAB>weld_json），默认 render_baiyu/filenames.txt
#   NUM_GPUS=8          GPU 张数（默认 nvidia-smi 自动探测；现在 2、以后 8 无需改脚本）
#   TIMEOUT=7200        单工件超时秒（默认 2 小时；0=不限）
#   START=0 LIMIT=0     从第 START 个工件起、最多处理 LIMIT 个（0=不限），小批验证用
#   FORCE=0             1=忽略已存在 per-seam pkl / .done 标记，强制重跑
#   PY=...              python 解释器（默认 /workspace/isaaclab/_isaac_sim/python.sh）
#
# 断点续跑：整工件跑完(rc=0)打 .done 标记，下次跳过；未打标记的工件重跑时，
# demo_scene.py 内部按 per-seam pkl 是否存在跳过已完成焊缝，只补未完成的。
#
# 故意不用 set -e：单工件崩溃/超时不能中断整卡队列，靠捕获退出码处理。
set -u

TASK="${TASK:?必须设置 TASK=type1|type2}"
OUT_ROOT="${OUT_ROOT:?必须设置 OUT_ROOT}"

# curobo dev 检出加进 PYTHONPATH（预编译 .so 已就位，无需 nvcc 重编）
export PYTHONPATH="/kpfs_dataset/dataset/baiyu/code/render/curobo/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"

# 切到项目根（本脚本在 scripts/bash/v1/ 下，上三级）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

FILELIST="${FILELIST:-/kpfs_dataset_ssd/dataset/render_baiyu/filenames.txt}"
TIMEOUT="${TIMEOUT:-7200}"
START="${START:-0}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"

# GPU 张数：未显式给则自动探测
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
if ! [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
    echo "[run] 探测不到 GPU（NUM_GPUS=$NUM_GPUS），退出"; exit 1
fi

[ -f "$FILELIST" ] || { echo "[run] 找不到清单：$FILELIST"; exit 1; }

LOG_DIR="$OUT_ROOT/_logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# 读清单到数组（跳过空行 / # 注释行）
mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$FILELIST")
TOTAL=${#LINES[@]}

# START/LIMIT 分片 [START, END)
END=$TOTAL
if [ "$LIMIT" -gt 0 ]; then
    END=$((START + LIMIT))
    [ "$END" -gt "$TOTAL" ] && END=$TOTAL
fi

echo "=========================================================="
echo "批处理障碍轨迹  TASK=$TASK"
echo "  FILELIST : $FILELIST（共 $TOTAL 工件，本次 [$START,$END)）"
echo "  OUT_ROOT : $OUT_ROOT"
echo "  PY       : $PY"
echo "  NUM_GPUS=$NUM_GPUS TIMEOUT=${TIMEOUT}s FORCE=$FORCE"
echo "=========================================================="

FORCE_FLAG=""
[ "$FORCE" = "1" ] && FORCE_FLAG="--force"

# 处理单个工件（$1=全局序号 $2=gpu）
run_one() {
    local idx="$1" gpu="$2"
    local line obj json stem log done_marker t0 rc status
    line="${LINES[$idx]}"
    obj="$(printf '%s' "$line" | cut -f1)"
    json="$(printf '%s' "$line" | cut -f2)"
    stem="$(basename "$obj")"; stem="${stem%.*}"
    log="$LOG_DIR/${stem}.log"
    done_marker="$LOG_DIR/${stem}.${TASK}.done"

    if [ "$FORCE" != "1" ] && [ -f "$done_marker" ]; then
        printf '[gpu%s][%d] skip 已完成 %s\n' "$gpu" "$idx" "$stem"
        return
    fi
    if [ ! -f "$obj" ] || [ ! -f "$json" ]; then
        printf '[gpu%s][%d] MISSING 输入缺失 %s\n' "$gpu" "$idx" "$stem"
        return
    fi

    printf '[gpu%s][%d] start %s\n' "$gpu" "$idx" "$stem"
    t0=$SECONDS
    if [ "$TIMEOUT" -gt 0 ] 2>/dev/null; then
        CUDA_VISIBLE_DEVICES="$gpu" timeout --kill-after=30 "$TIMEOUT" \
            "$PY" -u scripts/demo_scene.py --task "$TASK" \
                --obj "$obj" --weld-json "$json" --out-root "$OUT_ROOT" $FORCE_FLAG \
            >"$log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="$gpu" \
            "$PY" -u scripts/demo_scene.py --task "$TASK" \
                --obj "$obj" --weld-json "$json" --out-root "$OUT_ROOT" $FORCE_FLAG \
            >"$log" 2>&1
    fi
    rc=$?
    if [ "$rc" -eq 0 ]; then
        status="OK"; printf 'OK\n' >"$done_marker"
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        status="TIMEOUT"
    else
        status="CRASH(rc=$rc)"
    fi
    printf '[gpu%s][%d] %-14s %s (%ds)\n' "$gpu" "$idx" "$status" "$stem" "$((SECONDS - t0))"
}

# 每张卡 1 个后台 worker，处理 idx%NUM_GPUS==gpu 的工件（限 [START,END) 内）
worker() {
    local gpu="$1" idx
    for ((idx = START; idx < END; idx++)); do
        if [ $((idx % NUM_GPUS)) -eq "$gpu" ]; then
            run_one "$idx" "$gpu"
        fi
    done
    printf '[gpu%s] 队列完成\n' "$gpu"
}

for ((g = 0; g < NUM_GPUS; g++)); do
    worker "$g" &
done
wait

echo "=========================================================="
echo "完成。日志：$LOG_DIR/  |  结果：$OUT_ROOT/"
echo "RUN_OBSTACLE_DONE TASK=$TASK"
echo "=========================================================="
