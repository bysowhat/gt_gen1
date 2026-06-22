#!/bin/bash
# 批量：遍历 segment_output_sub 下全部 seam_*.pkl，每条 seam 先规划避障轨迹
# (place_obstacles.py)、再改造成边看边走的探索式 GT (place_obstacles_to_gt.py)，
# 处理完一条再下一条。单条失败/崩溃/超时都被隔离，绝不中断整个循环；可断点续跑。
#
# 在【服务器】上跑（与 place_obstacles.sh 同环境，用 /isaac-sim/python.sh + curobo src）：
#   ./scripts/bash/batch_place_obstacles.sh
#
# 目录映射（按 seam 命名空间隔离，避免不同 seam 的 scene_00 互相覆盖）：
#   输入   IN_ROOT /<part>/seam_22.pkl
#   避障   OBS_ROOT/<part>/seam_22/scene_00_<link>_<otype>.npz
#   探索式 GT_ROOT /<part>/seam_22/scene_00_<...>_gt.npz
#   日志   OBS_ROOT/_batch_logs/<part>__seam_22.obs.log / .gt.log + manifest.tsv
#
# 可用环境变量覆盖（缺省与 place_obstacles.sh 一致）：
#   LINKS=xiaoyu_accessory_link TYPES=plate SEED=0 OFFSET_CM=10 MAX_ATTEMPTS=20
#   IN_ROOT=... OBS_ROOT=... GT_ROOT=...
#   TIMEOUT_OBS=1800 TIMEOUT_GT=1800   # 秒；设为 0 关闭超时
#   START=0 LIMIT=0                     # 从第 START 条起、最多处理 LIMIT 条（0=不限）。先小批量验证用
#   FORCE=0                            # 1=忽略 .batch_done 标记，强制重做
#
# 注意：本脚本【故意不用 set -e】——单条 seam 出错要继续跑下一条，靠手动捕获退出码处理。
set -u

# 服务器上 /isaac-sim 自带 python 未装 curobo（dev 检出，预编译 .so 已就位）。
export PYTHONPATH="/kpfs_dataset/dataset/baiyu/code/render/curobo/src${PYTHONPATH:+:$PYTHONPATH}"

# 切到项目根目录（本脚本在 scripts/bash/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IN_ROOT="${IN_ROOT:-/kpfs_dataset/dataset/baiyu/dataset_v2/debug/segment_output_sub}"
OBS_ROOT="${OBS_ROOT:-/kpfs_dataset/dataset/baiyu/dataset_v2/debug/segment_output_sub_obstacles}"
GT_ROOT="${GT_ROOT:-/kpfs_dataset/dataset/baiyu/dataset_v2/debug/segment_output_sub_obstacles_gt}"

OFFSET_CM="${OFFSET_CM:-10}"
SEED="${SEED:-0}"
# LINKS 留空（默认）=每条 seam 从下面 6 个关键 link 里随机抽一个；显式设 LINKS=Link3 则全程固定。
LINKS="${LINKS:-}"
LINK_POOL=(Link2 Link3 Link4 Link5 Link6 xiaoyu_accessory_link)
# TYPES 留空（默认）=每条 seam 从下面 15 种里随机抽一种；显式设 TYPES=plate 则全程固定该类型。
TYPES="${TYPES:-}"
TYPE_POOL=(plate l_bracket u_channel open_box pipe parallel_pipes crossed_pipes \
           box_beam rect_frame gantry braced_frame tripod steps box_with_pipe frame_with_brace)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
TIMEOUT_OBS="${TIMEOUT_OBS:-300}"
TIMEOUT_GT="${TIMEOUT_GT:-300}"
START="${START:-0}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"

PY=/isaac-sim/python.sh
LOG_DIR="$OBS_ROOT/_batch_logs"
MANIFEST="$LOG_DIR/manifest.tsv"
mkdir -p "$LOG_DIR"
if [ ! -f "$MANIFEST" ]; then
    printf 'seam\tlink\totype\tobs_status\tn_scenes\tgt_status\tsecs\ttimestamp\n' >"$MANIFEST"
fi

# 按 SEED 播种 bash $RANDOM，使「每条 seam 随机抽类型」可复现（同 SEED → 同序列）。
RANDOM="$SEED"

# timeout 包装：TIMEOUT=0 时不包；否则 timeout --signal=KILL <秒> ...
run_with_timeout() {
    local secs="$1"; shift
    if [ "$secs" -gt 0 ] 2>/dev/null; then
        timeout --kill-after=30 "$secs" "$@"
    else
        "$@"
    fi
}

echo "=========================================================="
echo "批量放障碍 + 探索式 GT"
echo "  IN_ROOT : $IN_ROOT"
echo "  OBS_ROOT: $OBS_ROOT"
echo "  GT_ROOT : $GT_ROOT"
echo "  LINKS=${LINKS:-<随机6个>} TYPES=${TYPES:-<随机15种>} SEED=$SEED OFFSET_CM=$OFFSET_CM MAX_ATTEMPTS=$MAX_ATTEMPTS"
echo "  TIMEOUT_OBS=${TIMEOUT_OBS}s TIMEOUT_GT=${TIMEOUT_GT}s START=$START LIMIT=$LIMIT FORCE=$FORCE"
echo "=========================================================="

# 收集全部 seam（-print0 处理部件名里的 $、空格等特殊字符），稳定排序
mapfile -d '' SEAMS < <(find "$IN_ROOT" -name 'seam_*.pkl' -print0 | sort -z)
TOTAL=${#SEAMS[@]}
echo "发现 seam 总数: $TOTAL"
echo ""

# 统计
n_seen=0; n_proc=0; n_skip=0
n_obs_ok=0; n_obs_fail=0
n_gt_ok=0; n_gt_partial=0; n_gt_fail=0; n_gt_none=0

for seam in "${SEAMS[@]}"; do
    n_seen=$((n_seen + 1))
    # START / LIMIT 分片
    if [ "$n_seen" -le "$START" ]; then continue; fi
    if [ "$LIMIT" -gt 0 ] && [ "$n_proc" -ge "$LIMIT" ]; then break; fi
    n_proc=$((n_proc + 1))

    # 相对路径 -> <part>/<seam_stem>
    rel="${seam#"$IN_ROOT"/}"            # <part>/seam_22.pkl
    part="$(dirname "$rel")"             # <part>
    seam_file="$(basename "$seam")"      # seam_22.pkl
    seam_stem="${seam_file%.pkl}"        # seam_22

    obs_dir="$OBS_ROOT/$part/$seam_stem"
    gt_dir="$GT_ROOT/$part/$seam_stem"
    log_tag="${part}__${seam_stem}"
    log_tag="${log_tag//\//_}"           # 防 part 里有 / （正常没有，兜底）
    obs_log="$LOG_DIR/${log_tag}.obs.log"
    gt_log="$LOG_DIR/${log_tag}.gt.log"
    done_marker="$obs_dir/.batch_done"

    printf '[%d/%d] %s\n' "$n_proc" "$TOTAL" "$rel"

    # 断点续跑
    if [ "$FORCE" != "1" ] && [ -f "$done_marker" ]; then
        echo "    skip（已存在 .batch_done）"
        n_skip=$((n_skip + 1))
        continue
    fi

    mkdir -p "$obs_dir" "$gt_dir"
    t0=$SECONDS

    # 本条 seam 的关键 link 与障碍类型：显式给了 LINKS/TYPES 就固定用它；否则各自随机抽一个
    if [ -n "$LINKS" ]; then
        seam_link="$LINKS"
    else
        seam_link="${LINK_POOL[$((RANDOM % ${#LINK_POOL[@]}))]}"
    fi
    if [ -n "$TYPES" ]; then
        seam_type="$TYPES"
    else
        seam_type="${TYPE_POOL[$((RANDOM % ${#TYPE_POOL[@]}))]}"
    fi

    # ---------- ① 避障：place_obstacles.py ----------
    obs_status="OBS_FAIL"
    (
        run_with_timeout "$TIMEOUT_OBS" "$PY" -u scripts/place_obstacles.py \
            --seam "$seam" \
            --offset_cm "$OFFSET_CM" \
            --out_dir "$obs_dir" \
            --seed "$SEED" \
            --links "$seam_link" \
            --types "$seam_type" \
            --max_attempts "$MAX_ATTEMPTS"
    ) >"$obs_log" 2>&1
    obs_rc=$?

    # 实际产出的场景 npz（排除标记/隐藏文件）
    shopt -s nullglob
    scene_npzs=("$obs_dir"/scene_*.npz)
    shopt -u nullglob
    n_scenes=${#scene_npzs[@]}

    if [ "$obs_rc" -eq 124 ] || [ "$obs_rc" -eq 137 ]; then
        obs_status="OBS_TIMEOUT"
    elif [ "$obs_rc" -ne 0 ]; then
        obs_status="OBS_CRASH(rc=$obs_rc)"
    elif [ "$n_scenes" -gt 0 ]; then
        obs_status="OBS_OK"
    elif grep -q "PLACE_OBSTACLES_OK" "$obs_log" 2>/dev/null; then
        obs_status="OBS_NOSCENE"   # 正常结束但没产出可行场景（默认路径无碰 / 无绕行解等）
    else
        obs_status="OBS_FAIL"
    fi

    if [ "$n_scenes" -gt 0 ]; then
        n_obs_ok=$((n_obs_ok + 1))
    else
        n_obs_fail=$((n_obs_fail + 1))
    fi

    # ---------- ② 探索式 GT：对每个场景 npz 跑 place_obstacles_to_gt.py ----------
    gt_status="NONE"
    if [ "$n_scenes" -gt 0 ]; then
        : >"$gt_log"   # 清空，逐场景追加
        gt_ok=0; gt_partial=0; gt_fail=0
        for scene in "${scene_npzs[@]}"; do
            scene_stem="$(basename "$scene")"; scene_stem="${scene_stem%.npz}"
            gt_out="$gt_dir/${scene_stem}_gt.npz"
            {
                echo "######## scene: $scene -> $gt_out"
                run_with_timeout "$TIMEOUT_GT" "$PY" -u scripts/place_obstacles_to_gt.py \
                    --scene "$scene" \
                    --out "$gt_out"
                echo "######## gt_rc=$?"
            } >>"$gt_log" 2>&1
            gt_rc=$(grep -oE 'gt_rc=[0-9]+' "$gt_log" | tail -1 | cut -d= -f2)
            if [ "${gt_rc:-1}" -eq 124 ] || [ "${gt_rc:-1}" -eq 137 ]; then
                gt_fail=$((gt_fail + 1))   # 超时算失败
            elif [ "${gt_rc:-1}" -ne 0 ]; then
                gt_fail=$((gt_fail + 1))
            elif grep -q "PLACE_OBSTACLES_TO_GT_OK" "$gt_log"; then
                gt_ok=$((gt_ok + 1))
            elif grep -q "PLACE_OBSTACLES_TO_GT_PARTIAL" "$gt_log"; then
                gt_partial=$((gt_partial + 1))
            else
                gt_fail=$((gt_fail + 1))
            fi
        done
        gt_status="ok=${gt_ok},partial=${gt_partial},fail=${gt_fail}"
        n_gt_ok=$((n_gt_ok + gt_ok))
        n_gt_partial=$((n_gt_partial + gt_partial))
        n_gt_fail=$((n_gt_fail + gt_fail))
    else
        n_gt_none=$((n_gt_none + 1))
    fi

    secs=$((SECONDS - t0))

    # 完成标记 + 清单（只有进程没崩/没超时才打 .batch_done，崩溃/超时下次可重试）
    case "$obs_status" in
        OBS_OK|OBS_NOSCENE) printf '%s\n' "$obs_status" >"$done_marker" ;;
    esac
    printf '%s\t%s\t%s\t%s\t%d\t%s\t%d\t%s\n' \
        "$rel" "$seam_link" "$seam_type" "$obs_status" "$n_scenes" "$gt_status" "$secs" "$(date '+%F %T')" >>"$MANIFEST"

    printf '    link=%s type=%s obs=%s scenes=%d gt=%s (%ds)\n' "$seam_link" "$seam_type" "$obs_status" "$n_scenes" "$gt_status" "$secs"
done

echo ""
echo "=========================================================="
echo "完成。共扫描 $TOTAL 条，本次处理 $n_proc 条（跳过 $n_skip）"
echo "  避障  : 有场景 $n_obs_ok / 无场景或失败 $n_obs_fail"
echo "  探索式: OK $n_gt_ok / PARTIAL $n_gt_partial / FAIL $n_gt_fail / 无avoidance不跑 $n_gt_none"
echo "  清单  : $MANIFEST"
echo "  日志  : $LOG_DIR/"
echo "BATCH_PLACE_OBSTACLES_DONE"
echo "=========================================================="
