#!/usr/bin/env bash
# 在服务器上跑 render/render_seam.py：对某个工件的某条焊缝初始位姿(workpiece_pose7)，
# 把机械臂置于固定初始关节角(retract_config)，并行渲染左右目 RGB + 深度(EXR) + meta.npy。
# 工件随机贴库材质（无 UV → 世界/物体空间投影，详见 render/materials.py）。
#
# 跑在远程（/kpfs* 仅远程机器本地、ssh debug 可达；本机无挂载）：
#   ssh debug 'bash /kpfs_dataset/dataset/baiyu/code/render/gt_gen_hanfeng/scripts/bash/kejian/render_seam.sh'
#   # 覆盖示例：
#   STEM=柱_1JdzFk001Mz34qC38vE3On SEAM_IDX=40 bash .../render_seam.sh   # 指定工件+焊缝
#   OBJ=/path/a.obj SEAM_NPY=/path/seam_40.npy OUT=/tmp/r bash .../render_seam.sh  # 直接给路径
#   MAX_ENVS=4 SETTLE_STEPS=16 bash .../render_seam.sh                   # 调渲染参数
#   HEADLESS=0 bash .../render_seam.sh                                   # 带界面（默认 headless）
#
# 数据布局（与 run_plan_init_pose_kejian0.sh 一致）：
#   · 工件 mesh   : $SUB_DIR/<stem>_part/<stem>_part_watertight.obj
#   · 焊缝初始位姿: $INITPOSE_DIR/data/<stem>/seam_<idx>.npy   （含 workpiece_pose7）
#   · 材质库      : /kpfs_dataset/dataset/baiyu/haoyue/Materials/{Base,vMaterials_2}（已写死在 materials.py）
#
# 输出：
#   $OUT/<obj_stem>/<seam_stem>/pose_{p}/{left,right}_{rgb.png,depth.exr} + meta.npy

set -uo pipefail

# ---- 路径/参数（环境变量可覆盖）----
SUB_DIR="${SUB_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/segment_output_subs/part_0}"
INITPOSE_DIR="${INITPOSE_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/initpose_full}"
OUT="${OUT:-/kpfs_dataset_ssd/dataset/render_kejian/render_out}"
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"   # 远程 isaac python（须含 isaaclab）

# 机器人 cuRobo cfg（远程路径；render_seam 从 default.yaml 的 robot.cfg_path 读它，
# 这里把 cfg_path 换成远程值后生成临时 config，绝不改动仓库里的 configs/default.yaml）
ROBOT_CFG="${ROBOT_CFG:-/kpfs_dataset/dataset/baiyu/code/render/urdf_12e_260626_remote/ur12e_full.yml}"

# 要渲染哪个工件/焊缝：给 STEM(+SEAM_IDX) 自动拼路径；或直接给 OBJ/SEAM_NPY 覆盖
STEM="${STEM:-}"
SEAM_IDX="${SEAM_IDX:-40}"
OBJ="${OBJ:-}"
SEAM_NPY="${SEAM_NPY:-}"

# 渲染参数
MAX_ENVS="${MAX_ENVS:-2}"
SPACING="${SPACING:-40.0}"
SETTLE_STEPS="${SETTLE_STEPS:-12}"
HEADLESS="${HEADLESS:-1}"          # 1=无界面（服务器默认）；0=带界面
FORCE_CONVERT="${FORCE_CONVERT:-0}" # 1=强制重转 USD（改了 mesh/urdf 时用）

# 仓库根：本脚本在 scripts/bash/kejian/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCRIPT="$REPO/render/render_seam.py"
CONFIG_SRC="$REPO/configs/default.yaml"

# ---- 由 STEM 推导 OBJ / SEAM_NPY（未直接指定时）----
if [[ -z "$OBJ" && -n "$STEM" ]]; then
    OBJ="$SUB_DIR/${STEM}_part/${STEM}_part_watertight.obj"
fi
if [[ -z "$SEAM_NPY" && -n "$STEM" ]]; then
    SEAM_NPY="$INITPOSE_DIR/data/$STEM/seam_${SEAM_IDX}.npy"
fi

# ---- 校验 ----
[[ -x "$PY" ]]         || { echo "[render] python 不可执行: $PY" >&2; exit 1; }
[[ -f "$SCRIPT" ]]     || { echo "[render] 找不到 render_seam.py: $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG_SRC" ]] || { echo "[render] 找不到 default.yaml: $CONFIG_SRC" >&2; exit 1; }
[[ -f "$ROBOT_CFG" ]]  || { echo "[render] 机器人 cfg 不存在: $ROBOT_CFG（用 ROBOT_CFG= 覆盖）" >&2; exit 1; }
[[ -n "$OBJ" ]]        || { echo "[render] 未指定工件：设 STEM= 或 OBJ=" >&2; exit 1; }
[[ -n "$SEAM_NPY" ]]   || { echo "[render] 未指定焊缝：设 STEM=(+SEAM_IDX=) 或 SEAM_NPY=" >&2; exit 1; }
[[ -f "$OBJ" ]]        || { echo "[render] 工件 obj 不存在: $OBJ" >&2; exit 1; }
[[ -f "$SEAM_NPY" ]]   || { echo "[render] 焊缝 npy 不存在: $SEAM_NPY" >&2; exit 1; }
mkdir -p "$OUT"

# ---- 生成临时 config：把 robot.cfg_path 换成远程 ROBOT_CFG（不动仓库文件）----
# default.yaml 里激活的 `cfg_path:` 只有一行（远程那行是 `# cfg_path:`，带 # 不会被匹配）。
CONFIG="$(mktemp /tmp/render_seam_config.XXXXXX.yaml)"
trap 'rm -f "$CONFIG"' EXIT
sed -E 's|^([[:space:]]*cfg_path:).*|\1 "'"$ROBOT_CFG"'"|' "$CONFIG_SRC" > "$CONFIG"

# ---- 组装参数 ----
ARGS=(--obj "$OBJ" --seam-npy "$SEAM_NPY" --out "$OUT"
      --config "$CONFIG" --max-envs "$MAX_ENVS"
      --spacing "$SPACING" --settle-steps "$SETTLE_STEPS")
[[ "$HEADLESS" == "1" ]]      && ARGS+=(--headless)
[[ "$FORCE_CONVERT" == "1" ]] && ARGS+=(--force-convert)

echo "[render] python    : $PY"
echo "[render] script    : $SCRIPT"
echo "[render] robot cfg : $ROBOT_CFG"
echo "[render] obj       : $OBJ"
echo "[render] seam npy  : $SEAM_NPY"
echo "[render] out       : $OUT"
echo "[render] max_envs=$MAX_ENVS spacing=$SPACING settle_steps=$SETTLE_STEPS headless=$HEADLESS force_convert=$FORCE_CONVERT"

# ---- 运行 + 完成即强杀（靠哨兵文件，不依赖 stdout）------------------------
# 渲染跑完所有 pose 后会卡在 isaac 的 simulation_app.close()（不自己退出）。让 render_seam.py
# 在“全部 pose 落盘后”写一个哨兵文件（RENDER_DONE_FILE）；本脚本把 python 放进独立进程组
# 后台跑，输出经 tee 边显示边落日志，轮询到哨兵出现即判定成功并 SIGKILL 整个进程组
# （连带 isaac 子进程）。比 grep stdout 更鲁棒：不受编码/缓冲/print 文案改动影响。
# 多 GPU 调度器（render_dispatch.sh）会为每个作业传入唯一的 RENDER_DONE_FILE。
GRACE_SEC="${GRACE_SEC:-3}"          # 命中哨兵后等几秒让缓冲/文件落盘再杀
POLL_SEC="${POLL_SEC:-2}"           # 轮询哨兵间隔
DONE_FILE="${RENDER_DONE_FILE:-$(mktemp /tmp/render_seam_done.XXXXXX)}"
rm -f "$DONE_FILE"                   # 起跑前清空，避免命中上次残留
export RENDER_DONE_FILE="$DONE_FILE" # 传给 python：完成后写它

LOG="$(mktemp /tmp/render_seam_log.XXXXXX)"
# 清理临时 config 与 log；DONE_FILE 默认是本脚本的 mktemp，也一并清；若由调度器传入
# （/tmp/render_seam_done.* 之外的路径），保留给调度器判完成。
trap 'rm -f "$CONFIG" "$LOG"; [[ "$DONE_FILE" == /tmp/render_seam_done.* ]] && rm -f "$DONE_FILE"' EXIT

echo "[render] done file : $DONE_FILE"

# setsid：python 成为新进程组组长（PGID==其 PID），便于一次性杀掉整棵进程树
setsid env PYTHONUNBUFFERED=1 "$PY" -u "$SCRIPT" "${ARGS[@]}" > >(tee "$LOG") 2>&1 &
PGID=$!

rc=0
killed_on_done=0
while kill -0 "$PGID" 2>/dev/null; do
    if [[ -f "$DONE_FILE" ]]; then
        echo "[render] 检测到完成哨兵 $DONE_FILE，${GRACE_SEC}s 后强制结束进程组 $PGID ..."
        sleep "$GRACE_SEC"
        kill -KILL -- -"$PGID" 2>/dev/null
        killed_on_done=1
        break
    fi
    sleep "$POLL_SEC"
done

if [[ "$killed_on_done" == "1" ]]; then
    wait "$PGID" 2>/dev/null          # 回收，忽略被 KILL 的退出码
    rc=0
    echo "[render] 完成（已强杀挂起进程）-> $OUT"
else
    wait "$PGID"; rc=$?               # 进程自行退出：用其真实退出码
    if [[ $rc -eq 0 && -f "$DONE_FILE" ]]; then
        echo "[render] 完成 -> $OUT"
    else
        echo "[render] 失败 rc=$rc（未见完成哨兵）" >&2
        [[ $rc -eq 0 ]] && rc=1
    fi
fi
exit $rc
