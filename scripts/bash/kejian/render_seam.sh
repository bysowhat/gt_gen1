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

PYTHONUNBUFFERED=1 "$PY" -u "$SCRIPT" "${ARGS[@]}"
rc=$?
[[ $rc -eq 0 ]] && echo "[render] 完成 -> $OUT" || echo "[render] 失败 rc=$rc" >&2
exit $rc
