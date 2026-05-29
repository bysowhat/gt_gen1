#!/usr/bin/env bash
# M1.0: Phase 1 环境安装脚本
# 在 env_isaaclab 中安装 cuRobo (本地源码).
# 建图改用自实现 PyTorch 体素栅格 (M1.2), 不再装 nvblox.
#
# 用法:
#   bash setup_env.sh         # 完整安装
#   bash setup_env.sh --check # 只检查不装
#
# 前置:
#   - conda env env_isaaclab 已存在 (含 torch + isaacsim)
#   - GPU + CUDA 可用
#   - cuRobo 源码在 /home/a/Projects/Github/curobo

set -e

PYTHON=/home/a/miniforge3/envs/env_isaaclab/bin/python
PIP=/home/a/miniforge3/envs/env_isaaclab/bin/pip
CUROBO_SRC=/home/a/Projects/Github/curobo

step() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
}

# ---------- Step 0: 基础环境 ----------
step "Step 0: Python / CUDA / pytorch"
$PYTHON -c "
import torch, sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  Torch:  {torch.__version__}')
print(f'  CUDA:   {torch.version.cuda}  (avail={torch.cuda.is_available()})')
if not torch.cuda.is_available():
    sys.exit(1)
print(f'  Device: {torch.cuda.get_device_name(0)}')
"

if [ "${1:-}" = "--check" ]; then
    echo ""
    echo "[--check] 仅检查模式. 跑 verify_env.py:"
    $PYTHON "$(dirname "$0")/verify_env.py"
    exit 0
fi

# ---------- Step 1: 装 cuRobo ----------
step "Step 1: cuRobo (本地源码 build, 5-10 分钟)"
if $PYTHON -c "import curobo" 2>/dev/null; then
    echo "  cuRobo 已安装, 跳过"
else
    if [ ! -d "$CUROBO_SRC" ]; then
        echo "  ERROR: cuRobo 源码目录不存在: $CUROBO_SRC"
        exit 1
    fi
    echo "  清理旧 build (如果是 cpython-310)..."
    rm -rf "$CUROBO_SRC/build"
    cd "$CUROBO_SRC"
    $PIP install -e . --no-build-isolation
fi

# ---------- Step 2: 装 Python 辅助包 ----------
step "Step 2: 装额外 Python 依赖"
$PIP install --upgrade networkx 2>&1 | tail -1   # 全局图 + Dijkstra (M1.5)

# ---------- Step 3: 验证 ----------
step "Step 3: 验证安装"
$PYTHON "$(dirname "$0")/verify_env.py"

echo ""
echo "✓ 环境就绪. 下一步: 跑 curobo_hello.py (M1.0.5) 验证 ur12e 配置."
