#!/usr/bin/env bash
# M1.0: Phase 1 环境安装脚本
# 在 env_isaaclab 中安装 cuRobo (本地源码) 和 nvblox_torch (NVlabs git).
#
# 用法:
#   bash setup_env.sh         # 完整安装 (会调用 sudo)
#   bash setup_env.sh --check # 只检查不装
#
# 前置:
#   - conda env env_isaaclab 已存在 (含 torch + isaacsim)
#   - GPU + CUDA 可用
#   - cuRobo 源码在 /home/a/Projects/Github/curobo
#   - 你的 pytorch CXX11_ABI=True (已检查), 所以走简单路径
#
# 重要: 步骤 2/3 会调 sudo (apt install + make install), 请在交互终端运行.

set -e

PYTHON=/home/a/miniforge3/envs/env_isaaclab/bin/python
PIP=/home/a/miniforge3/envs/env_isaaclab/bin/pip
CUROBO_SRC=/home/a/Projects/Github/curobo
PKGS_DIR=/home/a/Projects/Github   # nvblox 和 nvblox_torch 都装到这下面
NVBLOX_SRC=$PKGS_DIR/nvblox
NVBLOX_TORCH_SRC=$PKGS_DIR/nvblox_torch

step() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
}

# ---------- Step 0: 基础环境 ----------
step "Step 0: Python / CUDA / pytorch ABI"
$PYTHON -c "
import torch, sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  Torch:  {torch.__version__}')
print(f'  CUDA:   {torch.version.cuda}  (avail={torch.cuda.is_available()})')
print(f'  Device: {torch.cuda.get_device_name(0)}')
print(f'  CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')
if not torch.cuda.is_available():
    sys.exit(1)
"
TORCH_ABI=$($PYTHON -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
echo "  → 使用 CXX11_ABI=$TORCH_ABI 路径"

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
    echo "  清理旧 build (上次是 cpython-310)..."
    rm -rf "$CUROBO_SRC/build"
    cd "$CUROBO_SRC"
    $PIP install -e . --no-build-isolation
fi

# ---------- Step 2: 装 nvblox C++ 库 ----------
step "Step 2: 装 nvblox C++ 库 (需要 sudo)"
if pkg-config --exists nvblox 2>/dev/null || [ -f /usr/local/lib/libnvblox_lib.so ] || [ -f /usr/local/lib/libnvblox.so ]; then
    echo "  nvblox C++ 库已安装, 跳过"
else
    echo "  装 apt 依赖..."
    sudo apt-get install -y libgoogle-glog-dev libgtest-dev libsqlite3-dev curl tcl libbenchmark-dev

    if [ ! -d "$NVBLOX_SRC" ]; then
        echo "  克隆 nvblox (valtsblukis fork) 到 $NVBLOX_SRC ..."
        git clone https://github.com/valtsblukis/nvblox.git "$NVBLOX_SRC"
    fi

    echo "  cmake + build + install ..."
    mkdir -p "$NVBLOX_SRC/nvblox/build"
    cd "$NVBLOX_SRC/nvblox/build"
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF
    make -j$(nproc)
    sudo make install
    sudo ldconfig
fi

# ---------- Step 3: 装 nvblox_torch ----------
step "Step 3: nvblox_torch (NVlabs git)"
if $PYTHON -c "import nvblox_torch" 2>/dev/null; then
    echo "  nvblox_torch 已安装, 跳过"
else
    if [ ! -d "$NVBLOX_TORCH_SRC" ]; then
        echo "  克隆 nvblox_torch 到 $NVBLOX_TORCH_SRC ..."
        git clone https://github.com/NVlabs/nvblox_torch.git "$NVBLOX_TORCH_SRC"
    fi
    cd "$NVBLOX_TORCH_SRC"
    CMAKE_PREFIX=$($PYTHON -c "import torch.utils; print(torch.utils.cmake_prefix_path)")
    echo "  install.sh 用 cmake_prefix_path=$CMAKE_PREFIX"
    sh install.sh "$CMAKE_PREFIX"
    $PIP install -e . --no-build-isolation
fi

# ---------- Step 4: 验证 ----------
step "Step 4: 验证安装"
$PYTHON "$(dirname "$0")/verify_env.py"

echo ""
echo "✓ 环境就绪. 下一步: 跑 curobo_hello.py (M1.0.5) 验证 ur12e 配置."
