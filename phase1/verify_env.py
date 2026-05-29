"""M1.0: Phase 1 环境验证.

验证项:
  1. Python 版本
  2. PyTorch + CUDA
  3. cuRobo 安装 + 关键模块 import
  4. networkx (M1.5 全局图用)
  5. Isaac Sim Python (omni / pxr 在 AppLauncher 启动后才可用)

跑法: /home/a/miniforge3/envs/env_isaaclab/bin/python verify_env.py
"""
from __future__ import annotations

import sys


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(name: str, fn) -> bool:
    try:
        info = fn()
        print(f"  {PASS} {name:30s} {info}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name:30s} {type(e).__name__}: {e}")
        return False


def _check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"需要 Python ≥ 3.10, 当前 {v.major}.{v.minor}")
    return f"{v.major}.{v.minor}.{v.micro}"


def _check_torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    return (
        f"torch={torch.__version__}  cuda={torch.version.cuda}  "
        f"device={torch.cuda.get_device_name(0)}"
    )


def _check_curobo():
    import curobo
    from curobo.types.base import TensorDeviceType  # noqa: F401
    from curobo.util_file import get_robot_configs_path

    return f"v={getattr(curobo, '__version__', 'dev')}  configs={get_robot_configs_path()}"


def _check_networkx():
    import networkx

    return f"v={networkx.__version__}"


def _check_isaacsim():
    import isaacsim

    return f"loaded from {isaacsim.__file__}"


def main() -> int:
    print()
    print("Phase 1 环境验证")
    print("─" * 60)

    results = [
        check("Python",       _check_python),
        check("PyTorch+CUDA", _check_torch_cuda),
        check("cuRobo",       _check_curobo),
        check("networkx",     _check_networkx),
        check("Isaac Sim",    _check_isaacsim),
    ]

    print("─" * 60)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"  {n_pass}/{n_total} 全部通过. 可以进 M1.0.5 (curobo_hello.py).")
        return 0
    print(f"  {n_pass}/{n_total} 通过, {n_total - n_pass} 失败.")
    print()
    print("处理建议:")
    print("  - cuRobo 失败:    检查 /home/a/Projects/Github/curobo, 重跑 setup_env.sh")
    print("  - networkx 失败:  pip install networkx")
    print("  - Isaac Sim 失败: conda activate env_isaaclab 没生效")
    return 1


if __name__ == "__main__":
    sys.exit(main())
