"""深度图保存助手（搬自 va_simulation/va_sim23_multi.py:104-193）。

用 OpenCV 把深度图写成 16-bit half float + ZIP 压缩的 EXR。
注意：OPENCV_IO_ENABLE_OPENEXR 必须在 import cv2 之前设置，故本模块在最顶
部就设置环境变量；任何模块只要先 import 本模块（或先于 cv2）即可。
"""
import os

# 启用 OpenCV 的 EXR 支持（必须在导入 cv2 之前设置）
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path

import cv2
import numpy as np


def to_numpy(data, dtype=np.float32):
    """将 numpy / torch.Tensor 统一转成指定 dtype 的 numpy 数组。"""
    try:
        import torch

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(dtype)
    except ImportError:
        pass
    if isinstance(data, np.ndarray):
        return data.astype(dtype)
    return np.array(data, dtype=dtype)


def _write_exr(fname, data, params=None):
    """用 OpenCV 把 2D/3D 数组写成 EXR。"""
    if Path(fname).suffix != ".exr":
        raise ValueError(f"只支持 .exr 后缀，收到: {fname}")
    data_np = to_numpy(data, dtype=np.float32)
    if data_np.ndim > 3 or data_np.ndim < 2:
        raise ValueError(f"图像需为 2D 或 3D，收到维度: {data_np.shape}")
    return cv2.imwrite(str(fname), data_np, params if params else [])


def store_depth(fname, data, params=None):
    """把深度图保存为 EXR（16-bit half float + ZIP 压缩）。

    Args:
        fname: 输出路径（.exr）
        data: 深度图（numpy 或 torch.Tensor），会 squeeze 成 2D
        params: 自定义 OpenCV imwrite 参数；缺省用 half float + ZIP
    Returns:
        bool: 是否保存成功
    """
    data_np = to_numpy(data, dtype=np.float32).squeeze()
    if data_np.ndim != 2:
        raise ValueError(f"深度图需为 2D，收到维度: {data_np.shape}")
    if params is None:
        params = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_ZIP,
        ]
    return _write_exr(fname, data_np, params=params)


def store_rgb(fname, rgb):
    """把 RGB（H,W,3 或 H,W,4，uint8）保存为 png（OpenCV 用 BGR）。"""
    rgb_np = to_numpy(rgb, dtype=np.uint8)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] >= 3:
        bgr = cv2.cvtColor(rgb_np[..., :3], cv2.COLOR_RGB2BGR)
    else:
        bgr = rgb_np
    return cv2.imwrite(str(fname), bgr)


def store_seg(fname, data):
    """把【实例分割 id 图】保存为 16-bit 单通道 PNG（每像素一个整数实例 id）。

    id 本身无颜色语义，靠整数值区分实例（不同实例=不同整数），配套 render_info 里
    逐帧的 seg_id_to_label 才知道每个整数是哪类物体。故这里落盘的是【原始整数 id】，
    不上色——渲染侧须设 colorize_instance_segmentation=False。

    Args:
        fname: 输出路径（必须 .png）
        data: id 图（numpy 或 torch.Tensor，整数），会 squeeze 成 2D
    Returns:
        bool: 是否保存成功
    """
    if Path(fname).suffix.lower() != ".png":
        raise ValueError(f"分割 id 图只支持 .png 后缀，收到: {fname}")
    # 先用宽整型防止 uint32→uint16 截断，校验后再降到 uint16
    data_np = to_numpy(data, dtype=np.int64).squeeze()
    if data_np.ndim != 2:
        raise ValueError(f"分割 id 图需为 2D，收到维度: {data_np.shape}")
    lo, hi = int(data_np.min()), int(data_np.max())
    if lo < 0 or hi > 65535:
        raise ValueError(f"实例 id 超出 [0,65535]（min={lo} max={hi}），16-bit PNG 无法表示，请改存 npy")
    return cv2.imwrite(str(fname), data_np.astype(np.uint16))
