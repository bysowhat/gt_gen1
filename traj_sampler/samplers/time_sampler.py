"""
固定时间步长采样
"""
from samplers.base_sampler import BaseSampler


class TimeSampler(BaseSampler):
    """固定时间步长采样器"""
    
    def __init__(self, poses, intrinsics, img_h, img_w, stride=1, **kwargs):
        """
        初始化固定时间步长采样器
        
        Args:
            poses: 位姿列表
            intrinsics: 相机内参矩阵
            img_h: 图像高度
            img_w: 图像宽度
            stride: 采样步长（每隔stride帧采样一次）
        """
        super().__init__(poses, intrinsics, img_h, img_w, **kwargs)
        self.stride = stride
    
    def sample(self):
        """
        执行固定步长采样
        
        Returns:
            selected_indices: list of int, 选中的帧索引
        """
        selected_indices = list(range(0, self.num_frames, self.stride))
        return selected_indices

