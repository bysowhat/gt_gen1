"""Step 2: 三态体素地图（FREE / OCCUPIED / UNKNOWN）。

见 docs/gt-generation-curobo-implementation.md §2.1、privileged-nbv.md §4.5。

约定：
- 坐标系 = 机械臂 **base_link**（与 cuRobo 规划世界一致 → Step 5 同步最省事）。
- `origin` = 体素 [0,0,0] 的**最小角**（lower corner），单位米。
- `voxel_to_world(idx)` 返回体素**中心**坐标。
- 下标用整型 (i,j,k)，i↔x、j↔y、k↔z；内部用 numpy uint8 网格，order='C'。
- 所有查询/设置方法都支持单个 (3,) 或批量 (N,3) 下标/点。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# 三态常量
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

STATE_NAMES = {UNKNOWN: "UNKNOWN", FREE: "FREE", OCCUPIED: "OCCUPIED"}


class ThreeStateVoxelMap:
    """ROI 包围盒内的三态占据栅格；初始全 UNKNOWN。

    职责：世界(base)坐标 <-> 体素下标互转、查询/设置、导出"非 FREE"掩码（供碰撞同步）。
    """

    def __init__(self, origin: Sequence[float], size_xyz: Sequence[float], voxel_size: float):
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.voxel_size = float(voxel_size)
        size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
        # 体素数：四舍五入，至少 1（避免退化）
        self.dims = np.maximum(np.round(size / self.voxel_size).astype(np.int64), 1)  # (nx,ny,nz)
        self.grid = np.full(tuple(self.dims), UNKNOWN, dtype=np.uint8)

    def __repr__(self):
        # 只打印形状/几何摘要，不展开 grid（避免 VSCode 调试器悬停/变量面板卡住）。
        d = tuple(int(x) for x in self.dims)
        o = tuple(round(float(x), 3) for x in self.origin)
        return f"ThreeStateVoxelMap(dims={d}, voxel_size={self.voxel_size}, origin={o})"

    __str__ = __repr__

    # ---- 形状/几何信息 ----
    @property
    def shape(self):
        return tuple(int(x) for x in self.dims)

    @property
    def num_voxels(self) -> int:
        return int(self.grid.size)

    @property
    def extent(self) -> np.ndarray:
        """物理尺寸 (米)，= dims * voxel_size。"""
        return self.dims * self.voxel_size

    @property
    def center(self) -> np.ndarray:
        """ROI 包围盒中心（base 坐标）；供 Step 5 映射到 cuRobo 的 center-based VoxelGrid。"""
        return self.origin + 0.5 * self.extent

    @property
    def upper(self) -> np.ndarray:
        """最大角（base 坐标）。"""
        return self.origin + self.extent

    # ---- 坐标互转 ----
    def world_to_voxel(self, p) -> np.ndarray:
        """base 坐标 -> 体素下标（向下取整）。p:(3,)->（3,)；(N,3)->(N,3)，dtype=int。

        不做边界裁剪——可能返回越界下标，用 in_bounds() 过滤。
        """
        p = np.asarray(p, dtype=np.float64)
        idx = np.floor((p - self.origin) / self.voxel_size).astype(np.int64)
        return idx

    def voxel_to_world(self, idx) -> np.ndarray:
        """体素下标 -> 体素中心 base 坐标。idx:(3,)->（3,)；(N,3)->(N,3)。"""
        idx = np.asarray(idx, dtype=np.float64)
        return self.origin + (idx + 0.5) * self.voxel_size

    def in_bounds(self, idx) -> np.ndarray:
        """下标是否在网格内。idx:(3,)->标量 bool；(N,3)->(N,) bool。"""
        idx = np.asarray(idx)
        ok = (idx >= 0) & (idx < self.dims)
        return np.all(ok, axis=-1)

    # ---- 查询 / 设置 ----
    def get(self, idx):
        """取状态。idx:(3,)->标量 int；(N,3)->(N,) uint8。越界返回 UNKNOWN（保守、仍属非 FREE）。"""
        idx = np.asarray(idx, dtype=np.int64)
        single = idx.ndim == 1
        idx2 = idx.reshape(1, 3) if single else idx
        inb = self.in_bounds(idx2)
        out = np.full(idx2.shape[0], UNKNOWN, dtype=np.uint8)
        if inb.any():
            ii = idx2[inb]
            out[inb] = self.grid[ii[:, 0], ii[:, 1], ii[:, 2]]
        return int(out[0]) if single else out

    def set_many(self, indices, state: int):
        """把一批体素设为某状态；越界下标静默跳过。indices:(3,) 或 (N,3)。"""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx.reshape(1, 3)
        inb = self.in_bounds(idx)
        if not inb.any():
            return 0
        ii = idx[inb]
        self.grid[ii[:, 0], ii[:, 1], ii[:, 2]] = np.uint8(state)
        return int(inb.sum())

    def set_world(self, points, state: int):
        """便捷：直接用 base 坐标点批量设置（内部转下标）。points:(3,) 或 (N,3)。"""
        return self.set_many(self.world_to_voxel(points), state)

    def get_world(self, points):
        """便捷：直接用 base 坐标点批量取三态（内部转下标）。points:(3,)->int；(N,3)->(N,)。

        与 MultiResVoxelMap.get_world 同名同签名，让「按世界坐标查」的消费者（candidates 的
        视线/站位判定）对单图/多分辨率壳一视同仁。越界按 get 约定返回 UNKNOWN。
        """
        return self.get(self.world_to_voxel(points))

    def fill(self, state: int):
        """整图置为某状态。"""
        self.grid[...] = np.uint8(state)

    # ---- 掩码 / 统计 / 导出 ----
    def non_free_mask(self) -> np.ndarray:
        """返回 OCCUPIED ∪ UNKNOWN 的占据掩码（bool, 形状=dims），喂给 cuRobo 碰撞世界。"""
        return self.grid != FREE

    def counts(self) -> dict:
        """各状态体素数 {UNKNOWN:.., FREE:.., OCCUPIED:..}。"""
        return {s: int((self.grid == s).sum()) for s in (UNKNOWN, FREE, OCCUPIED)}

    def state_centers(self, state: int) -> np.ndarray:
        """返回某状态全部体素的中心 base 坐标，形状 (M,3)；用于可视化/调试。"""
        ii = np.argwhere(self.grid == state)
        if ii.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return self.voxel_to_world(ii)


def build_roi_voxmap(config, margin_voxels: int = 1) -> ThreeStateVoxelMap:
    """按 config.roi 构建 coarse 单张空地图（全 UNKNOWN）：center/dims + voxel_size_m_coarse。

    与 init_curobo 的 cuRobo voxel 世界**同框**（同 center/dims/voxel）。额外向外扩 margin_voxels
    个体素，使 voxmap 比 cuRobo 网格稍大，包住其每轴 `1+floor(dim/voxel)` 多出的"+1"边界层
    （否则那层落在 voxmap 外 → 越界判 UNKNOWN→占据，在 ROI 边界形成多余"墙"）。
    """
    vs = config.voxel_size_m_coarse
    center = np.asarray(config.roi_center, dtype=np.float64).reshape(3)
    dims = np.asarray(config.roi_dims, dtype=np.float64).reshape(3)
    size = dims + 2.0 * margin_voxels * vs
    origin = center - 0.5 * size
    return ThreeStateVoxelMap(origin=origin, size_xyz=size, voxel_size=vs)


class MultiResVoxelMap:
    """多分辨率三态体素图：coarse（整 ROI，粗）+ fine（焊缝盒，细）两张 ThreeStateVoxelMap 的壳。

    动机/约束见 docs/缝周多分辨率voxmap-方案.md。仅在 STOMP-only 下可行（cuRobo VOXEL 世界不参与）。

    归属规则：世界坐标点落在 fine 盒内 → fine 网格；否则 → coarse。fine 盒 = fine 子网格的 AABB
             （半开区间 [origin, upper)），归属与 fine 网格边界严格一致 → 边界不留缝、不重复。
    挖空：coarse 落在 fine 盒内的格子视为不存在（查询/建世界/统计都排除），那块完全交给 fine；
         否则 coarse 的粗 UNKNOWN 会盖住 fine 雕出的 FREE 细通道，细化白做（方案 §3）。

    A 类（按世界坐标，壳透明分派）：get_world / set_world / state_centers / counts。
    B 类（裸下标/单一分辨率语义，无统一答案）：壳不提供。消费者（swept.voxelize_spheres、
         sensor.carve_observe、stomp_iface 建世界）直接取 .coarse / .fine 两张子网格 + in_fine_box()
         各跑一遍再合并/取并（方案 Row 2/3/4）。
    """

    def __init__(self, coarse: ThreeStateVoxelMap, fine: ThreeStateVoxelMap):
        self.coarse = coarse
        self.fine = fine
        # fine 盒 = fine 子网格的 AABB（world 坐标）；归属/挖空都以此为界
        self.fine_min = fine.origin.copy()
        self.fine_max = fine.upper.copy()
        # fine 盒须完全落在 coarse ROI 内，否则归属/挖空会漏
        assert (np.all(self.fine_min >= coarse.origin - 1e-9) and
                np.all(self.fine_max <= coarse.upper + 1e-9)), (
            f"fine 盒 [{self.fine_min}, {self.fine_max}] 须完全落在 coarse ROI "
            f"[{coarse.origin}, {coarse.upper}] 内")

    def __repr__(self):
        return (f"MultiResVoxelMap(coarse={self.coarse!r}, fine={self.fine!r}, "
                f"fine_box=[{np.round(self.fine_min, 3).tolist()}, {np.round(self.fine_max, 3).tolist()}])")

    __str__ = __repr__

    # ---- 点归属 ----
    def in_fine_box(self, points) -> np.ndarray:
        """世界坐标点是否落在 fine 盒内（半开 [min,max)）。points:(3,)->bool；(N,3)->(N,) bool。"""
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p2 = p.reshape(1, 3) if single else p
        inside = np.all((p2 >= self.fine_min) & (p2 < self.fine_max), axis=1)
        return bool(inside[0]) if single else inside

    # ---- A 类：按世界坐标查/设，透明分派 ----
    def get_world(self, points):
        """按世界坐标取三态。points:(3,)->int；(N,3)->(N,) uint8。fine 盒内查 fine，否则查 coarse。"""
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p2 = p.reshape(1, 3) if single else p
        out = np.full(p2.shape[0], UNKNOWN, dtype=np.uint8)
        inf = self.in_fine_box(p2)
        if inf.any():
            out[inf] = self.fine.get(self.fine.world_to_voxel(p2[inf]))
        rest = ~inf
        if rest.any():
            out[rest] = self.coarse.get(self.coarse.world_to_voxel(p2[rest]))
        return int(out[0]) if single else out

    def set_world(self, points, state: int) -> int:
        """按世界坐标批量写三态，分派到对应子网格；返回写入格数。points:(3,) 或 (N,3)。"""
        p = np.asarray(points, dtype=np.float64)
        p2 = p.reshape(1, 3) if p.ndim == 1 else p
        n = 0
        inf = self.in_fine_box(p2)
        if inf.any():
            n += self.fine.set_world(p2[inf], state)
        rest = ~inf
        if rest.any():
            n += self.coarse.set_world(p2[rest], state)
        return n

    def state_centers(self, state: int) -> np.ndarray:
        """某状态全部体素中心（world 坐标）：fine 全部 + coarse 排除 fine 盒内（挖空）。"""
        fc = self.fine.state_centers(state)
        cc = self.coarse.state_centers(state)
        if cc.shape[0]:
            cc = cc[~self.in_fine_box(cc)]
        if fc.shape[0] and cc.shape[0]:
            return np.concatenate([cc, fc], axis=0)
        return fc if fc.shape[0] else cc

    def counts(self) -> dict:
        """各状态体素数 {UNKNOWN,FREE,OCCUPIED}：fine 全部 + coarse 排除 fine 盒内（挖空）。"""
        out = {}
        for s in (UNKNOWN, FREE, OCCUPIED):
            cc = self.coarse.state_centers(s)
            n_c = int((~self.in_fine_box(cc)).sum()) if cc.shape[0] else 0
            n_f = int((self.fine.grid == s).sum())
            out[s] = n_c + n_f
        return out

    # ---- 兼容属性：给仍读单一网格语义的旧代码一个"代表值"（精度敏感处应改用子网格 vs）----
    @property
    def voxel_size(self) -> float:
        """代表值 = coarse 的 vs。精度敏感处（swept pad、carve near）须改用具体子网格 vs。"""
        return self.coarse.voxel_size

    @property
    def shape(self):
        """代表 = coarse 形状（壳无统一下标系；B 类消费者应各取子网格 shape）。"""
        return self.coarse.shape

    @property
    def origin(self):
        return self.coarse.origin

    @property
    def upper(self):
        return self.coarse.upper


def index_grid(voxmap):
    """返回消费者做「裸下标」运算应踩的那张单一网格：MultiResVoxelMap → coarse；单图 → 自身。

    动机：NBV 的视点选择目标 gain=|reveal ∩ B| 是【体素下标集合】运算，多分辨率下 fine/coarse
    两套下标无统一语义，故 B（阻塞 UNKNOWN 集）与 reveal（假设揭示集）统一在 coarse 索引空间算——
    这只影响「下一个相机看哪」的启发式，不影响 GT 正确性闸门（motion_stays_in_free 仍走整壳、fine
    精度，见 swept._motion_stays_in_free_multires），故近缝救援的核心收益不丢（方案 Row 3）。
    """
    return voxmap.coarse if isinstance(voxmap, MultiResVoxelMap) else voxmap


def build_roi_multires_voxmap(config, fine_center, fine_dims) -> MultiResVoxelMap:
    """构建多分辨率 voxmap：coarse=整 ROI（build_roi_voxmap，vs=coarse），fine=焊缝盒（vs=fine）。

    参数
    ----
    fine_center : (3,) 焊缝盒中心（base 系，米）。方案定 = 整条焊缝 AABB 的中心。
    fine_dims   : (3,) 焊缝盒尺寸（米）。方案定 = 焊缝 AABB 各方向外扩 10 cm 后的边长。

    fine 盒不喂 cuRobo VOXEL 世界，故不像 coarse 那样外扩 margin；直接按 center/dims 建。
    调用方（接入 main_loop 时）负责从 seam 几何算出 fine_center / fine_dims。
    """
    coarse = build_roi_voxmap(config)
    vs_f = float(config.voxel_size_m_fine)
    center = np.asarray(fine_center, dtype=np.float64).reshape(3)
    dims = np.asarray(fine_dims, dtype=np.float64).reshape(3)
    origin = center - 0.5 * dims
    fine = ThreeStateVoxelMap(origin=origin, size_xyz=dims, voxel_size=vs_f)
    return MultiResVoxelMap(coarse=coarse, fine=fine)


def fine_box_from_seam_points(seam_points_base, margin_m: float):
    """一批 base 系焊缝点 (N,3) → fine 盒 (center(3,), dims(3,))：轴对齐包围盒各方向外扩 margin_m。

    供 build_roi_multires_voxmap 的 fine_center / fine_dims 来源。焊缝点须已在 base 系
    （工件 mesh 系的 seam_line 需先经 T_workpiece_in_base 变换，见 Scene.fine_box_from_seam）。
    """
    p = np.asarray(seam_points_base, dtype=np.float64).reshape(-1, 3)
    assert p.shape[0] >= 1, "至少需要一个焊缝点"
    lo = p.min(axis=0) - margin_m
    hi = p.max(axis=0) + margin_m
    center = 0.5 * (lo + hi)
    dims = hi - lo
    return center, dims
