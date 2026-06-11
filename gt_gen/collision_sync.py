"""Step 5: voxmap -> cuRobo 碰撞世界同步（未知=障碍）。

见 docs/gt-generation-curobo-implementation.md §4.2。
关键：cuRobo 默认乐观（只把已知占据当障碍）；这里把"非 FREE"（OCCUPIED ∪ UNKNOWN）全部当占据。

实现：cuRobo VOXEL 世界用 ESDF（带符号距离场，约定【占据为正、自由为负】，
见 WorldVoxelCollision.get_sphere_distance(compute_esdf)）。本函数：
  1. 取 cuRobo voxel 网格的体素中心（base 系）；
  2. 在 voxmap 里查每个中心的三态 → 非 FREE = 占据；
  3. 用 scipy 距离变换从占据掩码算带符号 ESDF（占据=+到自由的距离，自由=-到占据的距离）；
  4. 写回 cuRobo（update_voxel_data）。
按"中心查表"而非按下标对齐，故 voxmap 分辨率/范围与 cuRobo 网格无需逐一相等，只需覆盖。
"""
from __future__ import annotations


def _world_centers(handle, vg):
    """cuRobo voxel 网格各体素中心的 base 系坐标 (M,3)。

    注意：get_voxel_grid 返回的 vg.pose 是 cuRobo 内部存的【逆位姿】(obj_in_world)，
    用 create_xyzr_tensor(transform_to_origin=True) 会把平移取反 → 整体错位。
    故这里取【局部中心】(transform_to_origin=False) 再用【已知世界位姿】handle.voxel["pose"]
    （init 时设的 [cx,cy,cz,qw,qx,qy,qz]）摆正。
    """
    import numpy as np
    from gt_gen.sensor import quat_wxyz_to_R

    local = vg.create_xyzr_tensor(transform_to_origin=False,
                                  tensor_args=handle.ta)[:, :3].detach().cpu().numpy()
    pose = np.asarray(handle.voxel["pose"], float)
    return local @ quat_wxyz_to_R(pose[3:7]).T + pose[:3]


def sync_collision_world(handle, voxmap, inflate_voxels=None) -> dict:
    """把 voxmap 的 (OCCUPIED ∪ UNKNOWN) 灌进 cuRobo 的 voxel 碰撞世界。

    inflate_voxels：把「非 FREE」掩码向 FREE 膨胀的体素层数（默认取
        handle.config.voxel_inflate_voxels，default.yaml planner.voxel_inflate_voxels=1）。
        作用：cuRobo voxel 规划器按 ESDF 只保证球心避障，而保守判据 motion_stays_in_free
        对整臂扫掠做 (r + √3/2·voxel) 的过近似，二者差约 1 体素 → cuRobo 规划的路径会贴着
        FREE/UNKNOWN 边界擦过几格 UNKNOWN（实测仅边界~1%格、0 真值碰撞）。把执行世界的障碍
        膨胀 1 层，使 cuRobo 规划留出与判据一致的余量 → GT 整臂扫掠真正 ⊆ FREE。

    返回 {'occupied':.., 'free':..} —— cuRobo 网格中判为占据/自由的体素数（膨胀后）。
    """
    import numpy as np
    import torch
    from scipy import ndimage

    from gt_gen.voxmap import FREE

    if inflate_voxels is None:
        inflate_voxels = int(getattr(handle.config, "voxel_inflate_voxels", 1))

    checker = handle.mg.world_coll_checker
    name = handle.voxel["name"]
    vg = checker.get_voxel_grid(name)
    shape, _, _ = vg.get_grid_shape()                       # [nx,ny,nz]
    # cuRobo 体素中心(base 系)；顺序 = meshgrid(x,y,z,'ij').reshape(-1) = C-order，与 shape 一致
    centers = _world_centers(handle, vg)
    states = np.asarray(voxmap.get(voxmap.world_to_voxel(centers)))
    occ = (states != FREE).reshape(shape)                   # 非 FREE = 占据（保守）

    # 障碍膨胀：把非 FREE 向 FREE 扩 inflate_voxels 层（26 邻接，覆盖对角），给 cuRobo 规划留余量。
    if inflate_voxels and occ.any() and not occ.all():
        st3 = ndimage.generate_binary_structure(3, 3)      # 3x3x3 全连通（含对角）
        occ = ndimage.binary_dilation(occ, structure=st3, iterations=int(inflate_voxels))

    vs = float(vg.voxel_size)
    mx = float(torch.as_tensor(checker.max_esdf_distance).reshape(-1)[0].item())
    if not occ.any():                                       # 全自由
        esdf = np.full(shape, -mx, dtype=np.float32)
    elif occ.all():                                         # 全占据（如初始全 UNKNOWN）
        esdf = np.full(shape, mx, dtype=np.float32)
    else:
        d_in = ndimage.distance_transform_edt(occ) * vs     # 占据体素到最近自由的距离（>0）
        d_out = ndimage.distance_transform_edt(~occ) * vs   # 自由体素到最近占据的距离（>0）
        esdf = (d_in - d_out).astype(np.float32)            # 占据=正、自由=负
        np.clip(esdf, -mx, mx, out=esdf)

    # 只更新特征(ESDF)，不碰位姿/尺寸——update_voxel_data 会对 vg.pose(逆位姿)再求逆而损坏内部位姿。
    feat = torch.as_tensor(esdf.reshape(-1, 1), device=handle.ta.device, dtype=torch.float32)
    checker.update_voxel_features(feat, name=name)
    return {"occupied": int(occ.sum()), "free": int((~occ).sum())}


def curobo_occupied_centers(handle, feature_threshold=None):
    """读回 cuRobo 碰撞世界【实际判为占据】的体素中心(base 系)，用于核对/可视化。

    占据判据 = ESDF feature > 阈值（cuRobo 默认 -0.5·voxel_size）。返回 (M,3) numpy。
    """
    import numpy as np

    checker = handle.mg.world_coll_checker
    vg = checker.get_voxel_grid(handle.voxel["name"])
    centers = _world_centers(handle, vg)
    feat = vg.feature_tensor.reshape(-1).detach().cpu().numpy()
    thr = -0.5 * float(vg.voxel_size) if feature_threshold is None else float(feature_threshold)
    return centers[feat > thr]

