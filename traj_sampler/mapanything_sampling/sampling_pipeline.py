"""
阶段 III：并行矩阵构建
利用 GPU 并行性，一次性计算所有帧对的共视率矩阵

阶段 II：整合约束的GPU采样策略
整合硬性节点约束和双重几何条件（共视率 + SE(3)距离）
"""
import numpy as np
from .covisibility_gpu import compute_score_for_pair
from .geometry_ops import unproject_depth_to_world
from core.geometry import _is_tensor, _to_tensor
from .action_metrics import build_action_matrix

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def build_overlap_matrix(poses_all, depths_all, K_all, img_h, img_w,
                        depth_assoc_error_thres=0.01,
                        depth_assoc_rel_error_thres=0.0,
                        depth_assoc_error_temp=0.0,
                        denominator_mode="valid_target_depth",
                        min_depth=0.04,
                        device=None,
                        precompute_world_pts=True,
                        chunk_size=None):
    """
    指令 III-1: 构建共视率矩阵（GPU优化）
    
    计算所有帧对的共视率矩阵。由于每个帧对的几何操作复杂且条件分支多，
    完全向量化所有帧对的计算非常困难。当前实现通过以下方式优化：
    1. 预计算所有帧的世界坐标点（避免重复计算）
    2. 使用GPU加速单个帧对的计算
    3. 分块处理避免内存溢出
    
    注意：由于几何操作的复杂性，仍需要Python循环遍历帧对。
    每个 compute_score_for_pair 调用内部已充分利用GPU并行性。
    
    Args:
        poses_all: 所有帧的位姿列表，每个元素是 (4, 4) 矩阵
        depths_all: 所有帧的深度图列表，每个元素是 (H, W) 矩阵
        K_all: 相机内参矩阵 (3, 3) 或列表
        img_h: 图像高度
        img_w: 图像宽度
        depth_assoc_error_thres: 深度关联绝对误差阈值（米）
        depth_assoc_rel_error_thres: 深度关联相对误差阈值
        depth_assoc_error_temp: 深度关联误差温度参数
        denominator_mode: 分母模式，"valid_target_depth" 或 "full"
        min_depth: 最小有效深度（米）
        device: torch device
        precompute_world_pts: 是否预计算世界坐标点（优化，强烈推荐）
        chunk_size: 分块大小，如果为None则一次性计算所有
    
    Returns:
        M_overlap: (N, N) 共视率矩阵，M_overlap[i, j] 表示帧i到帧j的共视率
    """
    num_frames = len(poses_all)
    use_torch = TORCH_AVAILABLE and device is not None and device.type == 'cuda'
    
    # 统一K的格式
    if not isinstance(K_all, (list, tuple)):
        K_all = [K_all] * num_frames
    
    # 预计算所有帧的世界坐标点（优化，避免重复计算）
    # 注意：确保 unproject_depth_to_world 总是处理单个帧（N=1），
    # 避免NumPy路径下的效率陷阱。在CPU端进行循环，每个帧单独处理。
    world_pts3d_all = None
    if precompute_world_pts:
        world_pts3d_all = []
        if use_torch:
            print(f"  - Precomputing world points for {num_frames} frames (GPU)...")
        else:
            print(f"  - Precomputing world points for {num_frames} frames (CPU)...")
        for i in range(num_frames):
            # 确保传入单个帧（不是批次），避免NumPy路径下的效率陷阱
            P_world, _ = unproject_depth_to_world(
                K_all[i], poses_all[i], depths_all[i], device=device
            )
            # 展平为点云格式
            if P_world.ndim == 3:
                H, W = P_world.shape[:2]
                P_world = P_world.reshape(H * W, 3)
            world_pts3d_all.append(P_world)
        print(f"  - Precomputation complete.")
    
    # 初始化矩阵
    if use_torch:
        M_overlap = torch.zeros((num_frames, num_frames), device=device)
    else:
        M_overlap = np.zeros((num_frames, num_frames))
    
    # 如果没有指定chunk_size，尝试一次性计算所有
    if chunk_size is None:
        chunk_size = num_frames
    
    # 性能说明：
    # 由于每个帧对的几何操作复杂（涉及条件分支、不同数量的有效点等），
    # 完全向量化所有帧对的计算非常困难。当前实现通过以下方式优化：
    # 1. 预计算世界坐标点（避免重复计算）
    # 2. 每个 compute_score_for_pair 调用内部已充分利用GPU并行性
    # 3. 分块处理避免内存溢出
    # 
    # 注意：虽然仍有Python循环，但每个循环内的计算都是GPU加速的。
    # 对于153帧，需要计算 153×153 = 23,409 个帧对，这是合理的。
    
    # 分块计算（避免内存溢出）
    total_pairs = num_frames * (num_frames - 1)  # 排除对角线
    print(f"  - Computing {total_pairs} frame pairs...")
    for i_start in range(0, num_frames, chunk_size):
        i_end = min(i_start + chunk_size, num_frames)
        
        for i in range(i_start, i_end):
            # 对角线元素：自己与自己的共视率为1
            if use_torch:
                M_overlap[i, i] = 1.0
            else:
                M_overlap[i, i] = 1.0
            
            # 预计算参考帧的世界坐标点（只计算一次，避免重复计算）
            world_pts3d_ref = world_pts3d_all[i] if world_pts3d_all is not None else None
            
            # 计算帧i到所有其他帧的共视率
            # 优化：使用预计算的世界坐标点（world_pts3d_ref），避免重复计算
            # 每个 compute_score_for_pair 调用内部已充分利用GPU并行性
            for j in range(num_frames):
                if i == j:
                    continue
                
                # 计算共视率（使用预计算的世界坐标点，避免重复unproject）
                overlap_score = compute_score_for_pair(
                    depths_all[i], poses_all[i],
                    depths_all[j], poses_all[j],
                    K_all[j], img_h, img_w,
                    depth_assoc_error_thres=depth_assoc_error_thres,
                    depth_assoc_rel_error_thres=depth_assoc_rel_error_thres,
                    depth_assoc_error_temp=depth_assoc_error_temp,
                    denominator_mode=denominator_mode,
                    min_depth=min_depth,
                    world_pts3d_ref=world_pts3d_ref,  # 使用预计算的点，避免重复计算
                    device=device
                )
                
                if use_torch:
                    M_overlap[i, j] = overlap_score
                else:
                    M_overlap[i, j] = overlap_score
            
            # 进度提示（每10个参考帧）
            if (i + 1) % 10 == 0:
                progress = (i + 1) / num_frames * 100
                print(f"    Progress: {i+1}/{num_frames} frames ({progress:.1f}%)")
        
        # 清理GPU内存
        if use_torch:
            torch.cuda.empty_cache()
    
    # 转换为numpy（如果需要）
    if use_torch:
        M_overlap = M_overlap.cpu().numpy()
    
    return M_overlap


def get_adaptive_indices(M_overlap, T_sample=16, O_target=0.6, O_min=0.3, O_max=0.8):
    """
    指令 III-2: 根据共视率矩阵生成自适应采样索引
    
    使用 M_overlap 和 O_target，按照 T_sample 的步长，生成最终的采样索引。
    
    Args:
        M_overlap: (N, N) 共视率矩阵
        T_sample: 采样序列长度（例如16）
        O_target: 目标共视率（例如0.6）
        O_min: 最小共视率阈值
        O_max: 最大共视率阈值
    
    Returns:
        indices_list: 最终的采样帧索引列表
    """
    num_frames = M_overlap.shape[0]
    indices_list = [0]  # 总是包含第一帧
    current_idx = 0
    
    # 生成T_sample长度的序列
    for step in range(T_sample - 1):
        # 寻找下一个满足共视率约束的帧
        best_idx = None
        best_score = -1
        
        # 搜索范围：从当前帧之后开始
        for next_idx in range(current_idx + 1, num_frames):
            overlap_score = M_overlap[current_idx, next_idx]
            
            # 检查是否在合理范围内
            if O_min <= overlap_score <= O_max:
                # 选择最接近O_target的帧
                score_diff = abs(overlap_score - O_target)
                if best_idx is None or score_diff < abs(best_score - O_target):
                    best_idx = next_idx
                    best_score = overlap_score
        
        # 如果找到了合适的帧，添加到列表
        if best_idx is not None:
            indices_list.append(best_idx)
            current_idx = best_idx
        else:
            # 如果没有找到，使用贪婪策略：选择共视率最高的帧
            if current_idx + 1 < num_frames:
                remaining_scores = M_overlap[current_idx, current_idx + 1:]
                best_remaining_idx = current_idx + 1 + np.argmax(remaining_scores)
                indices_list.append(best_remaining_idx)
                current_idx = best_remaining_idx
            else:
                # 已经到达末尾
                break
    
    return indices_list


def get_segmented_keyframes(M_overlap, M_action, nodes=None, D_target=0.1,
                            overlap_threshold=0.25, max_path_length=16,
                            max_jump_distance=10, device=None):
    """
    指令 II-A: 分段关键帧采样（整合硬性节点约束和双重几何条件）
    
    将整个轨迹按硬性节点分段，在每个段内使用双重约束（共视率 + SE(3)距离）
    进行图搜索，找到满足条件的路径。
    
    Args:
        M_overlap: (N, N) 共视率矩阵，M_overlap[i, j] 表示帧i到帧j的共视率
        M_action: (N, N) SE(3)距离矩阵，M_action[i, j] 表示从帧i到帧j的运动成本
        nodes: 硬性节点列表，如果为None则根据实际帧数自动生成
               - 自动生成规则：生成所有50的倍数节点，例如：
                 * 153帧 -> [0, 50, 100, 150]
                 * 201帧 -> [0, 50, 100, 150, 200]
                 * 49帧  -> [0] (没有50的倍数可选)
               - 如果手动指定，代码会自动修正为50的倍数，避免噪音帧
        D_target: 最小运动成本阈值（默认0.1）
        overlap_threshold: 最小共视率阈值（默认0.25）
        max_path_length: 每个段的最大路径长度（默认16）
        device: torch device
    
    Returns:
        keyframes: 最终采样的关键帧索引列表
    """
    num_frames = M_overlap.shape[0]
    use_torch = TORCH_AVAILABLE and device is not None and device.type == 'cuda'
    
    # 自动生成硬性节点：根据实际帧数生成所有50的倍数节点
    if nodes is None:
        nodes = []
        # 总是包含第一帧
        nodes.append(0)
        # 生成所有50的倍数的节点
        step = 50
        current = step
        # 生成所有不超过num_frames的50的倍数节点
        while current < num_frames:
            nodes.append(current)
            current += step
        # 处理最后一个节点：
        # 1. 如果轨迹长度正好是50的倍数（如50, 100, 150, 200），选择该节点-1（避免可能的噪音帧）
        #    例如：50帧  -> 选择49，100帧 -> 选择99，150帧 -> 选择149
        # 2. 如果轨迹长度不是50的倍数，选择最接近的50的倍数（向下取整）
        #    例如：151帧 -> 选择150，152帧 -> 选择150
        # 3. 如果少于50帧（如49帧），没有50的倍数可选，只保留[0]
        if num_frames % step == 0:
            # 如果正好是50的倍数，选择该节点-1（避免可能的噪音帧）
            # 例如：50帧 -> 49，100帧 -> 99，150帧 -> 149
            last_hard_node = num_frames - 1
            if last_hard_node >= 0 and last_hard_node not in nodes:
                nodes.append(last_hard_node)
        else:
            # 如果不是50的倍数，选择最接近的50的倍数（向下取整）
            last_node = (num_frames - 1) // step * step
            if last_node not in nodes and last_node >= 0:
                nodes.append(last_node)
        
        # 确保最后一个节点不超过num_frames-1
        if nodes and nodes[-1] >= num_frames:
            nodes[-1] = num_frames - 1
    else:
        # 如果提供了nodes，确保它们是50的倍数（除了首尾）
        nodes = sorted(set(nodes))
        # 验证并修正：确保中间节点是50的倍数
        corrected_nodes = [nodes[0]]  # 第一个节点（通常是0）
        for node in nodes[1:-1]:  # 中间节点
            # 找到最接近的50的倍数
            corrected = (node // 50) * 50
            if corrected not in corrected_nodes:
                corrected_nodes.append(corrected)
        # 最后一个节点：选择最接近的50的倍数（避免噪音帧）
        if len(nodes) > 1:
            last_node = nodes[-1]
            corrected_last = (last_node // 50) * 50
            if corrected_last not in corrected_nodes:
                corrected_nodes.append(corrected_last)
        nodes = corrected_nodes
    
    # 确保nodes是排序的且包含首尾帧
    nodes = sorted(set(nodes))
    if nodes[0] != 0:
        nodes.insert(0, 0)
    
    # 处理最后一个节点：如果轨迹长度不是50的倍数，选择最接近的50的倍数
    # 例如：151帧 -> 选择150，152帧 -> 选择150
    # 如果少于50帧，选择最接近的50的倍数（向下取整）
    last_ideal = (num_frames - 1) // 50 * 50
    if last_ideal not in nodes:
        nodes.append(last_ideal)
    
    # 确保最后一个节点不超过num_frames - 1
    if nodes[-1] >= num_frames:
        # 如果最后一个节点超出范围，选择最接近的有效节点
        # 优先选择50的倍数，如果都超出范围，则选择最后一个有效帧
        if last_ideal < num_frames:
            nodes[-1] = last_ideal
        else:
            # 如果所有50的倍数都超出范围，选择最后一个有效帧
            nodes[-1] = num_frames - 1
    
    print(f"  - Segmenting trajectory with {len(nodes)} hard nodes: {nodes}")
    print(f"    (Hard nodes are multiples of 50 to avoid noise frames)")
    
    # 转换为张量（如果使用GPU）
    if use_torch:
        M_overlap_t = torch.tensor(M_overlap, device=device)
        M_action_t = torch.tensor(M_action, device=device)
    else:
        M_overlap_t = M_overlap
        M_action_t = M_action
    
    # 步骤1: 生成双重约束掩码矩阵 M_filter
    # M_filter[i, j] = 1 if (M_overlap[i, j] > overlap_threshold) AND (M_action[i, j] >= D_target)
    if use_torch:
        overlap_mask = M_overlap_t > overlap_threshold
        action_mask = M_action_t >= D_target
        M_filter = (overlap_mask & action_mask).cpu().numpy().astype(bool)
    else:
        overlap_mask = M_overlap_t > overlap_threshold
        action_mask = M_action_t >= D_target
        M_filter = (overlap_mask & action_mask).astype(bool)
    
    print(f"  - Generated filter matrix: {np.sum(M_filter)}/{num_frames*num_frames} valid pairs")
    
    # 步骤2: 对每个段进行图搜索
    all_keyframes = []
    
    for seg_idx in range(len(nodes) - 1):
        start_node = nodes[seg_idx]
        end_node = nodes[seg_idx + 1]
        
        print(f"  - Segment {seg_idx + 1}/{len(nodes) - 1}: F_{start_node} -> F_{end_node}")
        
        # 在当前段内搜索路径
        # 设置最大跳跃距离，避免直接跳到硬性节点
        segment_length = end_node - start_node
        max_jump = min(10, segment_length // 4)  # 最大跳跃距离为段长的1/4，但不超过10
        if max_jump < 3:
            max_jump = 3  # 至少允许跳跃3帧
        
        segment_path = _search_path_in_segment(
            M_filter, M_overlap, M_action,
            start_node, end_node,
            max_path_length=max_path_length,
            overlap_threshold=overlap_threshold,
            D_target=D_target,
            max_jump_distance=max_jump_distance
        )
        
        if segment_path:
            # 添加路径（避免重复添加节点）
            if all_keyframes and segment_path[0] == all_keyframes[-1]:
                all_keyframes.extend(segment_path[1:])
            else:
                all_keyframes.extend(segment_path)
        else:
            # 如果找不到路径，至少添加起始和结束节点
            print(f"    Warning: No valid path found, using direct connection")
            if all_keyframes and start_node != all_keyframes[-1]:
                all_keyframes.append(start_node)
            if end_node not in all_keyframes:
                all_keyframes.append(end_node)
    
    # 确保采样结果包含第一帧和最后一帧（即使它们不是硬性节点）
    num_frames = M_overlap.shape[0]
    if 0 not in all_keyframes:
        all_keyframes.insert(0, 0)
    if (num_frames - 1) not in all_keyframes:
        all_keyframes.append(num_frames - 1)
    
    # 去重并排序
    all_keyframes = sorted(set(all_keyframes))
    
    return all_keyframes


def _search_path_in_segment(M_filter, M_overlap, M_action,
                           start_idx, end_idx,
                           max_path_length=16,
                           overlap_threshold=0.25,
                           D_target=0.1,
                           max_jump_distance=10):
    """
    在段内使用图搜索找到从start_idx到end_idx的有效路径
    
    使用贪心策略：在满足双重约束的前提下，优先选择共视率最接近目标值的帧
    避免直接跳到硬性节点，应该逐步搜索
    
    Args:
        max_jump_distance: 最大跳跃距离（默认10），避免一次跳过太多帧
    """
    if start_idx == end_idx:
        return [start_idx]
    
    current_idx = start_idx
    path = [start_idx]
    visited = {start_idx}
    
    # 贪心搜索：每次选择满足约束且共视率最高的下一个帧
    # 注意：路径长度限制应该更宽松，允许更多的中间节点
    while current_idx != end_idx and len(path) < max_path_length * 2:
        # 找到所有满足双重约束的候选帧
        candidates = []
        
        # 限制搜索范围：不要直接跳到end_idx，除非非常接近
        search_end = min(current_idx + max_jump_distance, end_idx)
        # 如果距离end_idx很近（<= max_jump_distance），允许直接跳到end_idx
        if end_idx - current_idx <= max_jump_distance:
            search_end = end_idx
        
        for next_idx in range(current_idx + 1, search_end + 1):
            if next_idx in visited:
                continue
            
            # 检查双重约束
            if M_filter[current_idx, next_idx]:
                overlap_score = M_overlap[current_idx, next_idx]
                action_cost = M_action[current_idx, next_idx]
                # 计算距离惩罚：距离越远，惩罚越大
                distance_penalty = (next_idx - current_idx) / max_jump_distance
                candidates.append((next_idx, overlap_score, action_cost, distance_penalty))
        
        if not candidates:
            # 如果没有满足约束的候选，尝试放宽约束（只要求共视率）
            for next_idx in range(current_idx + 1, search_end + 1):
                if next_idx in visited:
                    continue
                if M_overlap[current_idx, next_idx] > overlap_threshold * 0.5:  # 放宽到一半
                    overlap_score = M_overlap[current_idx, next_idx]
                    distance_penalty = (next_idx - current_idx) / max_jump_distance
                    candidates.append((next_idx, overlap_score, 0.0, distance_penalty))
        
        if not candidates:
            # 如果还是没有候选，尝试放宽约束并扩大搜索范围
            # 首先尝试在扩大范围内找满足共视率要求的帧
            expanded_search_end = min(current_idx + max_jump_distance * 2, end_idx)
            for next_idx in range(current_idx + 1, expanded_search_end + 1):
                if next_idx in visited:
                    continue
                if M_overlap[current_idx, next_idx] > overlap_threshold * 0.3:  # 更宽松的约束
                    overlap_score = M_overlap[current_idx, next_idx]
                    distance_penalty = (next_idx - current_idx) / max_jump_distance
                    candidates.append((next_idx, overlap_score, 0.0, distance_penalty))
                    break  # 只选择第一个满足条件的（最近的）
        
        if not candidates:
            # 如果还是没有候选，尝试选择最近的帧（即使共视率很低）
            # 这是为了确保能够继续前进，而不是卡住
            for next_idx in range(current_idx + 1, min(current_idx + max_jump_distance, end_idx + 1)):
                if next_idx in visited:
                    continue
                # 只要共视率 > 0，就作为候选
                if M_overlap[current_idx, next_idx] > 0.01:
                    overlap_score = M_overlap[current_idx, next_idx]
                    distance_penalty = (next_idx - current_idx) / max_jump_distance
                    candidates.append((next_idx, overlap_score, 0.0, distance_penalty))
                    break  # 只选择最近的
        
        if not candidates:
            # 如果还是没有候选（理论上不应该发生），选择下一个帧
            if current_idx + 1 <= end_idx:
                candidates.append((current_idx + 1, 0.0, 0.0, 0.1))
            else:
                # 如果已经到达或超过end_idx，退出循环
                break
        
        # 选择最佳候选：优先满足运动成本要求，然后选择共视率高的，最后选择距离近的
        candidates.sort(key=lambda x: (
            x[2] >= D_target,  # 优先满足运动成本要求
            -x[1],  # 然后选择共视率最高的
            x[3]  # 最后选择距离最近的（惩罚最小的）
        ), reverse=True)
        
        best_idx, best_overlap, best_action, _ = candidates[0]
        path.append(best_idx)
        visited.add(best_idx)
        current_idx = best_idx
    
    # 确保路径以end_idx结束
    if path[-1] != end_idx:
        path.append(end_idx)
    
    return path

