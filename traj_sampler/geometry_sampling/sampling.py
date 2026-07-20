import torch


def get_segmented_keyframes(M_action, traj, idx_3d, nodes=None, D_target=0.1, res_ratio = 0.5):
    """
    指令 II-A: 分段关键帧采样（整合硬性节点约束和双重几何条件）
    
    将整个轨迹按硬性节点分段，在每个段内使用双重约束（共视率 + SE(3)距离）
    进行图搜索，找到满足条件的路径。
    
    Args:
        M_action: (B, N, N) SE(3)距离矩阵，M_action[i, j] 表示从帧i到帧j的运动成本
        traj: (B, N, D) 原始轨迹关节角
        idx_3d: (B, N) 是否拍3d
        nodes: 硬性节点列表，如果为None则根据实际帧数自动生成
               - 自动生成规则：生成所有50的倍数节点，例如：
                 * 151帧 -> [0, 50, 100, 150]
                 * 201帧 -> [0, 50, 100, 150, 200]
               - 如果手动指定，代码会自动修正为50的倍数，避免噪音帧
        D_target: 最小运动成本阈值（默认0.1）
        max_path_length: 每个段的最大路径长度（默认16）
        device: torch device
    
    Returns:
        keyframes: 最终采样的关键帧索引列表
    """
    B, N = M_action.shape[:2]
    device = M_action.device
    dtype = M_action.dtype
    
    # 自动生成硬性节点：根据实际帧数生成所有50的倍数节点
    if nodes is None:
        nodes = []
        # 总是包含第一帧
        nodes.append(0)
        # 生成所有50的倍数的节点
        step = 50
        current = step
        # 生成所有不超过num_frames的50的倍数节点
        while current < N:
            nodes.append(current)
            current += step
        assert nodes[-1] == N-1, "the last node must be the final point"
    else:
        # 如果提供了nodes，确保它们是50的倍数（除了首尾）
        nodes = sorted(set(nodes))
        # 验证并修正：确保中间节点是50的倍数
        corrected_nodes = []
        for node in nodes:  # 中间节点
            # 找到最接近的50的倍数
            if node % 50 == 0:
                corrected_nodes.append(node)
            else:
                corrected = ((node + 25) // 50) * 50
                corrected_nodes.append(corrected)
        nodes = corrected_nodes
    
    # 确保nodes是排序的且包含首尾帧
    nodes = sorted(set(nodes))
    if nodes[0] != 0:
        nodes.insert(0, 0)
    
    # print(f"  - Segmenting trajectory with {len(nodes)} hard nodes: {nodes}")
    # print(f"    (Hard nodes are multiples of 50 to avoid noise frames)")
    
    # 步骤1: 生成约束下的掩码矩阵 M_filter
    idx = torch.argmin(torch.abs(M_action - D_target), dim=-1)      # (B, N)
    idx_min = torch.arange(N, device=device).unsqueeze(0).expand_as(idx) + 1
    idx_min[..., -1] = N-1
    next_idx = torch.max(idx, idx_min)   # (B, N)
    # action_mask = torch.zeros_like(M_action, dtype=dtype, device=device)
    # action_mask.scatter_(dim=-1, index=next_idx.unsqueeze(-1), value=1.0)
    # M_filter = action_mask.clone()  # (B, N, N)
    
    # 步骤2: 对每个段进行图搜索
    paths, _, _ = _search_path_in_segment(next_idx)     # (B, L)
    # 添加固定点，组成新的轨迹
    fixed = torch.as_tensor(nodes[1:], dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
    paths = torch.cat((paths, fixed), dim=-1)   # (B, L')
    paths = torch.sort(paths, dim=-1, descending=False)[0]
    # 去掉重复点
    diff = torch.ones_like(paths, dtype=paths.dtype, device=device).bool()
    diff[:, 1:] = paths[:, 1:] != paths[:, :-1]
    paths_new = paths.masked_fill(~diff, -1)
    paths_new = torch.sort(paths_new, dim=-1, descending=False)[0]

    # 步骤3：合并与固定点阈值内相近的点
    paths_final, valid = _update_paths_with_nodes(paths_new, nodes[1:], M_action, D_target / 3, res_ratio)     # (B, L')

    # 步骤4：提取需要的轨迹
    # 增加截止步
    final_q = traj[:, -1].unsqueeze(1)   # (B, 1, D)
    traj = torch.cat((traj, final_q), dim=1)    # (B, N+1, D)
    final_idx_3d = idx_3d[:, 0].unsqueeze(1)    # (B, 1)
    idx_3d = torch.cat((idx_3d, final_idx_3d), dim=1).to(traj.dtype)   # (B, N+1)
    actions_all = torch.cat((traj, idx_3d.unsqueeze(-1)), dim=-1)   # (B, N+1, D+1)
    # 提取actions
    actions = actions_all[valid].clone()    # (B', L', D+1)
    actions = torch.gather(input=actions, dim=1, 
                           index=paths_final.unsqueeze(-1).expand(-1, -1, actions_all.shape[-1]))     # (B', L', D+1)
    # print("actions.shape:", actions.shape)
    
    return actions, valid


def _search_path_in_segment(next_idx, start=0, end=None, max_steps=None):
    """
    next_idx: (B, N)
        next_idx[b, i] = j → 轨迹 b 上 i 跳到 j
    start: starting index (default 0)
    end: ending index (default N-1)
    max_steps: for safety, default N (prevents infinite loop)
    
    Returns:
        paths : (B, L), list[Tensor(B,)]  
        lengths : (B,) number of steps per trajectory
        has_cycle : (B,) whether cycle detected
    """
    device = next_idx.device
    B, N = next_idx.shape

    if end is None:
        end = N - 1

    if max_steps is None:
        max_steps = N  # upper bound

    cur = torch.full((B,), start, dtype=torch.long, device=device)
    finished = (cur == end)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    
    # For cycle detection
    visited = torch.zeros((B, N), dtype=torch.bool, device=device)
    visited[torch.arange(B), cur] = True
    has_cycle = torch.zeros(B, dtype=torch.bool, device=device)

    paths = [cur.clone()]  # store initial state

    step = 0
    while not torch.all(finished) and step < max_steps:
        step += 1

        cur_next = next_idx[torch.arange(B, device=device), cur]

        # Update only trajectories not finished
        active = ~finished
        cur[active] = cur_next[active]
        lengths[active] += 1

        # Check cycle
        cycle = visited[torch.arange(B), cur] & (~finished)
        has_cycle |= cycle

        # Mark visited
        visited[torch.arange(B), cur] = True

        paths.append(cur.clone())
        finished |= (cur == end)
    
    paths = torch.stack(paths, dim=-1)  # (B, L)

    return paths, lengths, has_cycle


def _update_paths_with_nodes(paths_new, nodes, M_action, D_min, res_ratio):
    """
    paths_new: (B, L+M) sorted tensor
    nodes: length M list of node indices
    M_action: (B, N, N)
    D_min: scalar threshold
    """
    device = paths_new.device
    B, LpM = paths_new.shape
    M = len(nodes)
    N = M_action.shape[1]

    nodes_t = torch.tensor(nodes, device=device)  # (M,)

    # 找 nodes 在 paths_new 中的位置
    pos = (paths_new.unsqueeze(-1) == nodes_t).nonzero(as_tuple=True)
    # pos: (2D indices: batch, position in paths)

    # 重构成 (B, M)
    positions = torch.full((B, M), -1, dtype=torch.long, device=device)
    positions[pos[0], pos[2]] = pos[1]      # 采样后轨迹长度LpM的idx

    # 找左右相邻点索引位置
    left_pos = torch.clamp(positions - 1, min=0)                # 采样后轨迹长度LpM的idx
    right_pos = torch.clamp(positions + 1, max=LpM - 1)         # 采样后轨迹长度LpM的idx

    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    # 取出相邻点索引值 (B, M)
    left_points = paths_new[batch_idx.clone(), left_pos]
    right_points = paths_new[batch_idx.clone(), right_pos]

    # 计算距离: M_action[b, a, node] / M_action[b, b, node]
    nodes_t_batched = nodes_t.unsqueeze(0).expand(B, M)  # (B, M)

    dist_left = M_action[batch_idx.clone(), left_points, nodes_t_batched]   # (B, M)
    dist_right = M_action[batch_idx.clone(), nodes_t_batched, right_points]

    # 拼成 (B, M, 2)
    distances = torch.stack([dist_left, dist_right], dim=-1)

    # 判断距离 < 阈值
    mask_left = (dist_left < D_min) * (dist_left != 0)       # (B, M)
    mask_right = (dist_right < D_min) * (dist_right != 0)

    # 将需要替换的位置改为 N-1
    paths_new = paths_new.clone()  # avoid modifying original input

    # 对左侧点更新
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(mask_left)    # (B, M)
    paths_new[batch_idx[mask_left], left_pos[mask_left]] = -1

    # 对右侧点更新
    paths_new[batch_idx[mask_right], right_pos[mask_right]] = -1

    # 将-1替换成N
    paths_new.masked_fill_(paths_new == -1, N)

    # 重新排序
    paths_new = torch.sort(paths_new, dim=-1, descending=False)[0]

    # 去除多余的N
    count = torch.sum((paths_new == N).int(), dim=-1)
    # print("count:", count)
    count_sort = torch.sort(count, dim=-1)[0]
    mean_count = torch.mean(count_sort[4:-4].float())
    # print("mean_count:", mean_count)
    valid = (count > mean_count * (1 - res_ratio))  # (B,)
    # print("valid:", valid)
    paths_new = paths_new[valid]

    count, _ = torch.min(torch.sum((paths_new == N).int(), dim=-1), dim=-1)
    # print("count:", torch.sum((paths_new == N).int(), dim=-1))
    # print(paths_new.shape)
    paths_new = paths_new[:, :-count+1]
    valid = torch.arange(B, device=device)[valid]

    return paths_new, valid


def _graph_search_keyframes_single(M_action, D_target, node_step=50):
    """单条轨迹（B=1）的关键帧【图搜索 + 固定节点 + 锚点近邻合并】，
    返回**升序去重**的关键帧下标 python list。

    这是 get_segmented_keyframes 步骤 1-3 的 B=1 安全版：复用 next_idx / _search_path_in_segment
    与"距锚点过近则合并"逻辑，**去掉**依赖 B>8 的跨批 count_sort[4:-4] / valid 筛选
    （新 Scene 每 (seam,hand) 仅 1 条，curobo GT 无需批内互筛；见方案附录 A.4）。

    node_step：固定锚点步长，每 node_step 帧保底一个关键帧（configs/default.yaml traj_downsample.node_step）。
    """
    device = M_action.device
    B, N = M_action.shape[:2]
    assert B == 1, "本函数只处理单条轨迹（B=1）"

    # 固定节点：0,node_step,2·node_step,... + 末帧（对任意 N 稳健，不要求 N-1 是 node_step 倍数）
    nodes = list(range(0, N, node_step))
    if nodes[-1] != N - 1:
        nodes.append(N - 1)

    # 步骤1：next_idx —— 累计运动量最接近 D_target 的下一帧，且强制前进（>当前）
    idx = torch.argmin(torch.abs(M_action - D_target), dim=-1)                 # (1,N)
    idx_min = torch.arange(N, device=device).unsqueeze(0).expand_as(idx) + 1
    idx_min[..., -1] = N - 1
    next_idx = torch.max(idx, idx_min)                                         # (1,N)

    # 步骤2：链式图搜索得贪心路径，并入固定节点，升序去重
    paths, _, _ = _search_path_in_segment(next_idx)                           # (1,L)
    fixed = torch.as_tensor(nodes, dtype=torch.long, device=device)
    S = torch.cat((paths.reshape(-1), fixed)).tolist()
    S = sorted(set(int(x) for x in S if 0 <= int(x) < N))

    # 步骤3：距固定锚点过近（SE(3) 累计距离 < D_target/3 且 !=0）的【非锚点】帧合并掉
    D_min = D_target / 3.0
    node_set = set(nodes)
    drop = set()
    for n in nodes:
        if n not in S:
            continue
        p = S.index(n)
        if p - 1 >= 0:
            left = S[p - 1]
            d = float(M_action[0, left, n])      # = d_sum[n]-d_sum[left] ≥ 0
            if left not in node_set and 0 < d < D_min:
                drop.add(left)
        if p + 1 < len(S):
            right = S[p + 1]
            d = float(M_action[0, n, right])
            if right not in node_set and 0 < d < D_min:
                drop.add(right)
    return [x for x in S if x not in drop]


def sample_keyframes_single(cam_poses, positions, observe, goal,
                            w_trans=1.0, w_rot=0.1, D_target=0.1, node_step=50):
    """单条 Scene 轨迹（B=1）关键帧稀疏采样 → (L,8) numpy 数组。

    列 = [q1..q6（6 关节角）, observe, goal]（决策 D5）。

    Args:
        cam_poses : (T,4,4) tensor —— 由 CameraFKConfig.forward_cam_pose(positions) 得的相机 base 位姿
                    （FK 全取自 config，见 kinematics.CameraFKConfig / 决策 D6）。
        positions : (T,6) 关节角序列（采样【只依据】此列的运动量 → 决策 D2）。
        observe   : (T,) 或 (T,1) int {0,1}，该帧是否被相机拍过。
        goal      : (T,) 或 (T,1) int {0,1}，该帧是否为某段到达目标。
        w_trans/w_rot/D_target/node_step : SE(3) 度量与图搜索参数
                    （configs/default.yaml traj_downsample 段）。

    流程：build_action_matrix(SE(3)累计距离) → 图搜索挑帧（B=1 安全版）→
         **强制并入** observe==1 / goal==1 的帧（决策 D4）→ 拼 (L,8)。
    """
    import numpy as np
    from .action_metrics import build_action_matrix

    cam_poses = torch.as_tensor(cam_poses)
    positions = np.asarray(positions, dtype=np.float64)
    observe = np.asarray(observe, dtype=np.int64).reshape(-1)
    goal = np.asarray(goal, dtype=np.int64).reshape(-1)
    T = positions.shape[0]
    assert cam_poses.shape[0] == T == observe.shape[0] == goal.shape[0], \
        f"帧数不一致：cam={cam_poses.shape[0]} pos={T} obs={observe.shape[0]} goal={goal.shape[0]}"

    if T == 1:
        S = [0]
    else:
        M_action = build_action_matrix(cam_poses.unsqueeze(0), w_trans, w_rot)   # (1,T,T)
        S = _graph_search_keyframes_single(M_action, D_target, node_step)

    # 强制并入 observe/goal 关键帧（D4）——无论运动量多少都不能丢
    forced = set(np.nonzero(observe == 1)[0].tolist()) | set(np.nonzero(goal == 1)[0].tolist())
    S = sorted(set(S) | forced)

    actions = np.concatenate([
        positions[S],                    # (L,6)
        observe[S, None],                # (L,1)
        goal[S, None],                   # (L,1)
    ], axis=1)                            # (L,8)
    return actions