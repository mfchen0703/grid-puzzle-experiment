"""
Softmax 模型拟合：用 agent 的策略预测人类每一步的 region 选择。

模型:
  每一步，对每个未着色 region 计算 utility:
    U(r) = beta_n * colored_neighbor_count(r) + beta_d * (-centroid_distance(r))
  通过 softmax 转化为选择概率:
    P(r) = exp(U(r)) / sum_r' exp(U(r'))
  拟合 beta_n, beta_d 使得人类选择的 log-likelihood 最大化。

用法:
    python fit_softmax.py                     # 拟合 data/ 目录下所有被试
    python fit_softmax.py path/to/data_dir    # 指定数据目录
"""

import os
import re
import sys
import math
import ctypes
import glob as glob_module
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# ═══════════════════════════════════════════════════════════════════
# 1. 地图生成 (与 App.tsx / dump_rounds.py 完全一致)
# ═══════════════════════════════════════════════════════════════════

ROWS, COLS = 12, 20
ROUND_SIZES = [20, 23, 26, 28, 31, 34, 37, 39, 42, 45]
ROUND_SEEDS = [42, 137, 256, 389, 512, 647, 783, 891, 1024, 1157]
PRACTICE_SIZES = [10, 10]
PRACTICE_SEEDS = [9999, 8888]
NUM_COLORS = 4


def _i32(x):
    return ctypes.c_int32(x & 0xFFFFFFFF).value


def _u32(x):
    return x & 0xFFFFFFFF


def _imul(a, b):
    return _i32(a * b)


def mulberry32(seed):
    s = _i32(seed)
    def rand():
        nonlocal s
        s = _i32(s + 0x6D2B79F5)
        t = _imul(s ^ _u32(_u32(s) >> 15), _i32(1 | s))
        t = _i32(t + _imul(t ^ _u32(_u32(t) >> 7), _i32(61 | t))) ^ t
        return _u32(t ^ _u32(_u32(t) >> 14)) / 4294967296
    return rand


def generate_map(num_regions, rng):
    grid = [[-1] * COLS for _ in range(ROWS)]
    regions = []

    for i in range(num_regions):
        while True:
            r = int(rng() * ROWS)
            c = int(rng() * COLS)
            if grid[r][c] == -1:
                break
        grid[r][c] = i
        regions.append([(r, c)])

    changed = True
    while changed:
        changed = False
        for i in range(num_regions):
            neighbors = []
            for (r, c) in regions[i]:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == -1:
                        neighbors.append((nr, nc))
            if neighbors:
                nr, nc = neighbors[int(rng() * len(neighbors))]
                if grid[nr][nc] == -1:
                    grid[nr][nc] = i
                    regions[i].append((nr, nc))
                    changed = True

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == -1:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != -1:
                        grid[r][c] = grid[nr][nc]
                        regions[grid[nr][nc]].append((r, c))
                        break

    adjacency = defaultdict(set)
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if nr < ROWS and nc < COLS:
                    if grid[r][c] != grid[nr][nc]:
                        adjacency[grid[r][c]].add(grid[nr][nc])
                        adjacency[grid[nr][nc]].add(grid[r][c])

    return grid, regions, adjacency


def centroid_distance(cells):
    cy, cx = ROWS / 2, COLS / 2
    avg_r = sum(r for r, c in cells) / len(cells)
    avg_c = sum(c for r, c in cells) / len(cells)
    return math.hypot(avg_r - cy, avg_c - cx)


# 预生成所有 round 的地图
def build_all_maps():
    """返回 {round_label: (regions, adjacency)} 字典。"""
    maps = {}
    for i, (size, seed) in enumerate(zip(PRACTICE_SIZES, PRACTICE_SEEDS)):
        label = f"P{i + 1}"
        rng = mulberry32(seed)
        _, regions, adjacency = generate_map(size, rng)
        maps[label] = (regions, adjacency)
    for i, (size, seed) in enumerate(zip(ROUND_SIZES, ROUND_SEEDS)):
        label = str(i + 1)
        rng = mulberry32(seed)
        _, regions, adjacency = generate_map(size, rng)
        maps[label] = (regions, adjacency)
    return maps


# ═══════════════════════════════════════════════════════════════════
# 2. 数据解析
# ═══════════════════════════════════════════════════════════════════

# 匹配 "Colored Region X with Color Y" 或 "Colored Region X with Eraser"
ACTION_RE = re.compile(
    r'Colored Region (\d+) with (Color (\d+)|Eraser)'
)


def parse_csv(filepath):
    """解析被试 CSV 文件，返回 actions 列表。

    每个 action = {
        'round': str,           # 'P1','P2','1',...,'10'
        'num_regions': int,
        'region': int | None,   # 0-based region id (None for Game Started)
        'color': int | None,    # 0-based color index (None for eraser / Game Started)
        'is_eraser': bool,
        'is_start': bool,
        'time': float,
    }
    """
    actions = []
    in_actions = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Actions]':
                in_actions = True
                continue
            if line == '[Adjacency]':
                break
            if not in_actions:
                continue
            if line.startswith('SessionID,'):
                continue  # header

            parts = line.split(',', 5)
            if len(parts) < 6:
                continue
            if not parts[2]:
                continue

            round_label = parts[1]
            num_regions = int(parts[2])
            action_text = parts[4].strip('"')
            time_taken = float(parts[5])

            if 'Game Started' in action_text:
                actions.append({
                    'round': round_label,
                    'num_regions': num_regions,
                    'region': None,
                    'color': None,
                    'is_eraser': False,
                    'is_start': True,
                    'time': time_taken,
                })
                continue

            m = ACTION_RE.match(action_text)
            if not m:
                continue

            region_1based = int(m.group(1))
            is_eraser = m.group(2) == 'Eraser'
            color_idx = None if is_eraser else int(m.group(3)) - 1  # 0-based

            actions.append({
                'round': round_label,
                'num_regions': num_regions,
                'region': region_1based - 1,  # 0-based
                'color': color_idx,
                'is_eraser': is_eraser,
                'is_start': False,
                'time': time_taken,
            })

    return actions


# ═══════════════════════════════════════════════════════════════════
# 3. 状态回放 & 构建拟合数据
# ═══════════════════════════════════════════════════════════════════

def is_color_legal(region_id, color, adjacency, current_colors):
    """颜色是否合法（不与邻居冲突）。"""
    for nb in adjacency[region_id]:
        if current_colors[nb] == color:
            return False
    return True


def count_effective_colors(region_id, adjacency, current_colors, used_colors):
    """计算 Canonical Color Normalization 下的有效颜色数。

    有效颜色数 = 已使用且合法的颜色数 + (1 if 存在未使用且合法的颜色)
    因为所有未使用颜色在规范化映射下等价，只算 1 个。
    """
    legal = [
        c for c in range(NUM_COLORS)
        if is_color_legal(region_id, c, adjacency, current_colors)
    ]
    n_used_legal = sum(1 for c in legal if c in used_colors)
    n_unused_legal = sum(1 for c in legal if c not in used_colors)
    return n_used_legal + (1 if n_unused_legal > 0 else 0)


def build_fitting_steps(actions, maps, include_practice=False):
    """回放人类操作，提取可用于拟合的步骤。

    只保留满足以下条件的步骤:
    - 是着色操作（非 eraser）
    - 被着色的 region 之前未着色
    - 所选颜色合法

    返回 list of dict:
    {
        'round': str,
        'uncolored': list[int],     # 当前未着色的 region id 列表
        'chosen_region': int,       # 人类选择的 region (0-based)
        'chosen_color': int,        # 人类选择的 color (0-based)
        'regions': list,            # region cells
        'adjacency': dict,          # adjacency graph
        'current_colors': list,     # 着色状态（copy）
        'num_legal_colors': int,    # 选中 region 的合法颜色数（原始）
        'num_effective_colors': int, # 规范化后的有效颜色数
    }
    """
    fitting_steps = []
    current_colors = None
    current_round = None
    regions = None
    adjacency = None
    used_colors = None  # 当前 round 中已使用过的颜色集合

    for act in actions:
        # 新的 round 开始
        if act['is_start']:
            current_round = act['round']
            if not include_practice and current_round.startswith('P'):
                current_colors = None
                continue
            if current_round not in maps:
                current_colors = None
                continue
            regions, adjacency = maps[current_round]
            current_colors = [None] * len(regions)
            used_colors = set()
            continue

        if current_colors is None:
            continue  # 跳过不拟合的 round

        rid = act['region']
        if rid is None:
            continue

        # Eraser: 更新状态但不计入拟合
        if act['is_eraser']:
            current_colors[rid] = None
            continue

        color = act['color']

        # 只拟合：未着色 region + 合法颜色 的步骤
        if current_colors[rid] is None and is_color_legal(rid, color, adjacency, current_colors):
            uncolored = [i for i in range(len(regions)) if current_colors[i] is None]

            # 原始合法颜色数
            n_legal = sum(
                1 for c in range(NUM_COLORS)
                if is_color_legal(rid, c, adjacency, current_colors)
            )

            # 规范化后的有效颜色数
            n_effective = count_effective_colors(
                rid, adjacency, current_colors, used_colors
            )

            fitting_steps.append({
                'round': current_round,
                'uncolored': uncolored,
                'chosen_region': rid,
                'chosen_color': color,
                'regions': regions,
                'adjacency': adjacency,
                'current_colors': list(current_colors),  # copy
                'num_legal_colors': n_legal,
                'num_effective_colors': n_effective,
            })

        # 不管是否计入拟合，都更新状态
        current_colors[rid] = color
        if color is not None:
            used_colors.add(color)

    return fitting_steps


# ═══════════════════════════════════════════════════════════════════
# 4. Softmax 模型
# ═══════════════════════════════════════════════════════════════════

def compute_utilities(uncolored, regions, adjacency, current_colors, params):
    """计算每个未着色 region 的 utility。"""
    beta_n, beta_d = params
    utilities = np.empty(len(uncolored))
    for i, rid in enumerate(uncolored):
        colored_nbrs = sum(
            1 for nb in adjacency[rid] if current_colors[nb] is not None
        )
        dist = centroid_distance(regions[rid])
        utilities[i] = beta_n * colored_nbrs + beta_d * (-dist)
    return utilities


def neg_log_likelihood(params, fitting_steps):
    """计算负 log-likelihood（用于最小化）。

    LL = Σ [log P(chosen_region) + log P(chosen_color | chosen_region)]

    其中:
    - P(region) = softmax(U(region)) over uncolored regions
    - P(color | region) = 1 / num_effective_colors
      (Canonical Color Normalization: 未使用颜色等价，只算 1 个)
    """
    total_nll = 0.0

    for step in fitting_steps:
        utilities = compute_utilities(
            step['uncolored'], step['regions'],
            step['adjacency'], step['current_colors'], params
        )
        chosen_idx = step['uncolored'].index(step['chosen_region'])

        # Region choice: softmax log-probability
        log_p_region = utilities[chosen_idx] - logsumexp(utilities)

        # Color choice: uniform over effective colors (canonical normalization)
        log_p_color = -np.log(step['num_effective_colors'])

        total_nll -= (log_p_region + log_p_color)

    return total_nll


def neg_log_likelihood_region_only(params, fitting_steps):
    """只拟合 region 选择（不含颜色选择）。"""
    total_nll = 0.0

    for step in fitting_steps:
        utilities = compute_utilities(
            step['uncolored'], step['regions'],
            step['adjacency'], step['current_colors'], params
        )
        chosen_idx = step['uncolored'].index(step['chosen_region'])
        log_p = utilities[chosen_idx] - logsumexp(utilities)
        total_nll -= log_p

    return total_nll


# ═══════════════════════════════════════════════════════════════════
# 5. 拟合 & 输出
# ═══════════════════════════════════════════════════════════════════

def fit_participant(filepath, maps, include_practice=False):
    """拟合单个被试的数据。"""
    actions = parse_csv(filepath)
    fitting_steps = build_fitting_steps(actions, maps, include_practice)

    if not fitting_steps:
        print(f"  警告: {filepath} 没有可拟合的步骤")
        return None

    # 初始参数
    x0 = np.array([1.0, 0.1])

    # 优化
    result = minimize(
        neg_log_likelihood,
        x0,
        args=(fitting_steps,),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8},
    )

    beta_n, beta_d = result.x
    nll = result.fun
    ll = -nll
    n_steps = len(fitting_steps)
    n_params = 2
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_steps) - 2 * ll

    # 随机模型的 LL（baseline，同样使用 effective colors）
    random_ll = 0.0
    for step in fitting_steps:
        n_choices = len(step['uncolored'])
        random_ll += -np.log(n_choices) + (-np.log(step['num_effective_colors']))

    # 计算准确率（人类选择是否为 utility 最高的 region）
    correct = 0
    for step in fitting_steps:
        utilities = compute_utilities(
            step['uncolored'], step['regions'],
            step['adjacency'], step['current_colors'], result.x
        )
        chosen_idx = step['uncolored'].index(step['chosen_region'])
        if chosen_idx == np.argmax(utilities):
            correct += 1
    accuracy = correct / n_steps

    # 每 round 的 LL
    round_stats = defaultdict(lambda: {'ll': 0.0, 'random_ll': 0.0, 'n': 0, 'correct': 0})
    for step in fitting_steps:
        rnd = step['round']
        utilities = compute_utilities(
            step['uncolored'], step['regions'],
            step['adjacency'], step['current_colors'], result.x
        )
        chosen_idx = step['uncolored'].index(step['chosen_region'])
        log_p = utilities[chosen_idx] - logsumexp(utilities)
        log_p_color = -np.log(step['num_effective_colors'])

        round_stats[rnd]['ll'] += log_p + log_p_color
        round_stats[rnd]['random_ll'] += -np.log(len(step['uncolored'])) + log_p_color
        round_stats[rnd]['n'] += 1
        if chosen_idx == np.argmax(utilities):
            round_stats[rnd]['correct'] += 1

    return {
        'filepath': filepath,
        'beta_n': beta_n,
        'beta_d': beta_d,
        'log_likelihood': ll,
        'random_ll': random_ll,
        'n_steps': n_steps,
        'aic': aic,
        'bic': bic,
        'accuracy': accuracy,
        'round_stats': dict(round_stats),
        'converged': result.success,
    }


def main():
    # 确定数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'data')

    data_dir = os.path.abspath(data_dir)
    csv_files = sorted(glob_module.glob(os.path.join(data_dir, 'data_*.csv')))

    if not csv_files:
        print(f"在 {data_dir} 中未找到 data_*.csv 文件")
        sys.exit(1)

    print(f"数据目录: {data_dir}")
    print(f"找到 {len(csv_files)} 个被试文件\n")

    # 预生成地图
    maps = build_all_maps()

    all_results = []

    for filepath in csv_files:
        name = os.path.basename(filepath)
        print(f"{'=' * 60}")
        print(f"被试: {name}")
        print(f"{'=' * 60}")

        result = fit_participant(filepath, maps)
        if result is None:
            continue
        all_results.append(result)

        print(f"  拟合步骤数:   {result['n_steps']}")
        print(f"  收敛:         {'是' if result['converged'] else '否'}")
        print(f"  beta_n (邻居): {result['beta_n']:.4f}")
        print(f"  beta_d (距离): {result['beta_d']:.4f}")
        print(f"  Log-likelihood:     {result['log_likelihood']:.2f}")
        print(f"  Random baseline LL: {result['random_ll']:.2f}")
        print(f"  AIC:                {result['aic']:.2f}")
        print(f"  BIC:                {result['bic']:.2f}")
        print(f"  准确率 (top-1):     {result['accuracy']:.1%}")
        print()

        # 每 round 的结果
        print(f"  {'Round':<8} {'Steps':>6} {'LL':>10} {'Random LL':>10} {'Accuracy':>10}")
        print(f"  {'-' * 44}")
        for rnd in sorted(result['round_stats'].keys(),
                          key=lambda x: (0, int(x)) if x.isdigit() else (-1, 0)):
            rs = result['round_stats'][rnd]
            acc = rs['correct'] / rs['n'] if rs['n'] > 0 else 0
            print(f"  {rnd:<8} {rs['n']:>6} {rs['ll']:>10.2f} {rs['random_ll']:>10.2f} {acc:>10.1%}")
        print()

    # 汇总
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"汇总 ({len(all_results)} 个被试)")
        print(f"{'=' * 60}")
        beta_ns = [r['beta_n'] for r in all_results]
        beta_ds = [r['beta_d'] for r in all_results]
        accs = [r['accuracy'] for r in all_results]
        print(f"  beta_n:  mean={np.mean(beta_ns):.4f}, std={np.std(beta_ns):.4f}")
        print(f"  beta_d:  mean={np.mean(beta_ds):.4f}, std={np.std(beta_ds):.4f}")
        print(f"  准确率:  mean={np.mean(accs):.1%}, std={np.std(accs):.1%}")


if __name__ == '__main__':
    main()
