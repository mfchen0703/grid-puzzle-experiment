"""
Global 策略统计：起点位置 & 空间连贯性

指标:
  1. 起点位置: 每 round 第一步选择区域的质心坐标及其到地图中心的距离
  2. 空间连贯性: 连续着色步骤之间区域质心的距离（均值、标准差）

用法:
    python global_stats.py                     # 分析 data/ 目录下所有被试
    python global_stats.py path/to/data_dir    # 指定数据目录
"""

import os
import sys
import math
import numpy as np

from fit_softmax import (
    ROWS, COLS, build_all_maps, parse_csv, centroid_distance,
)

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def region_centroid(cells):
    """返回区域质心坐标 (row, col)。"""
    avg_r = sum(r for r, c in cells) / len(cells)
    avg_c = sum(c for r, c in cells) / len(cells)
    return avg_r, avg_c


def centroid_pair_distance(cells_a, cells_b):
    """两个区域质心之间的欧氏距离。"""
    r1, c1 = region_centroid(cells_a)
    r2, c2 = region_centroid(cells_b)
    return math.hypot(r1 - r2, c1 - c2)


# ═══════════════════════════════════════════════════════════════════
# 提取每 round 的着色序列（含 eraser 信息）
# ═══════════════════════════════════════════════════════════════════

def extract_coloring_sequences(actions, maps, include_practice=False):
    """从原始 actions 中提取每 round 的有效着色序列。

    返回 dict: {round_label: [region_id, ...]}
    只保留"新着色"步骤（region 之前未着色 & 颜色合法），与 build_fitting_steps 一致。
    """
    from fit_softmax import is_color_legal, NUM_COLORS

    sequences = {}
    current_colors = None
    current_round = None
    regions = None
    adjacency = None

    for act in actions:
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
            sequences[current_round] = []
            continue

        if current_colors is None:
            continue

        rid = act['region']
        if rid is None:
            continue

        if act['is_eraser']:
            current_colors[rid] = None
            continue

        color = act['color']
        if current_colors[rid] is None and is_color_legal(rid, color, adjacency, current_colors):
            sequences[current_round].append(rid)

        current_colors[rid] = color

    return sequences


# ═══════════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════════

def compute_starting_point(sequences, maps):
    """计算每 round 起点区域的质心坐标和到中心距离。

    返回 list of dict:
        {'round': str, 'region': int,
         'centroid_row': float, 'centroid_col': float,
         'dist_to_center': float}
    """
    results = []
    for rnd, seq in sorted(sequences.items(),
                           key=lambda x: (0, int(x[0])) if x[0].isdigit() else (-1, 0)):
        if not seq:
            continue
        first_rid = seq[0]
        regions, _ = maps[rnd]
        cr, cc = region_centroid(regions[first_rid])
        dist = centroid_distance(regions[first_rid])
        results.append({
            'round': rnd,
            'region': first_rid,
            'centroid_row': cr,
            'centroid_col': cc,
            'dist_to_center': dist,
        })
    return results


def compute_spatial_coherence(sequences, maps):
    """计算每 round 连续步骤之间的质心距离。

    返回 list of dict:
        {'round': str, 'n_steps': int,
         'mean_step_dist': float, 'std_step_dist': float,
         'median_step_dist': float,
         'neighbor_transition_rate': float,  # 连续步骤是否为邻居的比例
         'step_distances': list[float]}
    """
    results = []
    for rnd, seq in sorted(sequences.items(),
                           key=lambda x: (0, int(x[0])) if x[0].isdigit() else (-1, 0)):
        if len(seq) < 2:
            continue
        regions, adjacency = maps[rnd]

        dists = []
        neighbor_transitions = 0
        for i in range(1, len(seq)):
            d = centroid_pair_distance(regions[seq[i - 1]], regions[seq[i]])
            dists.append(d)
            if seq[i] in adjacency[seq[i - 1]]:
                neighbor_transitions += 1

        dists_arr = np.array(dists)
        results.append({
            'round': rnd,
            'n_steps': len(seq),
            'mean_step_dist': float(np.mean(dists_arr)),
            'std_step_dist': float(np.std(dists_arr)),
            'median_step_dist': float(np.median(dists_arr)),
            'neighbor_transition_rate': neighbor_transitions / len(dists),
            'step_distances': dists,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def analyze_participant(filepath, maps):
    """分析单个被试的 global 策略指标。"""
    actions = parse_csv(filepath)
    sequences = extract_coloring_sequences(actions, maps)

    if not sequences:
        return None

    starting = compute_starting_point(sequences, maps)
    coherence = compute_spatial_coherence(sequences, maps)

    return {
        'filepath': filepath,
        'starting_points': starting,
        'spatial_coherence': coherence,
        'sequences': sequences,
    }


def main():
    import glob as glob_module

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

    maps = build_all_maps()

    print(f"数据目录: {data_dir}")
    print(f"找到 {len(csv_files)} 个被试文件\n")

    all_results = []

    for filepath in csv_files:
        name = os.path.splitext(os.path.basename(filepath))[0].replace('data_', '')
        result = analyze_participant(filepath, maps)
        if result is None:
            print(f"  警告: {name} 没有有效数据")
            continue
        all_results.append((name, result))

        print(f"{'=' * 70}")
        print(f"被试: {name}")
        print(f"{'=' * 70}")

        # ── 起点位置 ──
        print(f"\n  [起点位置]")
        print(f"  {'Round':<8} {'Region':>7} {'Row':>7} {'Col':>7} {'Dist→Center':>12}")
        print(f"  {'-' * 43}")
        dists = []
        for sp in result['starting_points']:
            print(f"  {sp['round']:<8} {sp['region']:>7} {sp['centroid_row']:>7.1f} "
                  f"{sp['centroid_col']:>7.1f} {sp['dist_to_center']:>12.2f}")
            dists.append(sp['dist_to_center'])
        print(f"  {'Mean':<8} {'':>7} {'':>7} {'':>7} {np.mean(dists):>12.2f}")
        print(f"  {'Std':<8} {'':>7} {'':>7} {'':>7} {np.std(dists):>12.2f}")

        # ── 空间连贯性 ──
        print(f"\n  [空间连贯性]")
        print(f"  {'Round':<8} {'Steps':>6} {'MeanDist':>10} {'StdDist':>10} "
              f"{'MedianDist':>11} {'NbrRate':>8}")
        print(f"  {'-' * 55}")
        all_mean = []
        all_nbr = []
        for sc in result['spatial_coherence']:
            print(f"  {sc['round']:<8} {sc['n_steps']:>6} {sc['mean_step_dist']:>10.2f} "
                  f"{sc['std_step_dist']:>10.2f} {sc['median_step_dist']:>11.2f} "
                  f"{sc['neighbor_transition_rate']:>8.1%}")
            all_mean.append(sc['mean_step_dist'])
            all_nbr.append(sc['neighbor_transition_rate'])
        print(f"  {'Mean':<8} {'':>6} {np.mean(all_mean):>10.2f} "
              f"{'':>10} {'':>11} {np.mean(all_nbr):>8.1%}")
        print()

    # ── 跨被试汇总 ──
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"跨被试汇总 ({len(all_results)} 人)")
        print(f"{'=' * 70}")

        # 起点距离
        participant_mean_start_dist = []
        for name, r in all_results:
            d = np.mean([sp['dist_to_center'] for sp in r['starting_points']])
            participant_mean_start_dist.append(d)
        print(f"\n  起点到中心距离:  mean={np.mean(participant_mean_start_dist):.2f}, "
              f"std={np.std(participant_mean_start_dist):.2f}")

        # 连贯性
        participant_mean_step_dist = []
        participant_mean_nbr_rate = []
        for name, r in all_results:
            md = np.mean([sc['mean_step_dist'] for sc in r['spatial_coherence']])
            nr = np.mean([sc['neighbor_transition_rate'] for sc in r['spatial_coherence']])
            participant_mean_step_dist.append(md)
            participant_mean_nbr_rate.append(nr)
        print(f"  步间质心距离:    mean={np.mean(participant_mean_step_dist):.2f}, "
              f"std={np.std(participant_mean_step_dist):.2f}")
        print(f"  邻居转移率:      mean={np.mean(participant_mean_nbr_rate):.1%}, "
              f"std={np.std(participant_mean_nbr_rate):.1%}")


if __name__ == '__main__':
    main()
