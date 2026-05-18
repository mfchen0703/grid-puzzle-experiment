"""
图结构分析：提取每一轮地图对应邻接图的结构特征。

输出内容分三类：
1. 基础图指标：节点数、边数、度分布、密度、直径、平均最短路等
2. 着色相关指标：团数下界、退化度上界、DSATUR/greedy 着色结果
3. 区域几何指标：区域面积、周长、紧致度、到中心距离

用法:
    python graph_analysis.py
    python graph_analysis.py --round 1
    python graph_analysis.py --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from statistics import mean, pstdev

from fit_softmax import ROWS, COLS, build_all_maps, generate_map, mulberry32

ROUND_SIZES = [20, 23, 26, 28, 31, 34, 37, 39, 42, 45]
ROUND_SEEDS = [42, 137, 256, 389, 512, 647, 783, 891, 1024, 1157]
PRACTICE_SIZES = [10, 10]
PRACTICE_SEEDS = [9999, 8888]


def sample_std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bfs_distances(start, adjacency):
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def connected_components(adjacency):
    unseen = set(adjacency)
    components = []
    while unseen:
        start = next(iter(unseen))
        reached = bfs_distances(start, adjacency)
        comp = set(reached)
        components.append(comp)
        unseen -= comp
    return components


def graph_distances(adjacency):
    all_dists = []
    eccentricities = {}
    for u in adjacency:
        dists = bfs_distances(u, adjacency)
        eccentricities[u] = max(dists.values()) if dists else 0
        for v, d in dists.items():
            if u < v:
                all_dists.append(d)
    diameter = max(eccentricities.values()) if eccentricities else 0
    radius = min(eccentricities.values()) if eccentricities else 0
    return all_dists, diameter, radius, eccentricities


def local_clustering(node, adjacency):
    neighbors = list(adjacency[node])
    k = len(neighbors)
    if k < 2:
        return 0.0
    links = 0
    for i in range(k):
        for j in range(i + 1, k):
            if neighbors[j] in adjacency[neighbors[i]]:
                links += 1
    return (2 * links) / (k * (k - 1))


def transitivity(adjacency):
    triangles_x3 = 0
    triples = 0
    for node in adjacency:
        neighbors = list(adjacency[node])
        k = len(neighbors)
        if k < 2:
            continue
        triples += k * (k - 1) / 2
        for i in range(k):
            for j in range(i + 1, k):
                if neighbors[j] in adjacency[neighbors[i]]:
                    triangles_x3 += 1
    if triples == 0:
        return 0.0
    return triangles_x3 / triples


def degeneracy_ordering(adjacency):
    remaining = {u: set(vs) for u, vs in adjacency.items()}
    order = []
    max_min_degree = 0
    while remaining:
        u = min(remaining, key=lambda x: len(remaining[x]))
        deg = len(remaining[u])
        max_min_degree = max(max_min_degree, deg)
        order.append(u)
        for v in list(remaining[u]):
            remaining[v].remove(u)
        del remaining[u]
    return order, max_min_degree


def greedy_coloring(adjacency, order):
    colors = {}
    for u in order:
        used = {colors[v] for v in adjacency[u] if v in colors}
        color = 0
        while color in used:
            color += 1
        colors[u] = color
    return colors


def dsatur_coloring(adjacency):
    colors = {}
    saturation = {u: set() for u in adjacency}
    degrees = {u: len(vs) for u, vs in adjacency.items()}

    while len(colors) < len(adjacency):
        candidates = [u for u in adjacency if u not in colors]
        u = max(candidates, key=lambda x: (len(saturation[x]), degrees[x], -x))
        used = {colors[v] for v in adjacency[u] if v in colors}
        color = 0
        while color in used:
            color += 1
        colors[u] = color
        for v in adjacency[u]:
            if v not in colors:
                saturation[v].add(color)
    return colors


def bron_kerbosch_max_clique(adjacency):
    best = []

    def expand(r, p, x):
        nonlocal best
        if not p and not x:
            if len(r) > len(best):
                best = list(r)
            return
        if len(r) + len(p) <= len(best):
            return
        pivot_candidates = p | x
        pivot = max(pivot_candidates, key=lambda u: len(adjacency[u])) if pivot_candidates else None
        pivot_neighbors = adjacency[pivot] if pivot is not None else set()
        for v in list(p - pivot_neighbors):
            expand(r | {v}, p & adjacency[v], x & adjacency[v])
            p.remove(v)
            x.add(v)

    expand(set(), set(adjacency), set())
    return best


def articulation_points_and_bridges(adjacency):
    time = 0
    disc = {}
    low = {}
    parent = {}
    articulation = set()
    bridges = []

    def dfs(u):
        nonlocal time
        time += 1
        disc[u] = low[u] = time
        child_count = 0

        for v in adjacency[u]:
            if v not in disc:
                parent[v] = u
                child_count += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and child_count > 1:
                    articulation.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    articulation.add(u)
                if low[v] > disc[u]:
                    bridges.append(tuple(sorted((u, v))))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for u in adjacency:
        if u not in disc:
            parent[u] = None
            dfs(u)

    bridges = sorted(set(bridges))
    return articulation, bridges


def shortest_cycle_length(adjacency):
    best = math.inf
    for start in adjacency:
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] * 2 + 1 >= best:
                continue
            for v in adjacency[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v:
                    best = min(best, dist[u] + dist[v] + 1)
    return None if best is math.inf else best


def region_perimeter(cells):
    cells_set = set(cells)
    perimeter = 0
    for r, c in cells:
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if (r + dr, c + dc) not in cells_set:
                perimeter += 1
    return perimeter


def centroid(cells):
    avg_r = sum(r for r, _ in cells) / len(cells)
    avg_c = sum(c for _, c in cells) / len(cells)
    return avg_r, avg_c


def centroid_dist_to_center(cells):
    cr, cc = centroid(cells)
    return math.hypot(cr - ROWS / 2, cc - COLS / 2)


def compactness(area, perimeter):
    if perimeter == 0:
        return 0.0
    return 4 * math.pi * area / (perimeter ** 2)


def exact_chromatic_number(adjacency):
    nodes = sorted(adjacency, key=lambda u: len(adjacency[u]), reverse=True)
    n = len(nodes)
    best = n
    colors = {}

    def backtrack(i, used_count):
        nonlocal best
        if used_count >= best:
            return
        if i == n:
            best = min(best, used_count)
            return

        u = nodes[i]
        forbidden = {colors[v] for v in adjacency[u] if v in colors}
        for color in range(used_count):
            if color not in forbidden:
                colors[u] = color
                backtrack(i + 1, used_count)
                del colors[u]

        colors[u] = used_count
        backtrack(i + 1, used_count + 1)
        del colors[u]

    backtrack(0, 0)
    return best


def rebuild_map(round_label):
    if round_label.startswith("P"):
        idx = int(round_label[1:]) - 1
        rng = mulberry32(PRACTICE_SEEDS[idx])
        return generate_map(PRACTICE_SIZES[idx], rng)
    idx = int(round_label) - 1
    rng = mulberry32(ROUND_SEEDS[idx])
    return generate_map(ROUND_SIZES[idx], rng)


def summarize_round(round_label, regions, adjacency):
    n = len(regions)
    m = sum(len(vs) for vs in adjacency.values()) // 2
    degrees = [len(adjacency[u]) for u in sorted(adjacency)]
    density = 0.0 if n < 2 else 2 * m / (n * (n - 1))

    components = connected_components(adjacency)
    connected = len(components) == 1
    dists, diameter, radius, eccentricities = graph_distances(adjacency)
    mean_path = mean(dists) if dists else 0.0

    clustering_values = [local_clustering(u, adjacency) for u in adjacency]
    articulation, bridges = articulation_points_and_bridges(adjacency)
    core_order, degeneracy = degeneracy_ordering(adjacency)
    reverse_deg_order = list(reversed(core_order))
    greedy_colors = greedy_coloring(adjacency, reverse_deg_order)
    dsatur_colors = dsatur_coloring(adjacency)
    clique = bron_kerbosch_max_clique(adjacency)
    girth = shortest_cycle_length(adjacency)
    chromatic = exact_chromatic_number(adjacency)

    areas = [len(cells) for cells in regions]
    perimeters = [region_perimeter(cells) for cells in regions]
    compactnesses = [compactness(a, p) for a, p in zip(areas, perimeters)]
    center_dists = [centroid_dist_to_center(cells) for cells in regions]

    return {
        "round": round_label,
        "nodes": n,
        "edges": m,
        "density": density,
        "is_connected": connected,
        "num_components": len(components),
        "diameter": diameter,
        "radius": radius,
        "avg_shortest_path": mean_path,
        "degree_min": min(degrees),
        "degree_max": max(degrees),
        "degree_mean": mean(degrees),
        "degree_std": sample_std(degrees),
        "degree_p25": percentile(degrees, 0.25),
        "degree_median": percentile(degrees, 0.50),
        "degree_p75": percentile(degrees, 0.75),
        "avg_local_clustering": mean(clustering_values),
        "transitivity": transitivity(adjacency),
        "articulation_points": len(articulation),
        "bridges": len(bridges),
        "degeneracy": degeneracy,
        "max_clique_size": len(clique),
        "girth": girth,
        "chromatic_number": chromatic,
        "greedy_color_count": max(greedy_colors.values()) + 1,
        "dsatur_color_count": max(dsatur_colors.values()) + 1,
        "eccentricity_mean": mean(eccentricities.values()),
        "area_mean": mean(areas),
        "area_std": sample_std(areas),
        "area_min": min(areas),
        "area_max": max(areas),
        "perimeter_mean": mean(perimeters),
        "perimeter_std": sample_std(perimeters),
        "compactness_mean": mean(compactnesses),
        "compactness_std": sample_std(compactnesses),
        "center_dist_mean": mean(center_dists),
        "center_dist_std": sample_std(center_dists),
    }


def format_table(rows):
    headers = [
        "round", "nodes", "edges", "degree_mean", "degree_max", "density",
        "diameter", "avg_shortest_path", "avg_local_clustering",
        "degeneracy", "max_clique_size", "chromatic_number", "dsatur_color_count",
    ]
    widths = {
        key: max(len(key), max(len(f"{row[key]:.3f}" if isinstance(row[key], float) else str(row[key])) for row in rows))
        for key in headers
    }
    lines = []
    lines.append("  ".join(key.ljust(widths[key]) for key in headers))
    lines.append("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        rendered = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                text = f"{value:.3f}"
            else:
                text = str(value)
            rendered.append(text.ljust(widths[key]))
        lines.append("  ".join(rendered))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", dest="round_label", help="分析指定 round，例如 1 或 P2")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    return parser.parse_args()


def main():
    args = parse_args()

    maps = build_all_maps()
    round_labels = [args.round_label] if args.round_label else list(maps.keys())

    rows = []
    for round_label in round_labels:
        if round_label not in maps:
            raise SystemExit(f"未知 round: {round_label}")
        _, regions, adjacency = rebuild_map(round_label)
        rows.append(summarize_round(round_label, regions, adjacency))

    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.format == "csv":
        writer = csv.DictWriter(
            __import__("sys").stdout,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
        return

    print(format_table(rows))


if __name__ == "__main__":
    main()
