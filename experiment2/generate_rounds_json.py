"""
Generate static experiment 2 round materials for the web app.

Output:
    experiment1/public/experiment2/rounds.json
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

ROWS = 12
COLS = 20
NUM_REGIONS = 45
COLORS = ["#377eb8", "#4daf4a", "#984ea3", "#ffff33"]
ROUND_SEEDS = [2021, 2037, 2053, 2069, 2081, 2099, 2111, 2137, 2153, 2179]


def mulberry32(seed: int):
    s = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = (s ^ (s >> 15)) * (1 | s)
        t &= 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        t ^= t >> 14
        return (t & 0xFFFFFFFF) / 4294967296

    return rand


def generate_map_data(num_regions: int, random):
    grid = [[-1] * COLS for _ in range(ROWS)]
    regions = []

    for i in range(num_regions):
        while True:
            r = int(random() * ROWS)
            c = int(random() * COLS)
            if grid[r][c] == -1:
                break
        grid[r][c] = i
        regions.append([(r, c)])

    changed = True
    while changed:
        changed = False
        for i in range(num_regions):
            neighbors = []
            for r, c in regions[i]:
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == -1:
                        neighbors.append((nr, nc))
            if neighbors:
                nr, nc = neighbors[int(random() * len(neighbors))]
                if grid[nr][nc] == -1:
                    grid[nr][nc] = i
                    regions[i].append((nr, nc))
                    changed = True

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] != -1:
                continue
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != -1:
                    grid[r][c] = grid[nr][nc]
                    regions[grid[nr][nc]].append((r, c))
                    break

    adjacency = {i: set() for i in range(num_regions)}
    for r in range(ROWS):
        for c in range(COLS):
            region1 = grid[r][c]
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr < ROWS and nc < COLS:
                    region2 = grid[nr][nc]
                    if region1 != region2:
                        adjacency[region1].add(region2)
                        adjacency[region2].add(region1)

    adjacency_pairs = []
    for region in range(num_regions):
        for neighbor in adjacency[region]:
            if region < neighbor:
                adjacency_pairs.append([region, neighbor])

    return {
        "grid": grid,
        "numRegions": num_regions,
        "adjacencyPairs": sorted(adjacency_pairs),
    }, adjacency


def shuffle(items, random):
    copy = list(items)
    for i in range(len(copy) - 1, 0, -1):
        j = int(random() * (i + 1))
        copy[i], copy[j] = copy[j], copy[i]
    return copy


def is_legal_color(region_id, color, adjacency, colors):
    return all(colors[nb] != color for nb in adjacency[region_id])


def solve_coloring(adjacency, random):
    num_regions = len(adjacency)
    order = sorted(range(num_regions), key=lambda idx: len(adjacency[idx]), reverse=True)
    colors = [-1] * num_regions

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True
        region_id = order[index]
        for color in shuffle(range(len(COLORS)), random):
            if not is_legal_color(region_id, color, adjacency, colors):
                continue
            colors[region_id] = color
            if backtrack(index + 1):
                return True
            colors[region_id] = -1
        return False

    if not backtrack(0):
        raise RuntimeError("Failed to construct a legal 4-coloring.")
    return colors


def get_conflict_edges(adjacency, colors):
    conflicts = []
    for region in range(len(colors)):
        for neighbor in adjacency[region]:
            if region < neighbor and colors[region] == colors[neighbor]:
                conflicts.append([region, neighbor])
    return conflicts


def get_conflict_regions(conflict_edges):
    regions = set()
    for a, b in conflict_edges:
        regions.add(a)
        regions.add(b)
    return regions


def is_legal_state(adjacency, colors):
    return len(get_conflict_edges(adjacency, colors)) == 0


def exists_solution_only_changing_allowed(adjacency, initial_colors, allowed_regions):
    colors = list(initial_colors)
    allowed = sorted(allowed_regions, key=lambda idx: len(adjacency[idx]), reverse=True)
    allowed_set = set(allowed)

    def backtrack(index):
        if index == len(allowed):
            return is_legal_state(adjacency, colors)

        region = allowed[index]
        current = colors[region]

        # Try all colors, including keeping the original color.
        for color in range(len(COLORS)):
            consistent = True
            for neighbor in adjacency[region]:
                neighbor_color = colors[neighbor]
                if neighbor not in allowed_set:
                    if neighbor_color == color:
                        consistent = False
                        break
                elif neighbor_color == color and neighbor in allowed[:index]:
                    consistent = False
                    break
            if not consistent:
                continue

            colors[region] = color
            if backtrack(index + 1):
                return True

        colors[region] = current
        return False

    return backtrack(0)


def get_relevant_search_regions(adjacency, conflict_edges, changed_regions):
    relevant = set(changed_regions)
    for a, b in conflict_edges:
        relevant.add(a)
        relevant.add(b)
        relevant.update(adjacency[a])
        relevant.update(adjacency[b])
    return sorted(relevant)


def has_solution_within_k_changes(adjacency, initial_colors, max_changes, region_ids):

    for n_changes in range(1, max_changes + 1):
        for changed_regions in itertools.combinations(region_ids, n_changes):
            alternative_colors = [
                [c for c in range(len(COLORS)) if c != initial_colors[region]]
                for region in changed_regions
            ]
            for new_colors in itertools.product(*alternative_colors):
                candidate = list(initial_colors)
                for region, color in zip(changed_regions, new_colors):
                    candidate[region] = color
                if is_legal_state(adjacency, candidate):
                    return True
    return False


def build_conflict_start_state(adjacency, solved_colors, random):
    region_ids = list(range(len(solved_colors)))

    for _ in range(8000):
        candidate = list(solved_colors)
        change_count = 4 + int(random() * 3)
        chosen_regions = shuffle(region_ids, random)[:change_count]

        for region in chosen_regions:
            neighbor_colors = [candidate[neighbor] for neighbor in adjacency[region]]
            preferred = shuffle(sorted({c for c in neighbor_colors if c >= 0 and c != candidate[region]}), random)
            fallback = shuffle([c for c in range(len(COLORS)) if c != candidate[region]], random)
            candidate[region] = preferred[0] if preferred else fallback[0]

        conflict_edges = get_conflict_edges(adjacency, candidate)
        if len(conflict_edges) < 3:
            continue
        changed_regions = {idx for idx, (a, b) in enumerate(zip(candidate, solved_colors)) if a != b}
        if len(changed_regions) < 4:
            continue
        conflict_regions = get_conflict_regions(conflict_edges)
        if exists_solution_only_changing_allowed(adjacency, candidate, conflict_regions):
            continue
        if changed_regions.issubset(conflict_regions):
            continue
        relevant_regions = get_relevant_search_regions(adjacency, conflict_edges, changed_regions)
        if has_solution_within_k_changes(
            adjacency,
            candidate,
            max_changes=3,
            region_ids=relevant_regions,
        ):
            continue
        return candidate, conflict_edges

    raise RuntimeError("Failed to build a planning-heavy initial state with min distance > 3.")


def build_rounds():
    rounds = []
    for seed in ROUND_SEEDS:
        random = mulberry32(seed)
        map_data, adjacency = generate_map_data(NUM_REGIONS, random)
        solved_colors = solve_coloring(adjacency, random)
        initial_colors, conflict_edges = build_conflict_start_state(adjacency, solved_colors, random)
        rounds.append(
            {
                "mapData": map_data,
                "initialColors": initial_colors,
                "solvedColors": solved_colors,
                "conflictEdges": conflict_edges,
            }
        )
    return rounds


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "experiment1" / "public" / "experiment2" / "rounds.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "rows": ROWS,
        "cols": COLS,
        "numRegions": NUM_REGIONS,
        "colors": COLORS,
        "rounds": build_rounds(),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
