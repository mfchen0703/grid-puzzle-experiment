"""
Generate experiment2_rounds_info.txt in the style of experiment1_rounds_info.txt,
with additional color information for each region.

Source:
    experiment1/public/experiment2/rounds.json
Output:
    experiment2_rounds_info.txt
"""

from __future__ import annotations

import json
from pathlib import Path


def build_regions_from_grid(grid: list[list[int]], num_regions: int) -> list[list[tuple[int, int]]]:
    regions: list[list[tuple[int, int]]] = [[] for _ in range(num_regions)]
    for r, row in enumerate(grid):
        for c, region_id in enumerate(row):
            regions[region_id].append((r, c))
    return regions


def build_adjacency_from_pairs(num_regions: int, pairs: list[list[int]]) -> list[list[int]]:
    adjacency = [[] for _ in range(num_regions)]
    for a, b in pairs:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "experiment1" / "public" / "experiment2" / "rounds.json"
    out = root / "experiment2_rounds_info.txt"

    payload = json.loads(src.read_text(encoding="utf-8"))
    rounds = payload["rounds"]

    lines: list[str] = []
    for round_idx, round_data in enumerate(rounds):
        map_data = round_data["mapData"]
        num_regions = int(map_data["numRegions"])
        grid = map_data["grid"]
        adjacency_pairs = map_data["adjacencyPairs"]
        initial_colors = round_data["initialColors"]
        solved_colors = round_data["solvedColors"]
        conflict_edges = round_data["conflictEdges"]

        regions = build_regions_from_grid(grid, num_regions)
        adjacency = build_adjacency_from_pairs(num_regions, adjacency_pairs)

        lines.append(f"===== Round {round_idx} ({num_regions} regions) =====")
        lines.append("")
        lines.append("--- Regions (id: cells) ---")
        for region_id, cells in enumerate(regions):
            cell_str = ", ".join(f"({r},{c})" for r, c in cells)
            lines.append(f"  Region {region_id:2d}: [{cell_str}]")
        lines.append("")

        lines.append("--- Adjacency ---")
        for region_id, neighbors in enumerate(adjacency):
            neighbor_str = ", ".join(str(n) for n in neighbors)
            lines.append(f"  Region {region_id:2d}: [{neighbor_str}]")
        lines.append("")

        lines.append("--- Colors (initial / solved) ---")
        for region_id, (initial_color, solved_color) in enumerate(zip(initial_colors, solved_colors)):
            lines.append(
                f"  Region {region_id:2d}: initial=Color {initial_color + 1}, solved=Color {solved_color + 1}"
            )
        lines.append("")

        lines.append("--- Conflict edges in initial state ---")
        for a, b in conflict_edges:
            lines.append(f"  ({a}, {b})")
        lines.append("")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
