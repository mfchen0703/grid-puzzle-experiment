"""Add two 20-region practice rounds to experiment 2 web materials."""

from __future__ import annotations

import json
from pathlib import Path

from generate_rounds_json import (
    COLORS,
    generate_map_data,
    get_conflict_edges,
    mulberry32,
    solve_coloring,
)


PRACTICE_NUM_REGIONS = 20
PRACTICE_SEEDS = [13001, 13002]


def make_practice_round(seed: int, practice_round: int) -> dict:
    random = mulberry32(seed)
    map_data, adjacency = generate_map_data(PRACTICE_NUM_REGIONS, random)
    solved_colors = solve_coloring(adjacency, random)
    initial_colors = list(solved_colors)

    edges = list(map(tuple, map_data["adjacencyPairs"]))
    if not edges:
        raise RuntimeError(f"Practice map seed {seed} has no adjacency edges.")

    # Create an easy, local conflict that can always be repaired by restoring
    # the changed region to its solved color.
    changed_region, neighbor = edges[int(random() * len(edges))]
    initial_colors[changed_region] = solved_colors[neighbor]
    conflict_edges = get_conflict_edges(adjacency, initial_colors)
    if not conflict_edges:
        raise RuntimeError(f"Practice map seed {seed} did not create a conflict.")

    return {
        "seed": seed,
        "conditionType": "practice",
        "generationMode": "practice_20_region",
        "mapData": map_data,
        "initialColors": initial_colors,
        "solvedColors": solved_colors,
        "conflictEdges": conflict_edges,
        "metadata": {
            "practiceRound": practice_round,
            "numRegions": PRACTICE_NUM_REGIONS,
            "changedRegion": changed_region,
            "repairColor": solved_colors[changed_region],
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    rounds_path = root / "experiment1" / "public" / "experiment2" / "rounds.json"
    payload = json.loads(rounds_path.read_text(encoding="utf-8-sig"))
    payload["practiceRounds"] = [
        make_practice_round(seed, index)
        for index, seed in enumerate(PRACTICE_SEEDS, start=1)
    ]
    rounds_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['practiceRounds'])} practice rounds to {rounds_path}")


if __name__ == "__main__":
    main()
