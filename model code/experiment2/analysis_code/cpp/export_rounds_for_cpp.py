"""Export Experiment 2 rounds JSON to compact C++ tool input.

The C++ recovery helpers intentionally avoid a JSON dependency. This script
converts the existing rounds payload into the compact text format consumed by
`tree_simulate`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round-limit", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    rounds = payload["rounds"] if isinstance(payload, dict) and "rounds" in payload else payload
    if args.round_limit is not None:
        rounds = rounds[: int(args.round_limit)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(f"ROUNDS {len(rounds)}\n")
        for round_index, round_data in enumerate(rounds, start=1):
            n_regions = int(round_data["mapData"]["numRegions"])
            edges = [
                (int(u), int(v))
                for u, v in round_data["mapData"]["adjacencyPairs"]
            ]
            f.write(f"ROUND {round_index} {n_regions} {len(edges)}\n")
            f.write(
                "INIT "
                + " ".join(str(int(color)) for color in round_data["initialColors"])
                + "\n"
            )
            for u, v in edges:
                f.write(f"EDGE {u} {v}\n")
    print(args.output)


if __name__ == "__main__":
    main()
