"""Export observed tree actions to compact C++ IBS task input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_state(text: str) -> list[int]:
    return [int(part) for part in str(text).split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds-json",
        type=Path,
        default=Path("experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json"),
    )
    parser.add_argument("--observed-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.rounds_json.read_text(encoding="utf-8-sig"))
    rounds = payload["rounds"] if isinstance(payload, dict) and "rounds" in payload else payload
    observed = pd.read_csv(args.observed_actions)
    observed = observed.sort_values(["round", "agent_step"]).reset_index(drop=True)
    round_ids = sorted({int(round_id) for round_id in observed["round"].tolist()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(f"ROUNDS {len(round_ids)}\n")
        for round_id in round_ids:
            round_data = rounds[round_id - 1]
            n_regions = int(round_data["mapData"]["numRegions"])
            edges = [
                (int(u), int(v))
                for u, v in round_data["mapData"]["adjacencyPairs"]
            ]
            f.write(f"ROUND {round_id} {n_regions} {len(edges)}\n")
            for u, v in edges:
                f.write(f"EDGE {u} {v}\n")

        f.write(f"OBS {len(observed)}\n")
        for obs_index, obs in observed.iterrows():
            state = parse_state(str(obs["state_before"]))
            f.write(
                "OBSROW "
                f"{int(obs_index)} "
                f"{int(obs['round'])} "
                f"{len(state)} "
                f"{' '.join(str(x) for x in state)} "
                f"{int(obs['region'])} "
                f"{int(obs['new_color'])}\n"
            )
    print(args.output)


if __name__ == "__main__":
    main()
