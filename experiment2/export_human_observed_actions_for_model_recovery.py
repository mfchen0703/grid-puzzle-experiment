"""Export segmented human Experiment 2 actions for C++ model recovery.

The C++ recovery tool expects one row per observed action with the map state
immediately before the action. Human logs only store the action and conflicts,
so this script reconstructs states from the experiment rounds and segmented
actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def state_text(state: list[int]) -> str:
    return " ".join(str(int(value)) for value in state)


def load_round_initial_states(path: Path, round_limit: int | None = None) -> dict[int, list[int]]:
    with path.open() as handle:
        data = json.load(handle)
    rounds = data["rounds"]
    if round_limit is not None and round_limit > 0:
        rounds = rounds[:round_limit]
    return {
        round_index: [int(value) for value in round_data["initialColors"]]
        for round_index, round_data in enumerate(rounds, start=1)
    }


def participant_from_segmented_path(path: Path) -> str:
    name = path.name
    prefix = "data_experiment2_"
    suffix = "_segmented_actions.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def export_one(
    segmented_path: Path,
    initial_states: dict[int, list[int]],
    output_dir: Path,
    attempts: str,
) -> Path:
    participant = participant_from_segmented_path(segmented_path)
    df = pd.read_csv(segmented_path)
    if attempts == "final":
        df = df[df["is_final_attempt"].astype(str).str.lower().isin(["true", "1"])]
    elif attempts == "abandoned":
        df = df[~df["is_final_attempt"].astype(str).str.lower().isin(["true", "1"])]
    elif attempts != "all":
        raise ValueError(f"Unknown attempts mode: {attempts}")

    columns = [
        "agent",
        "participant",
        "round",
        "attempt_id",
        "agent_step",
        "attempt_recolor_step",
        "state_before",
        "region",
        "old_color",
        "new_color",
        "source_step",
        "is_final_attempt",
    ]
    rows: list[dict] = []
    agent_step = 0
    for (round_id, attempt_id), group in df.groupby(["Round", "attempt_id"], sort=True):
        round_id = int(round_id)
        if round_id not in initial_states:
            continue
        state = list(initial_states[round_id])
        group = group.sort_values("Step")
        for _, row in group.iterrows():
            action_type = str(row["action_type"])
            if action_type != "recolor":
                continue
            region = int(row["region_zero_based"])
            new_color = int(row["color_zero_based"])
            old_color = int(state[region])
            rows.append(
                {
                    "agent": "human",
                    "participant": participant,
                    "round": round_id,
                    "attempt_id": int(attempt_id),
                    "agent_step": agent_step,
                    "attempt_recolor_step": int(row["attempt_recolor_step"]),
                    "state_before": state_text(state),
                    "region": region,
                    "old_color": old_color,
                    "new_color": new_color,
                    "source_step": int(row["Step"]),
                    "is_final_attempt": bool(row["is_final_attempt"]),
                }
            )
            state[region] = new_color
            agent_step += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{participant}_{attempts}_observed_actions.csv"
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmented-dir",
        type=Path,
        default=Path("experiment2/data/processed"),
        help="Directory containing *_segmented_actions.csv files.",
    )
    parser.add_argument(
        "--rounds-json",
        type=Path,
        default=Path("experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json"),
        help="Experiment 2 rounds JSON with initialColors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment2/data/processed/model_recovery_inputs"),
    )
    parser.add_argument(
        "--attempts",
        choices=["final", "abandoned", "all"],
        default="final",
        help="Which attempts to export.",
    )
    parser.add_argument("--round-limit", type=int, default=30)
    args = parser.parse_args()

    initial_states = load_round_initial_states(args.rounds_json, args.round_limit)
    paths = sorted(args.segmented_dir.glob("data_experiment2_*_segmented_actions.csv"))
    if not paths:
        raise FileNotFoundError(f"No segmented action files found in {args.segmented_dir}")

    for path in paths:
        out = export_one(path, initial_states, args.output_dir, args.attempts)
        n_rows = len(pd.read_csv(out)) if out.exists() else 0
        print(f"{out} rows={n_rows}")


if __name__ == "__main__":
    main()
