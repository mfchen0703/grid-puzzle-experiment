"""Segment Experiment 2 human action logs into reset-delimited attempts.

The exported CSV starts with a section marker line ("[Actions]") followed by
the real header. Within each round, "Round Reset" starts a new attempt. This
script writes:

- action-level rows with attempt ids and parsed recolor actions
- attempt-level summaries for model fitting or descriptive analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


RECOLOR_RE = re.compile(r"Recolored Region (\d+) to Color (\d+)")
START_RE = re.compile(r"Game Started \((\d+) regions, (\d+) conflicts\)")


def conflict_region_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len(text.split())


def parse_action(action: object) -> dict:
    text = str(action)
    recolor_match = RECOLOR_RE.search(text)
    if recolor_match:
        region_ui = int(recolor_match.group(1))
        color_ui = int(recolor_match.group(2))
        return {
            "action_type": "recolor",
            "region_ui": region_ui,
            "color_ui": color_ui,
            "region_zero_based": region_ui - 1,
            "color_zero_based": color_ui - 1,
            "started_regions": pd.NA,
            "started_conflicts": pd.NA,
        }

    start_match = START_RE.search(text)
    if start_match:
        return {
            "action_type": "game_started",
            "region_ui": pd.NA,
            "color_ui": pd.NA,
            "region_zero_based": pd.NA,
            "color_zero_based": pd.NA,
            "started_regions": int(start_match.group(1)),
            "started_conflicts": int(start_match.group(2)),
        }

    if "Round Reset" in text:
        return {
            "action_type": "round_reset",
            "region_ui": pd.NA,
            "color_ui": pd.NA,
            "region_zero_based": pd.NA,
            "color_zero_based": pd.NA,
            "started_regions": pd.NA,
            "started_conflicts": pd.NA,
        }

    return {
        "action_type": "other",
        "region_ui": pd.NA,
        "color_ui": pd.NA,
        "region_zero_based": pd.NA,
        "color_zero_based": pd.NA,
        "started_regions": pd.NA,
        "started_conflicts": pd.NA,
    }


def read_actions_csv(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        first_line = handle.readline().strip()
    skiprows = 1 if first_line == "[Actions]" else 0
    df = pd.read_csv(path, skiprows=skiprows)
    required = {
        "SessionID",
        "Experiment",
        "Round",
        "Step",
        "Action",
        "ConflictRegions",
        "TimeTaken(s)",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def segment_actions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (_, round_id), group in df.groupby(["SessionID", "Round"], sort=True):
        del round_id
        group = group.sort_values("Step")
        attempt_id = 1
        recolor_step = 0
        total_step = 0

        for _, row in group.iterrows():
            parsed = parse_action(row["Action"])
            action_type = parsed["action_type"]
            if action_type == "round_reset":
                attempt_id += 1
                recolor_step = 0
                total_step = 0

            if action_type == "recolor":
                recolor_step += 1
            total_step += 1

            out = row.to_dict()
            out.update(parsed)
            out["attempt_id"] = attempt_id
            out["attempt_total_step"] = total_step
            out["attempt_recolor_step"] = recolor_step if action_type == "recolor" else pd.NA
            out["conflict_region_count"] = conflict_region_count(row["ConflictRegions"])
            rows.append(out)

    segmented = pd.DataFrame(rows)
    max_attempt = (
        segmented.groupby(["SessionID", "Round"])["attempt_id"]
        .max()
        .rename("max_attempt_id")
        .reset_index()
    )
    segmented = segmented.merge(max_attempt, on=["SessionID", "Round"], how="left")
    segmented["is_final_attempt"] = segmented["attempt_id"] == segmented["max_attempt_id"]
    return segmented


def summarize_attempts(segmented: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped: Iterable[tuple[tuple[object, object, object], pd.DataFrame]] = segmented.groupby(
        ["SessionID", "Experiment", "Round", "attempt_id"],
        sort=True,
    )
    for (session_id, experiment, round_id, attempt_id), group in grouped:
        group = group.sort_values("Step")
        recolors = group[group["action_type"] == "recolor"]
        start_row = group.iloc[0]
        end_row = group.iloc[-1]
        is_final = bool(group["is_final_attempt"].iloc[0])
        completed = is_final and conflict_region_count(end_row["ConflictRegions"]) == 0
        end_reason = "completed" if completed else ("incomplete" if is_final else "reset")
        rows.append(
            {
                "SessionID": session_id,
                "Experiment": experiment,
                "Round": round_id,
                "attempt_id": attempt_id,
                "is_final_attempt": int(is_final),
                "end_reason": end_reason,
                "completed": int(completed),
                "start_step": int(start_row["Step"]),
                "end_step": int(end_row["Step"]),
                "n_rows": int(len(group)),
                "n_recolors": int(len(recolors)),
                "start_action_type": start_row["action_type"],
                "end_action_type": end_row["action_type"],
                "start_conflict_regions": "" if pd.isna(start_row["ConflictRegions"]) else str(start_row["ConflictRegions"]),
                "end_conflict_regions": "" if pd.isna(end_row["ConflictRegions"]) else str(end_row["ConflictRegions"]),
                "start_conflict_region_count": conflict_region_count(start_row["ConflictRegions"]),
                "end_conflict_region_count": conflict_region_count(end_row["ConflictRegions"]),
                "elapsed_time_s_all_rows": float(group["TimeTaken(s)"].sum()),
                "elapsed_time_s_recolors_only": float(recolors["TimeTaken(s)"].sum()),
            }
        )
    return pd.DataFrame(rows)


def output_prefix_for(path: Path) -> str:
    stem = path.stem
    if stem.startswith("data_"):
        return stem
    return f"{stem}_segmented"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiment2/data/data_experiment2_5201314.csv"),
        help="Experiment 2 human action CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment2/data/processed"),
        help="Directory for segmented outputs.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input stem.",
    )
    args = parser.parse_args()

    df = read_actions_csv(args.input)
    segmented = segment_actions(df)
    summary = summarize_attempts(segmented)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or output_prefix_for(args.input)
    actions_path = args.output_dir / f"{prefix}_segmented_actions.csv"
    summary_path = args.output_dir / f"{prefix}_attempt_summary.csv"

    segmented.to_csv(actions_path, index=False)
    summary.to_csv(summary_path, index=False)

    final_attempts = summary[summary["is_final_attempt"] == 1]
    print(f"segmented_actions: {actions_path}")
    print(f"attempt_summary: {summary_path}")
    print(f"rounds: {summary['Round'].nunique()}")
    print(f"attempts: {len(summary)}")
    print(f"rounds_with_reset: {(summary.groupby(['SessionID', 'Round']).size() > 1).sum()}")
    print(f"total_recolors: {(segmented['action_type'] == 'recolor').sum()}")
    print(f"final_attempt_recolors: {int(final_attempts['n_recolors'].sum())}")
    print(f"abandoned_attempt_recolors: {int(summary[summary['is_final_attempt'] == 0]['n_recolors'].sum())}")


if __name__ == "__main__":
    main()
