from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "experiment2" / "generated_map_search_planning1" / "agent_validation" / "failed_maps_value_backup_traces.json"
DIAG_PATH = ROOT / "experiment2" / "generated_map_search_planning1" / "agent_validation" / "failed_maps_value_backup_diagnostics.csv"
OUT_DIR = ROOT / "experiment2" / "generated_map_search_planning1" / "agent_validation" / "feature_comparison"

FEATURE_KEYS = [
    "repair",
    "opportunity",
    "spatial",
    "neighbor",
    "color",
    "region_preserve",
    "color_preserve",
    "direct_conflict_reduction",
    "environment_region",
    "delta_legal_conflict_colors_norm",
    "delta_same_color_blockers_removed_norm",
    "local_nonneighbor_color_match_norm",
]


def load_diagnostics(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["round"])] = row
    return out


def numeric(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def collect_rows(traces: dict, diagnostics: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for round_id, meta in diagnostics.items():
        if not meta.get("optimal_first_action_region"):
            continue
        correct_action = (
            int(float(meta["optimal_first_action_region"])),
            int(float(meta["optimal_first_action_color"])),
        )
        for step in traces.get(str(round_id), []):
            if step.get("status") != "action":
                continue
            candidates = step.get("candidate_actions", [])
            selected_action = (int(step["region"]), int(step["new_color"]))
            selected_candidate = None
            correct_candidate = None
            for cand in candidates:
                action = (int(cand["region"]), int(cand["new_color"]))
                if action == selected_action:
                    selected_candidate = cand
                if action == correct_action:
                    correct_candidate = cand
            if selected_candidate is None or correct_candidate is None:
                continue

            selected_features = selected_candidate.get("heuristic_features", {})
            correct_features = correct_candidate.get("heuristic_features", {})
            row = {
                "round": round_id,
                "agent_step": int(step["agent_step"]),
                "selected_region": selected_action[0],
                "selected_new_color": selected_action[1],
                "correct_region": correct_action[0],
                "correct_new_color": correct_action[1],
                "selected_is_correct": selected_action == correct_action,
                "selected_heuristic_score": numeric(selected_candidate.get("heuristic_score")),
                "correct_heuristic_score": numeric(correct_candidate.get("heuristic_score")),
                "heuristic_gap_selected_minus_correct": numeric(selected_candidate.get("heuristic_score"))
                - numeric(correct_candidate.get("heuristic_score")),
                "selected_score_component": numeric((selected_candidate.get("score") or [0.0, 0.0])[1]),
                "correct_score_component": numeric((correct_candidate.get("score") or [0.0, 0.0])[1]),
                "score_gap_selected_minus_correct": numeric((selected_candidate.get("score") or [0.0, 0.0])[1])
                - numeric((correct_candidate.get("score") or [0.0, 0.0])[1]),
            }
            for feature in FEATURE_KEYS:
                selected_value = numeric(selected_features.get(feature))
                correct_value = numeric(correct_features.get(feature))
                row[f"selected_{feature}"] = selected_value
                row[f"correct_{feature}"] = correct_value
                row[f"diff_{feature}"] = selected_value - correct_value
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {"all_visible_steps": rows}
    for row in rows:
        grouped.setdefault(f"round_{row['round']}", []).append(row)

    summary_rows: list[dict] = []
    for label, items in grouped.items():
        if not items:
            continue
        summary = {
            "group": label,
            "n_steps": len(items),
            "selected_is_correct_rate": sum(1 for item in items if item["selected_is_correct"]) / len(items),
            "mean_heuristic_gap_selected_minus_correct": sum(
                item["heuristic_gap_selected_minus_correct"] for item in items
            )
            / len(items),
            "mean_score_gap_selected_minus_correct": sum(
                item["score_gap_selected_minus_correct"] for item in items
            )
            / len(items),
        }
        for feature in FEATURE_KEYS:
            summary[f"mean_diff_{feature}"] = sum(item[f"diff_{feature}"] for item in items) / len(items)
        summary_rows.append(summary)
    return summary_rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_diffs(rows: list[dict], title: str, out_path: Path) -> None:
    if not rows:
        return
    means = []
    for feature in FEATURE_KEYS:
        means.append(sum(row[f"diff_{feature}"] for row in rows) / len(rows))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = list(range(len(FEATURE_KEYS)))
    ax.barh(y_pos, means, color=["#c44e52" if v > 0 else "#4c72b0" for v in means])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEATURE_KEYS)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("selected - correct")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_gaps(rows: list[dict], title: str, out_path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: (row["round"], row["agent_step"]))
    x = list(range(len(rows)))
    score_gaps = [row["score_gap_selected_minus_correct"] for row in rows]
    heur_gaps = [row["heuristic_gap_selected_minus_correct"] for row in rows]
    labels = [f"r{row['round']}-s{row['agent_step']}" for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, score_gaps, marker="o", label="score gap")
    ax.plot(x, heur_gaps, marker="s", label="heuristic gap")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("selected - correct")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    diagnostics = load_diagnostics(DIAG_PATH)

    rows = collect_rows(traces, diagnostics)
    summary_rows = summarize(rows)

    save_csv(OUT_DIR / "correct_vs_selected_feature_rows.csv", rows)
    save_csv(OUT_DIR / "correct_vs_selected_feature_summary.csv", summary_rows)

    plot_feature_diffs(
        rows,
        "Feature Difference: Selected minus Correct (All Visible Steps)",
        OUT_DIR / "all_visible_steps_feature_diff.png",
    )
    plot_score_gaps(
        rows,
        "Score And Heuristic Gaps: Selected minus Correct",
        OUT_DIR / "all_visible_steps_score_gap.png",
    )

    for round_id in sorted({row["round"] for row in rows}):
        round_rows = [row for row in rows if row["round"] == round_id]
        plot_feature_diffs(
            round_rows,
            f"Round {round_id}: Feature Difference (Selected minus Correct)",
            OUT_DIR / f"round_{round_id}_feature_diff.png",
        )
        plot_score_gaps(
            round_rows,
            f"Round {round_id}: Score And Heuristic Gaps",
            OUT_DIR / f"round_{round_id}_score_gap.png",
        )

    manifest = {
        "n_visible_steps": len(rows),
        "rounds_covered": sorted({row["round"] for row in rows}),
        "feature_keys": FEATURE_KEYS,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
