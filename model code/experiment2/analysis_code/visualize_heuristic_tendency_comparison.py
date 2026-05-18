from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from conflict_search_agent import load_materials, trace_tree_agent_on_round
from visualize_conflict_search_agent import (
    build_border_segments,
    build_region_cells,
    configure_chinese_font,
    render_state,
)


CONFIGS = [
    ("默认", None),
    ("区域保持高", {"region_preserve": 2.0}),
    ("颜色保持高", {"color_preserve": 0.02}),
]


def _action_steps(trace: list[dict]) -> list[dict]:
    return [step for step in trace if step.get("status") == "action"]


def _final_status(trace: list[dict]) -> str:
    if not trace:
        return "empty"
    last = trace[-1]
    return str(last.get("status", "action"))


def _step_title(step: dict) -> str:
    return (
        f"Step {step['agent_step']} | R{step['region'] + 1}: "
        f"C{step['old_color'] + 1} -> C{step['new_color'] + 1}\n"
        f"conflicts {len(step['conflict_edges_before'])} -> {len(step['conflict_edges_after'])}"
    )


def visualize_heuristic_tendency_comparison(
    output_dir: str | Path | None = None,
    rounds: list[int] | None = None,
    max_steps: int = 20,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 0.0,
) -> Path:
    plt.style.use("default")
    configure_chinese_font()

    materials = load_materials()
    target_rounds = rounds or list(range(1, len(materials["rounds"]) + 1))
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "results" / "heuristic_tendency_comparison_visualizations"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for round_index in target_rounds:
        round_data = materials["rounds"][round_index - 1]
        grid = round_data["mapData"]["grid"]
        region_cells = build_region_cells(grid)
        border_segments = build_border_segments(grid)

        traces = []
        for label, weights in CONFIGS:
            trace = trace_tree_agent_on_round(
                round_data,
                max_steps=max_steps,
                max_depth=max_depth,
                n_iterations=n_iterations,
                pruning_thresh=pruning_thresh,
                heuristic_weights=weights,
            )
            actions = _action_steps(trace)
            traces.append((label, weights, trace, actions))
            final_conflicts = (
                len(actions[-1]["conflict_edges_after"])
                if actions
                else len(round_data["mapData"].get("adjacencyPairs", []))
            )
            if trace and trace[-1].get("status") == "solved":
                final_conflicts = 0
            summary_rows.append(
                {
                    "round": round_index,
                    "condition": label,
                    "heuristic_weights": weights,
                    "rendered_action_steps": len(actions),
                    "trace_final_status": _final_status(trace),
                    "final_conflict_edges_in_trace": final_conflicts,
                    "max_steps": max_steps,
                    "max_depth": max_depth,
                    "n_iterations": n_iterations,
                    "pruning_thresh": pruning_thresh,
                }
            )

        max_rows = max(len(actions) for _, _, _, actions in traces)
        max_rows = max(max_rows, 1)
        fig, axes = plt.subplots(max_rows, len(traces), figsize=(5.4 * len(traces), 3.2 * max_rows))
        if max_rows == 1:
            axes = [axes]

        for col, (label, _, trace, actions) in enumerate(traces):
            for row in range(max_rows):
                ax = axes[row][col]
                if row >= len(actions):
                    ax.axis("off")
                    if row == 0 and not actions:
                        ax.set_title(f"{label}\n无动作", fontsize=12)
                    continue
                step = actions[row]
                render_state(
                    ax,
                    grid,
                    step["colors_before"],
                    border_segments,
                    region_cells,
                    step["conflict_edges_before"],
                    selected_region=step["region"],
                )
                ax.set_title(f"{label} | {_step_title(step)}", fontsize=10, pad=7)

            status = _final_status(trace)
            axes[0][col].text(
                0.5,
                1.18,
                f"status: {status}",
                transform=axes[0][col].transAxes,
                ha="center",
                va="bottom",
                fontsize=11,
            )

        fig.suptitle(
            f"Round {round_index}: heuristic 倾向对动作轨迹的影响",
            fontsize=18,
            y=0.997,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(output_root / f"round_{round_index:02d}_heuristic_tendency_comparison.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        output_root / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return output_root


if __name__ == "__main__":
    out = visualize_heuristic_tendency_comparison()
    print(out)
