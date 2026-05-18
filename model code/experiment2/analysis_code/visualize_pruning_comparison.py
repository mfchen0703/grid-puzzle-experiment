from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from conflict_search_agent import load_materials, trace_tree_agent_on_round
from visualize_conflict_search_agent import (
    build_border_segments,
    build_region_cells,
    configure_chinese_font,
    render_state,
)


CHANGED_ROUNDS = [1, 3, 6, 9]


def _action_steps(trace: list[dict]) -> list[dict]:
    return [step for step in trace if step.get("status") == "action"]


def _step_title(step: dict) -> str:
    return (
        f"Step {step['agent_step']} | R{step['region'] + 1}: "
        f"C{step['old_color'] + 1} -> C{step['new_color'] + 1}\n"
        f"conflicts {len(step['conflict_edges_before'])} -> {len(step['conflict_edges_after'])}"
    )


def visualize_pruning_comparison(
    output_dir: str | Path | None = None,
    rounds: list[int] | None = None,
    max_depth: int = 4,
    n_iterations: int = 20,
) -> Path:
    plt.style.use("default")
    configure_chinese_font()

    materials = load_materials()
    target_rounds = rounds or CHANGED_ROUNDS
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "results" / "pruning_thresh_comparison_visualizations"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for round_index in target_rounds:
        round_data = materials["rounds"][round_index - 1]
        grid = round_data["mapData"]["grid"]
        region_cells = build_region_cells(grid)
        border_segments = build_border_segments(grid)

        trace_small = _action_steps(
            trace_tree_agent_on_round(
                round_data,
                max_depth=max_depth,
                n_iterations=n_iterations,
                pruning_thresh=0,
            )
        )
        trace_large = _action_steps(
            trace_tree_agent_on_round(
                round_data,
                max_depth=max_depth,
                n_iterations=n_iterations,
                pruning_thresh=1,
            )
        )

        max_rows = max(len(trace_small), len(trace_large))
        fig, axes = plt.subplots(max_rows, 2, figsize=(12, max_rows * 3.4))
        if max_rows == 1:
            axes = [axes]

        for row in range(max_rows):
            for col, (label, trace) in enumerate([("prune=0", trace_small), ("prune=1", trace_large)]):
                ax = axes[row][col]
                if row >= len(trace):
                    ax.axis("off")
                    continue
                step = trace[row]
                render_state(
                    ax,
                    grid,
                    step["colors_before"],
                    border_segments,
                    region_cells,
                    step["conflict_edges_before"],
                    selected_region=step["region"],
                )
                ax.set_title(f"{label} | {_step_title(step)}", fontsize=11, pad=8)

        fig.suptitle(
            f"Round {round_index}: pruning_thresh 对动作序列的影响",
            fontsize=18,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(output_root / f"round_{round_index:02d}_pruning_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    return output_root


if __name__ == "__main__":
    out = visualize_pruning_comparison()
    print(out)
