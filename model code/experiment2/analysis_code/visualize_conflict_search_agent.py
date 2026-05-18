from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle

from conflict_search_agent import load_materials, trace_agent_on_round


BASE_COLORS = ["#ffffff", "#377eb8", "#4daf4a", "#984ea3", "#ffff33"]
CONFLICT_EDGE_COLOR = "#cc2f2f"
SELECTED_REGION_COLOR = "#111111"


def configure_chinese_font() -> str:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Source Han Sans SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Arial Unicode MS",
    ]
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False
    return chosen


def build_region_cells(grid: list[list[int]]) -> dict[int, list[tuple[int, int]]]:
    region_cells: dict[int, list[tuple[int, int]]] = {}
    for r, row in enumerate(grid):
        for c, region in enumerate(row):
            region_cells.setdefault(int(region), []).append((r, c))
    return region_cells


def build_border_segments(grid: list[list[int]]) -> tuple[list, list]:
    rows = len(grid)
    cols = len(grid[0])
    horizontal = []
    vertical = []
    for r in range(rows):
        for c in range(cols):
            region = grid[r][c]
            if r == 0 or grid[r - 1][c] != region:
                horizontal.append(((c, c + 1), (r, r)))
            if r == rows - 1 or grid[r + 1][c] != region:
                horizontal.append(((c, c + 1), (r + 1, r + 1)))
            if c == 0 or grid[r][c - 1] != region:
                vertical.append(((c, c), (r, r + 1)))
            if c == cols - 1 or grid[r][c + 1] != region:
                vertical.append(((c + 1, c + 1), (r, r + 1)))
    return horizontal, vertical


def render_state(ax, grid, colors, border_segments, region_cells, conflict_edges, selected_region=None):
    rows = len(grid)
    cols = len(grid[0])
    color_grid = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            region = grid[r][c]
            color_grid[r, c] = int(colors[region]) + 1

    ax.imshow(color_grid, cmap=ListedColormap(BASE_COLORS), vmin=0, vmax=len(BASE_COLORS) - 1, interpolation="nearest")
    h_segments, v_segments = border_segments
    for (xs, xe), (ys, ye) in h_segments:
        ax.plot([xs - 0.5, xe - 0.5], [ys - 0.5, ye - 0.5], color="black", linewidth=0.6, zorder=2)
    for (xs, xe), (ys, ye) in v_segments:
        ax.plot([xs - 0.5, xe - 0.5], [ys - 0.5, ye - 0.5], color="black", linewidth=0.6, zorder=2)

    conflict_regions = {x for edge in conflict_edges for x in edge}
    for region in sorted(conflict_regions):
        for r, c in region_cells[region]:
            for dr, dc, x0, x1, y0, y1 in [
                (-1, 0, c - 0.5, c + 0.5, r - 0.5, r - 0.5),
                (1, 0, c - 0.5, c + 0.5, r + 0.5, r + 0.5),
                (0, -1, c - 0.5, c - 0.5, r - 0.5, r + 0.5),
                (0, 1, c + 0.5, c + 0.5, r - 0.5, r + 0.5),
            ]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] != region:
                    ax.plot([x0, x1], [y0, y1], color=CONFLICT_EDGE_COLOR, linewidth=1.8, zorder=3)

    if selected_region is not None:
        for r, c in region_cells[selected_region]:
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=SELECTED_REGION_COLOR,
                    linewidth=1.6,
                    zorder=4,
                )
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("当前地图状态", pad=10)


def render_probability_heatmap(ax, grid, region_probabilities, border_segments, selected_region=None):
    rows = len(grid)
    cols = len(grid[0])
    prob_grid = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            region = int(grid[r][c])
            prob_grid[r, c] = float(region_probabilities.get(region, 0.0))

    im = ax.imshow(
        prob_grid,
        cmap="YlOrRd",
        norm=Normalize(vmin=0.0, vmax=max(0.25, float(prob_grid.max()) if prob_grid.size else 0.25)),
        interpolation="nearest",
    )
    h_segments, v_segments = border_segments
    for (xs, xe), (ys, ye) in h_segments:
        ax.plot([xs - 0.5, xe - 0.5], [ys - 0.5, ye - 0.5], color="black", linewidth=0.6, zorder=2)
    for (xs, xe), (ys, ye) in v_segments:
        ax.plot([xs - 0.5, xe - 0.5], [ys - 0.5, ye - 0.5], color="black", linewidth=0.6, zorder=2)

    if selected_region is not None:
        selected_mask = np.array([[int(grid[r][c]) == selected_region for c in range(cols)] for r in range(rows)])
        for r in range(rows):
            for c in range(cols):
                if selected_mask[r, c]:
                    ax.add_patch(
                        Rectangle(
                            (c - 0.5, r - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor=SELECTED_REGION_COLOR,
                            linewidth=1.6,
                            zorder=4,
                        )
                    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("区域选择概率热力图", pad=10)
    return im


def visualize_all_rounds(
    output_dir: str | Path | None = None,
    max_steps: int = 120,
    stop_on_cycle: bool = False,
    max_depth: int = 4,
    stuck_threshold: int = 3,
    random_tie_break: bool = False,
) -> Path:
    plt.style.use("default")
    configure_chinese_font()

    materials = load_materials()
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "results" / "conflict_search_agent_bfs_depth4_visualizations"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for round_index, round_data in enumerate(materials["rounds"], start=1):
        grid = round_data["mapData"]["grid"]
        region_cells = build_region_cells(grid)
        border_segments = build_border_segments(grid)
        round_dir = output_root / f"round_{round_index:02d}"
        round_dir.mkdir(exist_ok=True)

        trace = trace_agent_on_round(
            round_data,
            max_steps=max_steps,
            stop_on_cycle=stop_on_cycle,
            max_depth=max_depth,
            stuck_threshold=stuck_threshold,
            random_tie_break=random_tie_break,
        )
        n_rendered = 0

        for step in trace:
            status = step["status"]
            if status != "action":
                summary_rows.append(
                    {
                        "round": round_index,
                        "status": status,
                        "rendered_steps": n_rendered,
                        "max_depth": max_depth,
                        "stuck_threshold": stuck_threshold,
                        "random_tie_break": random_tie_break,
                    }
                )
                break

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            render_state(
                axes[0],
                grid,
                step["colors_before"],
                border_segments,
                region_cells,
                step["conflict_edges_before"],
                selected_region=step["region"],
            )
            im = render_probability_heatmap(
                axes[1],
                grid,
                step["region_probabilities"],
                border_segments,
                selected_region=step["region"],
            )
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle(
                f"Round {round_index} | Step {step['agent_step']} | planning-depth={step['planning_depth_used']} | "
                f"search-depth={step['search_depth']} | "
                f"R{step['region'] + 1}: C{step['old_color'] + 1} -> C{step['new_color'] + 1}",
                fontsize=15,
                y=0.98,
            )
            fig.text(
                0.5,
                0.02,
                f"conflicts: {len(step['conflict_edges_before'])} -> {len(step['conflict_edges_after'])} | "
                f"earliest-depth legal actions: {step['n_legal_actions_first_depth']} | "
                f"found-solution-within-depth: {step['selected_found_solution_within_depth']} | "
                f"next-planning-depth: {step['planning_depth_next']}",
                ha="center",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0.05, 1, 0.94))
            out_path = round_dir / f"step_{step['agent_step']:03d}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            n_rendered += 1
        else:
            summary_rows.append(
                {
                    "round": round_index,
                    "status": "max_steps_reached",
                    "rendered_steps": n_rendered,
                    "max_depth": max_depth,
                    "stuck_threshold": stuck_threshold,
                    "random_tie_break": random_tie_break,
                }
            )

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    return output_root


if __name__ == "__main__":
    out = visualize_all_rounds()
    print(out)
