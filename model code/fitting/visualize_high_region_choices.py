"""
可视化所有被试在高 region 数量轮次中的选择过程。

默认筛选 NumRegions > 35 的正式轮次（即 37/39/42/45），为每个被试每个轮次
生成一张包含完整操作过程的联系图，并保存到输出文件夹。

用法:
    python "model code/fitting/visualize_high_region_choices.py"
    python "model code/fitting/visualize_high_region_choices.py" --data-dir data --output-dir "model code/fitting/high_region_choice_process"
    python "model code/fitting/visualize_high_region_choices.py" --min-regions 40
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from fit_softmax import ROUND_SEEDS, ROUND_SIZES, generate_map, mulberry32

ROWS, COLS = 12, 20
FORMAL_ROUNDS = [str(i + 1) for i in range(len(ROUND_SIZES))]
ACTION_RE = re.compile(r"Colored Region (\d+) with (Color (\d+)|Eraser)")

# 与 App.tsx 保持一致；额外增加未着色和擦除高亮色。
COLOR_HEXES = [
    "#ffffff",  # 未着色
    "#377eb8",  # Color 1
    "#4daf4a",  # Color 2
    "#984ea3",  # Color 3
    "#ffff33",  # Color 4
]
ERASER_HIGHLIGHT = "#d62728"
MOVE_HIGHLIGHT = "#111111"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize participant choices for high-region rounds.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing data_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="model code/fitting/high_region_choice_process",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--min-regions",
        type=int,
        default=35,
        help="Only visualize formal rounds with NumRegions strictly greater than this threshold.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of panels per row in each contact sheet.",
    )
    return parser.parse_args()


def build_formal_maps():
    maps = {}
    for round_label, size, seed in zip(FORMAL_ROUNDS, ROUND_SIZES, ROUND_SEEDS):
        rng = mulberry32(seed)
        grid, regions, adjacency = generate_map(size, rng)
        maps[round_label] = {
            "grid": grid,
            "regions": regions,
            "adjacency": adjacency,
            "num_regions": size,
        }
    return maps


def ascii_label(text: str) -> str:
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return text.encode("unicode_escape").decode("ascii")


def parse_actions(csv_path: Path):
    actions = []
    in_actions = False

    with csv_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "[Actions]":
                in_actions = True
                continue
            if line == "[Adjacency]":
                break
            if not in_actions or not line or line.startswith("SessionID,"):
                continue

            parts = line.split(",", 5)
            if len(parts) < 6:
                continue

            round_label = parts[1]
            num_regions = int(parts[2])
            step = int(parts[3])
            action_text = parts[4].strip('"')
            time_taken = float(parts[5])

            if "Game Started" in action_text:
                actions.append(
                    {
                        "round": round_label,
                        "num_regions": num_regions,
                        "step": step,
                        "is_start": True,
                        "region": None,
                        "color": None,
                        "is_eraser": False,
                        "time_taken": time_taken,
                        "label": f"Start ({num_regions})",
                    }
                )
                continue

            match = ACTION_RE.match(action_text)
            if not match:
                continue

            region_id = int(match.group(1)) - 1
            is_eraser = match.group(2) == "Eraser"
            color_idx = None if is_eraser else int(match.group(3)) - 1

            if is_eraser:
                label = f"Erase R{region_id + 1}"
            else:
                label = f"R{region_id + 1} -> C{color_idx + 1}"

            actions.append(
                {
                    "round": round_label,
                    "num_regions": num_regions,
                    "step": step,
                    "is_start": False,
                    "region": region_id,
                    "color": color_idx,
                    "is_eraser": is_eraser,
                    "time_taken": time_taken,
                    "label": label,
                }
            )

    return actions


def group_visualization_steps(actions, min_regions: int):
    grouped = defaultdict(list)
    current_states = {}

    for action in actions:
        round_label = action["round"]
        num_regions = action["num_regions"]
        if round_label not in FORMAL_ROUNDS or num_regions <= min_regions:
            continue

        if action["is_start"]:
            current_states[round_label] = [None] * num_regions
        elif round_label not in current_states:
            current_states[round_label] = [None] * num_regions

        state = list(current_states[round_label])
        if not action["is_start"]:
            region_id = action["region"]
            state[region_id] = None if action["is_eraser"] else action["color"]
            current_states[round_label] = state

        grouped[round_label].append(
            {
                "snapshot": list(state),
                "step": action["step"],
                "label": action["label"],
                "region": action["region"],
                "is_eraser": action["is_eraser"],
                "time_taken": action["time_taken"],
            }
        )

    return grouped


def build_border_segments(grid):
    horizontal = []
    vertical = []

    for r in range(ROWS):
        for c in range(COLS):
            region = grid[r][c]
            if r == 0 or grid[r - 1][c] != region:
                horizontal.append(((c, c + 1), (r, r)))
            if r == ROWS - 1 or grid[r + 1][c] != region:
                horizontal.append(((c, c + 1), (r + 1, r + 1)))
            if c == 0 or grid[r][c - 1] != region:
                vertical.append(((c, c), (r, r + 1)))
            if c == COLS - 1 or grid[r][c + 1] != region:
                vertical.append(((c + 1, c + 1), (r, r + 1)))

    return horizontal, vertical


def render_snapshot(ax, grid, colors, border_segments, active_region=None, eraser=False):
    color_grid = [[0] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            region_id = grid[r][c]
            color = colors[region_id]
            color_grid[r][c] = 0 if color is None else color + 1

    cmap = ListedColormap(COLOR_HEXES)
    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=len(COLOR_HEXES) - 1, interpolation="nearest")

    horizontal, vertical = border_segments
    for (xs, xe), (ys, ye) in horizontal:
        ax.plot([xs - 0.5, xe - 0.5], [ys - 0.5, ye - 0.5], color="black", linewidth=0.6)
    for (xs, xe), (ys, ye) in vertical:
        ax.plot([xs - 0.5, xe - 0.5], [ys - 0.5, ye - 0.5], color="black", linewidth=0.6)

    if active_region is not None:
        outline_color = ERASER_HIGHLIGHT if eraser else MOVE_HIGHLIGHT
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == active_region:
                    ax.add_patch(
                        Rectangle(
                            (c - 0.5, r - 0.5),
                            1,
                            1,
                            fill=False,
                            linewidth=1.0,
                            edgecolor=outline_color,
                        )
                    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_aspect("equal")


def plot_round_contact_sheet(participant_id: str, round_label: str, round_info, steps, output_path: Path, ncols: int):
    if not steps:
        return

    grid = round_info["grid"]
    num_regions = round_info["num_regions"]
    border_segments = build_border_segments(grid)

    n_panels = len(steps)
    ncols = max(1, ncols)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.1, nrows * 2.9))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
    else:
        flattened = []
        for row in axes:
            if isinstance(row, (list, tuple)):
                flattened.extend(row)
            else:
                flattened.append(row)
        axes = flattened

    for ax, step_info in zip(axes, steps):
        render_snapshot(
            ax,
            grid=grid,
            colors=step_info["snapshot"],
            border_segments=border_segments,
            active_region=step_info["region"],
            eraser=step_info["is_eraser"],
        )
        title = f"Step {step_info['step']}\n{step_info['label']}"
        if step_info["step"] != 0:
            title += f" | {step_info['time_taken']:.1f}s"
        ax.set_title(title, fontsize=8)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Participant {ascii_label(participant_id)} | Round {round_label} | {num_regions} regions | {n_panels} snapshots",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_index(index_rows, output_dir: Path):
    index_path = output_dir / "index.tsv"
    with index_path.open("w", encoding="utf-8") as f:
        f.write("participant\tround\tnum_regions\tn_snapshots\tfile\n")
        for row in index_rows:
            f.write(
                f"{row['participant']}\t{row['round']}\t{row['num_regions']}\t{row['n_snapshots']}\t{row['file']}\n"
            )


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maps = build_formal_maps()
    csv_files = sorted(data_dir.glob("data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No data_*.csv files found in {data_dir}")

    index_rows = []
    created = 0

    for csv_path in csv_files:
        participant_id = csv_path.stem.replace("data_", "", 1)
        actions = parse_actions(csv_path)
        grouped = group_visualization_steps(actions, min_regions=args.min_regions)

        for round_label in sorted(grouped, key=lambda x: int(x)):
            output_path = output_dir / f"{participant_id}_round_{round_label}_{maps[round_label]['num_regions']}regions.png"
            plot_round_contact_sheet(
                participant_id=participant_id,
                round_label=round_label,
                round_info=maps[round_label],
                steps=grouped[round_label],
                output_path=output_path,
                ncols=args.cols,
            )
            index_rows.append(
                {
                    "participant": participant_id,
                    "round": round_label,
                    "num_regions": maps[round_label]["num_regions"],
                    "n_snapshots": len(grouped[round_label]),
                    "file": output_path.name,
                }
            )
            created += 1

    write_index(index_rows, output_dir)
    print(f"Created {created} figures in {output_dir}")


if __name__ == "__main__":
    main()
