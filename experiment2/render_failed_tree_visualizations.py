from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "experiment2" / "generated_map_search_planning1" / "agent_validation" / "failed_maps_correct_child_visibility_traces.json"
ROUND_INDEX_PATH = ROOT / "experiment2" / "generated_map_search_planning1" / "agent_validation" / "failed_maps_correct_child_round_index.csv"
OUTPUT_DIR = ROOT / "experiment2" / "generated_map_search_planning1" / "agent_validation" / "tree_visualizations"


def load_round_index(path: Path) -> dict[int, dict]:
    by_round: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rnd = int(row["round"])
            by_round[rnd] = row
    return by_round


def fmt_score(score) -> str:
    if score is None:
        return "-"
    return "(" + ", ".join(f"{float(v):.2f}" for v in score) + ")"


def fmt_action(region: int, new_color: int) -> str:
    return f"R{region + 1}->C{new_color + 1}"


def node_fill(candidate: dict, selected: tuple[int, int], correct: tuple[int, int] | None) -> str:
    action = (int(candidate["region"]), int(candidate["new_color"]))
    is_selected = action == selected
    is_correct = correct is not None and action == correct
    is_solution = bool(candidate.get("found_solution_within_depth", False))
    if is_selected and is_correct:
        return "#f4cccc"
    if is_selected:
        return "#fce5cd"
    if is_correct:
        return "#d9ead3"
    if is_solution:
        return "#cfe2f3"
    return "#f3f3f3"


def node_border(candidate: dict, selected: tuple[int, int], correct: tuple[int, int] | None) -> tuple[str, str]:
    action = (int(candidate["region"]), int(candidate["new_color"]))
    is_selected = action == selected
    is_correct = correct is not None and action == correct
    is_solution = bool(candidate.get("found_solution_within_depth", False))
    if is_selected and is_correct:
        return "#cc0000", "3"
    if is_selected:
        return "#d17b00", "3"
    if is_correct:
        return "#38761d", "3"
    if is_solution:
        return "#3d85c6", "2"
    return "#666666", "1"


def build_dot(round_id: int, round_meta: dict, trace: list[dict]) -> str:
    correct = None
    if round_meta.get("has_depth6_optimal_path", "").lower() == "true":
        correct = (
            int(float(round_meta["optimal_first_action_region"])),
            int(float(round_meta["optimal_first_action_color"])),
        )

    lines: list[str] = []
    lines.append("digraph G {")
    lines.append('  rankdir=LR;')
    lines.append('  splines=true;')
    lines.append('  graph [fontname="Helvetica", labelloc="t", fontsize=18];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];')
    lines.append('  edge [fontname="Helvetica", fontsize=10, color="#888888"];')

    title = f"Round {round_id}"
    if correct is not None:
        title += f" | correct first child: {fmt_action(*correct)}"
    else:
        title += " | no depth-6 optimal first child extracted"
    lines.append(f'  label="{title}";')

    lines.append('  legend [shape=note, fillcolor="#ffffff", color="#666666", label="orange border/fill = selected child\\ngreen border/fill = correct first child\\nblue border/fill = found_solution_within_depth\\nred border/fill = selected + correct"];')

    action_steps = [step for step in trace if step.get("status") == "action"]
    for step in action_steps:
        step_id = int(step["agent_step"])
        selected = (int(step["region"]), int(step["new_color"]))
        root_node = f"root_{step_id}"
        root_label = (
            f"step {step_id}\\n"
            f"conflicts {len(step['conflict_edges_before'])}->{len(step['conflict_edges_after'])}\\n"
            f"tree iters={step['tree_iterations_used']}"
        )
        lines.append(f'  {root_node} [fillcolor="#ffffff", color="#000000", penwidth=2, label="{root_label}"];')

        if step_id < len(action_steps) - 1:
            next_root = f"root_{step_id + 1}"
            lines.append(f'  {root_node} -> {next_root} [color="#000000", penwidth=2, label="selected path"];')

        for idx, cand in enumerate(step["candidate_actions"]):
            cand_id = f"s{step_id}_c{idx}"
            fill = node_fill(cand, selected, correct)
            border, penwidth = node_border(cand, selected, correct)
            label = (
                f"{fmt_action(int(cand['region']), int(cand['new_color']))}\\n"
                f"h={float(cand['heuristic_score']):.2f}\\n"
                f"score={fmt_score(cand['score'])}\\n"
                f"after_conf={int(cand['n_conflict_edges_after'])}"
            )
            if bool(cand.get("found_solution_within_depth", False)):
                label += "\\nSOLVABLE<=depth"
            lines.append(
                f'  {cand_id} [fillcolor="{fill}", color="{border}", penwidth={penwidth}, label="{label}"];'
            )
            lines.append(f'  {root_node} -> {cand_id};')

        same_rank = " ".join([root_node] + [f"s{step_id}_c{i}" for i in range(len(step["candidate_actions"]))])
        lines.append(f"  {{ rank=same; {same_rank}; }}")

    lines.append("}")
    return "\n".join(lines)


def render_round(round_id: int, round_meta: dict, trace: list[dict]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_path = OUTPUT_DIR / f"round_{round_id:02d}.dot"
    png_path = OUTPUT_DIR / f"round_{round_id:02d}.png"
    dot_path.write_text(build_dot(round_id, round_meta, trace), encoding="utf-8")
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    return dot_path, png_path


def main() -> None:
    traces = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    round_index = load_round_index(ROUND_INDEX_PATH)

    # Prefer rounds where a depth-6 optimal first action is available.
    chosen_rounds = [rnd for rnd, meta in round_index.items() if meta["has_depth6_optimal_path"].lower() == "true"]
    if not chosen_rounds:
        chosen_rounds = sorted(int(r) for r in traces.keys())

    print("Rendering rounds:", ", ".join(str(r) for r in chosen_rounds))
    for rnd in chosen_rounds:
        trace = traces.get(str(rnd), [])
        dot_path, png_path = render_round(rnd, round_index[rnd], trace)
        print(f"wrote {dot_path}")
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
