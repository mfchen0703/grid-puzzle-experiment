"""Generate maps that fourinarow-style tree agent fails.

Each generated candidate is validated with the current fourinarow-style BFS
tree agent.  A map is accepted only if all requested pruning thresholds fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "model code" / "experiment2" / "analysis_code"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import conflict_search_agent as tree_agent  # noqa: E402
from generate_rounds_json import get_conflict_regions  # noqa: E402
from search_generated_maps import (  # noqa: E402
    COLORS,
    NUM_REGIONS,
    adjacent_nonconflict_regions,
    build_adjacency_list,
    build_candidate,
    classify_difficulty,
    difficulty_score,
    find_optimal_solution_stats,
    seed_time_limit,
)


def parse_pruning_values(raw: str) -> list[float]:
    return [float(chunk) for chunk in raw.split(",") if chunk.strip()]


def build_generated_candidate(seed: int, attempts: int, args: argparse.Namespace) -> dict:
    condition_type = (
        "requires_nonconflict_region"
        if attempts % 2 == 1
        else "requires_adjacent_nonconflict_pair"
    )
    started = perf_counter()
    candidate = build_candidate(seed, condition_type, generation_mode=args.generation_mode)
    adjacency = build_adjacency_list(candidate.pop("_adjacency"))
    stats = find_optimal_solution_stats(
        adjacency,
        candidate["initialColors"],
        max_depth=args.max_depth,
        solution_limit=args.solution_limit,
    )
    conflict_regions = get_conflict_regions(candidate["conflictEdges"])
    adjacent_nonconflict = adjacent_nonconflict_regions(adjacency, conflict_regions)
    planning_requirement = stats["min_adjacent_nonconflict_changes_collected"]
    difficulty = classify_difficulty(stats["optimal_length"], planning_requirement)
    candidate["metadata"] = {
        "candidateIndex": 0,
        "seedAttemptIndex": attempts,
        "seed": seed,
        "generationMode": candidate.get("generationMode", args.generation_mode),
        "difficultyLabel": difficulty,
        "difficultyScore": difficulty_score(
            stats["optimal_length"],
            planning_requirement,
            args.max_depth,
        ),
        "optimalLength": stats["optimal_length"],
        "solvedWithinBound": stats["solved_within_bound"],
        "planningRequirementCollected": planning_requirement,
        "nCollectedOptimalSolutions": stats["n_collected_optimal_solutions"],
        "solutionCollectionCapped": stats["solution_collection_capped"],
        "searchNodes": stats["search_nodes"],
        "elapsedSeconds": round(perf_counter() - started, 4),
        "initialConflictRegions": sorted(conflict_regions),
        "adjacentNonconflictRegions": sorted(adjacent_nonconflict),
        "exampleOptimalSolution": stats["example_optimal_solution"],
        "source": "new_generated_fourinarow_tree_failed",
    }
    return candidate


def run_fourinarow_tree(round_data: dict, pruning_thresh: float, args: argparse.Namespace) -> dict:
    started = perf_counter()
    adjacency = tree_agent.build_adjacency_map(round_data["mapData"])
    trajectory, final_colors = tree_agent.run_bfs_tree_agent_on_round(
        round_data,
        max_steps=args.tree_max_steps,
        max_depth=args.tree_max_depth,
        max_expansions=args.tree_max_expansions,
        n_iterations=None,
        pruning_thresh=float(pruning_thresh),
        heuristic_eval_weight=args.tree_heuristic_eval_weight,
        tree_score_strategy=args.tree_score_strategy,
        gamma=args.tree_gamma,
        lapse_rate=args.tree_lapse_rate,
        random_tie_break=args.tree_random_tie_break,
    )
    final_conflicts = tree_agent.get_conflict_edges(adjacency, final_colors)
    action_steps = [row for row in trajectory if not row.get("terminated", False)]
    expansions = sum(int(row.get("bfs_expansions") or 0) for row in action_steps)
    max_step_expansions = max([int(row.get("bfs_expansions") or 0) for row in action_steps] or [0])
    return {
        "pruning_thresh": pruning_thresh,
        "success": len(final_conflicts) == 0,
        "steps": len(action_steps),
        "final_conflicts": len(final_conflicts),
        "expansions": expansions,
        "max_step_expansions": max_step_expansions,
        "elapsed_seconds": round(perf_counter() - started, 4),
        "conflict_trace": " ".join(
            str(int(row.get("n_conflict_edges_after")))
            for row in action_steps
            if row.get("n_conflict_edges_after") is not None
        ),
    }


def validate_candidate(candidate: dict, args: argparse.Namespace) -> list[dict]:
    return [
        run_fourinarow_tree(candidate, pruning_thresh, args)
        for pruning_thresh in parse_pruning_values(args.pruning_values)
    ]


def renumber_rows(rows: list[dict]) -> list[dict]:
    for index, row in enumerate(rows, start=1):
        metadata = row.setdefault("metadata", {})
        metadata["candidateIndex"] = index
        metadata["renumberedRound"] = index
    return rows


def write_outputs(rows: list[dict], validation_rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": 12,
        "cols": 20,
        "numRegions": NUM_REGIONS,
        "colors": COLORS,
        "rounds": rows,
    }
    json_path = output_dir / "generated_maps_sorted.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "generated_maps_sorted.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "round",
            "seed",
            "seedAttemptIndex",
            "difficultyLabel",
            "difficultyScore",
            "conditionType",
            "generationMode",
            "optimalLength",
            "planningRequirementCollected",
            "initialConflictEdges",
            "searchNodes",
            "elapsedSeconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            metadata = row["metadata"]
            writer.writerow(
                {
                    "round": index,
                    "seed": metadata["seed"],
                    "seedAttemptIndex": metadata["seedAttemptIndex"],
                    "difficultyLabel": metadata["difficultyLabel"],
                    "difficultyScore": metadata["difficultyScore"],
                    "conditionType": row["conditionType"],
                    "generationMode": row.get("generationMode", ""),
                    "optimalLength": metadata["optimalLength"],
                    "planningRequirementCollected": metadata["planningRequirementCollected"],
                    "initialConflictEdges": len(row["conflictEdges"]),
                    "searchNodes": metadata["searchNodes"],
                    "elapsedSeconds": metadata["elapsedSeconds"],
                }
            )

    validation_path = output_dir / "fourinarow_tree_pruning_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "round",
            "seed",
            "pruning_thresh",
            "success",
            "steps",
            "final_conflicts",
            "expansions",
            "max_step_expansions",
            "elapsed_seconds",
            "conflict_trace",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_rows)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {validation_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate maps unsolved by fourinarow-style tree agent."
    )
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--max-seed-attempts", type=int, default=500)
    parser.add_argument("--per-seed-timeout", type=float, default=20.0)
    parser.add_argument(
        "--generation-mode",
        choices=["legacy", "easy_perturb", "broad_perturb", "hard_perturb", "mixed_perturb"],
        default="legacy",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--solution-limit", type=int, default=200)
    parser.add_argument("--pruning-values", default="1,2,5")
    parser.add_argument("--tree-max-steps", type=int, default=100)
    parser.add_argument("--tree-max-depth", type=int, default=8)
    parser.add_argument("--tree-max-expansions", type=int, default=800)
    parser.add_argument("--tree-heuristic-eval-weight", type=float, default=0.0)
    parser.add_argument("--tree-score-strategy", default="task_first")
    parser.add_argument("--tree-gamma", type=float, default=None)
    parser.add_argument("--tree-lapse-rate", type=float, default=0.0)
    parser.add_argument("--tree-random-tie-break", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_fourinarow_tree_failed_maps_search_10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted: list[dict] = []
    validation_rows: list[dict] = []
    attempts = 0

    while len(accepted) < int(args.target_count) and attempts < int(args.max_seed_attempts):
        seed = int(args.start_seed) + attempts
        attempts += 1
        try:
            with seed_time_limit(args.per_seed_timeout):
                candidate = build_generated_candidate(seed, attempts, args)
                results = validate_candidate(candidate, args)
        except Exception as exc:
            print(f"skip seed={seed}: {exc}", flush=True)
            continue

        solved_any = any(result["success"] for result in results)
        metadata = candidate["metadata"]
        if solved_any:
            solved_pruning = [
                str(result["pruning_thresh"])
                for result in results
                if result["success"]
            ]
            print(
                f"reject seed={seed}: solved by pruning={','.join(solved_pruning)} "
                f"label={metadata['difficultyLabel']} optimal={metadata['optimalLength']}",
                flush=True,
            )
            continue

        accepted.append(candidate)
        round_index = len(accepted)
        metadata["fourinarowTreeFailedPruningValues"] = parse_pruning_values(args.pruning_values)
        for result in results:
            validation_rows.append(
                {
                    "round": round_index,
                    "seed": seed,
                    **result,
                }
            )
        print(
            f"accept {round_index}/{args.target_count} seed={seed}: "
            f"label={metadata['difficultyLabel']} optimal={metadata['optimalLength']} "
            f"planning={metadata['planningRequirementCollected']} "
            f"final_conflicts={[r['final_conflicts'] for r in results]}",
            flush=True,
        )

    if len(accepted) < int(args.target_count):
        raise RuntimeError(
            f"Only accepted {len(accepted)} maps after {attempts} attempts. "
            "Increase --max-seed-attempts or relax filters."
        )

    write_outputs(renumber_rows(accepted), validation_rows, args.output_dir)
    print(
        f"Done: accepted {len(accepted)} maps from {attempts} generated seed attempts.",
        flush=True,
    )


if __name__ == "__main__":
    main()
