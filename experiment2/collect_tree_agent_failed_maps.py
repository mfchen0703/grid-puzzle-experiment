"""
Build a map set containing only rounds the tree agent does not solve.

The script:
1. Loads an existing generated_maps_sorted.json file.
2. Re-runs the tree agent and keeps only unsolved rounds.
3. Generates additional candidates with search_generated_maps.py utilities.
4. Keeps only newly generated candidates that the same tree agent fails.
5. Re-numbers the combined output from 1..N and writes CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "model code" / "experiment2" / "analysis_code"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

# conflict_search_agent imports pandas for its batch DataFrame helpers, but this
# script only calls the single-round tree-agent function. Keep the dependency
# optional for lightweight environments.
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    pandas_stub = types.ModuleType("pandas")

    class _UnavailableDataFrame:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("pandas is required for DataFrame helpers")

    pandas_stub.DataFrame = _UnavailableDataFrame
    sys.modules["pandas"] = pandas_stub

import conflict_search_agent as tree_agent  # noqa: E402
from generate_rounds_json import get_conflict_edges, get_conflict_regions  # noqa: E402
from search_generated_maps import (  # noqa: E402
    adjacent_nonconflict_regions,
    build_adjacency_list,
    build_candidate,
    classify_difficulty,
    difficulty_score,
    find_optimal_solution_stats,
    write_outputs,
)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def load_rounds(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rounds", []))


def load_existing_summary(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["round"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    return out


def tree_agent_result(round_data: dict, args: argparse.Namespace) -> dict:
    started = perf_counter()
    adjacency = tree_agent.build_adjacency_map(round_data["mapData"])
    initial_colors = [int(color) for color in round_data["initialColors"]]
    initial_conflicts = tree_agent.get_conflict_edges(adjacency, initial_colors)
    trajectory, final_colors = tree_agent.run_tree_agent_on_round(
        round_data,
        max_steps=args.tree_max_steps,
        max_depth=args.tree_max_depth,
        n_iterations=args.tree_iterations,
        pruning_thresh=args.tree_pruning_thresh,
        heuristic_eval_weight=args.tree_heuristic_eval_weight,
        heuristic_frontier_weight=args.tree_heuristic_frontier_weight,
        force_expand_root=args.tree_force_expand_root,
        frontier_strategy=args.tree_frontier_strategy,
        tree_score_strategy=args.tree_score_strategy,
        history_penalty_weight=args.tree_history_penalty_weight,
        lapse_rate=args.tree_lapse_rate,
        gamma=args.tree_gamma,
        random_tie_break=args.tree_random_tie_break,
    )
    final_conflicts = tree_agent.get_conflict_edges(adjacency, final_colors)
    return {
        "solved": len(final_conflicts) == 0,
        "initial_conflict_edges": len(initial_conflicts),
        "final_conflict_edges": len(final_conflicts),
        "n_agent_steps": sum(1 for row in trajectory if not row.get("terminated", False)),
        "elapsed_seconds": round(perf_counter() - started, 4),
    }


def annotate_tree_result(row: dict, result: dict) -> None:
    metadata = row.setdefault("metadata", {})
    metadata["treeAgentSolved"] = bool(result["solved"])
    metadata["treeAgentFinalConflictEdges"] = int(result["final_conflict_edges"])
    metadata["treeAgentSteps"] = int(result["n_agent_steps"])
    metadata["treeAgentElapsedSeconds"] = float(result["elapsed_seconds"])


def renumber_rows(rows: list[dict]) -> list[dict]:
    for index, row in enumerate(rows, start=1):
        metadata = row.setdefault("metadata", {})
        metadata["candidateIndex"] = index
        metadata["renumberedRound"] = index
    return rows


def existing_seed_max(rows: list[dict]) -> int | None:
    seeds = []
    for row in rows:
        try:
            seeds.append(int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(seeds) if seeds else None


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
        "difficultyScore": difficulty_score(stats["optimal_length"], planning_requirement, args.max_depth),
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
        "source": "new_generated",
    }
    return candidate


def summary_tree_result(row: dict) -> dict:
    return {
        "solved": _is_true(row.get("solved")),
        "initial_conflict_edges": int(float(row.get("initial_conflict_edges") or 0)),
        "final_conflict_edges": int(float(row.get("final_conflict_edges") or 0)),
        "n_agent_steps": int(float(row.get("n_agent_steps") or 0)),
        "elapsed_seconds": float(row.get("elapsed_seconds") or 0.0),
    }


def keep_existing_unsolved(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    summary_by_round = {} if args.revalidate_existing else load_existing_summary(args.existing_summary_csv)
    if summary_by_round:
        print(f"Using existing tree-agent summary: {args.existing_summary_csv}", flush=True)
    kept = []
    for index, row in enumerate(rows, start=1):
        if index in summary_by_round:
            result = summary_tree_result(summary_by_round[index])
        else:
            result = tree_agent_result(row, args)
        annotate_tree_result(row, result)
        seed = row.get("seed", row.get("metadata", {}).get("seed", ""))
        if result["solved"]:
            print(f"drop existing round={index} seed={seed}: tree solved", flush=True)
            continue
        row.setdefault("metadata", {})["source"] = "existing_unsolved"
        kept.append(row)
        print(
            f"keep existing round={index} seed={seed}: "
            f"final_conflicts={result['final_conflict_edges']} steps={result['n_agent_steps']}",
            flush=True,
        )
    return kept


def generate_new_unsolved(args: argparse.Namespace, existing_rows: list[dict]) -> list[dict]:
    target = int(args.new_unsolved_count)
    if target <= 0:
        return []

    start_seed = args.start_seed
    if start_seed is None:
        max_seed = existing_seed_max(existing_rows)
        start_seed = 1 if max_seed is None else max_seed + 1

    accepted: list[dict] = []
    attempts = 0
    seen_seeds = {int(row["seed"]) for row in existing_rows if str(row.get("seed", "")).isdigit()}
    while len(accepted) < target and attempts < args.max_seed_attempts:
        seed = int(start_seed) + attempts
        attempts += 1
        if seed in seen_seeds:
            continue
        try:
            candidate = build_generated_candidate(seed, attempts, args)
        except RuntimeError as exc:
            print(f"skip seed={seed}: {exc}", flush=True)
            continue

        result = tree_agent_result(candidate, args)
        annotate_tree_result(candidate, result)
        metadata = candidate["metadata"]
        if result["solved"]:
            print(
                f"reject seed={seed}: tree solved "
                f"label={metadata['difficultyLabel']} optimal={metadata['optimalLength']}",
                flush=True,
            )
            continue

        accepted.append(candidate)
        print(
            f"accept {len(accepted)}/{target} seed={seed}: "
            f"label={metadata['difficultyLabel']} optimal={metadata['optimalLength']} "
            f"planning={metadata['planningRequirementCollected']} "
            f"final_conflicts={result['final_conflict_edges']}",
            flush=True,
        )

    if len(accepted) < target:
        raise RuntimeError(
            f"Only found {len(accepted)} new tree-unsolved maps after {attempts} attempts. "
            "Increase --max-seed-attempts or adjust generation settings."
        )
    return accepted


def write_tree_agent_summary(rows: list[dict], output_dir: Path) -> None:
    path = output_dir / "tree_agent_filter_summary.csv"
    fieldnames = [
        "round",
        "seed",
        "source",
        "difficultyLabel",
        "optimalLength",
        "planningRequirementCollected",
        "treeAgentSolved",
        "treeAgentSteps",
        "treeAgentFinalConflictEdges",
        "treeAgentElapsedSeconds",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            metadata = row.get("metadata", {})
            writer.writerow(
                {
                    "round": index,
                    "seed": row.get("seed", metadata.get("seed", "")),
                    "source": metadata.get("source", ""),
                    "difficultyLabel": metadata.get("difficultyLabel", ""),
                    "optimalLength": metadata.get("optimalLength", ""),
                    "planningRequirementCollected": metadata.get("planningRequirementCollected", ""),
                    "treeAgentSolved": metadata.get("treeAgentSolved", ""),
                    "treeAgentSteps": metadata.get("treeAgentSteps", ""),
                    "treeAgentFinalConflictEdges": metadata.get("treeAgentFinalConflictEdges", ""),
                    "treeAgentElapsedSeconds": metadata.get("treeAgentElapsedSeconds", ""),
                }
            )
    print(f"Wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep existing tree-agent failures and generate more tree-agent-failed maps."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_map_search_planning1" / "generated_maps_sorted.json",
    )
    parser.add_argument(
        "--existing-summary-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "generated_map_search_planning1"
        / "agent_validation"
        / "tree_agent_default_round_summary.csv",
        help="Existing tree-agent result table used to filter old maps. Use --revalidate-existing to ignore it.",
    )
    parser.add_argument(
        "--revalidate-existing",
        action="store_true",
        help="Re-run the current tree agent on existing maps instead of trusting --existing-summary-csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_tree_agent_failed_maps",
    )
    parser.add_argument("--new-unsolved-count", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-seed-attempts", type=int, default=500)
    parser.add_argument("--generation-mode", default="legacy")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--solution-limit", type=int, default=200)

    parser.add_argument("--tree-max-steps", type=int, default=100)
    parser.add_argument("--tree-max-depth", type=int, default=4)
    parser.add_argument("--tree-iterations", type=int, default=20)
    parser.add_argument("--tree-pruning-thresh", type=float, default=0.0)
    parser.add_argument("--tree-heuristic-eval-weight", type=float, default=0.0)
    parser.add_argument("--tree-heuristic-frontier-weight", type=float, default=0.0)
    parser.add_argument("--tree-history-penalty-weight", type=float, default=0.0)
    parser.add_argument("--tree-lapse-rate", type=float, default=0.0)
    parser.add_argument("--tree-gamma", type=float, default=None)
    parser.add_argument("--tree-frontier-strategy", default="global_frontier")
    parser.add_argument("--tree-score-strategy", default="task_first")
    parser.add_argument("--tree-force-expand-root", action="store_true")
    parser.add_argument("--tree-random-tie-break", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_rows = load_rounds(args.input_json)
    kept_existing = keep_existing_unsolved(existing_rows, args)
    new_unsolved = generate_new_unsolved(args, existing_rows)
    combined = renumber_rows(kept_existing + new_unsolved)
    write_outputs(combined, args.output_dir)
    write_tree_agent_summary(combined, args.output_dir)
    print(
        f"Done: kept {len(kept_existing)} existing unsolved maps, "
        f"added {len(new_unsolved)} new unsolved maps, total {len(combined)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
