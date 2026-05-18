"""Simulation-based recovery for full HSP2-vs-EFLOP repair trajectories.

This script generates full repair sequences from the current experiment-2
rounds, then asks whether a Monte Carlo recovery procedure can identify the
agent that generated each sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "model code" / "experiment2" / "analysis_code"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from conflict_repair_hsp2 import (  # noqa: E402
    apply_action,
    count_conflicts,
    is_solved,
    run_eflop,
    run_hsp2_planner,
    run_min_conflicts,
    state_to_key,
)
from conflict_repair_hsp2.solver import Action, AdjList  # noqa: E402


AgentName = str


@dataclass(frozen=True)
class RepairTrace:
    agent: AgentName
    round_id: int
    seed: int
    success: bool
    actions: Tuple[Action, ...]
    module_path: Tuple[str, ...]
    conflict_trace: Tuple[int, ...]


def load_rounds(path: Path) -> List[dict]:
    with path.open() as f:
        data = json.load(f)
    return list(data["rounds"])


def adjacency_from_round(round_data: dict) -> AdjList:
    n_regions = int(round_data["mapData"]["numRegions"])
    adj: AdjList = {idx: [] for idx in range(n_regions)}
    for u, v in round_data["mapData"]["adjacencyPairs"]:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return {node: sorted(neighbors) for node, neighbors in adj.items()}


def append_actions(
    current_state: Tuple[int, ...],
    adj_list: AdjList,
    actions: Iterable[Action],
    module: str,
    full_actions: List[Action],
    module_path: List[str],
    conflict_trace: List[int],
) -> Tuple[int, ...]:
    state = current_state
    for action in actions:
        state = apply_action(state, action)
        full_actions.append(action)
        module_path.append(module)
        conflict_trace.append(count_conflicts(state, adj_list))
    return state


def simulate_hsp2_repair(
    initial_state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    rng: random.Random,
    *,
    max_outer_loops: int,
    max_min_conflicts_steps: int,
) -> RepairTrace:
    current_state = state_to_key(initial_state)
    actions: List[Action] = []
    modules: List[str] = []
    conflict_trace = [count_conflicts(current_state, adj_list)]

    for _ in range(max_outer_loops):
        if is_solved(current_state, adj_list):
            break

        mc_state, mc_actions, _, solved, _ = run_min_conflicts(
            current_state,
            colors,
            adj_list,
            max_steps=max_min_conflicts_steps,
            rng=rng,
        )
        current_state = append_actions(
            current_state,
            adj_list,
            mc_actions,
            "min_conflicts",
            actions,
            modules,
            conflict_trace,
        )
        current_state = mc_state
        if solved or is_solved(current_state, adj_list):
            break

        plan, planned_state, _, _ = run_hsp2_planner(
            current_state,
            colors,
            adj_list,
            rng=rng,
        )
        if not plan:
            break
        current_state = append_actions(
            current_state,
            adj_list,
            plan,
            "hsp2",
            actions,
            modules,
            conflict_trace,
        )
        current_state = planned_state
        if is_solved(current_state, adj_list):
            break

    return RepairTrace(
        agent="hsp2",
        round_id=-1,
        seed=-1,
        success=is_solved(current_state, adj_list),
        actions=tuple(actions),
        module_path=tuple(modules),
        conflict_trace=tuple(conflict_trace),
    )


def simulate_eflop_repair(
    initial_state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    rng: random.Random,
    *,
    max_outer_loops: int,
    max_min_conflicts_steps: int,
    max_eflop_retries: int,
) -> RepairTrace:
    current_state = state_to_key(initial_state)
    actions: List[Action] = []
    modules: List[str] = []
    conflict_trace = [count_conflicts(current_state, adj_list)]

    for _ in range(max_outer_loops):
        if is_solved(current_state, adj_list):
            break

        mc_state, mc_actions, _, solved, _ = run_min_conflicts(
            current_state,
            colors,
            adj_list,
            max_steps=max_min_conflicts_steps,
            rng=rng,
        )
        current_state = append_actions(
            current_state,
            adj_list,
            mc_actions,
            "min_conflicts",
            actions,
            modules,
            conflict_trace,
        )
        current_state = mc_state
        if solved or is_solved(current_state, adj_list):
            break

        current_conflicts = count_conflicts(current_state, adj_list)
        accepted = False
        for _retry in range(max_eflop_retries):
            temp_state, ef_actions, _ = run_eflop(
                current_state,
                colors,
                adj_list,
                rng=rng,
            )
            temp_state2, mc_actions2, _, solved2, _ = run_min_conflicts(
                temp_state,
                colors,
                adj_list,
                max_steps=max_min_conflicts_steps,
                rng=rng,
            )
            temp_conflicts = count_conflicts(temp_state2, adj_list)
            if temp_conflicts < current_conflicts or solved2:
                current_state = append_actions(
                    current_state,
                    adj_list,
                    ef_actions,
                    "eflop",
                    actions,
                    modules,
                    conflict_trace,
                )
                current_state = append_actions(
                    current_state,
                    adj_list,
                    mc_actions2,
                    "min_conflicts",
                    actions,
                    modules,
                    conflict_trace,
                )
                current_state = temp_state2
                accepted = True
                break

        if not accepted:
            break

    return RepairTrace(
        agent="eflop",
        round_id=-1,
        seed=-1,
        success=is_solved(current_state, adj_list),
        actions=tuple(actions),
        module_path=tuple(modules),
        conflict_trace=tuple(conflict_trace),
    )


def simulate_agent(
    agent: AgentName,
    round_data: dict,
    seed: int,
    *,
    max_outer_loops: int,
    max_min_conflicts_steps: int,
    max_eflop_retries: int,
) -> RepairTrace:
    colors = list(range(4))
    adj_list = adjacency_from_round(round_data)
    rng = random.Random(seed)
    if agent == "hsp2":
        trace = simulate_hsp2_repair(
            round_data["initialColors"],
            colors,
            adj_list,
            rng,
            max_outer_loops=max_outer_loops,
            max_min_conflicts_steps=max_min_conflicts_steps,
        )
    elif agent == "eflop":
        trace = simulate_eflop_repair(
            round_data["initialColors"],
            colors,
            adj_list,
            rng,
            max_outer_loops=max_outer_loops,
            max_min_conflicts_steps=max_min_conflicts_steps,
            max_eflop_retries=max_eflop_retries,
        )
    else:
        raise ValueError(f"Unknown agent: {agent}")

    round_id = int(round_data.get("metadata", {}).get("renumberedRound", -1))
    return RepairTrace(
        agent=agent,
        round_id=round_id,
        seed=seed,
        success=trace.success,
        actions=trace.actions,
        module_path=trace.module_path,
        conflict_trace=trace.conflict_trace,
    )


def levenshtein(a: Sequence[Tuple], b: Sequence[Tuple]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def recovery_score(observed: RepairTrace, simulated: RepairTrace) -> float:
    action_dist = levenshtein(observed.actions, simulated.actions)
    module_dist = levenshtein(observed.module_path, simulated.module_path)
    length_penalty = abs(len(observed.actions) - len(simulated.actions))
    success_penalty = 5.0 if observed.success != simulated.success else 0.0
    conflict_penalty = abs(observed.conflict_trace[-1] - simulated.conflict_trace[-1])
    return (
        action_dist
        + 0.35 * module_dist
        + 0.15 * length_penalty
        + success_penalty
        + 0.5 * conflict_penalty
    )


def best_recovery_score(
    observed: RepairTrace,
    candidate_agent: AgentName,
    round_data: dict,
    candidate_seeds: Sequence[int],
    *,
    max_outer_loops: int,
    max_min_conflicts_steps: int,
    max_eflop_retries: int,
) -> Tuple[float, bool, int]:
    best_score = float("inf")
    exact_match = False
    best_seed = -1
    for seed in candidate_seeds:
        simulated = simulate_agent(
            candidate_agent,
            round_data,
            seed,
            max_outer_loops=max_outer_loops,
            max_min_conflicts_steps=max_min_conflicts_steps,
            max_eflop_retries=max_eflop_retries,
        )
        score = recovery_score(observed, simulated)
        if score < best_score:
            best_score = score
            best_seed = seed
        if observed.actions == simulated.actions and observed.module_path == simulated.module_path:
            exact_match = True
            best_score = 0.0
            best_seed = seed
            break
    return best_score, exact_match, best_seed


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=Path,
        default=ROOT / "experiment1" / "public" / "experiment2" / "rounds.json",
    )
    parser.add_argument("--n-sims", type=int, default=20)
    parser.add_argument("--max-outer-loops", type=int, default=50)
    parser.add_argument("--max-min-conflicts-steps", type=int, default=200)
    parser.add_argument("--max-eflop-retries", type=int, default=5)
    parser.add_argument("--round-limit", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "experiment2"
        / "generated_tree_agent_failed_maps"
        / "solver_validation",
    )
    args = parser.parse_args()

    rounds = load_rounds(args.rounds)
    if args.round_limit is not None:
        rounds = rounds[: args.round_limit]
    candidate_seeds = list(range(1, args.n_sims + 1))
    agents = ("hsp2", "eflop")
    detail_rows: List[Dict[str, object]] = []

    for round_index, round_data in enumerate(rounds, start=1):
        round_id = int(round_data.get("metadata", {}).get("renumberedRound", round_index))
        for true_agent in agents:
            obs_seed = 100000 + round_id * 10 + (0 if true_agent == "hsp2" else 1)
            observed = simulate_agent(
                true_agent,
                round_data,
                obs_seed,
                max_outer_loops=args.max_outer_loops,
                max_min_conflicts_steps=args.max_min_conflicts_steps,
                max_eflop_retries=args.max_eflop_retries,
            )
            hsp2_score, hsp2_exact, hsp2_seed = best_recovery_score(
                observed,
                "hsp2",
                round_data,
                candidate_seeds,
                max_outer_loops=args.max_outer_loops,
                max_min_conflicts_steps=args.max_min_conflicts_steps,
                max_eflop_retries=args.max_eflop_retries,
            )
            eflop_score, eflop_exact, eflop_seed = best_recovery_score(
                observed,
                "eflop",
                round_data,
                candidate_seeds,
                max_outer_loops=args.max_outer_loops,
                max_min_conflicts_steps=args.max_min_conflicts_steps,
                max_eflop_retries=args.max_eflop_retries,
            )
            predicted = "hsp2" if hsp2_score <= eflop_score else "eflop"
            detail_rows.append(
                {
                    "round": round_id,
                    "true_agent": true_agent,
                    "predicted_agent": predicted,
                    "correct": int(predicted == true_agent),
                    "observed_seed": obs_seed,
                    "observed_success": int(observed.success),
                    "observed_steps": len(observed.actions),
                    "observed_final_conflicts": observed.conflict_trace[-1],
                    "observed_hsp2_steps": observed.module_path.count("hsp2"),
                    "observed_eflop_steps": observed.module_path.count("eflop"),
                    "score_hsp2": round(hsp2_score, 6),
                    "score_eflop": round(eflop_score, 6),
                    "score_margin_hsp2_minus_eflop": round(hsp2_score - eflop_score, 6),
                    "exact_hsp2": int(hsp2_exact),
                    "exact_eflop": int(eflop_exact),
                    "best_seed_hsp2": hsp2_seed,
                    "best_seed_eflop": eflop_seed,
                }
            )
        print(f"finished round {round_id}/{len(rounds)}", flush=True)

    detail_path = args.out_dir / "hsp2_eflop_full_repair_recovery_detail.csv"
    write_csv(
        detail_path,
        detail_rows,
        [
            "round",
            "true_agent",
            "predicted_agent",
            "correct",
            "observed_seed",
            "observed_success",
            "observed_steps",
            "observed_final_conflicts",
            "observed_hsp2_steps",
            "observed_eflop_steps",
            "score_hsp2",
            "score_eflop",
            "score_margin_hsp2_minus_eflop",
            "exact_hsp2",
            "exact_eflop",
            "best_seed_hsp2",
            "best_seed_eflop",
        ],
    )

    summary_rows: List[Dict[str, object]] = []
    for true_agent in agents:
        rows = [row for row in detail_rows if row["true_agent"] == true_agent]
        summary_rows.append(
            {
                "true_agent": true_agent,
                "n": len(rows),
                "accuracy": round(sum(int(row["correct"]) for row in rows) / len(rows), 6),
                "mean_observed_steps": round(
                    sum(int(row["observed_steps"]) for row in rows) / len(rows), 6
                ),
                "success_rate": round(
                    sum(int(row["observed_success"]) for row in rows) / len(rows), 6
                ),
                "predicted_hsp2": sum(row["predicted_agent"] == "hsp2" for row in rows),
                "predicted_eflop": sum(row["predicted_agent"] == "eflop" for row in rows),
                "exact_hsp2": sum(int(row["exact_hsp2"]) for row in rows),
                "exact_eflop": sum(int(row["exact_eflop"]) for row in rows),
            }
        )
    summary_rows.append(
        {
            "true_agent": "overall",
            "n": len(detail_rows),
            "accuracy": round(
                sum(int(row["correct"]) for row in detail_rows) / len(detail_rows), 6
            ),
            "mean_observed_steps": round(
                sum(int(row["observed_steps"]) for row in detail_rows) / len(detail_rows),
                6,
            ),
            "success_rate": round(
                sum(int(row["observed_success"]) for row in detail_rows) / len(detail_rows),
                6,
            ),
            "predicted_hsp2": sum(row["predicted_agent"] == "hsp2" for row in detail_rows),
            "predicted_eflop": sum(row["predicted_agent"] == "eflop" for row in detail_rows),
            "exact_hsp2": sum(int(row["exact_hsp2"]) for row in detail_rows),
            "exact_eflop": sum(int(row["exact_eflop"]) for row in detail_rows),
        }
    )

    summary_path = args.out_dir / "hsp2_eflop_full_repair_recovery_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "true_agent",
            "n",
            "accuracy",
            "mean_observed_steps",
            "success_rate",
            "predicted_hsp2",
            "predicted_eflop",
            "exact_hsp2",
            "exact_eflop",
        ],
    )
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
