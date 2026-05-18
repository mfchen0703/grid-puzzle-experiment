"""Model parameter recovery for BFS tree agent using Inverse Binomial Sampling (IBS).

Follows the same approach as fourinarow: stochastic model (random tie-break
seeds), repeated runs per observed action, estimate log-likelihood via IBS.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import conflict_search_agent as agent


# Default parameter grids — small to keep fitting tractable.
DEFAULT_PRUNING_GRID: list[float] = [0.0, 1.0, 2.0, 5.0]
DEFAULT_N_ITERATIONS_GRID: list[int | None] = [None]
DEFAULT_GAMMA_GRID: list[float | None] = [None, 0.05, 0.1, 0.2, 0.5]
DEFAULT_LAPSE_GRID = [0.0, 0.1]
DEFAULT_HEURISTIC_EVAL_GRID = [0.0]


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------


def build_parameter_grid(
    pruning_grid: list[float] | None = None,
    n_iterations_grid: list[int | None] | None = None,
    gamma_grid: list[float | None] | None = None,
    lapse_grid: list[float] | None = None,
    heuristic_eval_grid: list[float] | None = None,
) -> pd.DataFrame:
    pruning_grid = DEFAULT_PRUNING_GRID if pruning_grid is None else list(pruning_grid)
    n_iterations_grid = (
        DEFAULT_N_ITERATIONS_GRID if n_iterations_grid is None else list(n_iterations_grid)
    )
    gamma_grid = DEFAULT_GAMMA_GRID if gamma_grid is None else list(gamma_grid)
    lapse_grid = DEFAULT_LAPSE_GRID if lapse_grid is None else list(lapse_grid)
    heuristic_eval_grid = (
        DEFAULT_HEURISTIC_EVAL_GRID if heuristic_eval_grid is None else list(heuristic_eval_grid)
    )

    rows = []
    for pruning_thresh in pruning_grid:
        for n_iterations in n_iterations_grid:
            for gamma in gamma_grid:
                for lapse_rate in lapse_grid:
                    for heuristic_eval_weight in heuristic_eval_grid:
                        rows.append(
                            {
                                "pruning_thresh": float(pruning_thresh),
                                "n_iterations": (
                                    int(n_iterations)
                                    if n_iterations is not None
                                    else None
                                ),
                                "gamma": float(gamma) if gamma is not None else None,
                                "lapse_rate": float(lapse_rate),
                                "heuristic_eval_weight": float(heuristic_eval_weight),
                            }
                        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic participant simulation
# ---------------------------------------------------------------------------


def simulate_synthetic_participant(
    pruning_thresh: float,
    n_iterations: int | None,
    lapse_rate: float,
    random_seed: int,
    heuristic_eval_weight: float = 0.0,
    gamma: float | None = None,
    max_depth: int = 8,
    max_expansions: int = 5000,
    rounds_subset: list[int] | None = None,
) -> pd.DataFrame:
    """Run BFS tree agent on all rounds and return per-step action data."""
    materials = agent.load_materials()
    step_rows: list[dict] = []
    rng = random.Random(int(random_seed))

    for round_index, round_data in enumerate(materials["rounds"], start=1):
        if rounds_subset is not None and round_index not in rounds_subset:
            continue
        trajectory, final_colors = agent.run_bfs_tree_agent_on_round(
            round_data,
            max_steps=500,
            max_depth=max_depth,
            max_expansions=max_expansions,
            n_iterations=n_iterations,
            pruning_thresh=pruning_thresh,
            lapse_rate=lapse_rate,
            heuristic_eval_weight=heuristic_eval_weight,
            gamma=gamma,
            random_tie_break=True,
            rng=rng,
            tree_score_strategy="task_first",
        )
        for step in trajectory:
            if step.get("terminated", False):
                continue
            step_rows.append(
                {
                    "round": round_index,
                    "agent_step": int(step["agent_step"]),
                    "region": int(step["region"]),
                    "old_color": int(step["old_color"]),
                    "new_color": int(step["new_color"]),
                    "n_conflict_edges_before": int(step["n_conflict_edges_before"]),
                }
            )
    df = pd.DataFrame(step_rows)
    for col in [
        "true_pruning_thresh",
        "true_n_iterations",
        "true_gamma",
        "true_lapse_rate",
        "true_heuristic_eval_weight",
    ]:
        df[col] = np.nan
    df["true_pruning_thresh"] = float(pruning_thresh)
    df["true_n_iterations"] = (
        int(n_iterations) if n_iterations is not None else np.nan
    )
    df["true_gamma"] = float(gamma) if gamma is not None else np.nan
    df["true_lapse_rate"] = float(lapse_rate)
    df["true_heuristic_eval_weight"] = float(heuristic_eval_weight)
    df["random_seed"] = int(random_seed)
    return df


# ---------------------------------------------------------------------------
# IBS log-likelihood
# ---------------------------------------------------------------------------

_IBS_MAX_TRIES = 2000  # safety cap per observed action


def ibs_loglik_for_actions(
    round_data_list: list[dict],
    observed_actions: list[dict],  # list of {round, agent_step, region, new_color}
    pruning_thresh: float,
    n_iterations: int | None,
    lapse_rate: float,
    heuristic_eval_weight: float = 0.0,
    gamma: float | None = None,
    max_depth: int = 8,
    max_expansions: int = 5000,
    ibs_samples: int = 20,
    base_seed: int = 1,
) -> float:
    """Estimate total negative log-likelihood via IBS for a sequence of actions.

    For each observed action, repeatedly run the BFS agent from the same
    preceding state with different random seeds until the observed action
    is matched ``ibs_samples`` times.  The per-action likelihood 1/p is
    estimated as tries / successes (negative binomial MLE).
    """
    materials = agent.load_materials()

    # Group observed actions by round
    actions_by_round: dict[int, list[dict]] = {}
    for obs in observed_actions:
        actions_by_round.setdefault(int(obs["round"]), []).append(obs)

    total_nll = 0.0

    for round_index, actions in sorted(actions_by_round.items()):
        round_data = materials["rounds"][round_index - 1]
        adjacency = agent.build_adjacency_map(round_data["mapData"])
        current_state = tuple(int(c) for c in round_data["initialColors"])

        for obs in sorted(actions, key=lambda a: int(a["agent_step"])):
            # Advance state to match the step where this action was taken
            # (actions are in order, so previous actions have already been applied.
            #  But we need to re-apply all previous actions to reach the correct state.)
            # Actually, since we process sequentially within the round, we've already
            # applied previous actions. We just need to verify state consistency.

            target_region = int(obs["region"])
            target_color = int(obs["new_color"])

            times_left = int(ibs_samples)
            tries = 0
            action_nll = 0.0

            while times_left > 0 and tries < _IBS_MAX_TRIES:
                tries += 1
                seed = base_seed + tries
                rng = __import__("random").Random(seed)
                trajectory, _ = agent.run_bfs_tree_agent_on_round(
                    {"mapData": round_data["mapData"], "initialColors": list(current_state)},
                    max_steps=500,
                    max_depth=max_depth,
                    max_expansions=max_expansions,
                    n_iterations=n_iterations,
                    pruning_thresh=pruning_thresh,
                    lapse_rate=lapse_rate,
                    heuristic_eval_weight=heuristic_eval_weight,
                    gamma=gamma,
                    random_tie_break=True,
                    rng=rng,
                    tree_score_strategy="task_first",
                )

                if not trajectory:
                    action_nll += 1.0 / (tries * ibs_samples)
                    continue

                first_action = trajectory[0]
                if first_action.get("terminated", False):
                    action_nll += 1.0 / (tries * ibs_samples)
                    continue

                if (
                    int(first_action["region"]) == target_region
                    and int(first_action["new_color"]) == target_color
                ):
                    times_left -= 1
                else:
                    action_nll += 1.0 / (tries * ibs_samples)

            if times_left > 0:
                action_nll += times_left * 3.5  # penalty for unmatched

            total_nll += action_nll

            # Advance to next state by applying the observed action
            current_state = agent.apply_action(
                current_state,
                {"region": target_region, "old_color": current_state[target_region], "new_color": target_color},
            )

    return total_nll


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_synthetic_participant(
    step_df: pd.DataFrame,
    pruning_grid: list[float] | None = None,
    n_iterations_grid: list[int | None] | None = None,
    gamma_grid: list[float | None] | None = None,
    lapse_grid: list[float] | None = None,
    heuristic_eval_grid: list[float] | None = None,
    max_depth: int = 8,
    ibs_samples: int = 20,
    only_first_action: bool = True,
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Fit BFS parameters by exhaustive grid search with IBS log-likelihood."""
    parameter_grid = build_parameter_grid(
        pruning_grid, n_iterations_grid, gamma_grid, lapse_grid, heuristic_eval_grid
    )

    # Extract observed actions
    step_df = step_df.copy()
    if "terminated" in step_df.columns:
        step_df = step_df[~step_df.get("terminated", pd.Series(False)).fillna(False)]
    step_df = step_df.sort_values(["round", "agent_step"])

    if only_first_action:
        step_df = step_df.groupby("round").first().reset_index()

    observed_actions = [
        {
            "round": int(row["round"]),
            "agent_step": int(row["agent_step"]),
            "region": int(row["region"]),
            "new_color": int(row["new_color"]),
        }
        for _, row in step_df.iterrows()
    ]

    candidate_rows = []
    for _, params in parameter_grid.iterrows():
        pruning_thresh = float(params["pruning_thresh"])

        n_iterations = params["n_iterations"]
        if isinstance(n_iterations, float) and np.isnan(n_iterations):
            n_iterations = None
        elif n_iterations is not None:
            n_iterations = int(n_iterations)

        gamma = params["gamma"]
        if isinstance(gamma, float) and np.isnan(gamma):
            gamma = None
        elif gamma is not None:
            gamma = float(gamma)

        nll = ibs_loglik_for_actions(
            round_data_list=[],
            observed_actions=observed_actions,
            pruning_thresh=pruning_thresh,
            n_iterations=n_iterations,
            lapse_rate=float(params["lapse_rate"]),
            heuristic_eval_weight=float(params["heuristic_eval_weight"]),
            gamma=gamma,
            max_depth=max_depth,
            ibs_samples=ibs_samples,
        )

        if verbose:
            print(
                f"  pruning={pruning_thresh} n_iter={n_iterations} "
                f"gamma={gamma} lapse={params['lapse_rate']} "
                f"heval={params['heuristic_eval_weight']} "
                f"nll={nll:.3f}",
                flush=True,
            )

        candidate_rows.append(
            {
                "pruning_thresh": (
                    float(pruning_thresh) if pruning_thresh is not None else None
                ),
                "n_iterations": n_iterations,
                "gamma": gamma,
                "lapse_rate": float(params["lapse_rate"]),
                "heuristic_eval_weight": float(params["heuristic_eval_weight"]),
                "nll": float(nll),
                "n_actions": len(observed_actions),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["nll", "pruning_thresh", "n_iterations", "gamma", "lapse_rate", "heuristic_eval_weight"],
        na_position="last",
    ).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()
    best["converged"] = True
    return best, candidate_df


# ---------------------------------------------------------------------------
# Recovery tasks
# ---------------------------------------------------------------------------


def build_recovery_tasks(
    pruning_grid: list[float] | None = None,
    n_iterations_grid: list[int | None] | None = None,
    gamma_grid: list[float | None] | None = None,
    lapse_grid: list[float] | None = None,
    heuristic_eval_grid: list[float] | None = None,
    n_repeats: int = 1,
    rounds_subset: list[int] | None = None,
) -> pd.DataFrame:
    grid_df = build_parameter_grid(
        pruning_grid, n_iterations_grid, gamma_grid, lapse_grid, heuristic_eval_grid
    )
    tasks = []
    task_id = 0
    for _, row in grid_df.iterrows():
        for repeat in range(int(n_repeats)):
            tasks.append(
                {
                    "task_id": int(task_id),
                    "true_pruning_thresh": (
                        float(row["pruning_thresh"])
                    ),
                    "true_n_iterations": (
                        int(row["n_iterations"])
                        if row["n_iterations"] is not None and not (isinstance(row["n_iterations"], float) and np.isnan(row["n_iterations"]))
                        else None
                    ),
                    "true_gamma": (
                        float(row["gamma"])
                        if row["gamma"] is not None and not (isinstance(row["gamma"], float) and np.isnan(row["gamma"]))
                        else None
                    ),
                    "true_lapse_rate": float(row["lapse_rate"]),
                    "true_heuristic_eval_weight": float(row["heuristic_eval_weight"]),
                    "random_seed": int(repeat + 1),
                    "rounds_subset": rounds_subset,
                }
            )
            task_id += 1
    return pd.DataFrame(tasks)


def _run_single_recovery_task(
    task: dict,
    fit_pruning_grid: list[float] | None,
    fit_n_iterations_grid: list[int | None] | None,
    fit_gamma_grid: list[float | None] | None,
    fit_lapse_grid: list[float] | None,
    fit_heuristic_eval_grid: list[float] | None,
    max_depth: int,
    ibs_samples: int,
    only_first_action: bool,
) -> tuple[dict, pd.DataFrame]:
    def _resolve(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        return val

    step_df = simulate_synthetic_participant(
        pruning_thresh=_resolve(task["true_pruning_thresh"]),
        n_iterations=_resolve(task["true_n_iterations"]),
        lapse_rate=float(task["true_lapse_rate"]),
        heuristic_eval_weight=float(task.get("true_heuristic_eval_weight", 0.0)),
        gamma=_resolve(task.get("true_gamma")),
        random_seed=int(task["random_seed"]),
        max_depth=max_depth,
        rounds_subset=task.get("rounds_subset"),
    )
    best, candidate_df = fit_synthetic_participant(
        step_df=step_df,
        pruning_grid=fit_pruning_grid,
        n_iterations_grid=fit_n_iterations_grid,
        gamma_grid=fit_gamma_grid,
        lapse_grid=fit_lapse_grid,
        heuristic_eval_grid=fit_heuristic_eval_grid,
        max_depth=max_depth,
        ibs_samples=ibs_samples,
        only_first_action=only_first_action,
    )
    row = {
        "task_id": int(task["task_id"]),
        "random_seed": int(task["random_seed"]),
        "true_pruning_thresh": task["true_pruning_thresh"],
        "true_n_iterations": task["true_n_iterations"],
        "true_gamma": task.get("true_gamma"),
        "true_lapse_rate": float(task["true_lapse_rate"]),
        "true_heuristic_eval_weight": float(task.get("true_heuristic_eval_weight", 0.0)),
        "hat_pruning_thresh": best["pruning_thresh"],
        "hat_n_iterations": best["n_iterations"],
        "hat_gamma": best["gamma"],
        "hat_lapse_rate": float(best["lapse_rate"]),
        "hat_heuristic_eval_weight": float(best["heuristic_eval_weight"]),
        "nll": float(best["nll"]),
        "n_actions": int(best["n_actions"]),
        "converged": bool(best["converged"]),
    }
    return row, candidate_df


def run_parameter_recovery(
    tasks_df: pd.DataFrame,
    fit_pruning_grid: list[float] | None = None,
    fit_n_iterations_grid: list[int | None] | None = None,
    fit_gamma_grid: list[float | None] | None = None,
    fit_lapse_grid: list[float] | None = None,
    fit_heuristic_eval_grid: list[float] | None = None,
    max_depth: int = 8,
    ibs_samples: int = 20,
    only_first_action: bool = True,
    save_every: int | None = None,
    output_prefix: str | Path | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    result_rows = []
    candidate_tables: dict[int, pd.DataFrame] = {}
    output_prefix = Path(output_prefix) if output_prefix is not None else None

    for idx, (_, task) in enumerate(tasks_df.iterrows(), start=1):
        row, candidate_df = _run_single_recovery_task(
            task,
            fit_pruning_grid,
            fit_n_iterations_grid,
            fit_gamma_grid,
            fit_lapse_grid,
            fit_heuristic_eval_grid,
            max_depth,
            ibs_samples,
            only_first_action,
        )
        result_rows.append(row)
        candidate_tables[int(task["task_id"])] = candidate_df

        if verbose:
            print(f"[{idx}/{len(tasks_df)}] task_id={task['task_id']}", flush=True)

        if output_prefix is not None and save_every is not None and idx % int(save_every) == 0:
            pd.DataFrame(result_rows).to_csv(
                output_prefix.with_name(f"{output_prefix.name}_partial.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    recovery_df = pd.DataFrame(result_rows)
    if output_prefix is not None:
        recovery_df.to_csv(
            output_prefix.with_name(f"{output_prefix.name}_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        summarize_recovery(recovery_df).to_csv(
            output_prefix.with_name(f"{output_prefix.name}_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    return recovery_df, candidate_tables


def run_parameter_recovery_parallel(
    tasks_df: pd.DataFrame,
    fit_pruning_grid: list[float] | None = None,
    fit_n_iterations_grid: list[int | None] | None = None,
    fit_gamma_grid: list[float | None] | None = None,
    fit_lapse_grid: list[float] | None = None,
    fit_heuristic_eval_grid: list[float] | None = None,
    max_depth: int = 8,
    ibs_samples: int = 20,
    only_first_action: bool = True,
    n_workers: int | None = None,
    save_every: int = 1,
    output_prefix: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, int(n_workers))
    output_prefix = (
        default_results_dir() / "experiment2_ibs_recovery"
        if output_prefix is None
        else Path(output_prefix)
    )

    result_rows: list[dict] = []
    candidate_tables: dict[int, pd.DataFrame] = {}
    task_records = tasks_df.to_dict("records")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_single_recovery_task,
                task,
                fit_pruning_grid,
                fit_n_iterations_grid,
                fit_gamma_grid,
                fit_lapse_grid,
                fit_heuristic_eval_grid,
                max_depth,
                ibs_samples,
                only_first_action,
            ): int(task["task_id"])
            for task in task_records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row, candidate_df = future.result()
            result_rows.append(row)
            candidate_tables[int(row["task_id"])] = candidate_df
            if verbose:
                print(f"completed {completed}/{len(task_records)} task_id={row['task_id']}", flush=True)
            if save_every > 0 and completed % int(save_every) == 0:
                pd.DataFrame(result_rows).sort_values("task_id").to_csv(
                    output_prefix.with_name(f"{output_prefix.name}_partial.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

    recovery_df = pd.DataFrame(result_rows).sort_values("task_id").reset_index(drop=True)
    recovery_df.to_csv(
        output_prefix.with_name(f"{output_prefix.name}_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summarize_recovery(recovery_df).to_csv(
        output_prefix.with_name(f"{output_prefix.name}_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return recovery_df, candidate_tables


def summarize_recovery(recovery_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param_name in [
        "pruning_thresh",
        "n_iterations",
        "gamma",
        "lapse_rate",
        "heuristic_eval_weight",
    ]:
        true_col = f"true_{param_name}"
        hat_col = f"hat_{param_name}"
        if true_col not in recovery_df.columns or hat_col not in recovery_df.columns:
            continue
        valid = recovery_df[[true_col, hat_col]].dropna()
        if len(valid) > 1:
            corr = valid.corr().iloc[0, 1]
        else:
            corr = np.nan
        mae = float((valid[true_col] - valid[hat_col]).abs().mean()) if len(valid) > 0 else np.nan
        rows.append(
            {
                "parameter": param_name,
                "correlation": float(corr) if pd.notna(corr) else np.nan,
                "mae": mae,
                "n_valid": len(valid),
            }
        )
    return pd.DataFrame(rows)
