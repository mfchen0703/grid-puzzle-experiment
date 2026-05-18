from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import conflict_search_agent as agent
import experiment2_model_recovery as grid_recovery


PARAMETER_BOUNDS = {
    "pruning_thresh": (0.0, 2.0),
    "gamma": (0.02, 0.5),
    "lapse_rate": (0.0, 0.5),
    "region_preserve": (-2.0, 3.0),
    "color_preserve": (-2.0, 3.0),
}

PLAUSIBLE_BOUNDS = {
    "pruning_thresh": (0.1, 1.2),
    "gamma": (0.04, 0.25),
    "lapse_rate": (0.01, 0.3),
    "region_preserve": (-0.5, 2.0),
    "color_preserve": (-0.5, 2.0),
}

DEFAULT_FIXED_PARAMS = {
    "pruning_thresh": 0.5,
    "gamma": 0.1,
    "lapse_rate": 0.0,
    "region_preserve": 0.0,
    "color_preserve": 0.0,
}

MODE_PARAMETERS = {
    "search": ["pruning_thresh", "gamma", "lapse_rate"],
    "heuristic": ["region_preserve", "color_preserve"],
    "full": ["pruning_thresh", "gamma", "lapse_rate", "region_preserve", "color_preserve"],
}


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def _require_bads():
    try:
        from pybads import BADS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pybads is not installed in the active Python environment. "
            "Install it with `python -m pip install -r requirements.txt`."
        ) from exc
    return BADS


def _observed_by_round(step_df: pd.DataFrame) -> dict[int, list[dict]]:
    valid_df = step_df.copy()
    if "terminated" in valid_df.columns:
        valid_df = valid_df[~valid_df["terminated"].fillna(False)]
    observed: dict[int, list[dict]] = {}
    for round_index, sub_df in valid_df.groupby("round"):
        observed[int(round_index)] = [
            {
                "region": int(row["region"]),
                "new_color": int(row["new_color"]),
            }
            for _, row in sub_df.sort_values("agent_step").iterrows()
        ]
    return observed


def _clip_params(params: dict[str, float]) -> dict[str, float]:
    out = {}
    for name, value in params.items():
        lo, hi = PARAMETER_BOUNDS[name]
        out[name] = float(np.clip(float(value), lo, hi))
    return out


def parameter_dict_from_vector(
    x: np.ndarray,
    parameter_names: list[str],
    fixed_params: dict[str, float] | None = None,
) -> dict[str, float]:
    params = dict(DEFAULT_FIXED_PARAMS)
    if fixed_params:
        params.update({key: float(value) for key, value in fixed_params.items()})
    for name, value in zip(parameter_names, np.asarray(x, dtype=float).ravel()):
        params[name] = float(value)
    return _clip_params(params)


def _single_pass_round_nll(
    step_df: pd.DataFrame,
    params: dict[str, float],
    max_depth: int = 4,
    epsilon: float = 1e-9,
    random_tie_break: bool = False,
    rng_seed: int | None = None,
) -> tuple[dict[int, float], int]:
    params = parameter_dict_from_vector(
        np.array([params[name] for name in MODE_PARAMETERS["full"]], dtype=float),
        MODE_PARAMETERS["full"],
    )
    heuristic_weights = {
        "region_preserve": float(params["region_preserve"]),
        "color_preserve": float(params["color_preserve"]),
    }
    n_iterations = agent.iterations_from_gamma(float(params["gamma"]))
    observed = _observed_by_round(step_df)
    materials = agent.load_materials()

    round_nll: dict[int, float] = {}
    n_actions = 0
    for round_index, round_data in enumerate(materials["rounds"], start=1):
        observed_actions = observed.get(round_index, [])
        if not observed_actions:
            continue

        adjacency = agent.build_adjacency_map(round_data["mapData"])
        root = agent.build_tree_node(adjacency, tuple(int(c) for c in round_data["initialColors"]))
        round_total_ll = 0.0
        round_rng = (
            random.Random(int(rng_seed) + round_index * 1000003)
            if (random_tie_break and rng_seed is not None)
            else None
        )

        for obs_action in observed_actions:
            (
                _best_children,
                tree_action_probs,
                _evaluated_actions,
                _conflict_edges,
                _conflict_regions,
                _layers,
                root,
            ) = agent.tree_policy_from_root(
                root,
                adjacency,
                max_depth=max_depth,
                n_iterations=n_iterations,
                pruning_thresh=float(params["pruning_thresh"]),
                heuristic_weights=heuristic_weights,
                random_tie_break=random_tie_break,
                rng=round_rng,
            )

            action_key = (int(obs_action["region"]), int(obs_action["new_color"]))
            if not root.children:
                tree_prob = 0.0
                lapse_prob = 0.0
            else:
                tree_prob = float(tree_action_probs.get(action_key, 0.0))
                child_keys = [agent.action_key(child.action_from_parent) for child in root.children]
                lapse_prob = (1.0 / len(child_keys)) if action_key in child_keys else 0.0

            action_prob = (1.0 - float(params["lapse_rate"])) * tree_prob
            action_prob += float(params["lapse_rate"]) * lapse_prob
            round_total_ll += float(np.log(max(action_prob, epsilon)))
            n_actions += 1

            chosen_child = None
            for child in root.children:
                if agent.action_key(child.action_from_parent) == action_key:
                    chosen_child = child
                    break

            if chosen_child is None:
                root = agent.build_tree_node(adjacency, agent.apply_action(root.state, obs_action))
            else:
                root = chosen_child
                root.parent = None
                agent.reset_subtree_depths(root, depth=0)

        round_nll[round_index] = float(-round_total_ll)

    return round_nll, int(n_actions)


def negative_log_likelihood(
    step_df: pd.DataFrame,
    params: dict[str, float],
    max_depth: int = 4,
    epsilon: float = 1e-9,
    round_repeats: dict[int, int] | None = None,
    random_tie_break: bool = False,
    base_seed: int = 12345,
) -> tuple[float, int]:
    if not round_repeats:
        round_nll, n_actions = _single_pass_round_nll(
            step_df,
            params,
            max_depth=max_depth,
            epsilon=epsilon,
            random_tie_break=random_tie_break,
            rng_seed=base_seed,
        )
        return float(sum(round_nll.values())), int(n_actions)

    max_repeats = max(1, max(int(v) for v in round_repeats.values()))
    stacked: dict[int, list[float]] = {int(k): [] for k in round_repeats}
    n_actions_out = 0
    for rep in range(max_repeats):
        round_nll, n_actions = _single_pass_round_nll(
            step_df,
            params,
            max_depth=max_depth,
            epsilon=epsilon,
            random_tie_break=random_tie_break,
            rng_seed=int(base_seed) + rep * 7919,
        )
        n_actions_out = n_actions
        for round_index, repeat_count in round_repeats.items():
            if rep < int(repeat_count):
                stacked[int(round_index)].append(float(round_nll.get(int(round_index), 0.0)))

    total_nll = 0.0
    for values in stacked.values():
        if values:
            total_nll += float(np.mean(values))
    return float(total_nll), int(n_actions_out)


def estimate_round_noise(
    step_df: pd.DataFrame,
    params: dict[str, float],
    max_depth: int = 4,
    n_preliminary: int = 8,
    epsilon: float = 1e-9,
    base_seed: int = 10000,
) -> dict[int, float]:
    per_round_values: dict[int, list[float]] = {}
    for i in range(max(1, int(n_preliminary))):
        round_nll, _ = _single_pass_round_nll(
            step_df,
            params,
            max_depth=max_depth,
            epsilon=epsilon,
            random_tie_break=True,
            rng_seed=int(base_seed) + i * 3571,
        )
        for round_index, value in round_nll.items():
            per_round_values.setdefault(int(round_index), []).append(float(value))

    round_std = {}
    for round_index, values in per_round_values.items():
        if len(values) <= 1:
            round_std[round_index] = 0.0
        else:
            round_std[round_index] = float(np.std(values, ddof=1))
    return round_std


def build_adaptive_round_repeats(
    round_std: dict[int, float],
    min_repeats: int = 1,
    max_repeats: int = 4,
) -> dict[int, int]:
    min_repeats = max(1, int(min_repeats))
    max_repeats = max(min_repeats, int(max_repeats))
    if not round_std:
        return {}

    std_values = np.array([float(v) for v in round_std.values()], dtype=float)
    positive = std_values[std_values > 0]
    if len(positive) == 0:
        return {int(round_index): min_repeats for round_index in round_std}

    avg_std = float(np.mean(positive))
    repeats: dict[int, int] = {}
    for round_index, std_value in round_std.items():
        scaled = float(std_value) / max(avg_std, 1e-9)
        proposed = int(np.rint(min_repeats * scaled))
        repeats[int(round_index)] = int(np.clip(proposed, min_repeats, max_repeats))
    return repeats


def build_bads_arrays(parameter_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in parameter_names], dtype=float)
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in parameter_names], dtype=float)
    plausible_lower = np.array([PLAUSIBLE_BOUNDS[name][0] for name in parameter_names], dtype=float)
    plausible_upper = np.array([PLAUSIBLE_BOUNDS[name][1] for name in parameter_names], dtype=float)
    x0 = (plausible_lower + plausible_upper) / 2.0
    return x0, lower, upper, plausible_lower, plausible_upper


def fit_synthetic_participant_bads(
    step_df: pd.DataFrame,
    parameter_names: list[str] | None = None,
    fixed_params: dict[str, float] | None = None,
    max_depth: int = 4,
    bads_options: dict[str, Any] | None = None,
    n_preliminary: int = 8,
    min_round_repeats: int = 1,
    max_round_repeats: int = 4,
    final_reevaluations: int = 10,
    noisy_objective: bool = True,
    preeval_seed: int = 10000,
) -> tuple[dict, pd.DataFrame]:
    BADS = _require_bads()
    parameter_names = list(MODE_PARAMETERS["search"] if parameter_names is None else parameter_names)
    fixed_params = {} if fixed_params is None else dict(fixed_params)
    x0, lower, upper, plausible_lower, plausible_upper = build_bads_arrays(parameter_names)
    evaluations: list[dict[str, float]] = []
    x0_params = parameter_dict_from_vector(x0, parameter_names, fixed_params=fixed_params)

    round_repeats: dict[int, int] | None
    preeval_round_std: dict[int, float] = {}
    if noisy_objective:
        preeval_round_std = estimate_round_noise(
            step_df,
            x0_params,
            max_depth=max_depth,
            n_preliminary=n_preliminary,
            base_seed=preeval_seed,
        )
        round_repeats = build_adaptive_round_repeats(
            preeval_round_std,
            min_repeats=min_round_repeats,
            max_repeats=max_round_repeats,
        )
    else:
        round_repeats = None

    eval_count = 0

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        params = parameter_dict_from_vector(x, parameter_names, fixed_params=fixed_params)
        nll, n_actions = negative_log_likelihood(
            step_df,
            params,
            max_depth=max_depth,
            round_repeats=round_repeats,
            random_tie_break=noisy_objective,
            base_seed=preeval_seed + eval_count * 1009,
        )
        row = {name: float(params[name]) for name in MODE_PARAMETERS["full"]}
        row["nll"] = float(nll)
        row["ll"] = float(-nll)
        row["n_actions"] = int(n_actions)
        row["eval_index"] = int(eval_count)
        evaluations.append(row)
        return float(nll)

    options = {
        "display": "off",
        "uncertainty_handling": bool(noisy_objective),
    }
    if bads_options:
        options.update(bads_options)

    bads = BADS(objective, x0, lower, upper, plausible_lower, plausible_upper, options=options)
    result = bads.optimize()
    best_x = np.asarray(result.get("x", result.get("x0", x0)), dtype=float).ravel()
    best_params = parameter_dict_from_vector(best_x, parameter_names, fixed_params=fixed_params)

    final_nll_samples: list[float] = []
    n_actions = 0
    for i in range(max(1, int(final_reevaluations))):
        nll_i, n_actions = negative_log_likelihood(
            step_df,
            best_params,
            max_depth=max_depth,
            round_repeats=round_repeats,
            random_tie_break=noisy_objective,
            base_seed=preeval_seed + 500000 + i * 8011,
        )
        final_nll_samples.append(float(nll_i))
    best_nll = float(np.mean(final_nll_samples))

    best = {name: float(best_params[name]) for name in MODE_PARAMETERS["full"]}
    best["ll"] = float(-best_nll)
    best["nll"] = float(best_nll)
    best["nll_std"] = float(np.std(final_nll_samples, ddof=1)) if len(final_nll_samples) > 1 else 0.0
    best["final_reevaluations"] = int(max(1, int(final_reevaluations)))
    best["n_actions"] = int(n_actions)
    best["converged"] = bool(result.get("success", True))
    best["bads_message"] = str(result.get("message", ""))
    best["n_evaluations"] = int(len(evaluations))
    best["adaptive_round_repeats"] = {int(k): int(v) for k, v in (round_repeats or {}).items()}
    best["preeval_round_std"] = {int(k): float(v) for k, v in preeval_round_std.items()}

    candidate_df = pd.DataFrame(evaluations)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values("nll").reset_index(drop=True)
    return best, candidate_df


def _run_single_bads_recovery_task(
    task: dict,
    parameter_names: list[str],
    fixed_params: dict[str, float],
    max_depth: int,
    bads_options: dict[str, Any] | None,
    n_preliminary: int,
    min_round_repeats: int,
    max_round_repeats: int,
    final_reevaluations: int,
    noisy_objective: bool,
    preeval_seed: int,
) -> tuple[dict, pd.DataFrame]:
    task_id = int(task["task_id"])
    _, step_df = grid_recovery.simulate_synthetic_participant(
        pruning_thresh=float(task["true_pruning_thresh"]),
        gamma=float(task["true_gamma"]),
        lapse_rate=float(task["true_lapse_rate"]),
        random_seed=int(task["random_seed"]),
        region_preserve=float(task.get("true_region_preserve", 0.0)),
        color_preserve=float(task.get("true_color_preserve", 0.0)),
        max_depth=max_depth,
    )
    best, candidate_df = fit_synthetic_participant_bads(
        step_df=step_df,
        parameter_names=parameter_names,
        fixed_params=fixed_params,
        max_depth=max_depth,
        bads_options=bads_options,
        n_preliminary=n_preliminary,
        min_round_repeats=min_round_repeats,
        max_round_repeats=max_round_repeats,
        final_reevaluations=final_reevaluations,
        noisy_objective=noisy_objective,
        preeval_seed=preeval_seed + task_id * 100003,
    )
    row = {
        "task_id": task_id,
        "random_seed": int(task["random_seed"]),
        "true_pruning_thresh": float(task["true_pruning_thresh"]),
        "true_gamma": float(task["true_gamma"]),
        "true_lapse_rate": float(task["true_lapse_rate"]),
        "true_region_preserve": float(task.get("true_region_preserve", 0.0)),
        "true_color_preserve": float(task.get("true_color_preserve", 0.0)),
        "hat_pruning_thresh": float(best["pruning_thresh"]),
        "hat_gamma": float(best["gamma"]),
        "hat_lapse_rate": float(best["lapse_rate"]),
        "hat_region_preserve": float(best["region_preserve"]),
        "hat_color_preserve": float(best["color_preserve"]),
        "ll": float(best["ll"]),
        "nll": float(best["nll"]),
        "n_actions": int(best["n_actions"]),
        "n_evaluations": int(best["n_evaluations"]),
        "nll_std": float(best["nll_std"]),
        "final_reevaluations": int(best["final_reevaluations"]),
        "adaptive_round_repeats": str(best.get("adaptive_round_repeats", {})),
        "converged": bool(best["converged"]),
        "bads_message": str(best["bads_message"]),
    }
    return row, candidate_df


def run_bads_recovery_parallel(
    tasks_df: pd.DataFrame,
    parameter_names: list[str],
    fixed_params: dict[str, float] | None = None,
    max_depth: int = 4,
    n_workers: int | None = None,
    output_prefix: str | Path | None = None,
    save_every: int = 1,
    bads_options: dict[str, Any] | None = None,
    n_preliminary: int = 8,
    min_round_repeats: int = 1,
    max_round_repeats: int = 4,
    final_reevaluations: int = 10,
    noisy_objective: bool = True,
    preeval_seed: int = 10000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, int(n_workers))
    fixed_params = {} if fixed_params is None else dict(fixed_params)
    output_prefix = (
        default_results_dir() / "experiment2_bads_recovery"
        if output_prefix is None
        else Path(output_prefix)
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_prefix.with_name(f"{output_prefix.name}_partial.csv")
    results_path = output_prefix.with_name(f"{output_prefix.name}_results.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    candidates_path = output_prefix.with_name(f"{output_prefix.name}_candidates.csv")

    result_rows: list[dict] = []
    candidate_tables: dict[int, pd.DataFrame] = {}
    task_records = tasks_df.to_dict("records")

    if n_workers == 1:
        for completed, task in enumerate(task_records, start=1):
            row, candidate_df = _run_single_bads_recovery_task(
                task,
                parameter_names,
                fixed_params,
                max_depth,
                bads_options,
                n_preliminary,
                min_round_repeats,
                max_round_repeats,
                final_reevaluations,
                noisy_objective,
                preeval_seed,
            )
            task_id = int(row["task_id"])
            result_rows.append(row)
            candidate_tables[task_id] = candidate_df
            if verbose:
                print(f"completed {completed}/{len(task_records)} task_id={task_id}", flush=True)
            if save_every > 0 and completed % int(save_every) == 0:
                pd.DataFrame(result_rows).sort_values("task_id").to_csv(
                    partial_path,
                    index=False,
                    encoding="utf-8-sig",
                )

        recovery_df = pd.DataFrame(result_rows).sort_values("task_id").reset_index(drop=True)
        recovery_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        grid_recovery.summarize_recovery(recovery_df).to_csv(summary_path, index=False, encoding="utf-8-sig")
        if candidate_tables:
            pd.concat(
                [table.assign(task_id=task_id) for task_id, table in sorted(candidate_tables.items())],
                ignore_index=True,
            ).to_csv(candidates_path, index=False, encoding="utf-8-sig")
        return recovery_df, candidate_tables

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_single_bads_recovery_task,
                task,
                parameter_names,
                fixed_params,
                max_depth,
                bads_options,
                n_preliminary,
                min_round_repeats,
                max_round_repeats,
                final_reevaluations,
                noisy_objective,
                preeval_seed,
            ): int(task["task_id"])
            for task in task_records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row, candidate_df = future.result()
            task_id = int(row["task_id"])
            result_rows.append(row)
            candidate_tables[task_id] = candidate_df
            if verbose:
                print(f"completed {completed}/{len(task_records)} task_id={task_id}", flush=True)
            if save_every > 0 and completed % int(save_every) == 0:
                pd.DataFrame(result_rows).sort_values("task_id").to_csv(
                    partial_path,
                    index=False,
                    encoding="utf-8-sig",
                )

    recovery_df = pd.DataFrame(result_rows).sort_values("task_id").reset_index(drop=True)
    recovery_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    grid_recovery.summarize_recovery(recovery_df).to_csv(summary_path, index=False, encoding="utf-8-sig")
    if candidate_tables:
        pd.concat(
            [table.assign(task_id=task_id) for task_id, table in sorted(candidate_tables.items())],
            ignore_index=True,
        ).to_csv(candidates_path, index=False, encoding="utf-8-sig")
    return recovery_df, candidate_tables
