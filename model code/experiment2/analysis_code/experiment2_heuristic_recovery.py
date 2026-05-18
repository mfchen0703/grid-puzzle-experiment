from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import conflict_search_agent as agent


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_REGION_PRESERVE_GRID = [0.0, 0.5]
DEFAULT_COLOR_PRESERVE_GRID = [0.0, 0.1]


def build_heuristic_grid(
    region_preserve_grid: list[float] | None = None,
    color_preserve_grid: list[float] | None = None,
) -> pd.DataFrame:
    region_preserve_grid = (
        DEFAULT_REGION_PRESERVE_GRID if region_preserve_grid is None else list(region_preserve_grid)
    )
    color_preserve_grid = (
        DEFAULT_COLOR_PRESERVE_GRID if color_preserve_grid is None else list(color_preserve_grid)
    )
    rows = []
    for region_preserve in region_preserve_grid:
        for color_preserve in color_preserve_grid:
            rows.append(
                {
                    "region_preserve": float(region_preserve),
                    "color_preserve": float(color_preserve),
                }
            )
    return pd.DataFrame(rows)


def simulate_heuristic_participant(
    region_preserve: float,
    color_preserve: float,
    random_seed: int = 1,
    max_steps: int = 120,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = {
        "region_preserve": float(region_preserve),
        "color_preserve": float(color_preserve),
    }
    round_df, step_df = agent.run_tree_agent_on_all_rounds(
        max_steps=max_steps,
        max_depth=max_depth,
        n_iterations=n_iterations,
        pruning_thresh=pruning_thresh,
        lapse_rate=0.0,
        heuristic_weights=weights,
        random_seed=random_seed,
    )
    for df in (round_df, step_df):
        df["true_region_preserve"] = float(region_preserve)
        df["true_color_preserve"] = float(color_preserve)
        df["random_seed"] = int(random_seed)
    return round_df, step_df


def _observed_by_round(step_df: pd.DataFrame) -> dict[int, list[dict]]:
    valid_df = step_df.copy()
    if "terminated" in valid_df.columns:
        valid_df = valid_df[~valid_df["terminated"].fillna(False)]
    out: dict[int, list[dict]] = {}
    for round_index, sub_df in valid_df.groupby("round"):
        out[int(round_index)] = [
            {
                "region": int(row["region"]),
                "new_color": int(row["new_color"]),
            }
            for _, row in sub_df.sort_values("agent_step").iterrows()
        ]
    return out


def fit_heuristic_participant(
    step_df: pd.DataFrame,
    region_preserve_grid: list[float] | None = None,
    color_preserve_grid: list[float] | None = None,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 1.0,
    epsilon: float = 1e-9,
) -> tuple[dict, pd.DataFrame]:
    materials = agent.load_materials()
    parameter_grid = build_heuristic_grid(region_preserve_grid, color_preserve_grid)
    observed = _observed_by_round(step_df)
    rows = []

    for _, params in parameter_grid.iterrows():
        weights = {
            "region_preserve": float(params["region_preserve"]),
            "color_preserve": float(params["color_preserve"]),
        }
        total_ll = 0.0
        n_actions = 0

        for round_index, round_data in enumerate(materials["rounds"], start=1):
            adjacency = agent.build_adjacency_map(round_data["mapData"])
            root = agent.build_tree_node(adjacency, tuple(int(c) for c in round_data["initialColors"]))

            for obs_action in observed.get(round_index, []):
                (
                    best_children,
                    tree_action_probs,
                    evaluated_actions,
                    conflict_edges,
                    conflict_regions,
                    layers,
                    root,
                ) = agent.tree_policy_from_root(
                    root,
                    adjacency,
                    max_depth=max_depth,
                    n_iterations=n_iterations,
                    pruning_thresh=pruning_thresh,
                    heuristic_weights=weights,
                    random_tie_break=False,
                    rng=None,
                )

                action_key = (int(obs_action["region"]), int(obs_action["new_color"]))
                prob = max(float(tree_action_probs.get(action_key, 0.0)), float(epsilon))
                total_ll += float(np.log(prob))
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

        rows.append(
            {
                **weights,
                "ll": total_ll,
                "nll": -total_ll,
                "n_actions": n_actions,
            }
        )

    candidate_df = pd.DataFrame(rows).sort_values(["nll", "region_preserve", "color_preserve"]).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()
    best["converged"] = True
    return best, candidate_df


def run_heuristic_recovery(
    region_preserve_grid: list[float] | None = None,
    color_preserve_grid: list[float] | None = None,
    random_seed: int = 1,
    max_steps: int = 120,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 1.0,
) -> pd.DataFrame:
    tasks = build_heuristic_grid(region_preserve_grid, color_preserve_grid)
    rows = []
    for task_id, task in tasks.iterrows():
        _, step_df = simulate_heuristic_participant(
            region_preserve=float(task["region_preserve"]),
            color_preserve=float(task["color_preserve"]),
            random_seed=random_seed,
            max_steps=max_steps,
            max_depth=max_depth,
            n_iterations=n_iterations,
            pruning_thresh=pruning_thresh,
        )
        best, _ = fit_heuristic_participant(
            step_df,
            region_preserve_grid=region_preserve_grid,
            color_preserve_grid=color_preserve_grid,
            max_depth=max_depth,
            n_iterations=n_iterations,
            pruning_thresh=pruning_thresh,
        )
        rows.append(
            {
                "task_id": int(task_id),
                "random_seed": int(random_seed),
                "true_region_preserve": float(task["region_preserve"]),
                "true_color_preserve": float(task["color_preserve"]),
                "hat_region_preserve": float(best["region_preserve"]),
                "hat_color_preserve": float(best["color_preserve"]),
                "ll": float(best["ll"]),
                "nll": float(best["nll"]),
                "n_actions": int(best["n_actions"]),
                "converged": bool(best["converged"]),
            }
        )
        print(f"completed {task_id + 1}/{len(tasks)}")

    recovery_df = pd.DataFrame(rows)
    recovery_df.to_csv(
        RESULTS_DIR / "experiment2_heuristic_weight_recovery_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_heuristic_recovery(recovery_df).to_csv(
        RESULTS_DIR / "experiment2_heuristic_weight_recovery_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return recovery_df


def _run_single_heuristic_recovery_task(
    task: dict,
    region_preserve_grid: list[float],
    color_preserve_grid: list[float],
    max_steps: int,
    max_depth: int,
    n_iterations: int,
    pruning_thresh: float,
) -> dict:
    task_id = int(task["task_id"])
    random_seed = int(task["random_seed"])
    true_region_preserve = float(task["true_region_preserve"])
    true_color_preserve = float(task["true_color_preserve"])

    _, step_df = simulate_heuristic_participant(
        region_preserve=true_region_preserve,
        color_preserve=true_color_preserve,
        random_seed=random_seed,
        max_steps=max_steps,
        max_depth=max_depth,
        n_iterations=n_iterations,
        pruning_thresh=pruning_thresh,
    )
    best, _ = fit_heuristic_participant(
        step_df,
        region_preserve_grid=region_preserve_grid,
        color_preserve_grid=color_preserve_grid,
        max_depth=max_depth,
        n_iterations=n_iterations,
        pruning_thresh=pruning_thresh,
    )
    return {
        "task_id": task_id,
        "random_seed": random_seed,
        "true_region_preserve": true_region_preserve,
        "true_color_preserve": true_color_preserve,
        "hat_region_preserve": float(best["region_preserve"]),
        "hat_color_preserve": float(best["color_preserve"]),
        "ll": float(best["ll"]),
        "nll": float(best["nll"]),
        "n_actions": int(best["n_actions"]),
        "converged": bool(best["converged"]),
    }


def run_heuristic_recovery_parallel(
    region_preserve_grid: list[float] | None = None,
    color_preserve_grid: list[float] | None = None,
    random_seed: int = 1,
    max_steps: int = 120,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 1.0,
    n_workers: int | None = None,
    save_every: int = 1,
    output_prefix: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    region_preserve_grid = (
        DEFAULT_REGION_PRESERVE_GRID if region_preserve_grid is None else list(region_preserve_grid)
    )
    color_preserve_grid = (
        DEFAULT_COLOR_PRESERVE_GRID if color_preserve_grid is None else list(color_preserve_grid)
    )
    tasks = build_heuristic_grid(region_preserve_grid, color_preserve_grid)
    tasks = tasks.reset_index(drop=True).reset_index(names="task_id")
    tasks = tasks.rename(
        columns={
            "region_preserve": "true_region_preserve",
            "color_preserve": "true_color_preserve",
        }
    )
    tasks["random_seed"] = int(random_seed)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, int(n_workers))

    if output_prefix is None:
        output_prefix = RESULTS_DIR / "experiment2_heuristic_weight_recovery_parallel"
    output_prefix = Path(output_prefix)
    partial_path = output_prefix.with_name(f"{output_prefix.name}_partial.csv")
    results_path = output_prefix.with_name(f"{output_prefix.name}_results.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")

    rows: list[dict] = []
    task_records = tasks.to_dict("records")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_single_heuristic_recovery_task,
                task,
                region_preserve_grid,
                color_preserve_grid,
                max_steps,
                max_depth,
                n_iterations,
                pruning_thresh,
            ): int(task["task_id"])
            for task in task_records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if verbose:
                print(f"completed {completed}/{len(task_records)} task_id={row['task_id']}", flush=True)
            if save_every > 0 and completed % int(save_every) == 0:
                pd.DataFrame(rows).sort_values("task_id").to_csv(
                    partial_path,
                    index=False,
                    encoding="utf-8-sig",
                )

    recovery_df = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    recovery_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summarize_heuristic_recovery(recovery_df).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return recovery_df


def summarize_heuristic_recovery(recovery_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param in ["region_preserve", "color_preserve"]:
        true_col = f"true_{param}"
        hat_col = f"hat_{param}"
        corr = recovery_df[[true_col, hat_col]].corr().iloc[0, 1] if len(recovery_df) > 1 else np.nan
        mae = float((recovery_df[true_col] - recovery_df[hat_col]).abs().mean())
        rows.append({"parameter": param, "correlation": corr, "mae": mae})
    return pd.DataFrame(rows)


def main() -> None:
    recovery_df = run_heuristic_recovery()
    print(recovery_df.to_string(index=False))
    print(summarize_heuristic_recovery(recovery_df).to_string(index=False))


if __name__ == "__main__":
    main()
