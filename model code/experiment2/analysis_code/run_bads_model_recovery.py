from __future__ import annotations

import argparse
from pathlib import Path

import experiment2_bads_recovery as bads_recovery
import experiment2_model_recovery as grid_recovery


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_optional_float_list(text: str) -> list[float] | None:
    text = text.strip()
    if text.lower() in {"", "none", "fixed"}:
        return None
    return _parse_float_list(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 BADS parameter recovery.")
    parser.add_argument("--mode", choices=sorted(bads_recovery.MODE_PARAMETERS), default="search")
    parser.add_argument("--true-pruning-grid", default="0,0.5,1")
    parser.add_argument("--true-gamma-grid", default="0.05,0.1,0.2")
    parser.add_argument("--true-lapse-grid", default="0,0.1,0.3")
    parser.add_argument("--true-region-grid", default="0")
    parser.add_argument("--true-color-grid", default="0")
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--max-fun-evals", type=int, default=80)
    parser.add_argument("--uncertainty-handling", action="store_true")
    parser.add_argument("--n-preliminary", type=int, default=8)
    parser.add_argument("--min-round-repeats", type=int, default=1)
    parser.add_argument("--max-round-repeats", type=int, default=4)
    parser.add_argument("--final-reevaluations", type=int, default=10)
    parser.add_argument("--deterministic-objective", action="store_true")
    parser.add_argument("--preeval-seed", type=int, default=10000)
    parser.add_argument(
        "--output-prefix",
        default=str(bads_recovery.default_results_dir() / "experiment2_bads_recovery"),
    )
    args = parser.parse_args()

    true_pruning_grid = _parse_float_list(args.true_pruning_grid)
    true_gamma_grid = _parse_float_list(args.true_gamma_grid)
    true_lapse_grid = _parse_float_list(args.true_lapse_grid)
    true_region_grid = _parse_optional_float_list(args.true_region_grid)
    true_color_grid = _parse_optional_float_list(args.true_color_grid)

    tasks_df = grid_recovery.build_recovery_tasks(
        pruning_grid=true_pruning_grid,
        gamma_grid=true_gamma_grid,
        lapse_grid=true_lapse_grid,
        region_preserve_grid=true_region_grid,
        color_preserve_grid=true_color_grid,
        n_repeats=args.n_repeats,
    )
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tasks_path = output_prefix.with_name(f"{output_prefix.name}_tasks.csv")
    tasks_df.to_csv(tasks_path, index=False, encoding="utf-8-sig")

    parameter_names = bads_recovery.MODE_PARAMETERS[args.mode]
    fixed_params = dict(bads_recovery.DEFAULT_FIXED_PARAMS)
    for name in parameter_names:
        fixed_params.pop(name, None)

    bads_options = {
        "max_fun_evals": int(args.max_fun_evals),
        "uncertainty_handling": bool(args.uncertainty_handling),
        "display": "off",
    }
    recovery_df, _ = bads_recovery.run_bads_recovery_parallel(
        tasks_df=tasks_df,
        parameter_names=parameter_names,
        fixed_params=fixed_params,
        max_depth=args.max_depth,
        n_workers=args.n_workers,
        output_prefix=output_prefix,
        bads_options=bads_options,
        n_preliminary=args.n_preliminary,
        min_round_repeats=args.min_round_repeats,
        max_round_repeats=args.max_round_repeats,
        final_reevaluations=args.final_reevaluations,
        noisy_objective=not args.deterministic_objective,
        preeval_seed=args.preeval_seed,
        verbose=True,
    )
    print(recovery_df.to_string(index=False))
    print(grid_recovery.summarize_recovery(recovery_df).to_string(index=False))


if __name__ == "__main__":
    main()
