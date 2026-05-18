from __future__ import annotations

import argparse
from pathlib import Path

import experiment2_heuristic_recovery as recovery


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment2 heuristic parameter recovery in parallel.")
    parser.add_argument("--region-grid", default="0,0.5,1,2")
    parser.add_argument("--color-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--pruning-thresh", type=float, default=0.5)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        default=str(recovery.RESULTS_DIR / "experiment2_heuristic_weight_recovery_parallel"),
    )
    args = parser.parse_args()

    recovery_df = recovery.run_heuristic_recovery_parallel(
        region_preserve_grid=_parse_float_list(args.region_grid),
        color_preserve_grid=_parse_float_list(args.color_grid),
        random_seed=args.random_seed,
        max_steps=args.max_steps,
        max_depth=args.max_depth,
        n_iterations=args.n_iterations,
        pruning_thresh=args.pruning_thresh,
        n_workers=args.n_workers,
        output_prefix=Path(args.output_prefix),
        verbose=True,
    )
    print(recovery_df.to_string(index=False))
    print(recovery.summarize_heuristic_recovery(recovery_df).to_string(index=False))


if __name__ == "__main__":
    main()
