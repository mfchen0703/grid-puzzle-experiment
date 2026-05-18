"""Small-scale IBS model recovery test for BFS tree agent."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import conflict_search_agent as agent
import experiment2_model_recovery as recovery

# Tiny parameter grids for fast testing
FIT_PRUNING = [None, 0.0]
FIT_N_ITER = [None, 10]
FIT_LAPSE = [0.0]
FIT_HEVAL = [0.0]

N_ROUNDS = 3
IBS_SAMPLES = 5
N_REPEATS = 1

print("=" * 70)
print("IBS Model Recovery — Small Test")
print(f"  rounds: {N_ROUNDS}, ibs_samples: {IBS_SAMPLES}, repeats: {N_REPEATS}")
print(f"  pruning grid: {FIT_PRUNING}")
print(f"  n_iterations grid: {FIT_N_ITER}")
print(f"  lapse grid: {FIT_LAPSE}")
print(f"  heuristic_eval grid: {FIT_HEVAL}")
print("=" * 70)

rounds_subset = list(range(1, N_ROUNDS + 1))

# Build tasks
tasks_df = recovery.build_recovery_tasks(
    pruning_grid=FIT_PRUNING,
    n_iterations_grid=FIT_N_ITER,
    lapse_grid=FIT_LAPSE,
    heuristic_eval_grid=FIT_HEVAL,
    n_repeats=N_REPEATS,
    rounds_subset=rounds_subset,
)
print(f"\nTasks: {len(tasks_df)}")
print(tasks_df.to_string(index=False))

# Run recovery (sequential, not parallel)
started = time.perf_counter()
recovery_df, candidate_tables = recovery.run_parameter_recovery(
    tasks_df=tasks_df,
    fit_pruning_grid=FIT_PRUNING,
    fit_n_iterations_grid=FIT_N_ITER,
    fit_lapse_grid=FIT_LAPSE,
    fit_heuristic_eval_grid=FIT_HEVAL,
    max_depth=8,
    ibs_samples=IBS_SAMPLES,
    only_first_action=True,
    output_prefix=_THIS_DIR.parent / "results" / "ibs_recovery_test",
    verbose=True,
)
elapsed = time.perf_counter() - started

print(f"\nElapsed: {elapsed:.1f}s")

# Show results
print("\n=== Recovery Results ===")
print(recovery_df.to_string(index=False))

print("\n=== Summary ===")
print(recovery.summarize_recovery(recovery_df).to_string(index=False))

# Show per-task candidate tables
for task_id, cand_df in sorted(candidate_tables.items()):
    true_row = tasks_df[tasks_df["task_id"] == task_id].iloc[0]
    print(f"\n--- Task {task_id} ---")
    print(f"  True: pruning={true_row['true_pruning_thresh']} "
          f"n_iter={true_row['true_n_iterations']} "
          f"lapse={true_row['true_lapse_rate']}")
    print(cand_df.to_string(index=False))

print("\nDone.")
