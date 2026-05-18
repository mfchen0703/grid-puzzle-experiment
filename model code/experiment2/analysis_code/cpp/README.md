# Experiment 2 C++ Recovery Tools

This directory contains C++ helpers for Experiment 2 recovery runs.

The Python recovery scripts are still available for BADS/grid parameter
recovery. This directory also contains a C++ model-recovery orchestrator so
that simulated observations and model scores can both come from C++ code.

## Current tools

### `hsp2_ibs_fast.cpp`

Standalone HSP2 likelihood helper used by
`experiment2_tree_hsp2_parameter_recovery.py`.

It supports two likelihood modes:

- `ibs`: inverse binomial sampling for HSP2 actions.
- `exact`: exact HSP2 action likelihood for deterministic planner plus lapse.

The Python flags are:

```bash
--cpp-hsp2-ibs
--cpp-hsp2-exact
```

`--cpp-hsp2-exact` is faster and more stable, but it is a C++ reimplementation
of the Python HSP2 planner. Use `--python-hsp2-exact` for strict Python-side
validation.

It also has a standalone C++ simulation mode:

```bash
"model code/experiment2/analysis_code/cpp/hsp2_ibs_fast" \
  --simulate \
  experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt \
  2.0 \
  0.1 \
  0.05 \
  50 \
  8 \
  1000 \
  1 \
  38 \
  > hsp2_cpp_actions.csv
```

Simulation arguments are:

```text
--simulate ROUNDS_FILE pruning gamma lapse max_steps max_depth max_expansions seed round_limit
```

### `tree_simulate.cpp`

Standalone C++ simulator for the current fourinarow-style tree agent.

It reads a compact rounds text file and writes per-action CSV to stdout. The
input format is:

```text
ROUNDS <n_rounds>
ROUND <round_id> <n_regions> <n_edges>
INIT <color_0> ... <color_n-1>
EDGE <u> <v>
...
```

Export that format from the current JSON rounds file with:

```bash
python3 "model code/experiment2/analysis_code/cpp/export_rounds_for_cpp.py" \
  --input experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json \
  --output experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt
```

Usage:

```bash
"model code/experiment2/analysis_code/cpp/tree_simulate" \
  rounds_cpp.txt \
  2.0 \
  0.1 \
  0.05 \
  100 \
  8 \
  1000 \
  1 \
  0 \
  > tree_cpp_actions.csv
```

Arguments are:

```text
ROUNDS_FILE pruning_thresh gamma lapse_rate max_steps max_depth max_expansions seed random_tie_break
```

The first version covers the recovery defaults:

- `tree_score_strategy=task_first`
- `heuristic_eval_weight=0`
- default heuristic weights

It is intended as the first C++ tree simulation kernel. Before using it for
final recovery claims, compare its actions against Python on a fixed set of
states and deterministic parameters.

The same executable also supports tree IBS likelihood:

```bash
python3 "model code/experiment2/analysis_code/cpp/export_tree_ibs_tasks.py" \
  --rounds-json experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json \
  --observed-actions experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_simulated_actions.csv \
  --output experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_ibs_tasks.txt

"model code/experiment2/analysis_code/cpp/tree_simulate" \
  --ibs \
  experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_ibs_tasks.txt \
  2.0 \
  0.1 \
  0.05 \
  5 \
  10 \
  8 \
  1000 \
  100 \
  8 \
  0
```

IBS arguments are:

```text
TASKS_FILE pruning_thresh gamma lapse_rate ibs_samples base_seed max_depth max_expansions max_tries n_workers random_tie_break
```

The tree parameter recovery script can call this C++ path directly:

```bash
python3 "model code/experiment2/analysis_code/experiment2_tree_hsp2_parameter_recovery.py" \
  --agent tree \
  --fit-method bads \
  --true-pruning-thresh 2.0 \
  --true-gamma 0.1 \
  --true-lapse-rate 0.05 \
  --round-limit 38 \
  --max-agent-steps 50 \
  --ibs-samples 5 \
  --bads-max-fun-evals 100 \
  --n-workers 64 \
  --cpp-tree-ibs \
  --output-prefix experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_bads_ibs_p2_g01_l005
```

With `--cpp-tree-ibs`, both synthetic action generation and IBS scoring use
`tree_simulate.cpp`.

### `eflop_simulate.cpp`

Standalone C++ simulator for full E-FLOP repair. It mirrors the Python
`simulate_eflop_repair` path:

- run min-conflicts
- if stuck, try E-FLOP perturbations
- accept an E-FLOP attempt when the follow-up min-conflicts pass improves
  conflicts or solves

Usage:

```bash
"model code/experiment2/analysis_code/cpp/eflop_simulate" \
  experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt \
  50 \
  200 \
  5 \
  1 \
  0 \
  > eflop_cpp_actions.csv
```

Arguments are:

```text
ROUNDS_FILE max_outer_loops max_min_conflicts_steps max_eflop_retries seed round_limit
```

It also supports action-level IBS:

```bash
python3 "model code/experiment2/analysis_code/cpp/export_tree_ibs_tasks.py" \
  --rounds-json experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json \
  --observed-actions eflop_cpp_actions.csv \
  --output eflop_ibs_tasks.txt

"model code/experiment2/analysis_code/cpp/eflop_simulate" \
  --ibs \
  eflop_ibs_tasks.txt \
  5 \
  10 \
  200 \
  5 \
  100 \
  8
```

IBS arguments are:

```text
TASKS_FILE ibs_samples base_seed max_min_conflicts_steps max_eflop_retries max_tries n_workers
```

The C++ E-FLOP implementation follows the Python logic in
`experiment2/hsp2_eflop_full_repair_recovery.py`: min-conflicts first, then
accepted E-FLOP perturbation plus a follow-up min-conflicts pass. It is logic
aligned with Python, but same numeric seeds do not imply identical action
chains because Python and C++ use different random-choice implementations.

## Recovery Workflows

### Tree agent parameter recovery

Implemented through `--cpp-tree-ibs` in
`experiment2_tree_hsp2_parameter_recovery.py`. Keep using the C++ path for both
simulation and scoring when the goal is C++-based parameter recovery.

### Three-agent model recovery

`cpp_model_recovery.cpp` performs model recovery over the three C++ agents:

- `tree`
- `hsp2`
- `eflop`

It can either score an existing action CSV or first simulate the observed
trajectory with one of the C++ agents and then recover the model.

Example: simulate tree observations and recover the model:

```bash
"model code/experiment2/analysis_code/cpp/cpp_model_recovery" \
  --rounds experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt \
  --simulate-agent tree \
  --output-prefix experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_model_recovery \
  --round-limit 38 \
  --random-seed 1 \
  --max-agent-steps 50 \
  --max-depth 8 \
  --max-expansions 1000 \
  --ibs-samples 5 \
  --ibs-max-tries 100 \
  --n-workers 64 \
  --tree-pruning 2.0 \
  --tree-gamma 0.1 \
  --tree-lapse 0.05 \
  --hsp2-pruning 2.0 \
  --hsp2-gamma 0.1 \
  --hsp2-lapse 0.05 \
  --hsp2-likelihood-mode exact
```

Example: score an existing observed action CSV:

```bash
"model code/experiment2/analysis_code/cpp/cpp_model_recovery" \
  --rounds experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt \
  --observed-actions observed_actions.csv \
  --output-prefix experiment2/generated_tree_agent_failed_maps/solver_validation/existing_cpp_model_recovery \
  --ibs-samples 5 \
  --ibs-max-tries 100 \
  --n-workers 64
```

Outputs are:

- `<output-prefix>_observed_actions.csv` when `--simulate-agent` is used
- `<output-prefix>_scores.csv`
- `<output-prefix>_summary.json`

Model recovery should compare actions generated and scored by the same
implementation family:

- C++ simulated actions recovered with C++ likelihoods
- or Python simulated actions recovered with Python likelihoods

Mixing Python simulation with C++ likelihood is useful for debugging, but not
for final recovery claims unless action-level equivalence has been verified.

## Build

From the repository root:

```bash
"model code/experiment2/analysis_code/cpp/build_cpp_tools.sh"
```

The Python scripts also auto-build `hsp2_ibs_fast` when needed.
