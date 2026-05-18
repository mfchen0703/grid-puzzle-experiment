"""Parameter recovery for experiment-2 Tree and BFS-HSP2 agents.

The likelihood estimator follows the fourinarow IBS pattern: for each observed
state/action pair, repeatedly sample the model's next action from that exact
state until the observed action is generated, accumulating the IBS negative
log-likelihood estimator.  BADS is used as a noisy optimizer over the same IBS
objective.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
import io
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import conflict_search_agent as tree_agent
from conflict_repair_hsp2.solver import (
    AdjList,
    apply_action as hsp2_apply_action,
    count_conflicts,
    generate_relevant_actions,
    get_dependency_candidate_nodes,
    is_solved,
    run_hsp2_planner,
    state_to_key,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = ROOT / "experiment2" / "generated_tree_agent_failed_maps" / "solver_validation"
CPP_DIR = Path(__file__).resolve().parent / "cpp"
CPP_HSP2_SOURCE = CPP_DIR / "hsp2_ibs_fast.cpp"
CPP_HSP2_BINARY = CPP_DIR / "hsp2_ibs_fast"
CPP_TREE_SOURCE = CPP_DIR / "tree_simulate.cpp"
CPP_TREE_BINARY = CPP_DIR / "tree_simulate"

PARAM_NAMES = ["pruning_thresh", "gamma", "lapse_rate"]
DEFAULT_GRID = {
    "pruning_thresh": [0.0, 1.0, 2.0, 5.0],
    "gamma": [0.05, 0.1, 0.2, 0.5],
    "lapse_rate": [0.0, 0.1],
}

_IBS_MAX_TRIES = 2000


def executor_context(n_workers: int, backend: str):
    workers = int(n_workers)
    if workers <= 1:
        return nullcontext(None)
    if backend == "thread":
        return ThreadPoolExecutor(max_workers=workers)
    if backend == "process":
        return ProcessPoolExecutor(max_workers=workers)
    raise ValueError(f"Unknown parallel backend: {backend}")


def parse_grid(text: str | None, default: Sequence[float]) -> list[float]:
    if text is None or text.strip() == "":
        return [float(x) for x in default]
    return [float(part) for part in text.split(",")]


def round_adjacency_for_hsp2(round_data: dict) -> AdjList:
    n_regions = int(round_data["mapData"]["numRegions"])
    adj: AdjList = {idx: [] for idx in range(n_regions)}
    for u, v in round_data["mapData"]["adjacencyPairs"]:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return {node: sorted(neighbors) for node, neighbors in adj.items()}


def hsp2_action_to_tree_action(action: tuple[int, int, int]) -> dict:
    region, old_color, new_color = action
    return {
        "search_depth": 0,
        "region": int(region),
        "old_color": int(old_color),
        "new_color": int(new_color),
    }


def state_to_text(state: Sequence[int]) -> str:
    return " ".join(str(int(x)) for x in state)


def text_to_state(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split())


def ensure_cpp_hsp2_binary() -> Path:
    """Build the optional C++ HSP2 IBS helper when needed."""

    if not CPP_HSP2_SOURCE.exists():
        raise RuntimeError(f"Missing C++ source: {CPP_HSP2_SOURCE}")
    needs_build = not CPP_HSP2_BINARY.exists()
    if not needs_build:
        needs_build = CPP_HSP2_BINARY.stat().st_mtime < CPP_HSP2_SOURCE.stat().st_mtime
    if needs_build:
        CPP_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "g++",
            "-O3",
            "-std=c++17",
            "-pthread",
            str(CPP_HSP2_SOURCE),
            "-o",
            str(CPP_HSP2_BINARY),
        ]
        subprocess.run(cmd, check=True)
    return CPP_HSP2_BINARY


def ensure_cpp_tree_binary() -> Path:
    """Build the optional C++ tree simulator/IBS helper when needed."""

    if not CPP_TREE_SOURCE.exists():
        raise RuntimeError(f"Missing C++ source: {CPP_TREE_SOURCE}")
    needs_build = not CPP_TREE_BINARY.exists()
    if not needs_build:
        needs_build = CPP_TREE_BINARY.stat().st_mtime < CPP_TREE_SOURCE.stat().st_mtime
    if needs_build:
        CPP_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "g++",
            "-O3",
            "-std=c++17",
            "-pthread",
            str(CPP_TREE_SOURCE),
            "-o",
            str(CPP_TREE_BINARY),
        ]
        subprocess.run(cmd, check=True)
    return CPP_TREE_BINARY


def write_cpp_tree_rounds_file(rounds: Sequence[dict]) -> Path:
    """Export rounds with initial states for C++ tree simulation."""

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tree_rounds.txt",
        delete=False,
    )
    with tmp:
        tmp.write(f"ROUNDS {len(rounds)}\n")
        for round_index, round_data in enumerate(rounds, start=1):
            n_regions = int(round_data["mapData"]["numRegions"])
            edges = [
                (int(u), int(v))
                for u, v in round_data["mapData"]["adjacencyPairs"]
            ]
            tmp.write(f"ROUND {round_index} {n_regions} {len(edges)}\n")
            tmp.write(
                "INIT "
                + " ".join(str(int(x)) for x in round_data["initialColors"])
                + "\n"
            )
            for u, v in edges:
                tmp.write(f"EDGE {u} {v}\n")
    return Path(tmp.name)


def write_cpp_tree_ibs_task_file(observed_df: pd.DataFrame) -> Path:
    """Export observed tree actions for C++ IBS likelihood."""

    materials = tree_agent.load_materials()
    observed_df = observed_df.sort_values(["round", "agent_step"]).reset_index(drop=True)
    round_ids = sorted({int(round_id) for round_id in observed_df["round"].tolist()})

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tree_ibs_tasks.txt",
        delete=False,
    )
    with tmp:
        tmp.write(f"ROUNDS {len(round_ids)}\n")
        for round_id in round_ids:
            round_data = materials["rounds"][round_id - 1]
            n_regions = int(round_data["mapData"]["numRegions"])
            edges = [
                (int(u), int(v))
                for u, v in round_data["mapData"]["adjacencyPairs"]
            ]
            tmp.write(f"ROUND {round_id} {n_regions} {len(edges)}\n")
            for u, v in edges:
                tmp.write(f"EDGE {u} {v}\n")

        tmp.write(f"OBS {len(observed_df)}\n")
        for obs_index, obs in observed_df.iterrows():
            state = text_to_state(str(obs["state_before"]))
            tmp.write(
                "OBSROW "
                f"{int(obs_index)} "
                f"{int(obs['round'])} "
                f"{len(state)} "
                f"{' '.join(str(int(x)) for x in state)} "
                f"{int(obs['region'])} "
                f"{int(obs['new_color'])}\n"
            )
    return Path(tmp.name)


def simulate_synthetic_actions_cpp_tree(
    theta: Sequence[float],
    *,
    random_seed: int,
    round_limit: int | None,
    max_agent_steps: int,
    max_depth: int,
    max_expansions: int,
    random_tie_break: bool = False,
) -> pd.DataFrame:
    """Generate tree actions with the C++ tree simulator."""

    binary = ensure_cpp_tree_binary()
    materials = tree_agent.load_materials()
    rounds = materials["rounds"]
    if round_limit is not None:
        rounds = rounds[: int(round_limit)]
    rounds_path = write_cpp_tree_rounds_file(rounds)
    theta = [float(x) for x in theta]
    try:
        cmd = [
            str(binary),
            str(rounds_path),
            f"{theta[0]:.17g}",
            f"{theta[1]:.17g}",
            f"{theta[2]:.17g}",
            str(int(max_agent_steps)),
            str(int(max_depth)),
            str(int(max_expansions)),
            str(int(random_seed)),
            "1" if random_tie_break else "0",
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
    finally:
        try:
            rounds_path.unlink()
        except FileNotFoundError:
            pass

    if not completed.stdout.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(completed.stdout))
    if df.empty:
        return df
    df.insert(0, "agent", "tree")
    df["true_pruning_thresh"] = float(theta[0])
    df["true_gamma"] = float(theta[1])
    df["true_lapse_rate"] = float(theta[2])
    df["random_seed"] = int(random_seed)
    return df


def ibs_loglik_cpp_tree(
    observed_df: pd.DataFrame,
    theta: Sequence[float],
    *,
    ibs_samples: int,
    base_seed: int,
    max_depth: int,
    max_expansions: int,
    max_tries: int,
    n_workers: int,
    random_tie_break: bool = False,
) -> np.ndarray:
    """Compute tree IBS action NLLs with the C++ tree simulator."""

    binary = ensure_cpp_tree_binary()
    task_path = write_cpp_tree_ibs_task_file(observed_df)
    theta = [float(x) for x in theta]
    try:
        cmd = [
            str(binary),
            "--ibs",
            str(task_path),
            f"{theta[0]:.17g}",
            f"{theta[1]:.17g}",
            f"{theta[2]:.17g}",
            str(int(ibs_samples)),
            str(int(base_seed)),
            str(int(max_depth)),
            str(int(max_expansions)),
            str(int(max_tries)),
            str(max(1, int(n_workers))),
            "1" if random_tie_break else "0",
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
    finally:
        try:
            task_path.unlink()
        except FileNotFoundError:
            pass

    per_action_nll = np.zeros(len(observed_df), dtype=float)
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        obs_index_text, nll_text = line.split()
        per_action_nll[int(obs_index_text)] = float(nll_text)
    return per_action_nll


def write_cpp_hsp2_task_file(observed_df: pd.DataFrame) -> Path:
    """Export round graphs and observed states in a compact text format."""

    materials = tree_agent.load_materials()
    observed_df = observed_df.sort_values(["round", "agent_step"]).reset_index(drop=True)
    round_ids = sorted({int(round_id) for round_id in observed_df["round"].tolist()})

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".hsp2_tasks.txt",
        delete=False,
    )
    with tmp:
        tmp.write(f"ROUNDS {len(round_ids)}\n")
        for round_id in round_ids:
            round_data = materials["rounds"][round_id - 1]
            n_regions = int(round_data["mapData"]["numRegions"])
            edges = [
                (int(u), int(v))
                for u, v in round_data["mapData"]["adjacencyPairs"]
            ]
            tmp.write(f"ROUND {round_id} {n_regions} {len(edges)}\n")
            for u, v in edges:
                tmp.write(f"EDGE {u} {v}\n")

        tmp.write(f"OBS {len(observed_df)}\n")
        for obs_index, obs in observed_df.iterrows():
            state = text_to_state(str(obs["state_before"]))
            tmp.write(
                "OBSROW "
                f"{int(obs_index)} "
                f"{int(obs['round'])} "
                f"{len(state)} "
                f"{' '.join(str(int(x)) for x in state)} "
                f"{int(obs['region'])} "
                f"{int(obs['new_color'])}\n"
            )
    return Path(tmp.name)


def ibs_loglik_cpp_hsp2(
    observed_df: pd.DataFrame,
    theta: Sequence[float],
    *,
    ibs_samples: int,
    base_seed: int,
    max_depth: int,
    max_expansions: int,
    max_tries: int,
    n_workers: int,
    exact: bool = False,
) -> np.ndarray:
    """Compute HSP2 IBS action NLLs with the optional C++ helper."""

    binary = ensure_cpp_hsp2_binary()
    task_path = write_cpp_hsp2_task_file(observed_df)
    theta = [float(x) for x in theta]
    try:
        cmd = [
            str(binary),
            str(task_path),
            f"{theta[0]:.17g}",
            f"{theta[1]:.17g}",
            f"{theta[2]:.17g}",
            str(int(ibs_samples)),
            str(int(base_seed)),
            str(int(max_depth)),
            str(int(max_expansions)),
            str(int(max_tries)),
            str(max(1, int(n_workers))),
            "exact" if exact else "ibs",
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
    finally:
        try:
            task_path.unlink()
        except FileNotFoundError:
            pass

    per_action_nll = np.zeros(len(observed_df), dtype=float)
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        obs_index_text, nll_text = line.split()
        per_action_nll[int(obs_index_text)] = float(nll_text)
    return per_action_nll


def sample_tree_next_action(
    round_data: dict,
    state: Sequence[int],
    theta: Sequence[float],
    rng: random.Random,
    *,
    max_depth: int,
    max_expansions: int,
) -> dict | None:
    pruning_thresh, gamma, lapse_rate = [float(x) for x in theta]
    one_step_round = {
        "mapData": round_data["mapData"],
        "initialColors": [int(x) for x in state],
    }
    trajectory, _ = tree_agent.run_bfs_tree_agent_on_round(
        one_step_round,
        max_steps=1,
        max_depth=max_depth,
        max_expansions=max_expansions,
        n_iterations=None,
        pruning_thresh=pruning_thresh,
        lapse_rate=lapse_rate,
        heuristic_eval_weight=0.0,
        gamma=gamma,
        random_tie_break=True,
        rng=rng,
        tree_score_strategy="task_first",
    )
    if not trajectory or trajectory[0].get("terminated", False):
        return None
    return trajectory[0]


def sample_hsp2_next_action(
    round_data: dict,
    state: Sequence[int],
    theta: Sequence[float],
    rng: random.Random,
    *,
    max_depth: int,
    max_expansions: int,
) -> dict | None:
    pruning_thresh, gamma, lapse_rate = [float(x) for x in theta]
    adj_list = round_adjacency_for_hsp2(round_data)
    colors = list(range(4))
    plan, _, _, _ = run_hsp2_planner(
        state,
        colors,
        adj_list,
        max_depth=max_depth,
        max_expansions=max_expansions,
        pruning_thresh=pruning_thresh,
        gamma=gamma,
        lapse_rate=lapse_rate,
        rng=rng,
    )
    if not plan:
        return None
    return hsp2_action_to_tree_action(plan[0])


def sample_next_action(
    agent_name: str,
    round_data: dict,
    state: Sequence[int],
    theta: Sequence[float],
    rng: random.Random,
    *,
    max_depth: int,
    max_expansions: int,
) -> dict | None:
    if agent_name == "tree":
        return sample_tree_next_action(
            round_data,
            state,
            theta,
            rng,
            max_depth=max_depth,
            max_expansions=max_expansions,
        )
    if agent_name == "hsp2":
        return sample_hsp2_next_action(
            round_data,
            state,
            theta,
            rng,
            max_depth=max_depth,
            max_expansions=max_expansions,
        )
    raise ValueError(f"Unknown agent: {agent_name}")


def exact_hsp2_action_nll(round_data: dict, state: Sequence[int], observed: dict, theta: Sequence[float], *, max_depth: int, max_expansions: int) -> float:
    """Exact HSP2 action NLL for deterministic planner plus lapse mixture."""

    pruning_thresh, gamma, lapse_rate = [float(x) for x in theta]
    adj_list = round_adjacency_for_hsp2(round_data)
    state_key = state_to_key(state)

    planner_action = sample_hsp2_next_action(
        round_data,
        state_key,
        [pruning_thresh, gamma, 0.0],
        random.Random(1),
        max_depth=max_depth,
        max_expansions=max_expansions,
    )
    planner_match = action_matches(planner_action, observed)
    probability = (1.0 - lapse_rate) if planner_match else 0.0

    if lapse_rate > 0.0 and count_conflicts(state_key, adj_list) > 0:
        candidate_nodes, candidate_reason = get_dependency_candidate_nodes(
            state_key,
            list(range(4)),
            adj_list,
            dependency_depth=3,
        )
        lapse_actions = generate_relevant_actions(
            state_key,
            list(range(4)),
            adj_list,
            candidate_nodes,
            candidate_reason=candidate_reason,
        )
        if lapse_actions:
            matches = sum(
                1
                for action in lapse_actions
                if action_matches(hsp2_action_to_tree_action(action), observed)
            )
            probability += lapse_rate * float(matches) / float(len(lapse_actions))
        elif planner_match:
            probability = 1.0

    if probability <= 0.0:
        return 50.0
    return float(-np.log(probability))


def exact_hsp2_loglik_python(
    observed_df: pd.DataFrame,
    theta: Sequence[float],
    *,
    max_depth: int,
    max_expansions: int,
) -> np.ndarray:
    materials = tree_agent.load_materials()
    theta = [float(x) for x in theta]
    observed_df = observed_df.sort_values(["round", "agent_step"]).reset_index(drop=True)
    per_action_nll = np.zeros(len(observed_df), dtype=float)

    for obs_index, obs in observed_df.iterrows():
        round_data = materials["rounds"][int(obs["round"]) - 1]
        observed = {
            "region": int(obs["region"]),
            "new_color": int(obs["new_color"]),
        }
        per_action_nll[int(obs_index)] = exact_hsp2_action_nll(
            round_data,
            text_to_state(str(obs["state_before"])),
            observed,
            theta,
            max_depth=max_depth,
            max_expansions=max_expansions,
        )
    return per_action_nll


def apply_observed_action(agent_name: str, state: Sequence[int], action: dict) -> tuple[int, ...]:
    if agent_name == "hsp2":
        return hsp2_apply_action(
            state_to_key(state),
            (int(action["region"]), int(action["old_color"]), int(action["new_color"])),
        )
    return tree_agent.apply_action(state, action)


def action_matches(sampled: dict | None, observed: dict) -> bool:
    if sampled is None:
        return False
    return (
        int(sampled["region"]) == int(observed["region"])
        and int(sampled["new_color"]) == int(observed["new_color"])
    )


def _ibs_single_action_nll(task: dict) -> tuple[int, float]:
    obs_index = int(task["obs_index"])
    round_data = task["round_data"]
    agent_name = str(task["agent_name"])
    theta = [float(x) for x in task["theta"]]
    state = text_to_state(task["state_before"])
    target = {
        "region": int(task["region"]),
        "new_color": int(task["new_color"]),
    }
    ibs_samples = int(task["ibs_samples"])
    max_tries = int(task["max_tries"])
    base_seed = int(task["base_seed"])
    max_depth = int(task["max_depth"])
    max_expansions = int(task["max_expansions"])

    action_nll = 0.0
    times_left = ibs_samples
    tries = 0
    while times_left > 0 and tries < max_tries:
        tries += 1
        seed = base_seed + obs_index * 1_000_003 + tries
        rng = random.Random(seed)
        sampled = sample_next_action(
            agent_name,
            round_data,
            state,
            theta,
            rng,
            max_depth=max_depth,
            max_expansions=max_expansions,
        )
        if action_matches(sampled, target):
            times_left -= 1
        else:
            action_nll += 1.0 / (tries * ibs_samples)

    if times_left > 0:
        action_nll += times_left * 3.5

    return obs_index, float(action_nll)


def simulate_synthetic_actions(
    agent_name: str,
    theta: Sequence[float],
    *,
    random_seed: int,
    round_limit: int | None,
    max_agent_steps: int,
    max_depth: int,
    max_expansions: int,
) -> pd.DataFrame:
    materials = tree_agent.load_materials()
    rounds = materials["rounds"]
    if round_limit is not None:
        rounds = rounds[: int(round_limit)]

    rng = random.Random(int(random_seed))
    rows: list[dict] = []
    for round_index, round_data in enumerate(rounds, start=1):
        state = tuple(int(x) for x in round_data["initialColors"])
        adj_hsp2 = round_adjacency_for_hsp2(round_data)
        for agent_step in range(int(max_agent_steps)):
            if count_conflicts(state, adj_hsp2) == 0:
                break
            state_before = state
            action = sample_next_action(
                agent_name,
                round_data,
                state_before,
                theta,
                rng,
                max_depth=max_depth,
                max_expansions=max_expansions,
            )
            if action is None:
                break
            state = apply_observed_action(agent_name, state_before, action)
            rows.append(
                {
                    "agent": agent_name,
                    "round": round_index,
                    "agent_step": agent_step,
                    "state_before": state_to_text(state_before),
                    "region": int(action["region"]),
                    "old_color": int(action["old_color"]),
                    "new_color": int(action["new_color"]),
                    "n_conflict_edges_before": count_conflicts(state_before, adj_hsp2),
                    "n_conflict_edges_after": count_conflicts(state, adj_hsp2),
                    "true_pruning_thresh": float(theta[0]),
                    "true_gamma": float(theta[1]),
                    "true_lapse_rate": float(theta[2]),
                    "random_seed": int(random_seed),
                }
            )
    return pd.DataFrame(rows)


def ibs_loglik(
    observed_df: pd.DataFrame,
    agent_name: str,
    theta: Sequence[float],
    *,
    ibs_samples: int,
    base_seed: int,
    max_depth: int,
    max_expansions: int,
    max_tries: int = _IBS_MAX_TRIES,
    n_workers: int = 1,
    executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None,
    cpp_hsp2_ibs: bool = False,
    cpp_hsp2_exact: bool = False,
    python_hsp2_exact: bool = False,
    cpp_tree_ibs: bool = False,
) -> np.ndarray:
    if cpp_tree_ibs:
        if agent_name != "tree":
            raise ValueError("C++ tree IBS can only be used with --agent tree")
        return ibs_loglik_cpp_tree(
            observed_df,
            theta,
            ibs_samples=ibs_samples,
            base_seed=base_seed,
            max_depth=max_depth,
            max_expansions=max_expansions,
            max_tries=max_tries,
            n_workers=n_workers,
            random_tie_break=True,
        )

    if python_hsp2_exact:
        if agent_name != "hsp2":
            raise ValueError("Python HSP2 exact likelihood can only be used with --agent hsp2")
        return exact_hsp2_loglik_python(
            observed_df,
            theta,
            max_depth=max_depth,
            max_expansions=max_expansions,
        )

    if cpp_hsp2_ibs or cpp_hsp2_exact:
        if agent_name != "hsp2":
            raise ValueError("C++ HSP2 likelihood can only be used with --agent hsp2")
        return ibs_loglik_cpp_hsp2(
            observed_df,
            theta,
            ibs_samples=ibs_samples,
            base_seed=base_seed,
            max_depth=max_depth,
            max_expansions=max_expansions,
            max_tries=max_tries,
            n_workers=n_workers,
            exact=cpp_hsp2_exact,
        )

    materials = tree_agent.load_materials()
    theta = [float(x) for x in theta]
    observed_df = observed_df.sort_values(["round", "agent_step"]).reset_index(drop=True)
    per_action_nll = np.zeros(len(observed_df), dtype=float)

    tasks: list[dict] = []
    for obs_index, obs in observed_df.iterrows():
        round_data = materials["rounds"][int(obs["round"]) - 1]
        tasks.append(
            {
                "obs_index": int(obs_index),
                "round_data": round_data,
                "agent_name": agent_name,
                "theta": theta,
                "state_before": str(obs["state_before"]),
                "region": int(obs["region"]),
                "new_color": int(obs["new_color"]),
                "ibs_samples": int(ibs_samples),
                "base_seed": int(base_seed),
                "max_depth": int(max_depth),
                "max_expansions": int(max_expansions),
                "max_tries": int(max_tries),
            }
        )

    if executor is not None:
        workers = max(1, int(n_workers))
        chunksize = max(1, len(tasks) // max(1, workers * 4))
        for obs_index, action_nll in executor.map(
            _ibs_single_action_nll,
            tasks,
            chunksize=chunksize,
        ):
            per_action_nll[obs_index] = action_nll
    elif int(n_workers) > 1:
        workers = max(1, int(n_workers))
        chunksize = max(1, len(tasks) // max(1, workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for obs_index, action_nll in pool.map(
                _ibs_single_action_nll,
                tasks,
                chunksize=chunksize,
            ):
                per_action_nll[obs_index] = action_nll
    else:
        for task in tasks:
            obs_index, action_nll = _ibs_single_action_nll(task)
            per_action_nll[obs_index] = action_nll

    return per_action_nll


def parameter_grid(
    pruning_grid: Iterable[float],
    gamma_grid: Iterable[float],
    lapse_grid: Iterable[float],
) -> list[list[float]]:
    return [
        [float(pruning), float(gamma), float(lapse)]
        for pruning in pruning_grid
        for gamma in gamma_grid
        for lapse in lapse_grid
    ]


def fit_grid_ibs(
    observed_df: pd.DataFrame,
    agent_name: str,
    grid: Sequence[Sequence[float]],
    *,
    ibs_samples: int,
    base_seed: int,
    max_depth: int,
    max_expansions: int,
    max_tries: int = _IBS_MAX_TRIES,
    n_workers: int = 1,
    parallel_backend: str = "process",
    cpp_hsp2_ibs: bool = False,
    cpp_hsp2_exact: bool = False,
    python_hsp2_exact: bool = False,
    cpp_tree_ibs: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    pool_context = (
        nullcontext(None)
        if (cpp_hsp2_ibs or cpp_hsp2_exact or python_hsp2_exact or cpp_tree_ibs)
        else executor_context(n_workers, parallel_backend)
    )
    with pool_context as pool:
        for theta in grid:
            per_action = ibs_loglik(
                observed_df,
                agent_name,
                theta,
                ibs_samples=ibs_samples,
                base_seed=base_seed,
                max_depth=max_depth,
                max_expansions=max_expansions,
                max_tries=max_tries,
                n_workers=n_workers,
                executor=pool,
                cpp_hsp2_ibs=cpp_hsp2_ibs,
                cpp_hsp2_exact=cpp_hsp2_exact,
                python_hsp2_exact=python_hsp2_exact,
                cpp_tree_ibs=cpp_tree_ibs,
            )
            rows.append(
                {
                    "pruning_thresh": float(theta[0]),
                    "gamma": float(theta[1]),
                    "lapse_rate": float(theta[2]),
                    "nll": float(np.sum(per_action)),
                    "n_actions": int(len(per_action)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["nll", "pruning_thresh", "gamma", "lapse_rate"]
    ).reset_index(drop=True)


def fit_bads_ibs(
    observed_df: pd.DataFrame,
    agent_name: str,
    *,
    x0: Sequence[float],
    max_fun_evals: int,
    ibs_samples: int,
    base_seed: int,
    max_depth: int,
    max_expansions: int,
    max_tries: int = _IBS_MAX_TRIES,
    n_workers: int = 1,
    parallel_backend: str = "process",
    progress_every: int = 0,
    cpp_hsp2_ibs: bool = False,
    cpp_hsp2_exact: bool = False,
    python_hsp2_exact: bool = False,
    cpp_tree_ibs: bool = False,
) -> dict:
    try:
        from pybads import BADS
    except ImportError as exc:
        raise RuntimeError("pybads is required for --fit-method bads") from exc

    lower = np.array([0.0, 0.02, 0.0], dtype=float)
    upper = np.array([5.0, 0.5, 0.3], dtype=float)
    plausible_lower = np.array([0.5, 0.05, 0.0], dtype=float)
    plausible_upper = np.array([3.0, 0.25, 0.15], dtype=float)
    x0_arr = np.clip(np.asarray(x0, dtype=float), lower, upper)
    eval_count = {"n": 0}
    pool_context = (
        nullcontext(None)
        if (cpp_hsp2_ibs or cpp_hsp2_exact or python_hsp2_exact or cpp_tree_ibs)
        else executor_context(n_workers, parallel_backend)
    )

    with pool_context as pool:
        def objective(theta: np.ndarray) -> float:
            theta = np.asarray(theta, dtype=float).ravel()
            eval_count["n"] += 1
            per_action = ibs_loglik(
                observed_df,
                agent_name,
                theta,
                ibs_samples=ibs_samples,
                base_seed=base_seed + eval_count["n"] * 10_000_019,
                max_depth=max_depth,
                max_expansions=max_expansions,
                max_tries=max_tries,
                n_workers=n_workers,
                executor=pool,
                cpp_hsp2_ibs=cpp_hsp2_ibs,
                cpp_hsp2_exact=cpp_hsp2_exact,
                python_hsp2_exact=python_hsp2_exact,
                cpp_tree_ibs=cpp_tree_ibs,
            )
            total = float(np.sum(per_action))
            if progress_every > 0 and eval_count["n"] % int(progress_every) == 0:
                print(
                    f"[bads eval {eval_count['n']}] nll={total:.6f} "
                    f"theta={[round(float(x), 6) for x in theta]}",
                    flush=True,
                )
            return total

        bads = BADS(
            objective,
            x0_arr,
            lower_bounds=lower,
            upper_bounds=upper,
            plausible_lower_bounds=plausible_lower,
            plausible_upper_bounds=plausible_upper,
            options={
                "max_fun_evals": int(max_fun_evals),
                "noise_final_samples": 0,
                "uncertainty_handling": True,
                "display": "off",
            },
        )
        result = bads.optimize()
        theta_hat = np.asarray(result["x"], dtype=float).ravel()
        final_nll = objective(theta_hat)
    return {
        "pruning_thresh": float(theta_hat[0]),
        "gamma": float(theta_hat[1]),
        "lapse_rate": float(theta_hat[2]),
        "nll": float(final_nll),
        "n_actions": int(len(observed_df)),
        "eval_count": int(eval_count["n"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["tree", "hsp2"], required=True)
    parser.add_argument("--fit-method", choices=["grid", "bads"], default="grid")
    parser.add_argument("--true-pruning-thresh", type=float, default=2.0)
    parser.add_argument("--true-gamma", type=float, default=0.1)
    parser.add_argument("--true-lapse-rate", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--round-limit", type=int, default=4)
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--ibs-samples", type=int, default=3)
    parser.add_argument("--ibs-max-tries", type=int, default=_IBS_MAX_TRIES)
    parser.add_argument("--base-seed", type=int, default=10)
    parser.add_argument("--pruning-grid", type=str, default=None)
    parser.add_argument("--gamma-grid", type=str, default=None)
    parser.add_argument("--lapse-grid", type=str, default=None)
    parser.add_argument("--bads-max-fun-evals", type=int, default=30)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--cpp-hsp2-ibs",
        action="store_true",
        help="Use the experimental C++ IBS likelihood fast path for --agent hsp2.",
    )
    parser.add_argument(
        "--cpp-hsp2-exact",
        action="store_true",
        help=(
            "Use the experimental C++ exact HSP2 action likelihood. "
            "This avoids IBS sampling because HSP2 is deterministic except lapse."
        ),
    )
    parser.add_argument(
        "--python-hsp2-exact",
        action="store_true",
        help="Use the Python exact HSP2 action likelihood for validation against C++.",
    )
    parser.add_argument(
        "--cpp-tree-ibs",
        action="store_true",
        help="Use C++ tree simulation and C++ tree IBS likelihood for --agent tree.",
    )
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    hsp2_likelihood_modes = [
        bool(args.cpp_hsp2_ibs),
        bool(args.cpp_hsp2_exact),
        bool(args.python_hsp2_exact),
    ]
    if sum(hsp2_likelihood_modes) > 1:
        raise ValueError("Use only one HSP2 likelihood mode at a time")
    if any(hsp2_likelihood_modes) and args.agent != "hsp2":
        raise ValueError("HSP2 likelihood modes can only be used with --agent hsp2")
    if args.cpp_tree_ibs and args.agent != "tree":
        raise ValueError("--cpp-tree-ibs can only be used with --agent tree")

    true_theta = [
        float(args.true_pruning_thresh),
        float(args.true_gamma),
        float(args.true_lapse_rate),
    ]
    if args.cpp_tree_ibs:
        observed_df = simulate_synthetic_actions_cpp_tree(
            true_theta,
            random_seed=args.random_seed,
            round_limit=args.round_limit,
            max_agent_steps=args.max_agent_steps,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            random_tie_break=True,
        )
    else:
        observed_df = simulate_synthetic_actions(
            args.agent,
            true_theta,
            random_seed=args.random_seed,
            round_limit=args.round_limit,
            max_agent_steps=args.max_agent_steps,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
        )
    if observed_df.empty:
        raise RuntimeError("No synthetic actions were generated.")

    if args.fit_method == "grid":
        grid = parameter_grid(
            parse_grid(args.pruning_grid, DEFAULT_GRID["pruning_thresh"]),
            parse_grid(args.gamma_grid, DEFAULT_GRID["gamma"]),
            parse_grid(args.lapse_grid, DEFAULT_GRID["lapse_rate"]),
        )
        fit_df = fit_grid_ibs(
            observed_df,
            args.agent,
            grid,
            ibs_samples=args.ibs_samples,
            base_seed=args.base_seed,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            max_tries=args.ibs_max_tries,
            n_workers=args.n_workers,
            parallel_backend=args.parallel_backend,
            cpp_hsp2_ibs=args.cpp_hsp2_ibs,
            cpp_hsp2_exact=args.cpp_hsp2_exact,
            python_hsp2_exact=args.python_hsp2_exact,
            cpp_tree_ibs=args.cpp_tree_ibs,
        )
        best = fit_df.iloc[0].to_dict()
    else:
        best = fit_bads_ibs(
            observed_df,
            args.agent,
            x0=true_theta,
            max_fun_evals=args.bads_max_fun_evals,
            ibs_samples=args.ibs_samples,
            base_seed=args.base_seed,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            max_tries=args.ibs_max_tries,
            n_workers=args.n_workers,
            parallel_backend=args.parallel_backend,
            progress_every=args.progress_every,
            cpp_hsp2_ibs=args.cpp_hsp2_ibs,
            cpp_hsp2_exact=args.cpp_hsp2_exact,
            python_hsp2_exact=args.python_hsp2_exact,
            cpp_tree_ibs=args.cpp_tree_ibs,
        )
        fit_df = pd.DataFrame([best])

    prefix = args.output_prefix
    if prefix is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        prefix = DEFAULT_OUT_DIR / f"{args.agent}_parameter_recovery_{args.fit_method}"

    observed_path = prefix.with_name(f"{prefix.name}_synthetic_actions.csv")
    fit_path = prefix.with_name(f"{prefix.name}_fit.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    observed_df.to_csv(observed_path, index=False, encoding="utf-8-sig")
    fit_df.to_csv(fit_path, index=False, encoding="utf-8-sig")
    summary = {
        "agent": args.agent,
        "fit_method": args.fit_method,
        "true": dict(zip(PARAM_NAMES, true_theta)),
        "best": {name: float(best[name]) for name in PARAM_NAMES},
        "nll": float(best["nll"]),
        "n_actions": int(best["n_actions"]),
        "n_workers": int(args.n_workers),
        "parallel_backend": args.parallel_backend,
        "cpp_hsp2_ibs": bool(args.cpp_hsp2_ibs),
        "cpp_hsp2_exact": bool(args.cpp_hsp2_exact),
        "python_hsp2_exact": bool(args.python_hsp2_exact),
        "cpp_tree_ibs": bool(args.cpp_tree_ibs),
        "ibs_max_tries": int(args.ibs_max_tries),
        "observed_path": str(observed_path),
        "fit_path": str(fit_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
