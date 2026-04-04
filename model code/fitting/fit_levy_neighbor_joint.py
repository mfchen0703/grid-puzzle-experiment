"""
联合模型拟合：Graph Levy step-length + 邻居 softmax + canonical effective colors

对每个被试单独拟合两个参数：
  - b: graph step length 的 Levy 指数，约束在 (1, 3)
  - beta: 在固定步长条件下，对 piece 的已填色邻居数权重

单步概率分解：
  p(action | state, b, beta)
  = p(L | b) * p(piece | state, L, beta) * p(color | piece, state)

其中：
  - L: 上一步位置到当前位置的图最短路步长
  - piece: 在当前 state 下、所有图步长等于 L 的未着色 region 中，
           按已填色邻居个数做 softmax 后选中的 region
  - color: 对选中 region 使用 canonical effective colors
"""

from __future__ import annotations

import math
import os
import glob as glob_module
from dataclasses import dataclass
from pathlib import Path

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from fit_softmax import (
    build_all_maps,
    parse_csv,
    is_color_legal,
    count_effective_colors,
)
from step_length_models import precompute_round_lengths


@dataclass(frozen=True)
class JointStep:
    participant: str
    round_label: str
    step_in_round: int
    prev_region: int
    chosen_region: int
    chosen_color: int
    graph_length: int
    available_graph_lengths: tuple[int, ...]
    candidate_regions_same_length: tuple[int, ...]
    colored_neighbor_counts: tuple[int, ...]
    chosen_colored_neighbor_count: int
    num_effective_colors: int


@dataclass
class PrecomputedSteps:
    """Pre-extracted step data for fast likelihood computation (no pandas in hot loop)."""
    graph_lengths: np.ndarray          # (N,) int
    n_effective_colors: np.ndarray     # (N,) int
    neighbor_counts: List[np.ndarray]  # per-step colored neighbor count arrays
    chosen_indices: np.ndarray         # (N,) index of chosen region in candidate list
    n_candidates: np.ndarray           # (N,) number of candidates per step


def precompute_steps(df: pd.DataFrame) -> PrecomputedSteps:
    graph_lengths = df["graph_length"].values.astype(int)
    n_effective_colors = df["num_effective_colors"].values.astype(int)

    neighbor_counts: List[np.ndarray] = []
    chosen_indices_list: list[int] = []
    n_candidates_list: list[int] = []

    candidates_col = df["candidate_regions_same_length"].tolist()
    counts_col = df["colored_neighbor_counts"].tolist()
    chosen_col = df["chosen_region"].tolist()

    for candidates, counts, chosen in zip(candidates_col, counts_col, chosen_col):
        candidates_t = tuple(candidates)
        neighbor_counts.append(np.asarray(counts, dtype=float))
        n_candidates_list.append(len(candidates_t))
        chosen_indices_list.append(candidates_t.index(chosen) if chosen in candidates_t else -1)

    return PrecomputedSteps(
        graph_lengths=graph_lengths,
        n_effective_colors=n_effective_colors,
        neighbor_counts=neighbor_counts,
        chosen_indices=np.array(chosen_indices_list, dtype=int),
        n_candidates=np.array(n_candidates_list, dtype=int),
    )


def default_data_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def build_joint_steps(data_dir: str | Path | None = None, include_practice: bool = False) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    maps, round_info = precompute_round_lengths()
    rows: list[dict] = []

    for filepath in sorted(data_dir.glob("data_*.csv")):
        participant = filepath.stem.replace("data_", "")
        actions = parse_csv(str(filepath))

        current_colors = None
        used_colors = None
        current_round = None
        prev_region = None
        step_in_round = -1
        adjacency = None
        regions = None

        for act in actions:
            if act["is_start"]:
                current_round = act["round"]
                if not include_practice and current_round.startswith("P"):
                    current_colors = None
                    used_colors = None
                    prev_region = None
                    continue
                if current_round not in maps:
                    current_colors = None
                    used_colors = None
                    prev_region = None
                    continue
                regions, adjacency = maps[current_round]
                current_colors = [None] * len(regions)
                used_colors = set()
                prev_region = None
                step_in_round = 0
                continue

            if current_colors is None:
                continue

            rid = act["region"]
            if rid is None:
                continue

            if act["is_eraser"]:
                current_colors[rid] = None
                continue

            color = act["color"]
            if current_colors[rid] is None:
                if not is_color_legal(rid, color, adjacency, current_colors):
                    continue
                if prev_region is not None:
                    graph_length = int(round_info[current_round]["graph"][prev_region, rid])
                    if graph_length <= 0:
                        prev_region = rid
                        step_in_round += 1
                        current_colors[rid] = color
                        if color is not None:
                            used_colors.add(color)
                        continue
                    available_graph_lengths = []
                    same_length_candidates = []
                    same_length_neighbor_counts = []

                    for candidate in range(len(regions)):
                        if current_colors[candidate] is not None:
                            continue
                        cand_len = int(round_info[current_round]["graph"][prev_region, candidate])
                        if cand_len > 0:
                            available_graph_lengths.append(cand_len)
                        if cand_len == graph_length:
                            same_length_candidates.append(candidate)
                            same_length_neighbor_counts.append(
                                sum(1 for nb in adjacency[candidate] if current_colors[nb] is not None)
                            )

                    num_effective_colors = count_effective_colors(
                        rid, adjacency, current_colors, used_colors
                    )
                    chosen_neighbor_count = sum(
                        1 for nb in adjacency[rid] if current_colors[nb] is not None
                    )

                    rows.append(
                        JointStep(
                            participant=participant,
                            round_label=current_round,
                            step_in_round=step_in_round,
                            prev_region=int(prev_region),
                            chosen_region=int(rid),
                            chosen_color=int(color),
                            graph_length=graph_length,
                            available_graph_lengths=tuple(int(x) for x in available_graph_lengths),
                            candidate_regions_same_length=tuple(int(x) for x in same_length_candidates),
                            colored_neighbor_counts=tuple(int(x) for x in same_length_neighbor_counts),
                            chosen_colored_neighbor_count=int(chosen_neighbor_count),
                            num_effective_colors=int(num_effective_colors),
                        ).__dict__
                    )
                prev_region = rid
                step_in_round += 1

            current_colors[rid] = color
            if color is not None:
                used_colors.add(color)

    return pd.DataFrame(rows)


def get_global_graph_length_support(joint_steps_df: pd.DataFrame) -> np.ndarray:
    all_lengths: list[int] = []
    for lengths in joint_steps_df["available_graph_lengths"]:
        all_lengths.extend(int(x) for x in lengths if int(x) > 0)
    if not all_lengths:
        return np.array([], dtype=int)
    return np.sort(np.unique(np.array(all_lengths, dtype=int)))


def log_p_length(graph_length: int, b: float, support: np.ndarray) -> float:
    if not (1.0 < b < 3.0):
        return -math.inf
    weights = support.astype(float) ** (-b)
    probs = weights / np.sum(weights)
    match = np.where(support == int(graph_length))[0]
    if len(match) == 0:
        return -math.inf
    p = float(probs[match[0]])
    return math.log(max(p, 1e-300))


def log_p_piece(step_row: pd.Series, beta: float) -> float:
    counts = np.array(step_row["colored_neighbor_counts"], dtype=float)
    candidates = tuple(step_row["candidate_regions_same_length"])
    if len(candidates) == 0:
        return -math.inf
    if step_row["chosen_region"] not in candidates:
        return -math.inf
    utilities = beta * counts
    chosen_idx = candidates.index(step_row["chosen_region"])
    return float(utilities[chosen_idx] - logsumexp(utilities))


def log_p_piece_uniform(step_row: pd.Series) -> float:
    candidates = tuple(step_row["candidate_regions_same_length"])
    if len(candidates) == 0:
        return -math.inf
    if step_row["chosen_region"] not in candidates:
        return -math.inf
    return -math.log(len(candidates))


def log_p_color(step_row: pd.Series) -> float:
    n_effective = int(step_row["num_effective_colors"])
    if n_effective <= 0:
        return -math.inf
    return -math.log(n_effective)


def log_p_length_uniform(graph_length: int, support: np.ndarray) -> float:
    if len(support) == 0:
        return -math.inf
    if int(graph_length) not in set(int(x) for x in support):
        return -math.inf
    return -math.log(len(support))


# ---------------------------------------------------------------------------
# Fast batch likelihood computation (avoids pandas in optimizer hot loop)
# ---------------------------------------------------------------------------

def _build_length_to_idx(support: np.ndarray) -> dict[int, int]:
    return {int(L): i for i, L in enumerate(support)}


def _fast_ll_length(precomp: PrecomputedSteps, b: float,
                    support: np.ndarray, length_to_idx: dict[int, int]) -> float:
    log_weights = -b * np.log(support.astype(float))
    log_probs = log_weights - logsumexp(log_weights)
    idxs = np.array([length_to_idx.get(int(gl), -1) for gl in precomp.graph_lengths])
    if np.any(idxs < 0):
        return -math.inf
    return float(np.sum(log_probs[idxs]))


def _fast_ll_length_uniform(precomp: PrecomputedSteps,
                            support: np.ndarray, length_to_idx: dict[int, int]) -> float:
    if len(support) == 0:
        return -math.inf
    for gl in precomp.graph_lengths:
        if int(gl) not in length_to_idx:
            return -math.inf
    return float(-math.log(len(support)) * len(precomp.graph_lengths))


def _fast_ll_piece(precomp: PrecomputedSteps, beta: float) -> float:
    total = 0.0
    for i in range(len(precomp.chosen_indices)):
        if precomp.n_candidates[i] == 0 or precomp.chosen_indices[i] < 0:
            return -math.inf
        utilities = beta * precomp.neighbor_counts[i]
        total += float(utilities[precomp.chosen_indices[i]] - logsumexp(utilities))
    return total


def _fast_ll_piece_uniform(precomp: PrecomputedSteps) -> float:
    if np.any(precomp.n_candidates == 0) or np.any(precomp.chosen_indices < 0):
        return -math.inf
    return float(-np.sum(np.log(precomp.n_candidates.astype(float))))


def _fast_ll_color(precomp: PrecomputedSteps) -> float:
    neff = precomp.n_effective_colors.astype(float)
    if np.any(neff <= 0):
        return -math.inf
    return float(-np.sum(np.log(neff)))


def _fast_components(ll_length: float, ll_piece: float, ll_color: float) -> dict[str, float]:
    return {
        "ll_length": ll_length,
        "ll_piece": ll_piece,
        "ll_color": ll_color,
        "ll_total": ll_length + ll_piece + ll_color,
    }


# ---------------------------------------------------------------------------
# Public component functions (accept DataFrame for backward compatibility)
# ---------------------------------------------------------------------------

def joint_loglikelihood_components(
    participant_steps: pd.DataFrame,
    b: float,
    beta: float,
    support: np.ndarray,
) -> dict[str, float]:
    precomp = precompute_steps(participant_steps)
    length_to_idx = _build_length_to_idx(support)
    return _fast_components(
        _fast_ll_length(precomp, b, support, length_to_idx),
        _fast_ll_piece(precomp, beta),
        _fast_ll_color(precomp),
    )


def length_only_loglikelihood_components(
    participant_steps: pd.DataFrame,
    b: float,
    support: np.ndarray,
) -> dict[str, float]:
    precomp = precompute_steps(participant_steps)
    length_to_idx = _build_length_to_idx(support)
    return _fast_components(
        _fast_ll_length(precomp, b, support, length_to_idx),
        _fast_ll_piece_uniform(precomp),
        _fast_ll_color(precomp),
    )


def piece_only_loglikelihood_components(
    participant_steps: pd.DataFrame,
    beta: float,
    support: np.ndarray,
) -> dict[str, float]:
    precomp = precompute_steps(participant_steps)
    length_to_idx = _build_length_to_idx(support)
    return _fast_components(
        _fast_ll_length_uniform(precomp, support, length_to_idx),
        _fast_ll_piece(precomp, beta),
        _fast_ll_color(precomp),
    )


# ---------------------------------------------------------------------------
# Negative log-likelihood objectives (fast path, used by optimizer)
# ---------------------------------------------------------------------------

def _neg_ll_joint(params: np.ndarray, precomp: PrecomputedSteps,
                  support: np.ndarray, length_to_idx: dict[int, int]) -> float:
    b, beta = float(params[0]), float(params[1])
    if not (1.0 < b < 3.0):
        return np.inf
    ll = (_fast_ll_length(precomp, b, support, length_to_idx)
          + _fast_ll_piece(precomp, beta)
          + _fast_ll_color(precomp))
    return -ll if np.isfinite(ll) else np.inf


def _neg_ll_length_only(params: np.ndarray, precomp: PrecomputedSteps,
                        support: np.ndarray, length_to_idx: dict[int, int],
                        ll_piece_uniform: float, ll_color: float) -> float:
    b = float(params[0])
    if not (1.0 < b < 3.0):
        return np.inf
    ll = _fast_ll_length(precomp, b, support, length_to_idx) + ll_piece_uniform + ll_color
    return -ll if np.isfinite(ll) else np.inf


def _neg_ll_piece_only(params: np.ndarray, precomp: PrecomputedSteps,
                       support: np.ndarray, length_to_idx: dict[int, int],
                       ll_length_uniform: float, ll_color: float) -> float:
    beta = float(params[0])
    ll = ll_length_uniform + _fast_ll_piece(precomp, beta) + ll_color
    return -ll if np.isfinite(ll) else np.inf


# ---------------------------------------------------------------------------
# Legacy neg-log-likelihood wrappers (DataFrame interface)
# ---------------------------------------------------------------------------

def neg_log_likelihood_joint(params: np.ndarray, participant_steps: pd.DataFrame, support: np.ndarray) -> float:
    b, beta = float(params[0]), float(params[1])
    if not (1.0 < b < 3.0):
        return np.inf
    comps = joint_loglikelihood_components(participant_steps, b, beta, support)
    if not np.isfinite(comps["ll_total"]):
        return np.inf
    return -comps["ll_total"]


def neg_log_likelihood_length_only(params: np.ndarray, participant_steps: pd.DataFrame, support: np.ndarray) -> float:
    b = float(params[0])
    if not (1.0 < b < 3.0):
        return np.inf
    comps = length_only_loglikelihood_components(participant_steps, b, support)
    if not np.isfinite(comps["ll_total"]):
        return np.inf
    return -comps["ll_total"]


def neg_log_likelihood_piece_only(params: np.ndarray, participant_steps: pd.DataFrame, support: np.ndarray) -> float:
    beta = float(params[0])
    comps = piece_only_loglikelihood_components(participant_steps, beta, support)
    if not np.isfinite(comps["ll_total"]):
        return np.inf
    return -comps["ll_total"]


# ---------------------------------------------------------------------------
# Per-participant fitting (uses fast path internally)
# ---------------------------------------------------------------------------

def fit_participant_joint(participant_steps: pd.DataFrame, support: np.ndarray) -> dict[str, float | str]:
    participant = str(participant_steps["participant"].iloc[0])
    precomp = precompute_steps(participant_steps)
    length_to_idx = _build_length_to_idx(support)

    result = minimize(
        _neg_ll_joint,
        np.array([1.5, 1.0]),
        args=(precomp, support, length_to_idx),
        method="L-BFGS-B",
        bounds=[(1.01, 2.99), (-50.0, 50.0)],
    )

    b_hat, beta_hat = float(result.x[0]), float(result.x[1])
    comps = _fast_components(
        _fast_ll_length(precomp, b_hat, support, length_to_idx),
        _fast_ll_piece(precomp, beta_hat),
        _fast_ll_color(precomp),
    )
    n_steps = len(participant_steps)
    ll = comps["ll_total"]
    n_params = 2
    aic = 2 * n_params - 2 * ll
    bic = n_params * math.log(max(n_steps, 1)) - 2 * ll

    return {
        "participant": participant,
        "b_hat": b_hat,
        "beta_hat": beta_hat,
        "ll": ll,
        "nll": -ll,
        "aic": aic,
        "bic": bic,
        "n_steps": n_steps,
        "ll_length": comps["ll_length"],
        "ll_piece": comps["ll_piece"],
        "ll_color": comps["ll_color"],
        "avg_loglik_per_step": ll / n_steps,
        "converged": bool(result.success),
    }


def fit_participant_length_only(participant_steps: pd.DataFrame, support: np.ndarray) -> dict[str, float | str]:
    participant = str(participant_steps["participant"].iloc[0])
    precomp = precompute_steps(participant_steps)
    length_to_idx = _build_length_to_idx(support)
    ll_piece_uniform = _fast_ll_piece_uniform(precomp)
    ll_color = _fast_ll_color(precomp)

    result = minimize(
        _neg_ll_length_only,
        np.array([1.5]),
        args=(precomp, support, length_to_idx, ll_piece_uniform, ll_color),
        method="L-BFGS-B",
        bounds=[(1.01, 2.99)],
    )

    b_hat = float(result.x[0])
    comps = _fast_components(
        _fast_ll_length(precomp, b_hat, support, length_to_idx),
        ll_piece_uniform,
        ll_color,
    )
    n_steps = len(participant_steps)
    ll = comps["ll_total"]
    n_params = 1
    aic = 2 * n_params - 2 * ll
    bic = n_params * math.log(max(n_steps, 1)) - 2 * ll

    return {
        "participant": participant,
        "model": "levy_only",
        "b_hat": b_hat,
        "beta_hat": np.nan,
        "ll": ll,
        "nll": -ll,
        "aic": aic,
        "bic": bic,
        "n_steps": n_steps,
        "ll_length": comps["ll_length"],
        "ll_piece": comps["ll_piece"],
        "ll_color": comps["ll_color"],
        "avg_loglik_per_step": ll / n_steps,
        "converged": bool(result.success),
    }


def fit_participant_piece_only(participant_steps: pd.DataFrame, support: np.ndarray) -> dict[str, float | str]:
    participant = str(participant_steps["participant"].iloc[0])
    precomp = precompute_steps(participant_steps)
    length_to_idx = _build_length_to_idx(support)
    ll_length_uniform = _fast_ll_length_uniform(precomp, support, length_to_idx)
    ll_color = _fast_ll_color(precomp)

    result = minimize(
        _neg_ll_piece_only,
        np.array([1.0]),
        args=(precomp, support, length_to_idx, ll_length_uniform, ll_color),
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0)],
    )

    beta_hat = float(result.x[0])
    comps = _fast_components(
        ll_length_uniform,
        _fast_ll_piece(precomp, beta_hat),
        ll_color,
    )
    n_steps = len(participant_steps)
    ll = comps["ll_total"]
    n_params = 1
    aic = 2 * n_params - 2 * ll
    bic = n_params * math.log(max(n_steps, 1)) - 2 * ll

    return {
        "participant": participant,
        "model": "neighbor_only",
        "b_hat": np.nan,
        "beta_hat": beta_hat,
        "ll": ll,
        "nll": -ll,
        "aic": aic,
        "bic": bic,
        "n_steps": n_steps,
        "ll_length": comps["ll_length"],
        "ll_piece": comps["ll_piece"],
        "ll_color": comps["ll_color"],
        "avg_loglik_per_step": ll / n_steps,
        "converged": bool(result.success),
    }


def fit_all_participants_joint(
    data_dir: str | Path | None = None,
    include_practice: bool = False,
) -> pd.DataFrame:
    joint_steps = build_joint_steps(data_dir=data_dir, include_practice=include_practice)
    if joint_steps.empty:
        raise ValueError("No valid joint-model steps were constructed.")

    support = get_global_graph_length_support(joint_steps)
    results = []
    for participant, sub_df in joint_steps.groupby("participant"):
        sub_df = sub_df.reset_index(drop=True)
        results.append(fit_participant_joint(sub_df, support))

    results_df = pd.DataFrame(results).sort_values("participant").reset_index(drop=True)
    results_df["support"] = [tuple(int(x) for x in support)] * len(results_df)
    return results_df


def fit_all_participants_model_comparison(
    data_dir: str | Path | None = None,
    include_practice: bool = False,
) -> pd.DataFrame:
    joint_steps = build_joint_steps(data_dir=data_dir, include_practice=include_practice)
    if joint_steps.empty:
        raise ValueError("No valid joint-model steps were constructed.")

    support = get_global_graph_length_support(joint_steps)
    results = []

    for participant, sub_df in joint_steps.groupby("participant"):
        sub_df = sub_df.reset_index(drop=True)

        joint_res = fit_participant_joint(sub_df, support)
        joint_res["model"] = "joint"
        results.append(joint_res)
        results.append(fit_participant_length_only(sub_df, support))
        results.append(fit_participant_piece_only(sub_df, support))

    results_df = pd.DataFrame(results).sort_values(["participant", "model"]).reset_index(drop=True)
    results_df["support"] = [tuple(int(x) for x in support)] * len(results_df)

    baseline = (
        results_df[results_df["model"] == "joint"][
            ["participant", "ll", "nll", "aic", "bic"]
        ]
        .rename(
            columns={
                "ll": "ll_joint",
                "nll": "nll_joint",
                "aic": "aic_joint",
                "bic": "bic_joint",
            }
        )
    )
    results_df = results_df.merge(baseline, on="participant", how="left")
    results_df["delta_nll_vs_joint"] = results_df["nll"] - results_df["nll_joint"]
    results_df["delta_aic_vs_joint"] = results_df["aic"] - results_df["aic_joint"]
    results_df["delta_bic_vs_joint"] = results_df["bic"] - results_df["bic_joint"]
    return results_df


def main():
    if len(os.sys.argv) > 1:
        data_dir = os.sys.argv[1]
    else:
        data_dir = default_data_dir()

    results_df = fit_all_participants_joint(data_dir=data_dir, include_practice=False)
    print(results_df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
