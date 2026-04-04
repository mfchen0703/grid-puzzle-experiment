from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fit_softmax import build_all_maps, build_fitting_steps, parse_csv
from global_stats import region_centroid
from graph_analysis import bfs_distances


@dataclass(frozen=True)
class TransitionRecord:
    participant: str
    round_label: str
    step_in_round: int
    prev_region: int
    chosen_region: int
    candidate_regions: tuple[int, ...]
    graph_lengths: tuple[float, ...]
    euclidean_lengths: tuple[float, ...]
    chosen_graph_length: float
    chosen_euclidean_length: float


def default_data_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def _sorted_round_labels(labels):
    return sorted(labels, key=lambda x: (0, int(x)) if x.isdigit() else (-1, x))


def _pairwise_graph_lengths(adjacency) -> np.ndarray:
    n = len(adjacency)
    dmat = np.full((n, n), np.inf, dtype=float)
    for node in range(n):
        dmat[node, node] = 0.0
        dists = bfs_distances(node, adjacency)
        for other, dist in dists.items():
            dmat[node, other] = float(dist)
    return dmat


def _pairwise_euclidean_lengths(regions) -> np.ndarray:
    centroids = np.array([region_centroid(cells) for cells in regions], dtype=float)
    diffs = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=2))


def precompute_round_lengths():
    maps = build_all_maps()
    info = {}
    for round_label, (regions, adjacency) in maps.items():
        info[round_label] = {
            "regions": regions,
            "adjacency": adjacency,
            "graph": _pairwise_graph_lengths(adjacency),
            "euclidean": _pairwise_euclidean_lengths(regions),
        }
    return maps, info


def build_transition_table(data_dir: str | Path | None = None, include_practice: bool = False) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    maps, round_info = precompute_round_lengths()
    rows: list[TransitionRecord] = []

    for filepath in sorted(data_dir.glob("data_*.csv")):
        participant = filepath.stem.replace("data_", "")
        actions = parse_csv(str(filepath))
        steps = build_fitting_steps(actions, maps, include_practice=include_practice)

        prev_round = None
        prev_region = None
        step_in_round = -1

        for step in steps:
            round_label = step["round"]
            if round_label != prev_round:
                prev_round = round_label
                prev_region = None
                step_in_round = 0
            else:
                step_in_round += 1

            if prev_region is not None:
                candidates = tuple(int(r) for r in step["uncolored"])
                graph_lengths = tuple(
                    float(round_info[round_label]["graph"][prev_region, rid])
                    for rid in candidates
                )
                euclidean_lengths = tuple(
                    float(round_info[round_label]["euclidean"][prev_region, rid])
                    for rid in candidates
                )
                chosen_region = int(step["chosen_region"])

                rows.append(
                    TransitionRecord(
                        participant=participant,
                        round_label=round_label,
                        step_in_round=step_in_round,
                        prev_region=int(prev_region),
                        chosen_region=chosen_region,
                        candidate_regions=candidates,
                        graph_lengths=graph_lengths,
                        euclidean_lengths=euclidean_lengths,
                        chosen_graph_length=float(round_info[round_label]["graph"][prev_region, chosen_region]),
                        chosen_euclidean_length=float(round_info[round_label]["euclidean"][prev_region, chosen_region]),
                    )
                )

            prev_region = step["chosen_region"]

    df = pd.DataFrame([row.__dict__ for row in rows])
    if df.empty:
        return df

    df["round_num"] = df["round_label"].map(lambda x: int(x) if x.isdigit() else -1)
    return df.sort_values(["participant", "round_num", "step_in_round"]).reset_index(drop=True)


def _weights_random(lengths: np.ndarray, _: float | None = None) -> np.ndarray:
    return np.ones(len(lengths), dtype=float)


def _weights_neighbor(lengths: np.ndarray, _: float | None = None) -> np.ndarray:
    nearest = np.min(lengths)
    return (np.isclose(lengths, nearest)).astype(float)


def _weights_jump(lengths: np.ndarray, _: float | None = None) -> np.ndarray:
    nearest = np.min(lengths)
    weights = (~np.isclose(lengths, nearest)).astype(float)
    if np.all(weights == 0):
        return np.ones(len(lengths), dtype=float)
    return weights


def _weights_levy(lengths: np.ndarray, mu: float | None = None) -> np.ndarray:
    if mu is None:
        raise ValueError("Levy model requires mu.")
    safe_lengths = np.maximum(lengths, 1e-9)
    return safe_lengths ** (-mu)


MODEL_SPECS: dict[str, dict[str, object]] = {
    "random": {"n_params": 0, "weight_fn": _weights_random},
    "neighbor": {"n_params": 0, "weight_fn": _weights_neighbor},
    "jump": {"n_params": 0, "weight_fn": _weights_jump},
    "levy": {"n_params": 1, "weight_fn": _weights_levy},
}


def get_length_array(row: pd.Series, metric: str) -> np.ndarray:
    key = "graph_lengths" if metric == "graph" else "euclidean_lengths"
    return np.array(row[key], dtype=float)


def get_chosen_length(row: pd.Series, metric: str) -> float:
    key = "chosen_graph_length" if metric == "graph" else "chosen_euclidean_length"
    return float(row[key])


def normalized_weights(lengths: np.ndarray, model_name: str, mu: float | None = None) -> np.ndarray:
    weight_fn: Callable[[np.ndarray, float | None], np.ndarray] = MODEL_SPECS[model_name]["weight_fn"]  # type: ignore[index]
    weights = weight_fn(lengths, mu)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full(len(lengths), 1.0 / len(lengths), dtype=float)
    return weights / total


def row_log_prob(row: pd.Series, metric: str, model_name: str, mu: float | None = None) -> float:
    lengths = get_length_array(row, metric)
    probs = normalized_weights(lengths, model_name, mu)
    chosen_mask = np.isclose(lengths, get_chosen_length(row, metric))
    if not np.any(chosen_mask):
        raise ValueError("Chosen length does not match any candidate length.")
    chosen_prob = float(np.sum(probs[chosen_mask]))
    return math.log(max(chosen_prob, 1e-300))


def compute_step_loglikelihoods(
    transitions_df: pd.DataFrame,
    metric: str,
    model_name: str,
    mu: float | None = None,
) -> pd.DataFrame:
    rows = []
    for idx, row in transitions_df.reset_index(drop=True).iterrows():
        log_prob = row_log_prob(row, metric, model_name, mu)
        rows.append(
            {
                "step_index": idx,
                "participant": row["participant"],
                "round_label": row["round_label"],
                "step_in_round": row["step_in_round"],
                "metric": metric,
                "model": model_name,
                "mu": mu if model_name == "levy" else np.nan,
                "chosen_length": get_chosen_length(row, metric),
                "loglik": log_prob,
                "likelihood": math.exp(log_prob),
            }
        )
    return pd.DataFrame(rows)


def fit_participant_model(participant_df: pd.DataFrame, metric: str, model_name: str):
    n_steps = len(participant_df)

    if model_name != "levy":
        ll = float(sum(row_log_prob(row, metric, model_name) for _, row in participant_df.iterrows()))
        n_params = int(MODEL_SPECS[model_name]["n_params"])
        return {
            "participant": participant_df["participant"].iloc[0],
            "metric": metric,
            "model": model_name,
            "mu": np.nan,
            "ll": ll,
            "nll": -ll,
            "aic": 2 * n_params - 2 * ll,
            "bic": n_params * math.log(max(n_steps, 1)) - 2 * ll,
            "n_steps": n_steps,
        }

    def objective(mu: float) -> float:
        return -sum(row_log_prob(row, metric, "levy", mu) for _, row in participant_df.iterrows())

    result = minimize_scalar(objective, bounds=(1.01, 2.99), method="bounded")
    mu_hat = float(result.x)
    ll = -float(result.fun)
    n_params = int(MODEL_SPECS["levy"]["n_params"])
    return {
        "participant": participant_df["participant"].iloc[0],
        "metric": metric,
        "model": "levy",
        "mu": mu_hat,
        "ll": ll,
        "nll": -ll,
        "aic": 2 * n_params - 2 * ll,
        "bic": n_params * math.log(max(n_steps, 1)) - 2 * ll,
        "n_steps": n_steps,
    }


def fit_pooled_model(transitions_df: pd.DataFrame, metric: str, model_name: str):
    n_steps = len(transitions_df)

    if model_name != "levy":
        ll = float(sum(row_log_prob(row, metric, model_name) for _, row in transitions_df.iterrows()))
        n_params = int(MODEL_SPECS[model_name]["n_params"])
        return {
            "metric": metric,
            "model": model_name,
            "mu": np.nan,
            "ll": ll,
            "nll": -ll,
            "aic": 2 * n_params - 2 * ll,
            "bic": n_params * math.log(max(n_steps, 1)) - 2 * ll,
            "n_steps": n_steps,
        }

    def objective(mu: float) -> float:
        return -sum(row_log_prob(row, metric, "levy", mu) for _, row in transitions_df.iterrows())

    result = minimize_scalar(objective, bounds=(1.01, 2.99), method="bounded")
    mu_hat = float(result.x)
    ll = -float(result.fun)
    n_params = int(MODEL_SPECS["levy"]["n_params"])
    return {
        "metric": metric,
        "model": "levy",
        "mu": mu_hat,
        "ll": ll,
        "nll": -ll,
        "aic": 2 * n_params - 2 * ll,
        "bic": n_params * math.log(max(n_steps, 1)) - 2 * ll,
        "n_steps": n_steps,
    }


def fit_all_models(transitions_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in {"graph", "euclidean"}:
        raise ValueError("metric must be 'graph' or 'euclidean'.")

    results = []
    for participant, sub_df in transitions_df.groupby("participant"):
        sub_df = sub_df.reset_index(drop=True)
        for model_name in MODEL_SPECS:
            results.append(fit_participant_model(sub_df, metric, model_name))
    results_df = pd.DataFrame(results)
    results_df["avg_loglik_per_step"] = results_df["ll"] / results_df["n_steps"]
    results_df["avg_likelihood_per_step"] = np.exp(results_df["avg_loglik_per_step"])
    return results_df.sort_values(["participant", "bic", "aic"]).reset_index(drop=True)


def fit_all_models_pooled(transitions_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in {"graph", "euclidean"}:
        raise ValueError("metric must be 'graph' or 'euclidean'.")

    results = [fit_pooled_model(transitions_df.reset_index(drop=True), metric, model_name) for model_name in MODEL_SPECS]
    results_df = pd.DataFrame(results)
    results_df["avg_loglik_per_step"] = results_df["ll"] / results_df["n_steps"]
    results_df["avg_likelihood_per_step"] = np.exp(results_df["avg_loglik_per_step"])
    return results_df.sort_values(["bic", "aic"]).reset_index(drop=True)


def compute_pooled_step_loglikelihoods(transitions_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    fitted = fit_all_models_pooled(transitions_df, metric)
    step_tables = []
    for _, result in fitted.iterrows():
        mu = None if result["model"] != "levy" else float(result["mu"])
        step_tables.append(
            compute_step_loglikelihoods(
                transitions_df,
                metric=metric,
                model_name=result["model"],
                mu=mu,
            )
        )
    return pd.concat(step_tables, ignore_index=True)


def summarize_best_models(results_df: pd.DataFrame, criterion: str = "bic") -> pd.DataFrame:
    idx = results_df.groupby(["participant", "metric"])[criterion].idxmin()
    best_df = results_df.loc[idx].sort_values(["metric", "participant"]).reset_index(drop=True)
    return best_df


def best_model_counts(best_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        best_df.groupby(["metric", "model"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["metric", "count", "model"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return counts


def add_random_baseline_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [col for col in ["participant", "metric"] if col in results_df.columns]
    baseline = (
        results_df[results_df["model"] == "random"][
            key_cols + ["ll", "aic", "bic", "avg_loglik_per_step", "avg_likelihood_per_step"]
        ]
        .rename(
            columns={
                "ll": "ll_random",
                "aic": "aic_random",
                "bic": "bic_random",
                "avg_loglik_per_step": "avg_loglik_per_step_random",
                "avg_likelihood_per_step": "avg_likelihood_per_step_random",
            }
        )
    )
    merged = results_df.merge(baseline, on=key_cols, how="left")
    merged["delta_ll_vs_random"] = merged["ll"] - merged["ll_random"]
    merged["delta_aic_vs_random"] = merged["aic"] - merged["aic_random"]
    merged["delta_bic_vs_random"] = merged["bic"] - merged["bic_random"]
    merged["delta_avg_loglik_per_step_vs_random"] = (
        merged["avg_loglik_per_step"] - merged["avg_loglik_per_step_random"]
    )
    merged["avg_likelihood_ratio_vs_random"] = (
        merged["avg_likelihood_per_step"] / merged["avg_likelihood_per_step_random"]
    )
    return merged


def summarize_model_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_random_baseline_deltas(results_df)
    summary = (
        enriched.groupby(["metric", "model"], as_index=False)
        .agg(
            participants=("participant", "nunique"),
            mean_ll=("ll", "mean"),
            mean_aic=("aic", "mean"),
            mean_bic=("bic", "mean"),
            mean_avg_loglik_per_step=("avg_loglik_per_step", "mean"),
            mean_avg_likelihood_per_step=("avg_likelihood_per_step", "mean"),
            mean_delta_ll_vs_random=("delta_ll_vs_random", "mean"),
            mean_delta_aic_vs_random=("delta_aic_vs_random", "mean"),
            mean_delta_bic_vs_random=("delta_bic_vs_random", "mean"),
            mean_likelihood_ratio_vs_random=("avg_likelihood_ratio_vs_random", "mean"),
            sd_delta_ll_vs_random=("delta_ll_vs_random", "std"),
            sd_delta_aic_vs_random=("delta_aic_vs_random", "std"),
            sd_delta_bic_vs_random=("delta_bic_vs_random", "std"),
        )
        .sort_values(["metric", "model"])
        .reset_index(drop=True)
    )
    return summary


def summarize_pooled_model_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_random_baseline_deltas(results_df)
    cols = [
        "metric",
        "model",
        "mu",
        "n_steps",
        "ll",
        "nll",
        "aic",
        "bic",
        "avg_loglik_per_step",
        "avg_likelihood_per_step",
        "delta_ll_vs_random",
        "delta_aic_vs_random",
        "delta_bic_vs_random",
        "delta_avg_loglik_per_step_vs_random",
        "avg_likelihood_ratio_vs_random",
    ]
    return enriched[cols].sort_values(["metric", "bic", "aic"]).reset_index(drop=True)


def marginal_length_support_and_counts(transitions_df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    observed = transitions_df[f"chosen_{metric}_length"].to_numpy(dtype=float)
    support, counts = np.unique(observed, return_counts=True)
    return support.astype(float), counts.astype(float)


def candidate_length_support(transitions_df: pd.DataFrame, metric: str) -> np.ndarray:
    key = "graph_lengths" if metric == "graph" else "euclidean_lengths"
    all_lengths: list[float] = []
    for lengths in transitions_df[key]:
        all_lengths.extend(float(x) for x in lengths if float(x) > 0)
    if not all_lengths:
        return np.array([], dtype=float)
    return np.unique(np.array(all_lengths, dtype=float))


def fit_marginal_levy_distribution(transitions_df: pd.DataFrame, metric: str) -> dict[str, float | np.ndarray]:
    support, counts = marginal_length_support_and_counts(transitions_df, metric)
    positive_mask = support > 0
    support = support[positive_mask]
    counts = counts[positive_mask]
    if len(support) == 0:
        return {"mu": np.nan, "support": support, "probs": np.array([], dtype=float)}

    def objective(mu: float) -> float:
        weights = support ** (-mu)
        probs = weights / np.sum(weights)
        return -float(np.sum(counts * np.log(np.maximum(probs, 1e-300))))

    result = minimize_scalar(objective, bounds=(1.01, 2.99), method="bounded")
    mu_hat = float(result.x)
    weights = support ** (-mu_hat)
    probs = weights / np.sum(weights)
    return {"mu": mu_hat, "support": support, "probs": probs}


def marginal_model_distribution(
    transitions_df: pd.DataFrame,
    metric: str,
    model_name: str,
    mu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    support = candidate_length_support(transitions_df, metric)
    if len(support) == 0:
        return support, np.array([], dtype=float)

    if model_name == "levy":
        if mu is None:
            raise ValueError("Marginal Levy distribution requires mu.")
        weights = support ** (-mu)
        probs = weights / np.sum(weights)
        return support, probs

    probs_by_length = {float(length): 0.0 for length in support}
    for _, row in transitions_df.iterrows():
        lengths = get_length_array(row, metric)
        positive_lengths = lengths[lengths > 0]
        if len(positive_lengths) == 0:
            continue
        candidate_probs = normalized_weights(positive_lengths, model_name, None)
        uniq_lengths, inverse = np.unique(positive_lengths, return_inverse=True)
        for idx, length in enumerate(uniq_lengths):
            probs_by_length[float(length)] += float(np.sum(candidate_probs[inverse == idx]))

    probs = np.array([probs_by_length[float(length)] for length in support], dtype=float)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return support, probs


def pooled_length_histogram(
    transitions_df: pd.DataFrame,
    metric: str,
    model_name: str,
    mu_by_participant: dict[str, float] | None = None,
    bins: int | np.ndarray = 20,
    density: bool = True,
):
    observed = transitions_df[f"chosen_{metric}_length"].to_numpy(dtype=float)
    hist_obs, edges = np.histogram(observed, bins=bins, density=density)
    hist_pred = np.zeros_like(hist_obs, dtype=float)

    for _, row in transitions_df.iterrows():
        lengths = get_length_array(row, metric)
        if mu_by_participant is None:
            mu = None
        elif row["participant"] in mu_by_participant:
            mu = mu_by_participant[row["participant"]]
        elif "__pooled__" in mu_by_participant:
            mu = mu_by_participant["__pooled__"]
        else:
            mu = None
        probs = normalized_weights(lengths, model_name, mu)
        row_hist = np.zeros_like(hist_obs, dtype=float)
        for length, prob in zip(lengths, probs):
            idx = np.searchsorted(edges, length, side="right") - 1
            idx = min(max(idx, 0), len(hist_obs) - 1)
            row_hist[idx] += prob
        hist_pred += row_hist

    if hist_pred.sum() > 0:
        if density:
            hist_pred = hist_pred / np.sum(hist_pred) / np.diff(edges)
        else:
            hist_pred = hist_pred / np.sum(hist_pred) * len(observed)
    return hist_obs, hist_pred, edges


def plot_best_model_counts(best_df: pd.DataFrame, metric: str, ax=None):
    ax = ax or plt.gca()
    sub = best_df[best_df["metric"] == metric]
    order = ["random", "neighbor", "jump", "levy"]
    counts = sub["model"].value_counts().reindex(order, fill_value=0)
    ax.bar(counts.index, counts.values, color=["0.6", "tab:blue", "tab:red", "tab:green"])
    ax.set_title(f"{metric.title()} metric: best model counts")
    ax.set_ylabel("Participants")
    return ax


def plot_pooled_histogram(
    transitions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    metric: str,
    ax=None,
    y_mode: str = "count",
    levy_plot_mode: str = "marginal",
):
    ax = ax or plt.gca()
    density = y_mode != "count"
    if metric == "graph":
        max_len = int(np.max(transitions_df["chosen_graph_length"]))
        bins = np.arange(0.5, max_len + 1.5, 1.0)
    else:
        bins = 20

    model_order = ["random", "neighbor", "jump", "levy"]
    colors = {"random": "0.5", "neighbor": "tab:blue", "jump": "tab:red", "levy": "tab:green"}
    support_obs, count_obs = marginal_length_support_and_counts(transitions_df, metric)
    support_obs = support_obs[support_obs > 0]
    count_obs = count_obs[-len(support_obs):] if len(support_obs) else count_obs
    if "participant" in results_df.columns:
        levy_map = (
            results_df[(results_df["metric"] == metric) & (results_df["model"] == "levy")]
            .set_index("participant")["mu"]
            .to_dict()
        )
    else:
        levy_rows = results_df[(results_df["metric"] == metric) & (results_df["model"] == "levy")]
        levy_mu = None if levy_rows.empty else float(levy_rows["mu"].iloc[0])
        levy_map = None

    hist_obs, _, edges = pooled_length_histogram(
        transitions_df, metric, "random", None, bins=bins, density=density
    )
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    ax.bar(centers, hist_obs, width=widths, alpha=0.35, color="black", label="human")

    for model_name in model_order:
        if levy_plot_mode == "conditional":
            if model_name == "levy":
                if levy_map is not None:
                    _, hist_pred, pred_edges = pooled_length_histogram(
                        transitions_df, metric, model_name, levy_map, bins=bins, density=density
                    )
                else:
                    _, hist_pred, pred_edges = pooled_length_histogram(
                        transitions_df,
                        metric,
                        model_name,
                        {"__pooled__": levy_mu} if levy_mu is not None else None,
                        bins=bins,
                        density=density,
                    )
            else:
                _, hist_pred, pred_edges = pooled_length_histogram(
                    transitions_df, metric, model_name, None, bins=bins, density=density
                )
            pred_centers = (pred_edges[:-1] + pred_edges[1:]) / 2
        else:
            if model_name == "levy":
                levy_fit = fit_marginal_levy_distribution(transitions_df, metric)
                pred_support, pred_probs = marginal_model_distribution(
                    transitions_df,
                    metric,
                    model_name,
                    mu=float(levy_fit["mu"]),
                )
            else:
                pred_support, pred_probs = marginal_model_distribution(
                    transitions_df, metric, model_name
                )
            pred_centers = pred_support
            if density:
                hist_pred = pred_probs
            else:
                hist_pred = pred_probs * len(transitions_df)
        ax.plot(pred_centers, hist_pred, marker="o", linewidth=2, color=colors[model_name], label=model_name)

    ax.set_title(f"{metric.title()} step-length distribution")
    ax.set_xlabel("Step length")
    ax.set_ylabel("Count" if not density else "Density")
    ax.legend()
    return ax


def plot_metric_deltas_vs_random(results_df: pd.DataFrame, metric: str, axes=None):
    enriched = add_random_baseline_deltas(results_df)
    sub = enriched[enriched["metric"] == metric].copy()
    order = ["random", "neighbor", "jump", "levy"]
    stats = (
        sub.groupby("model")
        .agg(
            mean_delta_ll=("delta_ll_vs_random", "mean"),
            mean_delta_aic=("delta_aic_vs_random", "mean"),
            mean_delta_bic=("delta_bic_vs_random", "mean"),
            sd_delta_ll=("delta_ll_vs_random", "std"),
            sd_delta_aic=("delta_aic_vs_random", "std"),
            sd_delta_bic=("delta_bic_vs_random", "std"),
        )
        .reindex(order)
    )

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(len(order))
    configs = [
        ("mean_delta_ll", "sd_delta_ll", "Delta LL vs random", "tab:green"),
        ("mean_delta_aic", "sd_delta_aic", "Delta AIC vs random", "tab:blue"),
        ("mean_delta_bic", "sd_delta_bic", "Delta BIC vs random", "tab:purple"),
    ]

    for ax, (mean_col, sd_col, title, color) in zip(axes, configs):
        means = stats[mean_col].to_numpy(dtype=float)
        sds = stats[sd_col].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, means, yerr=sds, capsize=4, color=color, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_title(f"{metric.title()}: {title}")
    return axes


def plot_pooled_metric_deltas_vs_random(results_df: pd.DataFrame, metric: str, axes=None):
    sub = add_random_baseline_deltas(results_df)
    sub = sub[sub["metric"] == metric].copy()
    order = ["random", "neighbor", "jump", "levy"]
    sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
    sub = sub.sort_values("model")

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(len(sub))
    configs = [
        ("delta_ll_vs_random", "Delta LL vs random", "tab:green"),
        ("delta_aic_vs_random", "Delta AIC vs random", "tab:blue"),
        ("delta_bic_vs_random", "Delta BIC vs random", "tab:purple"),
    ]

    for ax, (col, title, color) in zip(axes, configs):
        ax.bar(x, sub[col].to_numpy(dtype=float), color=color, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist())
        ax.set_title(f"{metric.title()}: {title}")
    return axes


def plot_step_loglikelihood_distribution(step_ll_df: pd.DataFrame, metric: str, ax=None, kind: str = "box"):
    ax = ax or plt.gca()
    sub = step_ll_df[step_ll_df["metric"] == metric].copy()
    order = ["random", "neighbor", "jump", "levy"]
    data = [sub[sub["model"] == model]["loglik"].to_numpy(dtype=float) for model in order]

    if kind == "box":
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_ylabel("Per-step log-likelihood")
    elif kind == "violin":
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order)
        ax.set_ylabel("Per-step log-likelihood")
    else:
        raise ValueError("kind must be 'box' or 'violin'.")

    ax.set_title(f"{metric.title()} per-step log-likelihoods (all steps)")
    return ax


def plot_observed_length_histogram(transitions_df: pd.DataFrame, metric: str, ax=None):
    ax = ax or plt.gca()
    observed = transitions_df[f"chosen_{metric}_length"].to_numpy(dtype=float)
    if metric == "graph":
        max_len = int(np.max(observed))
        bins = np.arange(0.5, max_len + 1.5, 1.0)
    else:
        bins = 20
    ax.hist(observed, bins=bins, color="0.3", alpha=0.8, density=True)
    ax.set_title(f"{metric.title()} observed step lengths (n={len(observed)})")
    ax.set_xlabel("Step length")
    ax.set_ylabel("Density")
    return ax


def participant_model_matrix(results_df: pd.DataFrame, metric: str, criterion: str = "bic") -> pd.DataFrame:
    sub = results_df[results_df["metric"] == metric].copy()
    return (
        sub.pivot(index="participant", columns="model", values=criterion)
        .reindex(columns=["random", "neighbor", "jump", "levy"])
        .sort_index()
    )


def print_model_definition():
    print("Model definitions")
    print("- random: all currently available targets are equally likely")
    print("- neighbor: choose among shortest available steps under the current metric")
    print("- jump: avoid shortest available steps whenever longer options exist")
    print("- levy: choose with probability proportional to L^{-mu}, 1 < mu < 3")
