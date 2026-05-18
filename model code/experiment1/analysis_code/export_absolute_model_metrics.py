from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from unified_model_comparison import (
    build_unified_steps,
    default_data_dir,
    summarize_deltas,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_RESULTS_CSV = RESULTS_DIR / "all_models_comparison_results.csv"


def _available_length_support(step: pd.Series) -> list[int]:
    return sorted({int(x) for x in step["available_graph_lengths"] if int(x) > 0})


def random_step_logprob(step: pd.Series) -> float:
    """Uniform random baseline in the model-1 action space.

    The random policy first samples an available graph step length uniformly,
    then samples a region uniformly among uncolored regions at that length,
    then samples one currently legal color uniformly.
    """
    support = _available_length_support(step)
    same_length_n = len(step["candidate_regions_same_length"])
    n_legal_colors = int(step["num_legal_colors"])
    if len(support) == 0 or same_length_n == 0 or n_legal_colors <= 0:
        return -math.inf
    return -math.log(len(support)) - math.log(same_length_n) - math.log(n_legal_colors)


def compute_random_baseline(steps_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, sub_df in steps_df.groupby("participant", sort=True):
        ll = float(sum(random_step_logprob(step) for _, step in sub_df.iterrows()))
        n_steps = int(len(sub_df))
        nll = float(-ll)
        rows.append(
            {
                "participant": str(participant),
                "random_ll": ll,
                "random_nll": nll,
                "random_n_steps": n_steps,
                "random_mean_log_likelihood_per_step": ll / n_steps,
                "random_mean_nll_per_step": nll / n_steps,
                "random_geomean_likelihood_per_step": math.exp(ll / n_steps),
            }
        )
    return pd.DataFrame(rows)


def build_absolute_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = pd.read_csv(MODEL_RESULTS_CSV)
    steps_df = build_unified_steps(default_data_dir())
    random_df = compute_random_baseline(steps_df)

    merged = results_df.merge(random_df, on="participant", how="left")
    merged["mean_log_likelihood_per_step"] = merged["ll"] / merged["n_steps"]
    merged["mean_nll_per_step"] = merged["nll"] / merged["n_steps"]
    merged["geomean_likelihood_per_step"] = np.exp(merged["mean_log_likelihood_per_step"])
    merged["ll_gain_vs_random"] = merged["ll"] - merged["random_ll"]
    merged["nll_reduction_vs_random"] = merged["random_nll"] - merged["nll"]
    merged["mean_log_likelihood_gain_vs_random_per_step"] = (
        merged["mean_log_likelihood_per_step"] - merged["random_mean_log_likelihood_per_step"]
    )
    merged["mean_nll_reduction_vs_random_per_step"] = (
        merged["random_mean_nll_per_step"] - merged["mean_nll_per_step"]
    )
    merged["likelihood_ratio_vs_random_per_step"] = np.exp(
        merged["mean_log_likelihood_gain_vs_random_per_step"]
    )

    model_summary = (
        merged.groupby(["model_key", "model"], as_index=False)
        .agg(
            n_participants=("participant", "nunique"),
            total_steps=("n_steps", "sum"),
            total_ll=("ll", "sum"),
            total_nll=("nll", "sum"),
            total_random_ll=("random_ll", "sum"),
            total_random_nll=("random_nll", "sum"),
            total_aic=("aic", "sum"),
            total_bic=("bic", "sum"),
            mean_ll=("ll", "mean"),
            mean_nll=("nll", "mean"),
            mean_aic=("aic", "mean"),
            mean_bic=("bic", "mean"),
            mean_log_likelihood_per_step=("mean_log_likelihood_per_step", "mean"),
            mean_nll_per_step=("mean_nll_per_step", "mean"),
            geomean_likelihood_per_step=("geomean_likelihood_per_step", "mean"),
            mean_log_likelihood_gain_vs_random_per_step=(
                "mean_log_likelihood_gain_vs_random_per_step",
                "mean",
            ),
            mean_nll_reduction_vs_random_per_step=(
                "mean_nll_reduction_vs_random_per_step",
                "mean",
            ),
            likelihood_ratio_vs_random_per_step=("likelihood_ratio_vs_random_per_step", "mean"),
        )
        .sort_values("mean_nll")
        .reset_index(drop=True)
    )
    model_summary["global_log_likelihood_per_step"] = (
        model_summary["total_ll"] / model_summary["total_steps"]
    )
    model_summary["global_nll_per_step"] = (
        model_summary["total_nll"] / model_summary["total_steps"]
    )
    model_summary["global_geomean_likelihood_per_step"] = np.exp(
        model_summary["global_log_likelihood_per_step"]
    )
    model_summary["global_random_log_likelihood_per_step"] = (
        model_summary["total_random_ll"] / model_summary["total_steps"]
    )
    model_summary["global_random_nll_per_step"] = (
        model_summary["total_random_nll"] / model_summary["total_steps"]
    )
    model_summary["global_log_likelihood_gain_vs_random_per_step"] = (
        model_summary["global_log_likelihood_per_step"]
        - model_summary["global_random_log_likelihood_per_step"]
    )
    model_summary["global_nll_reduction_vs_random_per_step"] = (
        model_summary["global_random_nll_per_step"] - model_summary["global_nll_per_step"]
    )
    model_summary["global_likelihood_ratio_vs_random_per_step"] = np.exp(
        model_summary["global_log_likelihood_gain_vs_random_per_step"]
    )

    random_summary = pd.DataFrame(
        [
            {
                "model_key": "random",
                "model": "随机模型",
                "n_participants": random_df["participant"].nunique(),
                "total_steps": int(random_df["random_n_steps"].sum()),
                "total_ll": float(random_df["random_ll"].sum()),
                "total_nll": float(random_df["random_nll"].sum()),
                "total_random_ll": float(random_df["random_ll"].sum()),
                "total_random_nll": float(random_df["random_nll"].sum()),
                "total_aic": np.nan,
                "total_bic": np.nan,
                "mean_ll": float(random_df["random_ll"].mean()),
                "mean_nll": float(random_df["random_nll"].mean()),
                "mean_aic": np.nan,
                "mean_bic": np.nan,
                "mean_log_likelihood_per_step": float(
                    random_df["random_mean_log_likelihood_per_step"].mean()
                ),
                "mean_nll_per_step": float(random_df["random_mean_nll_per_step"].mean()),
                "geomean_likelihood_per_step": float(
                    random_df["random_geomean_likelihood_per_step"].mean()
                ),
                "mean_log_likelihood_gain_vs_random_per_step": 0.0,
                "mean_nll_reduction_vs_random_per_step": 0.0,
                "likelihood_ratio_vs_random_per_step": 1.0,
                "global_log_likelihood_per_step": float(
                    random_df["random_ll"].sum() / random_df["random_n_steps"].sum()
                ),
                "global_nll_per_step": float(
                    random_df["random_nll"].sum() / random_df["random_n_steps"].sum()
                ),
                "global_geomean_likelihood_per_step": float(
                    math.exp(random_df["random_ll"].sum() / random_df["random_n_steps"].sum())
                ),
                "global_random_log_likelihood_per_step": float(
                    random_df["random_ll"].sum() / random_df["random_n_steps"].sum()
                ),
                "global_random_nll_per_step": float(
                    random_df["random_nll"].sum() / random_df["random_n_steps"].sum()
                ),
                "global_log_likelihood_gain_vs_random_per_step": 0.0,
                "global_nll_reduction_vs_random_per_step": 0.0,
                "global_likelihood_ratio_vs_random_per_step": 1.0,
            }
        ]
    )
    summary_with_random = pd.concat([model_summary, random_summary], ignore_index=True)

    return merged, summary_with_random


def main() -> None:
    participant_metrics, model_summary = build_absolute_metrics()
    participant_metrics.to_csv(
        RESULTS_DIR / "all_models_absolute_metrics_by_participant.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_summary.to_csv(
        RESULTS_DIR / "all_models_absolute_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_cols = [
        "model",
        "total_nll",
        "mean_nll",
        "global_nll_per_step",
        "global_geomean_likelihood_per_step",
        "global_log_likelihood_gain_vs_random_per_step",
        "global_likelihood_ratio_vs_random_per_step",
    ]
    print(model_summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
