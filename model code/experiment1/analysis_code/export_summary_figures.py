from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from fit_softmax import (
    build_all_maps,
    build_fitting_steps,
    compute_utilities,
    parse_csv,
)
from global_stats import analyze_participant
from step_length_models import (
    build_transition_table,
    fit_all_models_pooled,
    summarize_pooled_model_comparison,
    plot_pooled_histogram,
)
from fit_levy_neighbor_joint import fit_all_participants_model_comparison


ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT.parent / "data").resolve()
OUT_DIR = (ROOT / "summary_figures").resolve()


def configure_matplotlib() -> None:
    plt.style.use("ggplot")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.unicode_minus"] = False


def _fit_basic_choice_models() -> pd.DataFrame:
    maps = build_all_maps()
    rows = []
    data_files = sorted(DATA_DIR.glob("data_*.csv"))

    def nll_neighbor_only(params, fitting_steps):
        total = 0.0
        for step in fitting_steps:
            utilities = compute_utilities(
                step["uncolored"], step["regions"], step["adjacency"], step["current_colors"], np.array([params[0], 0.0])
            )
            chosen_idx = step["uncolored"].index(step["chosen_region"])
            total -= utilities[chosen_idx] - logsumexp(utilities)
        return total

    def nll_distance_only(params, fitting_steps):
        total = 0.0
        for step in fitting_steps:
            utilities = compute_utilities(
                step["uncolored"], step["regions"], step["adjacency"], step["current_colors"], np.array([0.0, params[0]])
            )
            chosen_idx = step["uncolored"].index(step["chosen_region"])
            total -= utilities[chosen_idx] - logsumexp(utilities)
        return total

    def nll_neighbor_distance(params, fitting_steps):
        total = 0.0
        for step in fitting_steps:
            utilities = compute_utilities(
                step["uncolored"], step["regions"], step["adjacency"], step["current_colors"], np.array([params[0], params[1]])
            )
            chosen_idx = step["uncolored"].index(step["chosen_region"])
            total -= utilities[chosen_idx] - logsumexp(utilities)
        return total

    model_specs = [
        ("random", None, None, 0),
        ("neighbor", nll_neighbor_only, np.array([1.0]), 1),
        ("distance", nll_distance_only, np.array([0.1]), 1),
        ("neighbor+distance", nll_neighbor_distance, np.array([1.0, 0.1]), 2),
    ]

    for fp in data_files:
        participant = fp.stem.replace("data_", "")
        actions = parse_csv(str(fp))
        steps = build_fitting_steps(actions, maps, include_practice=False)
        if not steps:
            continue
        n = len(steps)
        random_nll = sum(-math.log(1.0 / len(step["uncolored"])) for step in steps)

        for model_name, nll_func, x0, n_params in model_specs:
            if model_name == "random":
                nll = random_nll
                ll = -nll
            else:
                res = minimize(
                    nll_func,
                    x0,
                    args=(steps,),
                    method="Nelder-Mead",
                    options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
                )
                nll = float(res.fun)
                ll = -nll
            aic = 2 * n_params - 2 * ll
            bic = n_params * math.log(n) - 2 * ll
            rows.append(
                {
                    "participant": participant,
                    "model": model_name,
                    "n_steps": n,
                    "ll": ll,
                    "nll": nll,
                    "aic": aic,
                    "bic": bic,
                }
            )

    df = pd.DataFrame(rows)
    baseline = (
        df[df["model"] == "random"][["participant", "ll", "nll", "aic", "bic"]]
        .rename(columns={"ll": "ll_random", "nll": "nll_random", "aic": "aic_random", "bic": "bic_random"})
    )
    df = df.merge(baseline, on="participant", how="left")
    df["delta_ll_vs_random"] = df["ll"] - df["ll_random"]
    df["delta_nll_vs_random"] = df["nll"] - df["nll_random"]
    df["delta_aic_vs_random"] = df["aic"] - df["aic_random"]
    df["delta_bic_vs_random"] = df["bic"] - df["bic_random"]
    return df


def save_basic_choice_comparison() -> Path:
    df = _fit_basic_choice_models()
    order = ["random", "distance", "neighbor", "neighbor+distance"]
    summary = (
        df.groupby("model", as_index=False)[["delta_nll_vs_random", "delta_aic_vs_random", "delta_bic_vs_random"]]
        .mean()
        .set_index("model")
        .loc[order]
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    configs = [
        ("delta_nll_vs_random", "Mean ΔNLL vs random", "tab:green"),
        ("delta_aic_vs_random", "Mean ΔAIC vs random", "tab:blue"),
        ("delta_bic_vs_random", "Mean ΔBIC vs random", "tab:purple"),
    ]
    for ax, (col, title, color) in zip(axes, configs):
        ax.bar(summary["model"], summary[col], color=color, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Basic Region-Choice Models", y=1.03, fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "01_basic_choice_models.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_descriptive_stats() -> Path:
    maps = build_all_maps()
    rows = []
    for fp in sorted(DATA_DIR.glob("data_*.csv")):
        participant = fp.stem.replace("data_", "")
        result = analyze_participant(str(fp), maps)
        if result is None:
            continue
        start_mean = float(np.mean([x["dist_to_center"] for x in result["starting_points"]]))
        coherence = result["spatial_coherence"]
        step_mean = float(np.mean([x["mean_step_dist"] for x in coherence])) if coherence else np.nan
        neighbor_rate = float(np.mean([x["neighbor_transition_rate"] for x in coherence])) if coherence else np.nan
        rows.append(
            {
                "participant": participant,
                "start_dist": start_mean,
                "step_dist": step_mean,
                "neighbor_rate": neighbor_rate,
            }
        )
    df = pd.DataFrame(rows).sort_values("participant")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = [
        ("start_dist", "Start Distance to Center", "Distance"),
        ("step_dist", "Mean Step Distance", "Distance"),
        ("neighbor_rate", "Neighbor Transition Rate", "Rate"),
    ]
    x = np.arange(len(df))
    for ax, (col, title, ylabel) in zip(axes, metrics):
        ax.scatter(x, df[col], s=40, alpha=0.8)
        ax.axhline(df[col].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
        ax.set_xlabel("Participant index")
    fig.suptitle("Descriptive Spatial Statistics", y=1.03, fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "02_descriptive_spatial_stats.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_step_length_distributions() -> list[Path]:
    transitions = build_transition_table(include_practice=False)
    out_paths = []
    for metric, idx in [("graph", "03"), ("euclidean", "04")]:
        results = fit_all_models_pooled(transitions, metric)
        summary = summarize_pooled_model_comparison(results)
        levy_row = summary[summary["model"] == "levy"].iloc[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_pooled_histogram(transitions, results, metric=metric, ax=ax, y_mode="count")
        ax.set_title(
            f"{metric.title()} step-length distribution\n"
            f"levy-random ΔLL={levy_row['delta_ll_vs_random']:.2f}, "
            f"ΔAIC={levy_row['delta_aic_vs_random']:.2f}, ΔBIC={levy_row['delta_bic_vs_random']:.2f}"
        )
        fig.tight_layout()
        out = OUT_DIR / f"{idx}_{metric}_step_length_distribution.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def save_joint_model_comparison() -> Path:
    results = fit_all_participants_model_comparison(include_practice=False)
    summary = (
        results.groupby("model", as_index=False)[["delta_nll_vs_joint", "delta_aic_vs_joint", "delta_bic_vs_joint"]]
        .mean()
        .set_index("model")
        .loc[["joint", "levy_only", "neighbor_only"]]
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    configs = [
        ("delta_nll_vs_joint", "Mean ΔNLL vs joint", "tab:green"),
        ("delta_aic_vs_joint", "Mean ΔAIC vs joint", "tab:blue"),
        ("delta_bic_vs_joint", "Mean ΔBIC vs joint", "tab:purple"),
    ]
    for ax, (col, title, color) in zip(axes, configs):
        ax.bar(summary["model"], summary[col], color=color, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(title)
    fig.suptitle("Levy + Neighbor Joint Model Comparison", y=1.03, fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "05_joint_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_joint_parameter_estimates() -> Path:
    results = fit_all_participants_model_comparison(include_practice=False)
    joint = results[results["model"] == "joint"].sort_values("participant")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].scatter(np.arange(len(joint)), joint["b_hat"], s=45)
    axes[0].axhline(joint["b_hat"].mean(), color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Joint Model b_hat")
    axes[0].set_ylabel("b_hat")
    axes[0].set_xticks(np.arange(len(joint)))
    axes[0].set_xticklabels([str(i + 1) for i in range(len(joint))], fontsize=8)
    axes[0].set_xlabel("Participant index")

    axes[1].scatter(np.arange(len(joint)), joint["beta_hat"], s=45, color="tab:red")
    axes[1].axhline(joint["beta_hat"].mean(), color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Joint Model beta_hat")
    axes[1].set_ylabel("beta_hat")
    axes[1].set_xticks(np.arange(len(joint)))
    axes[1].set_xticklabels([str(i + 1) for i in range(len(joint))], fontsize=8)
    axes[1].set_xlabel("Participant index")

    fig.suptitle("Joint Model Parameter Estimates", y=1.03, fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "06_joint_model_parameters.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.append(save_basic_choice_comparison())
    outputs.append(save_descriptive_stats())
    outputs.extend(save_step_length_distributions())
    outputs.append(save_joint_model_comparison())
    outputs.append(save_joint_parameter_estimates())

    print("Saved figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
