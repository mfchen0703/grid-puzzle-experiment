from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_CSV = RESULTS_DIR / "all_models_absolute_metrics_summary.csv"


def configure_chinese_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Arial Unicode MS",
    ]
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _model_colors(models: list[str]) -> dict[str, str]:
    palette = ["#7cc6d7", "#6a87bd", "#6fbd7f", "#dc8757", "#9583be", "#9a9a9a"]
    return {model: palette[i % len(palette)] for i, model in enumerate(models)}


def plot_absolute_metrics(summary_df: pd.DataFrame) -> None:
    configure_chinese_font()
    df = summary_df.copy()
    df = df.sort_values("global_nll_per_step", ascending=True).reset_index(drop=True)
    colors = _model_colors(df["model"].tolist())
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = [
        ("global_nll_per_step", "每步 NLL", "越低越好"),
        ("global_geomean_likelihood_per_step", "每步几何平均 likelihood", "越高越好"),
        ("global_likelihood_ratio_vs_random_per_step", "相对随机模型 likelihood ratio", "随机模型 = 1"),
    ]

    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        vals = df[col].to_numpy(dtype=float)
        ax.bar(x, vals, color=[colors[m] for m in df["model"]], edgecolor="black", linewidth=1.0)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(subtitle, fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=12)
        ax.grid(axis="y", alpha=0.18)
        if "ratio" in col:
            ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
        for xi, val in zip(x, vals):
            ax.text(xi, val, f"{val:.2f}" if val >= 1 else f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=colors[row["model"]], markersize=10)
        for _, row in df.iterrows()
    ]
    labels = [f"{i + 1}. {row['model']}" for i, row in df.iterrows()]
    fig.legend(
        handles,
        labels,
        title="模型",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=15,
    )
    fig.suptitle("实验1模型绝对拟合指标", fontsize=18, y=1.02)
    fig.subplots_adjust(left=0.06, right=0.76, bottom=0.16, top=0.83, wspace=0.32)
    fig.savefig(RESULTS_DIR / "all_models_absolute_metrics_combined.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_metric(summary_df: pd.DataFrame, col: str, ylabel: str, filename: str) -> None:
    configure_chinese_font()
    df = summary_df.copy().sort_values("global_nll_per_step", ascending=True).reset_index(drop=True)
    colors = _model_colors(df["model"].tolist())
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vals = df[col].to_numpy(dtype=float)
    ax.bar(x, vals, color=[colors[m] for m in df["model"]], edgecolor="black", linewidth=1.0)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=12)
    ax.grid(axis="y", alpha=0.18)
    if "ratio" in col:
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    for xi, val in zip(x, vals):
        ax.text(xi, val, f"{val:.2f}" if val >= 1 else f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=colors[row["model"]], markersize=10)
        for _, row in df.iterrows()
    ]
    labels = [f"{i + 1}. {row['model']}" for i, row in df.iterrows()]
    fig.legend(
        handles,
        labels,
        title="模型",
        loc="center right",
        bbox_to_anchor=(1.33, 0.5),
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )
    fig.subplots_adjust(left=0.12, right=0.68, bottom=0.16, top=0.9)
    fig.savefig(RESULTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary_df = pd.read_csv(SUMMARY_CSV)
    plot_absolute_metrics(summary_df)
    plot_single_metric(summary_df, "global_nll_per_step", "每步 NLL", "all_models_absolute_nll_per_step.png")
    plot_single_metric(
        summary_df,
        "global_geomean_likelihood_per_step",
        "每步几何平均 likelihood",
        "all_models_absolute_likelihood_per_step.png",
    )
    plot_single_metric(
        summary_df,
        "global_likelihood_ratio_vs_random_per_step",
        "相对随机模型 likelihood ratio",
        "all_models_likelihood_ratio_vs_random.png",
    )


if __name__ == "__main__":
    main()
