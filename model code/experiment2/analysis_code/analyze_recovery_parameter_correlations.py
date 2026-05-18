from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RECOVERY_CSV = RESULTS_DIR / "experiment2_parameter_recovery_results.csv"


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


def add_recovery_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for param in ["pruning_thresh", "gamma", "lapse_rate"]:
        out[f"error_{param}"] = out[f"hat_{param}"] - out[f"true_{param}"]
    return out


def plot_corr_heatmap(corr: pd.DataFrame, title: str, output_path: Path) -> None:
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=25, ha="right", fontsize=11)
    ax.set_yticklabels(corr.index, fontsize=11)
    ax.set_title(title, fontsize=16, pad=12)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson r", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_scatter(df: pd.DataFrame, output_path: Path) -> None:
    configure_chinese_font()
    pairs = [
        ("error_pruning_thresh", "error_gamma"),
        ("error_pruning_thresh", "error_lapse_rate"),
        ("error_gamma", "error_lapse_rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    rng = np.random.default_rng(0)
    for ax, (x_col, y_col) in zip(axes, pairs):
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        xj = x + rng.normal(0, 0.004, size=len(x))
        yj = y + rng.normal(0, 0.004, size=len(y))
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(xj, yj, s=42, alpha=0.78, color="#2a6f97", edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
        ax.axvline(0, color="#999999", linewidth=1.0, linestyle="--")
        ax.set_xlabel(x_col.replace("error_", "误差: "), fontsize=12)
        ax.set_ylabel(y_col.replace("error_", "误差: "), fontsize=12)
        ax.set_title(f"r = {r:.2f}", fontsize=14)
        ax.grid(alpha=0.18)
    fig.suptitle("Experiment 2 recovery 误差相关散点图", fontsize=17, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = add_recovery_errors(pd.read_csv(RECOVERY_CSV))
    hat_cols = ["hat_pruning_thresh", "hat_gamma", "hat_lapse_rate"]
    error_cols = ["error_pruning_thresh", "error_gamma", "error_lapse_rate"]

    hat_corr = df[hat_cols].corr()
    error_corr = df[error_cols].corr()
    hat_corr.to_csv(RESULTS_DIR / "experiment2_recovery_hat_parameter_correlations.csv", encoding="utf-8-sig")
    error_corr.to_csv(RESULTS_DIR / "experiment2_recovery_error_correlations.csv", encoding="utf-8-sig")

    plot_corr_heatmap(
        hat_corr,
        "恢复参数之间的相关性",
        RESULTS_DIR / "experiment2_recovery_hat_parameter_correlations.png",
    )
    plot_corr_heatmap(
        error_corr,
        "恢复误差之间的相关性",
        RESULTS_DIR / "experiment2_recovery_error_correlations.png",
    )
    plot_error_scatter(
        df,
        RESULTS_DIR / "experiment2_recovery_error_correlation_scatter.png",
    )

    print("Recovered parameter correlations")
    print(hat_corr.to_string())
    print("\nRecovery error correlations")
    print(error_corr.to_string())


if __name__ == "__main__":
    main()
