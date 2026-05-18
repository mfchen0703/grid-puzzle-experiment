from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RECOVERY_CSV = RESULTS_DIR / "experiment2_parameter_recovery_results.csv"


def configure_chinese_font() -> None:
    candidates = [
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plot_true_vs_hat(ax, df: pd.DataFrame, true_col: str, hat_col: str, title: str) -> None:
    x = df[true_col].to_numpy(dtype=float)
    y = df[hat_col].to_numpy(dtype=float)
    uniq_x = sorted(df[true_col].unique().tolist())
    uniq_y = sorted(df[hat_col].unique().tolist())

    jitter_scale = 0.015 if max(len(uniq_x), len(uniq_y)) > 2 else 0.01
    rng = np.random.default_rng(0)
    xj = x + rng.normal(0, jitter_scale, size=len(x))
    yj = y + rng.normal(0, jitter_scale, size=len(y))

    ax.scatter(xj, yj, s=55, alpha=0.8, color="#2a6f97", edgecolor="white", linewidth=0.7)
    lo = min(min(uniq_x), min(uniq_y))
    hi = max(max(uniq_x), max(uniq_y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#999999", linewidth=1.5)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("设定参数", fontsize=13)
    ax.set_ylabel("恢复参数", fontsize=13)
    ax.set_xticks(uniq_x)
    ax.set_yticks(uniq_y)
    pad = (hi - lo) * 0.12 if hi > lo else 0.1
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(alpha=0.2)


def _plot_confusion(ax, df: pd.DataFrame, true_col: str, hat_col: str, title: str) -> None:
    table = pd.crosstab(df[true_col], df[hat_col]).sort_index().sort_index(axis=1)
    matrix = table.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("恢复参数", fontsize=13)
    ax.set_ylabel("设定参数", fontsize=13)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels([str(v) for v in table.columns], fontsize=11)
    ax.set_yticklabels([str(v) for v in table.index], fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=11)
    return im


def main() -> None:
    configure_chinese_font()
    df = pd.read_csv(RECOVERY_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    _plot_true_vs_hat(axes[0], df, "true_pruning_thresh", "hat_pruning_thresh", "pruning_thresh")
    _plot_true_vs_hat(axes[1], df, "true_gamma", "hat_gamma", "gamma")
    _plot_true_vs_hat(axes[2], df, "true_lapse_rate", "hat_lapse_rate", "lapse_rate")
    fig.suptitle("Experiment 2 参数恢复：设定值 vs 恢复值", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment2_parameter_recovery_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ims = []
    ims.append(_plot_confusion(axes[0], df, "true_pruning_thresh", "hat_pruning_thresh", "pruning_thresh"))
    ims.append(_plot_confusion(axes[1], df, "true_gamma", "hat_gamma", "gamma"))
    ims.append(_plot_confusion(axes[2], df, "true_lapse_rate", "hat_lapse_rate", "lapse_rate"))
    fig.suptitle("Experiment 2 参数恢复：混淆矩阵", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment2_parameter_recovery_confusion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
