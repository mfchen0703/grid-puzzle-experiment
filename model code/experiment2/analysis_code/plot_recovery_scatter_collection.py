from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "recovery_scatter_plots"

SEARCH_RECOVERY_CSV = RESULTS_DIR / "experiment2_search_parameter_recovery_parallel_1repeat_current_results.csv"
HEURISTIC_RECOVERY_CSV = RESULTS_DIR / "experiment2_heuristic_weight_recovery_parallel_broad_results.csv"


def configure_chinese_font() -> None:
    candidates = [
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def nice_metric_text(x: np.ndarray, y: np.ndarray) -> str:
    r = safe_corr(x, y)
    mae = float(np.mean(np.abs(y - x))) if len(x) else float("nan")
    if np.isnan(r):
        return f"r = NA\nMAE = {mae:.3f}"
    return f"r = {r:.3f}\nMAE = {mae:.3f}"


def plot_true_vs_hat(
    ax: plt.Axes,
    df: pd.DataFrame,
    true_col: str,
    hat_col: str,
    title: str,
    color: str,
) -> None:
    x = df[true_col].to_numpy(dtype=float)
    y = df[hat_col].to_numpy(dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    rng = np.random.default_rng(7)
    all_values = np.concatenate([x, y]) if len(x) else np.array([0.0, 1.0])
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    span = hi - lo
    jitter = max(span * 0.012, 0.006)

    ax.scatter(
        x + rng.normal(0, jitter, size=len(x)),
        y + rng.normal(0, jitter, size=len(y)),
        s=70,
        alpha=0.82,
        color=color,
        edgecolor="white",
        linewidth=0.8,
    )

    if span == 0:
        span = max(abs(lo) * 0.2, 0.2)
        lo -= span / 2
        hi += span / 2
    pad = span * 0.12
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#8a8a8a", linewidth=1.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    true_ticks = sorted(pd.unique(df[true_col].dropna()).tolist())
    hat_ticks = sorted(pd.unique(df[hat_col].dropna()).tolist())
    xticks = sorted(set(true_ticks + hat_ticks))
    if len(xticks) <= 8:
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("设定参数", fontsize=15)
    ax.set_ylabel("恢复参数", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.2)
    ax.text(
        0.04,
        0.96,
        nice_metric_text(x, y),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )


def save_scatter_grid(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str, str]],
    title: str,
    output_path: Path,
    ncols: int = 3,
) -> None:
    nplots = len(specs)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.7 * ncols, 5.2 * nrows), squeeze=False)
    for ax, (true_col, hat_col, panel_title, color) in zip(axes.ravel(), specs):
        plot_true_vs_hat(ax, df, true_col, hat_col, panel_title, color)
    for ax in axes.ravel()[nplots:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_chinese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    search_df = pd.read_csv(SEARCH_RECOVERY_CSV)
    heuristic_df = pd.read_csv(HEURISTIC_RECOVERY_CSV)

    search_specs = [
        ("true_pruning_thresh", "hat_pruning_thresh", "pruning_thresh", "#2a9d8f"),
        ("true_gamma", "hat_gamma", "gamma", "#457b9d"),
        ("true_lapse_rate", "hat_lapse_rate", "lapse_rate", "#e76f51"),
    ]
    heuristic_specs = [
        ("true_region_preserve", "hat_region_preserve", "region_preserve", "#7b2cbf"),
        ("true_color_preserve", "hat_color_preserve", "color_preserve", "#f4a261"),
    ]

    save_scatter_grid(
        search_df,
        search_specs,
        "Experiment 2 搜索参数恢复",
        OUTPUT_DIR / "search_parameter_recovery_scatter.png",
        ncols=3,
    )
    save_scatter_grid(
        heuristic_df,
        heuristic_specs,
        "Experiment 2 heuristic 权重恢复",
        OUTPUT_DIR / "heuristic_weight_recovery_scatter.png",
        ncols=2,
    )

    combined_df = pd.concat(
        [
            search_df.assign(source="search"),
            heuristic_df.assign(source="heuristic"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_specs = search_specs + heuristic_specs
    save_scatter_grid(
        combined_df,
        combined_specs,
        "Experiment 2 当前 recovery 散点图汇总",
        OUTPUT_DIR / "all_current_recovery_scatter.png",
        ncols=3,
    )

    print(f"saved scatter plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
