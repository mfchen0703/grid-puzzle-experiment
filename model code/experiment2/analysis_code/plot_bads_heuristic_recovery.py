from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PARAMS = ["region_preserve", "color_preserve"]
COLORS = {
    "region_preserve": "#7b2cbf",
    "color_preserve": "#f4a261",
}


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


def metric_text(true: np.ndarray, hat: np.ndarray) -> str:
    r = safe_corr(true, hat)
    mae = float(np.mean(np.abs(hat - true)))
    return f"r = {r:.3f}\nMAE = {mae:.3f}"


def plot_scatter(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    rng = np.random.default_rng(13)
    for ax, param in zip(axes, PARAMS):
        true = results_df[f"true_{param}"].to_numpy(dtype=float)
        hat = results_df[f"hat_{param}"].to_numpy(dtype=float)
        values = np.concatenate([true, hat])
        lo, hi = float(values.min()), float(values.max())
        span = max(hi - lo, 0.1)
        jitter = span * 0.012
        ax.scatter(
            true + rng.normal(0, jitter, len(true)),
            hat + rng.normal(0, jitter, len(hat)),
            s=78,
            color=COLORS[param],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        pad = span * 0.14
        ax.plot([lo, hi], [lo, hi], "--", color="#8c8c8c", linewidth=1.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ticks = sorted(set(results_df[f"true_{param}"].round(6).tolist() + results_df[f"hat_{param}"].round(6).tolist()))
        if len(ticks) <= 8:
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        ax.text(
            0.04,
            0.96,
            metric_text(true, hat),
            transform=ax.transAxes,
            va="top",
            fontsize=13,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        ax.set_title(param, fontsize=18, pad=12)
        ax.set_xlabel("设定参数", fontsize=15)
        ax.set_ylabel("恢复参数", fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.22)
    fig.suptitle("BADS heuristic 参数恢复：设定值 vs 恢复值", fontsize=22, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, param in zip(axes, PARAMS):
        err = results_df[f"hat_{param}"].to_numpy(dtype=float) - results_df[f"true_{param}"].to_numpy(dtype=float)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1.5)
        ax.hist(err, bins=min(10, max(4, len(np.unique(np.round(err, 6))))), color=COLORS[param], alpha=0.8, edgecolor="white")
        ax.scatter(err, np.zeros_like(err), color="#1f1f1f", s=24, alpha=0.68, zorder=3)
        ax.set_title(param, fontsize=17, pad=10)
        ax.set_xlabel("恢复误差：hat - true", fontsize=14)
        ax.set_ylabel("任务数", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("BADS heuristic 参数恢复误差分布", fontsize=21, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_true_hat_heatmaps(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, param in zip(axes, PARAMS):
        true_vals = sorted(results_df[f"true_{param}"].unique())
        hat_bins = np.linspace(
            min(results_df[f"hat_{param}"].min(), min(true_vals)),
            max(results_df[f"hat_{param}"].max(), max(true_vals)),
            8,
        )
        binned = pd.cut(results_df[f"hat_{param}"], bins=hat_bins, include_lowest=True)
        table = pd.crosstab(results_df[f"true_{param}"], binned).reindex(index=true_vals, fill_value=0)
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap="Purples" if param == "region_preserve" else "YlOrBr")
        ax.set_title(param, fontsize=17, pad=10)
        ax.set_xlabel("恢复参数区间", fontsize=14)
        ax.set_ylabel("设定参数", fontsize=14)
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels([f"{v:g}" for v in table.index], fontsize=11)
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels([str(c) for c in table.columns], rotation=35, ha="right", fontsize=9)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = int(table.iloc[i, j])
                if val:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=11, color="black")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("BADS heuristic 参数恢复：设定值 × 恢复区间", fontsize=21, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_diagnostics(candidates_df: pd.DataFrame, output_path: Path) -> None:
    if candidates_df.empty:
        return
    progress_rows = []
    for task_id, sub_df in candidates_df.groupby("task_id"):
        sub_df = sub_df.sort_values("eval_index").copy()
        sub_df["best_nll_so_far"] = sub_df["nll"].cummin()
        progress_rows.append(sub_df[["task_id", "eval_index", "best_nll_so_far"]])
    progress_df = pd.concat(progress_rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for _, sub_df in progress_df.groupby("task_id"):
        axes[0].plot(sub_df["eval_index"], sub_df["best_nll_so_far"], color="#7b2cbf", alpha=0.25, linewidth=1)
    axes[0].set_title("每个 task 的 best NLL 下降轨迹", fontsize=16, pad=10)
    axes[0].set_xlabel("BADS evaluation index", fontsize=13)
    axes[0].set_ylabel("best NLL so far", fontsize=13)
    axes[0].grid(alpha=0.22)

    eval_counts = candidates_df.groupby("task_id")["eval_index"].max()
    axes[1].hist(eval_counts, bins=12, color="#f4a261", alpha=0.82, edgecolor="white")
    axes[1].set_title("每个 task 的 evaluation 次数", fontsize=16, pad=10)
    axes[1].set_xlabel("max eval_index", fontsize=13)
    axes[1].set_ylabel("任务数", fontsize=13)
    axes[1].grid(axis="y", alpha=0.22)

    fig.suptitle("BADS heuristic candidate 搜索诊断", fontsize=20, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_error_summary(results_df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for param in PARAMS:
        true = results_df[f"true_{param}"].to_numpy(dtype=float)
        hat = results_df[f"hat_{param}"].to_numpy(dtype=float)
        error = hat - true
        rows.append(
            {
                "parameter": param,
                "correlation": safe_corr(true, hat),
                "mae": float(np.mean(np.abs(error))),
                "bias_mean_hat_minus_true": float(np.mean(error)),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "min_error": float(np.min(error)),
                "max_error": float(np.max(error)),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BADS heuristic recovery results.")
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "server_bads_heuristic_1repeat"),
    )
    args = parser.parse_args()

    configure_chinese_font()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(results_dir / "experiment2_bads_heuristic_results.csv")
    candidates_path = results_dir / "experiment2_bads_heuristic_candidates.csv"
    candidates_df = pd.read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()

    plot_scatter(results_df, figures_dir / "bads_heuristic_recovery_scatter.png")
    plot_error_distribution(results_df, figures_dir / "bads_heuristic_recovery_error_distribution.png")
    plot_true_hat_heatmaps(results_df, figures_dir / "bads_heuristic_recovery_true_hat_heatmaps.png")
    plot_candidate_diagnostics(candidates_df, figures_dir / "bads_heuristic_candidate_diagnostics.png")
    write_error_summary(results_df, figures_dir / "bads_heuristic_recovery_error_summary.csv")

    print(f"saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
