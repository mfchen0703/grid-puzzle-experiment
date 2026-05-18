from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from fit_softmax import (
    build_all_maps,
    centroid_distance,
    count_effective_colors,
    is_color_legal,
    parse_csv,
)
from step_length_models import precompute_round_lengths


MODEL_LABELS = {
    "spatial_neighbor_color": "空间步长+邻居+颜色",
    "levy_neighbor_color": "Levy步长+邻居+颜色",
    "spatial_neighbor": "空间步长+邻居",
    "levy_neighbor": "Levy步长+邻居",
    "neighbor_color": "邻居+颜色",
}


def default_data_dir() -> Path:
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def configure_chinese_font() -> str:
    import matplotlib as mpl
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Source Han Sans SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Arial Unicode MS",
    ]
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False
    return chosen


def _count_colored_neighbors(region_id: int, adjacency, current_colors: list[int | None]) -> int:
    return sum(1 for nb in adjacency[region_id] if current_colors[nb] is not None)


def _count_legal_colors(region_id: int, adjacency, current_colors: list[int | None]) -> int:
    return sum(1 for c in range(4) if is_color_legal(region_id, c, adjacency, current_colors))


def build_unified_steps(
    data_dir: str | Path | None = None,
    include_practice: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    maps = build_all_maps()
    _, round_info = precompute_round_lengths()
    rows: list[dict] = []

    for filepath in sorted(data_dir.glob("data_*.csv")):
        participant = filepath.stem.replace("data_", "")
        actions = parse_csv(str(filepath))

        current_colors = None
        used_colors = None
        current_round = None
        prev_region = None
        prev_color = None
        regions = None
        adjacency = None
        step_in_round = -1

        for act in actions:
            if act["is_start"]:
                current_round = act["round"]
                if not include_practice and current_round.startswith("P"):
                    current_colors = None
                    used_colors = None
                    prev_region = None
                    prev_color = None
                    continue
                if current_round not in maps:
                    current_colors = None
                    used_colors = None
                    prev_region = None
                    prev_color = None
                    continue
                regions, adjacency = maps[current_round]
                current_colors = [None] * len(regions)
                used_colors = set()
                prev_region = None
                prev_color = None
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

                    uncolored = [i for i in range(len(regions)) if current_colors[i] is None]

                    candidate_graph_lengths = []
                    candidate_neighbor_counts = []
                    candidate_distances = []
                    candidate_legal_colors = []
                    candidate_effective_colors = []
                    candidate_regions_same_length = []
                    colored_neighbor_counts_same_length = []
                    available_graph_lengths = []

                    for candidate in uncolored:
                        cand_len = int(round_info[current_round]["graph"][prev_region, candidate])
                        cand_neighbors = _count_colored_neighbors(candidate, adjacency, current_colors)
                        cand_distance = centroid_distance(regions[candidate])
                        cand_legal = _count_legal_colors(candidate, adjacency, current_colors)
                        cand_effective = count_effective_colors(
                            candidate,
                            adjacency,
                            current_colors,
                            used_colors,
                        )

                        candidate_graph_lengths.append(cand_len)
                        candidate_neighbor_counts.append(cand_neighbors)
                        candidate_distances.append(cand_distance)
                        candidate_legal_colors.append(cand_legal)
                        candidate_effective_colors.append(cand_effective)

                        if cand_len > 0:
                            available_graph_lengths.append(cand_len)
                        if cand_len == graph_length:
                            candidate_regions_same_length.append(candidate)
                            colored_neighbor_counts_same_length.append(cand_neighbors)

                    chosen_idx = uncolored.index(rid)
                    rows.append(
                        {
                            "participant": participant,
                            "round_label": current_round,
                            "step_in_round": step_in_round,
                            "prev_region": int(prev_region),
                            "chosen_region": int(rid),
                            "chosen_color": int(color),
                            "prev_color": None if prev_color is None else int(prev_color),
                            "chosen_index": int(chosen_idx),
                            "graph_length": int(graph_length),
                            "available_graph_lengths": tuple(int(x) for x in available_graph_lengths),
                            "uncolored": tuple(int(x) for x in uncolored),
                            "candidate_graph_lengths": tuple(int(x) for x in candidate_graph_lengths),
                            "candidate_neighbor_counts": tuple(int(x) for x in candidate_neighbor_counts),
                            "candidate_distances": tuple(float(x) for x in candidate_distances),
                            "candidate_legal_colors": tuple(int(x) for x in candidate_legal_colors),
                            "candidate_effective_colors": tuple(int(x) for x in candidate_effective_colors),
                            "candidate_regions_same_length": tuple(int(x) for x in candidate_regions_same_length),
                            "colored_neighbor_counts_same_length": tuple(
                                int(x) for x in colored_neighbor_counts_same_length
                            ),
                            "num_legal_colors": int(candidate_legal_colors[chosen_idx]),
                            "num_effective_colors": int(candidate_effective_colors[chosen_idx]),
                            "prev_color_legal": False
                            if prev_color is None
                            else bool(is_color_legal(rid, int(prev_color), adjacency, current_colors)),
                        }
                    )
                prev_region = rid
                prev_color = color
                step_in_round += 1

            current_colors[rid] = color
            if color is not None:
                used_colors.add(color)

    return pd.DataFrame(rows)


def _available_length_support(step: pd.Series) -> np.ndarray:
    values = sorted({int(x) for x in step["available_graph_lengths"] if int(x) > 0})
    return np.array(values, dtype=int)


def _log_p_length_levy(graph_length: int, b: float, support: np.ndarray) -> float:
    if not (1.0 < b < 3.0):
        return -math.inf
    match = np.where(support == int(graph_length))[0]
    if len(match) == 0:
        return -math.inf
    log_weights = -b * np.log(support.astype(float))
    log_probs = log_weights - logsumexp(log_weights)
    return float(log_probs[match[0]])


def _log_p_length_spatial(graph_length: int, theta_s: float, support: np.ndarray) -> float:
    if len(support) == 0:
        return -math.inf
    match = np.where(support == int(graph_length))[0]
    if len(match) == 0:
        return -math.inf
    utilities = theta_s * (-support.astype(float))
    return float(utilities[match[0]] - logsumexp(utilities))


def _region_logprob_neighbor_all(step: pd.Series, beta_neighbor: float) -> float:
    neighbors = np.array(step["candidate_neighbor_counts"], dtype=float)
    utilities = beta_neighbor * neighbors
    chosen_idx = int(step["chosen_index"])
    return float(utilities[chosen_idx] - logsumexp(utilities))


def _piece_logprob_same_length(step: pd.Series, beta_neighbor: float) -> float:
    counts = np.array(step["colored_neighbor_counts_same_length"], dtype=float)
    candidates = tuple(step["candidate_regions_same_length"])
    if len(candidates) == 0 or int(step["chosen_region"]) not in candidates:
        return -math.inf
    chosen_idx = candidates.index(int(step["chosen_region"]))
    utilities = beta_neighbor * counts
    return float(utilities[chosen_idx] - logsumexp(utilities))


def _piece_logprob_uniform(step: pd.Series) -> float:
    n_candidates = len(step["candidate_regions_same_length"])
    if n_candidates == 0:
        return -math.inf
    return -math.log(n_candidates)


def _color_logprob_same_color(step: pd.Series, phi_same_color: float) -> float:
    n_legal = int(step["num_legal_colors"])
    if n_legal <= 0:
        return -math.inf

    prev_color = step["prev_color"]
    chosen_color = int(step["chosen_color"])
    if pd.isna(prev_color):
        return -math.log(n_legal)

    prev_color = int(prev_color)
    prev_color_legal = bool(step["prev_color_legal"])

    if not prev_color_legal:
        return -math.log(n_legal)

    log_denom = math.log(math.exp(phi_same_color) + max(n_legal - 1, 0))
    if chosen_color == prev_color:
        return phi_same_color - log_denom
    return -log_denom


def _color_logprob_effective(step: pd.Series) -> float:
    n = int(step["num_effective_colors"])
    if n <= 0:
        return -math.inf
    return -math.log(n)


def neg_log_likelihood_model(
    params: np.ndarray,
    participant_steps: pd.DataFrame,
    model_name: str,
) -> float:
    total_ll = 0.0

    if model_name == "spatial_neighbor_color":
        theta_s, beta_neighbor, phi_same_color = map(float, params)
        for _, step in participant_steps.iterrows():
            support = _available_length_support(step)
            total_ll += _log_p_length_spatial(int(step["graph_length"]), theta_s, support)
            total_ll += _piece_logprob_same_length(step, beta_neighbor)
            total_ll += _color_logprob_same_color(step, phi_same_color)
    elif model_name == "levy_neighbor_color":
        b, beta_neighbor, phi_same_color = map(float, params)
        if not (1.0 < b < 3.0):
            return np.inf
        for _, step in participant_steps.iterrows():
            support = _available_length_support(step)
            total_ll += _log_p_length_levy(int(step["graph_length"]), b, support)
            total_ll += _piece_logprob_same_length(step, beta_neighbor)
            total_ll += _color_logprob_same_color(step, phi_same_color)
    elif model_name == "spatial_neighbor":
        theta_s, beta_neighbor = map(float, params)
        for _, step in participant_steps.iterrows():
            support = _available_length_support(step)
            total_ll += _log_p_length_spatial(int(step["graph_length"]), theta_s, support)
            total_ll += _piece_logprob_same_length(step, beta_neighbor)
            total_ll += _color_logprob_effective(step)
    elif model_name == "levy_neighbor":
        b, beta_neighbor = map(float, params)
        if not (1.0 < b < 3.0):
            return np.inf
        for _, step in participant_steps.iterrows():
            support = _available_length_support(step)
            total_ll += _log_p_length_levy(int(step["graph_length"]), b, support)
            total_ll += _piece_logprob_same_length(step, beta_neighbor)
            total_ll += _color_logprob_effective(step)
    elif model_name == "neighbor_color":
        beta_neighbor, phi_same_color = map(float, params)
        for _, step in participant_steps.iterrows():
            total_ll += _region_logprob_neighbor_all(step, beta_neighbor)
            total_ll += _color_logprob_same_color(step, phi_same_color)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return -total_ll if np.isfinite(total_ll) else np.inf


def fit_participant_model(
    participant_steps: pd.DataFrame,
    model_name: str,
) -> dict[str, float | str]:
    participant = str(participant_steps["participant"].iloc[0])
    n_steps = len(participant_steps)

    if model_name == "spatial_neighbor_color":
        x0 = np.array([1.0, 1.0, 1.0])
        bounds = [(-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0)]
        param_names = ["theta_s", "beta_neighbor", "phi_same_color"]
    elif model_name == "levy_neighbor_color":
        x0 = np.array([1.5, 1.0, 1.0])
        bounds = [(1.01, 2.99), (-50.0, 50.0), (-50.0, 50.0)]
        param_names = ["b", "beta_neighbor", "phi_same_color"]
    elif model_name == "spatial_neighbor":
        x0 = np.array([1.0, 1.0])
        bounds = [(-50.0, 50.0), (-50.0, 50.0)]
        param_names = ["theta_s", "beta_neighbor"]
    elif model_name == "levy_neighbor":
        x0 = np.array([1.5, 1.0])
        bounds = [(1.01, 2.99), (-50.0, 50.0)]
        param_names = ["b", "beta_neighbor"]
    elif model_name == "neighbor_color":
        x0 = np.array([1.0, 1.0])
        bounds = [(-50.0, 50.0), (-50.0, 50.0)]
        param_names = ["beta_neighbor", "phi_same_color"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    result = minimize(
        neg_log_likelihood_model,
        x0,
        args=(participant_steps, model_name),
        method="L-BFGS-B",
        bounds=bounds,
    )

    nll = float(result.fun)
    ll = -nll
    n_params = len(param_names)
    fitted = {
        "participant": participant,
        "model_key": model_name,
        "model": MODEL_LABELS[model_name],
        "ll": ll,
        "nll": nll,
        "aic": 2 * n_params - 2 * ll,
        "bic": n_params * math.log(max(n_steps, 1)) - 2 * ll,
        "n_steps": n_steps,
        "converged": bool(result.success),
    }

    for name, value in zip(param_names, result.x):
        fitted[name] = float(value)

    return fitted


def compare_all_models(
    data_dir: str | Path | None = None,
    include_practice: bool = False,
    baseline_model_key: str = "spatial_neighbor_color",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps_df = build_unified_steps(data_dir=data_dir, include_practice=include_practice)
    if steps_df.empty:
        raise ValueError("No valid comparable steps were constructed.")
    model_keys = [
        "spatial_neighbor_color",
        "levy_neighbor_color",
        "spatial_neighbor",
        "levy_neighbor",
        "neighbor_color",
    ]

    results = []
    for participant, sub_df in steps_df.groupby("participant"):
        sub_df = sub_df.reset_index(drop=True)
        for model_key in model_keys:
            results.append(fit_participant_model(sub_df, model_key))

    results_df = pd.DataFrame(results).sort_values(["participant", "model"]).reset_index(drop=True)
    baseline = (
        results_df[results_df["model_key"] == baseline_model_key][
            ["participant", "ll", "nll", "aic", "bic"]
        ]
        .rename(
            columns={
                "ll": "ll_baseline",
                "nll": "nll_baseline",
                "aic": "aic_baseline",
                "bic": "bic_baseline",
            }
        )
    )
    results_df = results_df.merge(baseline, on="participant", how="left")
    results_df["delta_nll"] = results_df["nll"] - results_df["nll_baseline"]
    results_df["delta_aic"] = results_df["aic"] - results_df["aic_baseline"]
    results_df["delta_bic"] = results_df["bic"] - results_df["bic_baseline"]
    results_df["baseline_model"] = MODEL_LABELS[baseline_model_key]
    return steps_df, results_df


def summarize_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    def _se(x: pd.Series) -> float:
        if len(x) <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    return (
        results_df.groupby(["model_key", "model"], as_index=False)
        .agg(
            participants=("participant", "nunique"),
            mean_nll=("nll", "mean"),
            mean_aic=("aic", "mean"),
            mean_bic=("bic", "mean"),
            mean_delta_nll=("delta_nll", "mean"),
            se_delta_nll=("delta_nll", _se),
            mean_delta_aic=("delta_aic", "mean"),
            se_delta_aic=("delta_aic", _se),
            mean_delta_bic=("delta_bic", "mean"),
            se_delta_bic=("delta_bic", _se),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )


def plot_delta_metrics(results_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    configure_chinese_font()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 17,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )

    order = [
        "空间步长+邻居+颜色",
        "Levy步长+邻居+颜色",
        "空间步长+邻居",
        "Levy步长+邻居",
        "邻居+颜色",
    ]
    colors = {
        "空间步长+邻居+颜色": "#64b5cd",
        "Levy步长+邻居+颜色": "#4c72b0",
        "空间步长+邻居": "#55a868",
        "Levy步长+邻居": "#dd8452",
        "邻居+颜色": "#8172b3",
    }
    summary = summarize_deltas(results_df)
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model")

    participant_order = sorted(results_df["participant"].unique())
    participant_colors = {
        participant: plt.cm.tab20(i % 20) for i, participant in enumerate(participant_order)
    }

    fig, axes = plt.subplots(1, 3, figsize=(21.5, 6.4), sharex=True)
    configs = [
        ("delta_aic", "mean_delta_aic", "se_delta_aic", "ΔAIC"),
        ("delta_bic", "mean_delta_bic", "se_delta_bic", "ΔBIC"),
        ("delta_nll", "mean_delta_nll", "se_delta_nll", "ΔNLL"),
    ]

    x = np.arange(len(order))
    for ax, (raw_col, mean_col, se_col, title) in zip(axes, configs):
        means = summary[mean_col].to_numpy(dtype=float)
        ses = summary[se_col].fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x,
            means,
            yerr=ses,
            capsize=5,
            width=0.68,
            color=[colors[m] for m in summary["model"]],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )
        for j, model in enumerate(order):
            sub = results_df[results_df["model"] == model].sort_values("participant")
            offsets = np.linspace(-0.12, 0.12, len(sub)) if len(sub) > 1 else np.array([0.0])
            for offset, (_, row) in zip(offsets, sub.iterrows()):
                ax.scatter(
                    j + offset,
                    row[raw_col],
                    s=42,
                    color=participant_colors[row["participant"]],
                    edgecolor="none",
                    alpha=0.95,
                    zorder=3,
                )
        ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--", zorder=0)
        ax.set_title(title, pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(len(order))], rotation=0, ha="center")
        ax.set_xlabel("模型编号")
        ax.set_ylabel(f"{title} (model - 空间步长+邻居+颜色)")
        ax.grid(axis="y", linestyle=":", alpha=0.25, zorder=0)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[name], markeredgecolor="black", markersize=10)
        for name in order
    ]
    fig.legend(
        legend_handles,
        [f"{i + 1}. {name}" for i, name in enumerate(order)],
        loc="center left",
        bbox_to_anchor=(0.80, 0.5, 0.19, 0.0),
        frameon=False,
        title="模型",
        mode="expand",
        alignment="left",
        fontsize=16,
        title_fontsize=18,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.78, bottom=0.14)
    return fig, axes


def plot_single_delta_metric(results_df: pd.DataFrame, metric: str, ax=None):
    import matplotlib.pyplot as plt

    configure_chinese_font()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 17,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )

    metric_map = {
        "aic": ("delta_aic", "mean_delta_aic", "se_delta_aic", "ΔAIC"),
        "bic": ("delta_bic", "mean_delta_bic", "se_delta_bic", "ΔBIC"),
        "nll": ("delta_nll", "mean_delta_nll", "se_delta_nll", "ΔNLL"),
    }
    if metric not in metric_map:
        raise ValueError("metric must be one of: 'aic', 'bic', 'nll'.")

    raw_col, mean_col, se_col, title = metric_map[metric]
    order = [
        "空间步长+邻居+颜色",
        "Levy步长+邻居+颜色",
        "空间步长+邻居",
        "Levy步长+邻居",
        "邻居+颜色",
    ]
    colors = {
        "空间步长+邻居+颜色": "#64b5cd",
        "Levy步长+邻居+颜色": "#4c72b0",
        "空间步长+邻居": "#55a868",
        "Levy步长+邻居": "#dd8452",
        "邻居+颜色": "#8172b3",
    }

    summary = summarize_deltas(results_df)
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model")
    participant_order = sorted(results_df["participant"].unique())
    participant_colors = {
        participant: plt.cm.tab20(i % 20) for i, participant in enumerate(participant_order)
    }

    ax = ax or plt.gca()
    x = np.arange(len(order))
    means = summary[mean_col].to_numpy(dtype=float)
    ses = summary[se_col].fillna(0.0).to_numpy(dtype=float)
    ax.bar(
        x,
        means,
        yerr=ses,
        capsize=5,
        width=0.68,
        color=[colors[m] for m in summary["model"]],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        zorder=1,
    )
    for j, model in enumerate(order):
        sub = results_df[results_df["model"] == model].sort_values("participant")
        offsets = np.linspace(-0.12, 0.12, len(sub)) if len(sub) > 1 else np.array([0.0])
        for offset, (_, row) in zip(offsets, sub.iterrows()):
            ax.scatter(
                j + offset,
                row[raw_col],
                s=42,
                color=participant_colors[row["participant"]],
                edgecolor="none",
                alpha=0.95,
                zorder=3,
            )
    ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--", zorder=0)
    ax.set_title(title, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(order))], rotation=0, ha="center")
    ax.set_xlabel("模型编号")
    ax.set_ylabel(f"{title} (model - 空间步长+邻居+颜色)")
    ax.grid(axis="y", linestyle=":", alpha=0.25, zorder=0)
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[name], markeredgecolor="black", markersize=9)
        for name in order
    ]
    ax.legend(
        legend_handles,
        [f"{i + 1}. {name}" for i, name in enumerate(order)],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="模型",
    )
    return ax
