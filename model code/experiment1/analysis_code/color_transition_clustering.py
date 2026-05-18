from __future__ import annotations

import os
from pathlib import Path

# Avoid OpenMP / shared-memory issues in the current local environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from fit_softmax import build_all_maps, is_color_legal, parse_csv


def default_data_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def _run_kmeans(X: np.ndarray, n_clusters: int, n_init: int = 50, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for _ in range(n_init):
        seed_idx = rng.choice(len(X), size=n_clusters, replace=False)
        centers = X[seed_idx].copy()

        for _iter in range(200):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = centers.copy()
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = X[rng.integers(len(X))]

            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        inertia = float(np.sum((X - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(X):
        return float("nan")

    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    scores = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if np.any(same):
            a = D[i, same].mean()
        else:
            a = 0.0

        b = np.inf
        for lab in unique_labels:
            if lab == labels[i]:
                continue
            other = labels == lab
            if np.any(other):
                b = min(b, D[i, other].mean())

        if not np.isfinite(b):
            continue
        denom = max(a, b)
        scores.append(0.0 if denom == 0 else (b - a) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var = (S ** 2) / max(len(X) - 1, 1)
    ratio = var / var.sum()
    return coords[:, :2], ratio[:2]


def build_valid_color_steps(
    data_dir: str | Path | None = None,
    include_practice: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    maps = build_all_maps()
    rows: list[dict] = []

    for filepath in sorted(data_dir.glob("data_*.csv")):
        participant = filepath.stem.replace("data_", "")
        actions = parse_csv(str(filepath))

        current_colors = None
        current_round = None
        current_step_in_round = 0
        valid_step_in_round = 0
        adjacency = None
        regions = None

        for act in actions:
            if act["is_start"]:
                current_round = act["round"]
                if not include_practice and current_round.startswith("P"):
                    current_colors = None
                    continue
                if current_round not in maps:
                    current_colors = None
                    continue
                regions, adjacency = maps[current_round]
                current_colors = [None] * len(regions)
                current_step_in_round = 0
                valid_step_in_round = 0
                continue

            if current_colors is None:
                continue

            current_step_in_round += 1
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

                rows.append(
                    {
                        "participant": participant,
                        "round_label": current_round,
                        "round_num": int(current_round) if current_round.isdigit() else -1,
                        "step_in_round_raw": current_step_in_round,
                        "step_in_round_valid": valid_step_in_round,
                        "chosen_region": int(rid),
                        "chosen_color": int(color),
                    }
                )
                valid_step_in_round += 1

            current_colors[rid] = color

    return pd.DataFrame(rows)


def compute_color_transition_rates(valid_steps_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    for (participant, round_label), sub_df in valid_steps_df.groupby(["participant", "round_label"]):
        sub_df = sub_df.sort_values("step_in_round_valid").reset_index(drop=True)
        colors = sub_df["chosen_color"].to_numpy(dtype=int)
        n_steps = len(colors)
        if n_steps <= 1:
            transitions = 0
            transition_rate = 0.0
        else:
            transitions = int(np.sum(colors[1:] != colors[:-1]))
            transition_rate = transitions / (n_steps - 1)

        rows.append(
            {
                "participant": participant,
                "round_label": round_label,
                "round_num": int(round_label) if round_label.isdigit() else -1,
                "n_valid_steps": n_steps,
                "n_color_transitions": transitions,
                "transition_opportunities": max(n_steps - 1, 0),
                "color_transition_rate": transition_rate,
            }
        )

    rate_long = pd.DataFrame(rows).sort_values(["participant", "round_num"]).reset_index(drop=True)
    rate_wide = (
        rate_long[rate_long["round_num"] > 0]
        .pivot(index="participant", columns="round_num", values="color_transition_rate")
        .reindex(columns=sorted(rate_long.loc[rate_long["round_num"] > 0, "round_num"].unique()))
        .sort_index()
    )
    rate_wide.columns = [f"round_{c}" for c in rate_wide.columns]
    return rate_long, rate_wide


def choose_k_by_silhouette(rate_wide_df: pd.DataFrame, k_values=range(2, 7)) -> pd.DataFrame:
    X = rate_wide_df.to_numpy(dtype=float)
    Xz, _, _ = _standardize(X)

    rows = []
    for k in k_values:
        if k >= len(rate_wide_df):
            continue
        labels, centers, inertia = _run_kmeans(Xz, n_clusters=k, n_init=50, random_state=42)
        score = _silhouette_score(Xz, labels)
        rows.append({"k": k, "silhouette": score, "inertia": inertia})
    return pd.DataFrame(rows)


def cluster_transition_profiles(rate_wide_df: pd.DataFrame, n_clusters: int = 3):
    X = rate_wide_df.to_numpy(dtype=float)
    Xz, mean, std = _standardize(X)
    labels, centers, inertia = _run_kmeans(Xz, n_clusters=n_clusters, n_init=50, random_state=42)

    out = rate_wide_df.copy()
    out["cluster"] = labels
    out["cluster"] = out["cluster"].astype(int)
    meta = {
        "mean": mean,
        "std": std,
        "centers": centers,
        "inertia": inertia,
    }
    return out, Xz, meta


def hierarchical_order(rate_wide_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    X = rate_wide_df.to_numpy(dtype=float)
    Xz, _, _ = _standardize(X)
    Z = linkage(pdist(Xz), method="ward")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    ordered_participants = rate_wide_df.index[leaves].tolist()
    return ordered_participants, Z


def plot_transition_rate_heatmap(clustered_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    feature_cols = [c for c in clustered_df.columns if c.startswith("round_")]
    order = clustered_df.sort_values(["cluster"] + feature_cols).index.tolist()
    plot_df = clustered_df.loc[order, feature_cols]

    im = ax.imshow(plot_df.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels([c.replace("round_", "R") for c in feature_cols], rotation=0)
    ax.set_title("颜色转移率热图")
    ax.set_xlabel("Round")
    ax.set_ylabel("Participant")
    return im, order


def plot_dendrogram(rate_wide_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    _, Z = hierarchical_order(rate_wide_df)
    dendrogram(Z, labels=rate_wide_df.index.tolist(), leaf_rotation=90, ax=ax)
    ax.set_title("层次聚类树状图")
    ax.set_ylabel("Ward distance")
    return ax


def plot_cluster_mean_profiles(clustered_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    feature_cols = [c for c in clustered_df.columns if c.startswith("round_")]
    means = clustered_df.groupby("cluster")[feature_cols].mean()
    for cluster_id, row in means.iterrows():
        ax.plot(
            np.arange(1, len(feature_cols) + 1),
            row.to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=f"Cluster {cluster_id}",
        )
    ax.set_xticks(np.arange(1, len(feature_cols) + 1))
    ax.set_xlabel("Round")
    ax.set_ylabel("颜色转移率")
    ax.set_title("各簇平均颜色转移率轨迹")
    ax.legend()
    ax.set_ylim(0, 1)
    return ax


def plot_pca_clusters(clustered_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    feature_cols = [c for c in clustered_df.columns if c.startswith("round_")]
    X = clustered_df[feature_cols].to_numpy(dtype=float)
    Xz, _, _ = _standardize(X)
    coords, explained = _pca_2d(Xz)

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        mask = clustered_df["cluster"].to_numpy() == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1], s=55, alpha=0.9, label=f"Cluster {cluster_id}")
    for (x, y), name in zip(coords, clustered_df.index):
        ax.text(x + 0.03, y + 0.03, name, fontsize=9)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title("颜色转移率向量的 PCA")
    ax.legend()
    return ax
