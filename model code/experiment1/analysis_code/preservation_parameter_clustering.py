from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from color_transition_clustering import _run_kmeans, _silhouette_score, _standardize
from unified_model_comparison import configure_chinese_font


def default_results_path() -> Path:
    return (Path(__file__).resolve().parent.parent / "results" / "all_models_comparison_results.csv").resolve()


def load_preservation_parameters(results_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(results_path) if results_path is not None else default_results_path()
    df = pd.read_csv(path)
    sub = df[df["model_key"] == "spatial_color_preservation"].copy()
    keep = ["participant", "theta_s", "phi_same_color", "nll", "aic", "bic"]
    return sub[keep].sort_values("participant").reset_index(drop=True)


def choose_k_by_silhouette(param_df: pd.DataFrame, k_values=range(2, 7)) -> pd.DataFrame:
    X = param_df[["theta_s", "phi_same_color"]].to_numpy(dtype=float)
    Xz, _, _ = _standardize(X)
    rows = []
    for k in k_values:
        if k >= len(param_df):
            continue
        labels, centers, inertia = _run_kmeans(Xz, n_clusters=k, n_init=100, random_state=42)
        score = _silhouette_score(Xz, labels)
        rows.append({"k": k, "silhouette": score, "inertia": inertia})
    return pd.DataFrame(rows)


def cluster_preservation_parameters(param_df: pd.DataFrame, n_clusters: int = 3):
    X = param_df[["theta_s", "phi_same_color"]].to_numpy(dtype=float)
    Xz, mean, std = _standardize(X)
    labels, centers, inertia = _run_kmeans(Xz, n_clusters=n_clusters, n_init=100, random_state=42)
    out = param_df.copy()
    out["cluster"] = labels.astype(int)
    meta = {
        "mean": mean,
        "std": std,
        "centers_z": centers,
        "centers_raw": centers * std + mean,
        "inertia": inertia,
    }
    return out, Xz, meta


def plot_parameter_scatter(clustered_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    colors = plt.cm.Set2(np.linspace(0, 1, clustered_df["cluster"].nunique()))
    cluster_to_color = {c: colors[i] for i, c in enumerate(sorted(clustered_df["cluster"].unique()))}

    for _, row in clustered_df.iterrows():
        ax.scatter(
            row["theta_s"],
            row["phi_same_color"],
            s=70,
            color=cluster_to_color[row["cluster"]],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.text(row["theta_s"] + 0.03, row["phi_same_color"] + 0.03, str(row["participant"]), fontsize=9)

    ax.set_xlabel(r"$\theta_s$")
    ax.set_ylabel(r"$\phi_{\mathrm{same}}$")
    ax.set_title("被试 preservation 参数聚类")
    ax.grid(alpha=0.25, linestyle=":")
    return ax


def plot_cluster_centers(meta: dict, ax=None):
    ax = ax or plt.gca()
    centers = np.asarray(meta["centers_raw"], dtype=float)
    colors = plt.cm.Set2(np.linspace(0, 1, len(centers)))
    for idx, (theta_s, phi_same) in enumerate(centers):
        ax.scatter(theta_s, phi_same, s=180, color=colors[idx], edgecolor="black", linewidth=1.0)
        ax.text(theta_s + 0.03, phi_same + 0.03, f"Cluster {idx}", fontsize=10, weight="bold")
    ax.set_xlabel(r"$\theta_s$")
    ax.set_ylabel(r"$\phi_{\mathrm{same}}$")
    ax.set_title("各簇中心（原始参数空间）")
    ax.grid(alpha=0.25, linestyle=":")
    return ax


def run_preservation_parameter_clustering(
    results_path: str | Path | None = None,
    n_clusters: int | None = None,
):
    param_df = load_preservation_parameters(results_path)
    k_df = choose_k_by_silhouette(param_df)
    if n_clusters is None:
        if k_df.empty:
            n_clusters = 2
        else:
            n_clusters = int(k_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
    clustered_df, Xz, meta = cluster_preservation_parameters(param_df, n_clusters=n_clusters)
    return param_df, k_df, clustered_df, meta


def main() -> None:
    configure_chinese_font()
    param_df, k_df, clustered_df, meta = run_preservation_parameter_clustering()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    param_out = results_dir / "preservation_parameter_estimates.csv"
    k_out = results_dir / "preservation_parameter_k_selection.csv"
    cluster_out = results_dir / "preservation_parameter_cluster_assignments.csv"
    scatter_out = results_dir / "preservation_parameter_clusters.png"
    centers_out = results_dir / "preservation_parameter_cluster_centers.png"

    param_df.to_csv(param_out, index=False, encoding="utf-8-sig")
    k_df.to_csv(k_out, index=False, encoding="utf-8-sig")
    clustered_df.to_csv(cluster_out, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_parameter_scatter(clustered_df, ax=ax)
    fig.tight_layout()
    fig.savefig(scatter_out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    plot_cluster_centers(meta, ax=ax)
    fig.tight_layout()
    fig.savefig(centers_out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(param_out)
    print(k_out)
    print(cluster_out)
    print(scatter_out)
    print(centers_out)


if __name__ == "__main__":
    main()
