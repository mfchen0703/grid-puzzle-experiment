from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_NAME = "preservation_parameter_clustering.ipynb"


def main() -> None:
    analysis_dir = Path(__file__).resolve().parent
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Preservation 参数聚类

这个 notebook 只针对新模型 `空间+颜色保持` 的两个参数做被试聚类：

- `theta_s`
- `phi_same_color`

聚类单位是 participant，不是 round。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

analysis_dir = Path.cwd()
if not (analysis_dir / "preservation_parameter_clustering.py").exists():
    analysis_dir = Path.cwd() / "model code" / "experiment1" / "analysis_code"
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

import pandas as pd
import matplotlib.pyplot as plt

from preservation_parameter_clustering import (
    run_preservation_parameter_clustering,
    plot_parameter_scatter,
    plot_cluster_centers,
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """param_df, k_df, clustered_df, meta = run_preservation_parameter_clustering()

print(f"被试数: {len(param_df)}")
param_df"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 聚类数选择

这里用 silhouette 在 `k=2..6` 间选聚类数。"""
        )
    )

    cells.append(nbf.v4.new_code_cell("""k_df"""))

    cells.append(
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(k_df["k"], k_df["silhouette"], marker="o")
ax.set_xlabel("k")
ax.set_ylabel("silhouette")
ax.set_title("聚类数选择")
ax.grid(alpha=0.25, linestyle=":")
plt.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 被试聚类散点图"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(7, 5.5))
plot_parameter_scatter(clustered_df, ax=ax)
plt.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 各簇中心"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(6.2, 4.8))
plot_cluster_centers(meta, ax=ax)
plt.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 聚类结果表"""
        )
    )

    cells.append(nbf.v4.new_code_cell("""clustered_df.sort_values(['cluster', 'participant']).reset_index(drop=True)"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    }

    out_path = analysis_dir / NOTEBOOK_NAME
    nbf.write(nb, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
