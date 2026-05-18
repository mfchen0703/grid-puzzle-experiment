from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_NAME = "color_transition_clustering.ipynb"


def main() -> None:
    fit_dir = Path(__file__).resolve().parent
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 颜色转移率与聚类分析

这个 notebook 关注被试在每一轮中的颜色使用动态。

这里先定义一个简单指标：

- 对某个被试、某一轮，取该轮所有**合法有效着色**步骤的颜色序列
- 比较相邻两步的颜色是否发生变化
- 颜色转移率 = `换色次数 / (有效步数 - 1)`

因此，每个被试最终会得到一个 `10 × 1` 的向量，表示 10 轮正式实验中的颜色转移率。然后我们基于这个向量做聚类分析。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

fit_dir = Path.cwd()
if not (fit_dir / "color_transition_clustering.py").exists():
    fit_dir = Path.cwd() / "model code" / "fitting"
if str(fit_dir) not in sys.path:
    sys.path.insert(0, str(fit_dir))

import pandas as pd
import matplotlib.pyplot as plt

from color_transition_clustering import (
    build_valid_color_steps,
    compute_color_transition_rates,
    choose_k_by_silhouette,
    cluster_transition_profiles,
    plot_transition_rate_heatmap,
    plot_dendrogram,
    plot_cluster_mean_profiles,
    plot_pca_clusters,
)
from unified_model_comparison import configure_chinese_font

plt.style.use("default")
font_name = configure_chinese_font()
print(f"当前使用字体: {font_name}")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """valid_steps_df = build_valid_color_steps(include_practice=False)
rate_long_df, rate_wide_df = compute_color_transition_rates(valid_steps_df)

print(f"有效着色 step 数: {len(valid_steps_df)}")
print(f"被试数: {rate_wide_df.shape[0]}")
print(f"正式 round 数: {rate_wide_df.shape[1]}")

rate_long_df.head()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 每个被试的 round 向量

下面这张表就是后续聚类分析用到的 `participant × round` 颜色转移率矩阵。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """rate_wide_df"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 描述统计

先看每一轮颜色转移率的均值和标准差。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """round_summary = (
    rate_long_df[rate_long_df['round_num'] > 0]
    .groupby('round_num', as_index=False)
    .agg(
        mean_transition_rate=('color_transition_rate', 'mean'),
        sd_transition_rate=('color_transition_rate', 'std'),
    )
)
round_summary"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    round_summary['round_num'],
    round_summary['mean_transition_rate'],
    yerr=round_summary['sd_transition_rate'],
    marker='o',
    linewidth=2,
    capsize=4,
)
ax.set_xlabel('Round')
ax.set_ylabel('颜色转移率')
ax.set_title('每一轮颜色转移率的均值 ± SD')
ax.set_ylim(0, 1)
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 选择聚类数

这里用 silhouette score 作为一个简单参考。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """k_eval_df = choose_k_by_silhouette(rate_wide_df, k_values=range(2, 7))
k_eval_df"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """best_k = int(k_eval_df.sort_values('silhouette', ascending=False).iloc[0]['k'])
print(f'建议的聚类数: {best_k}')"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 聚类结果

下面默认使用 silhouette 最高的 `k`。如果你想手动指定别的聚类数，只要把下一格的 `n_clusters` 改掉。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """clustered_df, Xz, cluster_meta = cluster_transition_profiles(rate_wide_df, n_clusters=best_k)
clustered_df"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(16, 12))

im, order = plot_transition_rate_heatmap(clustered_df, ax=axes[0, 0])
fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

plot_dendrogram(rate_wide_df, ax=axes[0, 1])
plot_cluster_mean_profiles(clustered_df, ax=axes[1, 0])
plot_pca_clusters(clustered_df, ax=axes[1, 1])

fig.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 输出文件

把长表、宽表和聚类结果都保存下来，方便后续分析。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """rate_long_path = fit_dir / 'color_transition_rates_long.csv'
rate_wide_path = fit_dir / 'color_transition_rates_wide.csv'
cluster_path = fit_dir / 'color_transition_cluster_assignments.csv'

rate_long_df.to_csv(rate_long_path, index=False, encoding='utf-8-sig')
rate_wide_df.to_csv(rate_wide_path, encoding='utf-8-sig')
clustered_df.to_csv(cluster_path, encoding='utf-8-sig')

print(rate_long_path)
print(rate_wide_path)
print(cluster_path)"""
        )
    )

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

    out_path = fit_dir / NOTEBOOK_NAME
    nbf.write(nb, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
