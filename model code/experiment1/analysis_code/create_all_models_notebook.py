from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_NAME = "all_models_comparison.ipynb"


def main() -> None:
    fit_dir = Path(__file__).resolve().parent
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 五模型统一比较

这个 notebook 按新的理论划分比较 5 个模型：

1. 空间步长+邻居+颜色  
2. Levy步长+邻居+颜色  
3. 空间步长+邻居  
4. Levy步长+邻居  
5. 邻居+颜色

这里统一使用“每轮从第二个有效着色动作开始”的 step 来拟合，这样步长类模型和 region-choice 类模型比较时使用的是同一批数据。

说明：
- `空间步长+邻居+颜色` 被当作 baseline，用来计算 `ΔAIC / ΔBIC / ΔNLL`
- “空间步长”表示优先选择更短的 graph step length
- “Levy步长”表示按 Levy 规则选择 graph step length
- “邻居”表示在给定步长下，按已填色邻居数 softmax 选择 region
- “颜色”表示在选定 region 后，按是否保持上一步颜色做 softmax 选择颜色"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

fit_dir = Path.cwd()
if not (fit_dir / "unified_model_comparison.py").exists():
    fit_dir = Path.cwd() / "model code" / "experiment1" / "analysis_code"
if str(fit_dir) not in sys.path:
    sys.path.insert(0, str(fit_dir))

import pandas as pd
import matplotlib.pyplot as plt

from unified_model_comparison import (
    MODEL_LABELS,
    compare_all_models,
    summarize_deltas,
    plot_delta_metrics,
    plot_single_delta_metric,
    configure_chinese_font,
)

plt.style.use("default")
font_name = configure_chinese_font()
print(f"当前使用字体: {font_name}")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """steps_df, results_df = compare_all_models(include_practice=False)

print(f"可比较 step 数: {len(steps_df)}")
print(f"被试数: {steps_df['participant'].nunique()}")
print(f"baseline: {results_df['baseline_model'].iloc[0]}")

results_df.head()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 每个被试、每个模型的拟合结果

下面这张表已经包含：
- `ll / nll / aic / bic`
- 相对于 `空间步长+邻居+颜色` 的 `delta_nll / delta_aic / delta_bic`
- 各模型对应的参数估计"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display_cols = [
    "participant",
    "model",
    "n_steps",
    "ll",
    "nll",
    "aic",
    "bic",
    "delta_nll",
    "delta_aic",
    "delta_bic",
    "b",
    "beta_neighbor",
    "theta_s",
    "phi_same_color",
    "converged",
]

results_df[display_cols].sort_values(["participant", "model"]).reset_index(drop=True)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 各模型平均表现

这里汇总每个模型的平均 `NLL / AIC / BIC`，以及相对于 baseline 的平均 delta 和标准误。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """summary_df = summarize_deltas(results_df)
summary_df"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Delta 指标图

柱子表示被试平均值，误差线是标准误，黑点是各被试。  
正的 delta 表示该模型比 baseline `空间步长+邻居+颜色` 更差。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plot_delta_metrics(results_df)
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 单独的 Delta 图

如果你想单独看 `ΔAIC / ΔBIC / ΔNLL`，下面三张图是拆开的版本。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """for metric in ['aic', 'bic', 'nll']:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    plot_single_delta_metric(results_df, metric, ax=ax)
    fig.tight_layout()
    plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """out_path = fit_dir / "all_models_comparison_results.csv"
results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"结果已保存到: {out_path}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """for metric in ['aic', 'bic', 'nll']:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    plot_single_delta_metric(results_df, metric, ax=ax)
    fig.tight_layout()
    fig_path = fit_dir / f"all_models_delta_{metric}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {fig_path}")"""
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
