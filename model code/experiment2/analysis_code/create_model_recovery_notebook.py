from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_NAME = "experiment2_model_recovery.ipynb"


def main() -> None:
    analysis_dir = Path(__file__).resolve().parent
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 实验2 Model Recovery

这个 notebook 用于给实验2当前的 tree-search agent 做 `parameter recovery`。

当前目标不是立刻完成最终拟合，而是先检查：

1. 给定一组真实参数，agent 能否生成稳定的模拟动作序列  
2. 用同一个模型回拟合这些模拟数据时，能否大致找回原参数  
3. `pruning_thresh / gamma / lapse_rate` 三个参数之间是否容易混淆

建议先做 `parameter recovery`，后面再扩展到更完整的 `model recovery`。"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 当前计划

第一版 recovery 建议分四步：

1. 定义一组代表性的参数网格  
2. 用当前 agent 在 10 个 round 上生成 synthetic trajectories  
3. 对每条 synthetic trajectory 做参数回拟合  
4. 比较真实参数与恢复参数的对应关系

当前 notebook 先搭骨架：
- 参数网格
- 单个 synthetic participant 的模拟入口
- recovery 结果表结构
- 预留可视化位置"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import importlib
import sys

analysis_dir = Path.cwd()
if not (analysis_dir / "conflict_search_agent.py").exists():
    analysis_dir = Path.cwd() / "model code" / "experiment2" / "analysis_code"
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import conflict_search_agent as agent
import experiment2_model_recovery as recovery
importlib.reload(agent)
importlib.reload(recovery)

from visualize_conflict_search_agent import configure_chinese_font

plt.style.use("default")
font_name = configure_chinese_font()
print(f"当前使用字体: {font_name}")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 参数网格

这里先给出一个小型参数网格，方便快速试跑。  
后面如果 recovery 表现稳定，再扩大网格。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """PRUNING_GRID = recovery.DEFAULT_PRUNING_GRID
GAMMA_GRID = recovery.DEFAULT_GAMMA_GRID
LAPSE_GRID = recovery.DEFAULT_LAPSE_GRID

grid_df = recovery.build_parameter_grid(
    pruning_grid=PRUNING_GRID,
    gamma_grid=GAMMA_GRID,
    lapse_grid=LAPSE_GRID,
)
grid_df"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 单个 synthetic participant 的模拟入口

这一格先定义一个最小模拟函数。  
当前版本直接复用 agent 的 `run_tree_agent_on_all_rounds()`，之后可以改成更贴近拟合输入格式的导出。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """round_df_example, step_df_example = recovery.simulate_synthetic_participant(
    pruning_thresh=0.5,
    gamma=0.1,
    lapse_rate=0.1,
    random_seed=1,
)

round_df_example"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 模拟动作序列表

这里只先确认 synthetic data 的基本结构。  
后续正式 recovery 时，需要把它转成拟合函数需要的输入格式。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """step_df_example.head(20)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 拟合接口骨架

下面实现一个第一版 recovery 拟合器。  
这一版先用参数网格搜索，不直接上连续优化。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fit_synthetic_participant = recovery.fit_synthetic_participant
run_parameter_recovery = recovery.run_parameter_recovery
summarize_recovery = recovery.summarize_recovery"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Recovery 任务表骨架

这一格先生成未来要跑的 recovery 任务表。  
例如每组参数模拟多个 synthetic participants。"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 单个 synthetic participant 的 recovery 示例

先对前面生成的 `step_df_example` 跑一遍网格搜索，确认输出格式。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """best_fit_example, candidate_df_example = fit_synthetic_participant(step_df_example)

best_fit_example"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """candidate_df_example.head(10)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """N_REPEATS = 3

tasks_df = recovery.build_recovery_tasks(
    pruning_grid=PRUNING_GRID,
    gamma_grid=GAMMA_GRID,
    lapse_grid=LAPSE_GRID,
    n_repeats=N_REPEATS,
)
tasks_df.head(12)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Recovery 结果表结构

后面正式跑 recovery 时，建议至少保存这些列：

- 真实参数：`true_pruning_thresh / true_gamma / true_lapse_rate`
- 恢复参数：`hat_pruning_thresh / hat_gamma / hat_lapse_rate`
- 拟合优度：`ll / nll / converged`
- synthetic participant 标识：`task_id / random_seed`"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """recovery_template = pd.DataFrame(
    columns=[
        "task_id",
        "random_seed",
        "true_pruning_thresh",
        "true_gamma",
        "true_lapse_rate",
        "hat_pruning_thresh",
        "hat_gamma",
        "hat_lapse_rate",
        "ll",
        "nll",
        "converged",
    ]
)

recovery_template"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 小规模 recovery 示例

先只取前几个 task 试跑，确认整个 recovery pipeline 可用。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """demo_tasks_df = tasks_df.head(4).copy()
demo_recovery_df, demo_candidate_tables = run_parameter_recovery(demo_tasks_df)
demo_recovery_df"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """summarize_recovery(demo_recovery_df)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 之后建议加入的图

正式 recovery 做完之后，可以在这里放：

1. `true vs recovered` 散点图  
2. 参数相关矩阵  
3. 不同参数区间的 recovery bias  
4. synthetic 序列长度、是否解开 round 的分布"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# 预留：后续在这里画 recovery 图
# 例如：
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# ..."""
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

    out_path = analysis_dir / NOTEBOOK_NAME
    nbf.write(nb, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
