# Grid Puzzle Experiment

这个仓库包含地图着色任务的网页实验、实验材料生成脚本、行为数据分析代码，以及 Experiment 2 使用的 agent simulation、parameter recovery 和 model recovery 工具。

当前网页入口只保留 Experiment 2。被试会先完成 2 张练习地图，然后进入 30 张正式地图。正式地图均来自 fourinarow-style tree agent 难以解决的地图集合，并已验证在 `pruning_thresh = 1.0 / 2.0 / 5.0` 下无法被该 tree agent 解开。

## 项目结构

- `experiment1/`: React + Vite 网页实验。虽然目录名仍是 `experiment1`，当前入口只运行 Experiment 2。
- `experiment1/public/experiment2/rounds.json`: 网页实验实际读取的 Experiment 2 地图文件，包含 `practiceRounds` 和正式 `rounds`。
- `experiment2/`: Experiment 2 地图生成、筛选、验证和可视化脚本。
- `experiment2/generated_fourinarow_tree_failed_maps_38/`: 当前正式地图材料。目录名保留历史命名，但现在只保留 30 张正式地图。
- `experiment2/generated_tree_agent_failed_maps/solver_validation/`: solver validation、parameter sweep、recovery 输出等中间结果。
- `model code/experiment2/analysis_code/`: Experiment 2 的 Python recovery、agent 和分析脚本。
- `model code/experiment2/analysis_code/cpp/`: C++ simulation、IBS/exact likelihood 和 model recovery 工具。
- `data/`: 被试导出的 CSV 数据。
- `requirements.txt`: Python 分析和 recovery 依赖。
- `PROJECT_OVERVIEW.md`: 更偏背景介绍的项目说明。

## 当前网页实验

Experiment 2 的任务是：地图一开始已经全部着色，但存在若干相邻区域颜色冲突。被试需要修改各个区域的颜色，使整个地图最终没有任何相邻区域为相同颜色。

当前设置：

- 练习阶段：2 张地图，`numRegions = 20`。
- 正式阶段：30 张地图，`numRegions = 45`。
- 正式地图来源：`experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json`。
- 网页读取文件：`experiment1/public/experiment2/rounds.json`。
- 跳过规则：正式实验中超过 3 分钟未完成可以跳过本轮。
- 网页入口：输入被试 ID 后直接进入 Experiment 2，不再显示实验选择页面。

## 本地运行网页

需要 Node.js 和 npm。

```bash
cd experiment1
npm install
npm run dev
```

默认地址：

```text
http://localhost:3000/
```

也可以直接访问：

```text
http://localhost:3000/experiment2?id=test
```

常用检查命令：

```bash
cd experiment1
npm run lint
npm run build
```

## Python 环境

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要依赖包括 `numpy`、`pandas`、`scipy`、`matplotlib`、`notebook` 和 `pybads`。

在服务器或无图形界面环境中运行画图或 BADS 时，建议加上：

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib
```

## 地图材料

当前正式地图保存在：

```text
experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json
```

验证结果保存在：

```text
experiment2/generated_fourinarow_tree_failed_maps_38/fourinarow_tree_pruning_validation.csv
```

当前验证文件应包含 90 行，即：

```text
30 maps x 3 pruning thresholds
```

如果修改正式地图或练习地图，需要同步更新网页读取的：

```text
experiment1/public/experiment2/rounds.json
```

生成或刷新练习地图：

```bash
python3 experiment2/add_experiment2_practice_rounds.py
```

## C++ 工具

C++ 工具集中在：

```text
model code/experiment2/analysis_code/cpp/
```

更详细的工具说明见：

```text
model code/experiment2/analysis_code/cpp/README.md
```

从仓库根目录编译：

```bash
"model code/experiment2/analysis_code/cpp/build_cpp_tools.sh"
```

服务器上不要直接使用本地下载过去的二进制文件，应该在服务器重新编译。否则可能出现：

```text
cannot execute binary file: Exec format error
```

如果二进制没有执行权限：

```bash
chmod +x "model code/experiment2/analysis_code/cpp/cpp_model_recovery"
```

## 导出 C++ 地图格式

C++ 工具读取 compact rounds text，需要先从 JSON 导出：

```bash
python3 "model code/experiment2/analysis_code/cpp/export_rounds_for_cpp.py" \
  --input experiment2/generated_fourinarow_tree_failed_maps_38/generated_maps_sorted.json \
  --output experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt
```

## Parameter Recovery

Tree agent 和 BFS-HSP2 的 parameter recovery 入口是：

```text
model code/experiment2/analysis_code/experiment2_tree_hsp2_parameter_recovery.py
```

Tree agent 使用 C++ IBS 的示例：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
python3 "model code/experiment2/analysis_code/experiment2_tree_hsp2_parameter_recovery.py" \
  --agent tree \
  --fit-method bads \
  --true-pruning-thresh 2.0 \
  --true-gamma 0.1 \
  --true-lapse-rate 0.05 \
  --round-limit 30 \
  --max-agent-steps 50 \
  --max-depth 8 \
  --max-expansions 1000 \
  --ibs-samples 5 \
  --ibs-max-tries 100 \
  --bads-max-fun-evals 100 \
  --n-workers 64 \
  --cpp-tree-ibs \
  --progress-every 1 \
  --output-prefix experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_bads_ibs_p2_g01_l005
```

BFS-HSP2 使用 C++ exact likelihood 的示例：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
python3 "model code/experiment2/analysis_code/experiment2_tree_hsp2_parameter_recovery.py" \
  --agent hsp2 \
  --fit-method bads \
  --true-pruning-thresh 2.0 \
  --true-gamma 0.1 \
  --true-lapse-rate 0.05 \
  --round-limit 30 \
  --max-agent-steps 50 \
  --max-depth 8 \
  --max-expansions 1000 \
  --ibs-samples 5 \
  --ibs-max-tries 100 \
  --bads-max-fun-evals 100 \
  --n-workers 64 \
  --cpp-hsp2-exact \
  --progress-every 1 \
  --output-prefix experiment2/generated_tree_agent_failed_maps/solver_validation/hsp2_cpp_exact_bads_p2_g01_l005
```

参数含义：

- `pruning_thresh`: fourinarow-style pruning threshold。
- `gamma`: 按 fourinarow 定义控制搜索/选择噪声。
- `lapse_rate`: 按 fourinarow 定义的 lapse probability。
- `n-workers`: 并行 worker 数。服务器有 128 cores 时可以先试 `64`。
- `round-limit`: 当前正式实验建议使用 `30`。

## Model Recovery

三类 agent 的 C++ model recovery 工具是：

```text
model code/experiment2/analysis_code/cpp/cpp_model_recovery
```

它支持：

- `tree`
- `hsp2`
- `eflop`

单次示例：先用 tree simulate observed actions，再比较三个 model 的 likelihood。

```bash
"model code/experiment2/analysis_code/cpp/cpp_model_recovery" \
  --rounds experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt \
  --simulate-agent tree \
  --output-prefix experiment2/generated_tree_agent_failed_maps/solver_validation/tree_cpp_model_recovery \
  --round-limit 30 \
  --random-seed 1 \
  --max-agent-steps 50 \
  --max-depth 8 \
  --max-expansions 1000 \
  --max-outer-loops 50 \
  --max-min-conflicts-steps 200 \
  --max-eflop-retries 5 \
  --ibs-samples 5 \
  --ibs-max-tries 100 \
  --n-workers 64 \
  --tree-pruning 2.0 \
  --tree-gamma 0.1 \
  --tree-lapse 0.05 \
  --hsp2-pruning 2.0 \
  --hsp2-gamma 0.1 \
  --hsp2-lapse 0.05 \
  --hsp2-likelihood-mode exact
```

输出文件：

- `<output-prefix>_observed_actions.csv`
- `<output-prefix>_scores.csv`
- `<output-prefix>_summary.json`

## 批量 Model Recovery 示例

下面示例会跑 50 个 seed，每个 seed 下 tree 和 hsp2 各测试 5 组参数，并分别用 `tree`、`hsp2`、`eflop` 生成 observed actions 后做 model recovery。

```bash
tree_params=(
  "0.0 0.05 0.00"
  "1.0 0.05 0.05"
  "2.0 0.10 0.05"
  "5.0 0.20 0.10"
  "2.0 0.20 0.00"
)

hsp2_params=(
  "0.0 0.05 0.00"
  "1.0 0.05 0.05"
  "2.0 0.10 0.05"
  "5.0 0.20 0.10"
  "2.0 0.20 0.00"
)

for seed in $(seq 1 50); do
  for i in 0 1 2 3 4; do
    read tree_p tree_g tree_l <<< "${tree_params[$i]}"
    read hsp2_p hsp2_g hsp2_l <<< "${hsp2_params[$i]}"

    for true_agent in tree hsp2 eflop; do
      "model code/experiment2/analysis_code/cpp/cpp_model_recovery" \
        --rounds experiment2/generated_fourinarow_tree_failed_maps_38/rounds_cpp.txt \
        --simulate-agent "$true_agent" \
        --output-prefix "experiment2/generated_tree_agent_failed_maps/solver_validation/model_recovery_true_${true_agent}_param${i}_seed${seed}" \
        --round-limit 30 \
        --random-seed "$seed" \
        --max-agent-steps 50 \
        --max-depth 8 \
        --max-expansions 1000 \
        --max-outer-loops 50 \
        --max-min-conflicts-steps 200 \
        --max-eflop-retries 5 \
        --ibs-samples 5 \
        --ibs-max-tries 100 \
        --n-workers 64 \
        --tree-pruning "$tree_p" \
        --tree-gamma "$tree_g" \
        --tree-lapse "$tree_l" \
        --hsp2-pruning "$hsp2_p" \
        --hsp2-gamma "$hsp2_g" \
        --hsp2-lapse "$hsp2_l" \
        --hsp2-likelihood-mode exact
    done
  done
done
```

汇总结果：

```bash
python3 - <<'PY'
import json
import re
from pathlib import Path
import pandas as pd

base = Path("experiment2/generated_tree_agent_failed_maps/solver_validation")
rows = []

for path in sorted(base.glob("model_recovery_true_*_param*_seed*_summary.json")):
    data = json.loads(path.read_text())
    match = re.search(r"true_(.*?)_param(\d+)_seed(\d+)_summary", path.name)
    rows.append({
        "file": path.name,
        "true_agent": data["true_agent"],
        "predicted_agent": data["predicted_agent"],
        "correct": int(data["true_agent"] == data["predicted_agent"]),
        "param_id": int(match.group(2)) if match else None,
        "seed": int(match.group(3)) if match else None,
        "n_actions": data["n_actions"],
        **{f"nll_{score['model']}": score["nll"] for score in data["scores"]},
    })

df = pd.DataFrame(rows)
out = base / "model_recovery_3agents_50seeds_5params_summary.csv"
df.to_csv(out, index=False)

print("overall accuracy:", df["correct"].mean())
print(df.groupby("true_agent")["correct"].agg(["count", "mean"]))
print(df.groupby(["param_id", "true_agent"])["correct"].agg(["count", "mean"]))
print("saved:", out)
PY
```

## 结果和数据文件

网页实验导出的数据通常是 CSV，保存在 `data/` 或用户手动下载的位置。正式分析时需要区分：

- practice actions: 练习阶段操作，不应混入正式实验分析。
- formal actions: 30 张正式地图上的操作。
- skipped rounds: 超时或主动跳过的轮次，需要在分析中单独标记。

Recovery 和 solver validation 的常见输出位置：

```text
experiment2/generated_tree_agent_failed_maps/solver_validation/
```

## 常见问题

`py_compile` 没有输出是否正常？

正常。`python3 -m py_compile ...` 没有输出通常表示语法检查通过。

`Permission denied` 是什么意思？

通常是 C++ 二进制没有执行权限。运行：

```bash
chmod +x "model code/experiment2/analysis_code/cpp/cpp_model_recovery"
```

`Exec format error` 是什么意思？

通常是把 macOS 上编译的二进制拿到 Linux 服务器运行，或架构不匹配。需要在服务器上重新编译：

```bash
"model code/experiment2/analysis_code/cpp/build_cpp_tools.sh"
```

如何查看 CPU 使用情况？

```bash
top
```

或：

```bash
ps -o pid,ppid,pcpu,pmem,etime,stat,cmd -C python3
```

`top` 中 `%Cpu(s)` 的 `id` 表示 idle。比如 `99.0 id` 说明大部分 CPU 处于空闲状态。

## 维护建议

- 修改网页实验后运行 `npm run build`。
- 修改 Python recovery 脚本后至少运行 `python3 -m py_compile`。
- 修改 C++ 工具后重新运行 `build_cpp_tools.sh`，并用小规模 smoke test 确认能输出 CSV/JSON。
- 大规模 recovery 建议在服务器上运行，并把 `round-limit` 固定为当前正式实验的 `30`。
