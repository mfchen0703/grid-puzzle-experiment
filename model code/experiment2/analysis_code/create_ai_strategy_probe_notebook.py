from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_NAME = "ai_strategy_probe_template.ipynb"


def main() -> None:
    analysis_dir = Path(__file__).resolve().parent
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# AI Strategy Probe Template

这个 notebook 用来做一件事：

- 给 AI 一个 **1 冲突状态**
- 让 AI 输出它认为的 **后续行动路径**
- 并为 **每一步** 给出简短理由

目标不是让 AI 解释我们当前 agent 的评分，而是看它在 near-terminal 状态下会采用什么解题策略。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

analysis_dir = Path.cwd()
if not (analysis_dir / "conflict_search_agent.py").exists():
    analysis_dir = Path.cwd() / "model code" / "experiment2" / "analysis_code"
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

import conflict_search_agent as agent

agent = importlib.reload(agent)

ROOT = analysis_dir.parents[2]
print(f"analysis_dir = {analysis_dir}")
print(f"repo_root     = {ROOT}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 输出要求

AI 只需要返回两个核心部分：

1. `action_path`
   - 一个行动序列
   - 每一步包含：
     - `step`
     - `region`
     - `new_color`
     - `reason`

2. `strategy_summary`
   - 一段简短总结
   - 说明它整体在用什么策略

推荐返回 JSON 结构：

```json
{
  "action_path": [
    {"step": 1, "region": 22, "new_color": 1, "reason": "..."},
    {"step": 2, "region": 17, "new_color": 3, "reason": "..."}
  ],
  "strategy_summary": "..."
}
```"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """DEFAULT_AGENT_PARAMS = dict(
    max_steps=100,
    max_depth=6,
    n_iterations=20,
    pruning_thresh=1,
    heuristic_eval_weight=0.5,
    frontier_strategy="best_path",
    tree_score_strategy="value_backup",
)


DEFAULT_GENERATED_MATERIALS_PATH = (
    ROOT / "experiment2" / "generated_map_search_planning1" / "generated_maps_sorted.json"
)


def load_materials_flexible(materials_path: str | Path | None = None, prefer_generated: bool = True) -> dict:
    if materials_path is not None:
        path = Path(materials_path)
        return json.loads(path.read_text(encoding="utf-8"))

    if prefer_generated and DEFAULT_GENERATED_MATERIALS_PATH.exists():
        return json.loads(DEFAULT_GENERATED_MATERIALS_PATH.read_text(encoding="utf-8"))

    return agent.load_materials(None)


def load_round(
    round_index: int,
    materials_path: str | Path | None = None,
    prefer_generated: bool = True,
) -> dict:
    materials = load_materials_flexible(materials_path, prefer_generated=prefer_generated)
    rounds = materials["rounds"]
    if round_index < 1 or round_index > len(rounds):
        raise IndexError(
            f"round_index={round_index} out of range; available rounds: 1..{len(rounds)}"
        )
    return rounds[round_index - 1]


def get_final_one_conflict_snapshot(
    round_index: int,
    *,
    materials_path: str | Path | None = None,
    prefer_generated: bool = True,
    agent_params: dict | None = None,
) -> dict:
    params = dict(DEFAULT_AGENT_PARAMS)
    if agent_params:
        params.update(agent_params)

    round_data = load_round(round_index, materials_path, prefer_generated=prefer_generated)
    trace = agent.trace_tree_agent_on_round(round_data, **params)
    action_steps = [row for row in trace if row["status"] == "action"]
    if not action_steps:
        raise ValueError(f"round {round_index} has no action steps")

    target = None
    for row in reversed(action_steps):
        if len(row["conflict_edges_before"]) == 1:
            target = row
            break
    if target is None:
        raise ValueError(
            f"round {round_index} never reaches a 1-conflict state under current params"
        )

    snapshot = {
        "round_index": round_index,
        "agent_step": int(target["agent_step"]),
        "colors_before": target["colors_before"],
        "conflict_edges_before": target["conflict_edges_before"],
        "conflict_regions_before": target["conflict_regions_before"],
        "candidate_actions": target["candidate_actions"],
        "selected_action": {
            "region": target["region"],
            "new_color": target["new_color"],
        },
        "n_candidates": len(target["candidate_actions"]),
        "agent_params": params,
    }
    return snapshot


snapshot = get_final_one_conflict_snapshot(round_index=13)
snapshot["n_candidates"], snapshot["selected_action"]"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """def simplify_candidate_actions(candidate_actions: list[dict], top_k: int | None = None) -> list[dict]:
    rows = []
    subset = candidate_actions if top_k is None else candidate_actions[:top_k]
    for item in subset:
        rows.append(
            {
                "region": int(item["region"]),
                "old_color": int(item["old_color"]),
                "new_color": int(item["new_color"]),
                "search_depth": int(item["search_depth"]),
                "n_conflict_edges_after": int(item["n_conflict_edges_after"]),
                "conflict_delta": int(item["conflict_delta"]),
                "next_depth": item["next_depth"],
                "solves_after_one_move": bool(item["solves_after_one_move"]),
                "found_solution_within_depth": bool(item["found_solution_within_depth"]),
            }
        )
    return rows


def build_user_prompt(snapshot: dict, top_k_candidates: int | None = None, max_plan_steps: int = 12) -> str:
    payload = {
        "round_index": snapshot["round_index"],
        "agent_step": snapshot["agent_step"],
        "colors_before": snapshot["colors_before"],
        "conflict_edges_before": snapshot["conflict_edges_before"],
        "conflict_regions_before": snapshot["conflict_regions_before"],
        "candidate_actions": simplify_candidate_actions(
            snapshot["candidate_actions"],
            top_k=top_k_candidates,
        ),
    }
    return dedent(
        f\"\"\"\
        You are analyzing a graph-coloring repair state with exactly one remaining conflict edge.

        Your task:
        1. Propose a plausible action path from the current state.
        2. For each step, explain why that action is chosen.
        3. Summarize the overall strategy briefly.

        Requirements:
        - Output valid JSON only.
        - Use this exact top-level structure:
          {{
            "action_path": [
              {{"step": 1, "region": ..., "new_color": ..., "reason": "..."}},
              ...
            ],
            "strategy_summary": "..."
          }}
        - Plan at most {max_plan_steps} steps.
        - Base your reasoning on the state and legal-action structure, not on any hidden scoring function.
        - Do not mention heuristic scores or ranking scores from another agent.
        - Keep the reasons concise and state-based.
        - Do not describe the UI or repeat the prompt.

        State payload:
        {json.dumps(payload, ensure_ascii=False, indent=2)}
        \"\"\"
    )


SYSTEM_PROMPT = dedent(
    \"\"\"\
    You are a careful puzzle-solving assistant.
    Focus on how to solve a one-conflict graph-coloring repair state.
    Prefer concrete step-by-step state-based reasoning over generic advice.
    Return valid JSON only.
    \"\"\"
)


user_prompt = build_user_prompt(snapshot, top_k_candidates=12, max_plan_steps=12)
print(user_prompt[:3000])"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """EXPECTED_SCHEMA = {
    "type": "object",
    "required": ["action_path", "strategy_summary"],
    "properties": {
        "action_path": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step", "region", "new_color", "reason"],
                "properties": {
                    "step": {"type": "integer"},
                    "region": {"type": "integer"},
                    "new_color": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
        "strategy_summary": {"type": "string"},
    },
}


def parse_json_response(text: str) -> dict:
    return json.loads(text)


def validate_minimal_response(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Response must be a JSON object.")
    if "action_path" not in payload or "strategy_summary" not in payload:
        raise ValueError("Response must contain action_path and strategy_summary.")
    if not isinstance(payload["action_path"], list):
        raise TypeError("action_path must be a list.")
    if not isinstance(payload["strategy_summary"], str):
        raise TypeError("strategy_summary must be a string.")
    for item in payload["action_path"]:
        if not isinstance(item, dict):
            raise TypeError("Each action_path item must be an object.")
        for key in ["step", "region", "new_color", "reason"]:
            if key not in item:
                raise ValueError(f"Missing key in action_path item: {key}")


example_response = {
    "action_path": [
        {
            "step": 1,
            "region": 22,
            "new_color": 1,
            "reason": "This move most directly releases color space near the conflict endpoint.",
        },
        {
            "step": 2,
            "region": 17,
            "new_color": 3,
            "reason": "After the first unlock, this move continues reducing local color pressure.",
        },
    ],
    "strategy_summary": "First unlock the conflict endpoint, then continue with the move that most directly reduces local color pressure.",
}
validate_minimal_response(example_response)
example_response"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Qwen API 调用骨架

下面这格默认使用新版 `openai` Python SDK 去调用 **Qwen / DashScope 的 OpenAI 兼容接口**。

运行前准备：

1. 安装依赖：
   - `pip install openai`
2. 设置环境变量：
   - `QWEN_API_KEY`
   - 可选：`QWEN_BASE_URL`
   - 可选：`QWEN_MODEL`

默认值建议：

- `QWEN_BASE_URL`
  - 中国（北京）：`https://dashscope.aliyuncs.com/compatible-mode/v1`
  - 新加坡：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- `QWEN_MODEL`
  - 默认先用：`qwen-plus`"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

print("QWEN_BASE_URL =", QWEN_BASE_URL)
print("QWEN_MODEL    =", QWEN_MODEL)
print("QWEN_API_KEY  =", "(set)" if QWEN_API_KEY else "(missing)")


def call_qwen_strategy_probe(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Please install the openai package first: pip install openai") from exc

    api_key = api_key or QWEN_API_KEY
    if not api_key:
        raise EnvironmentError("QWEN_API_KEY is not set.")

    base_url = base_url or QWEN_BASE_URL
    model = model or QWEN_MODEL

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.output_text
    payload = parse_json_response(text)
    validate_minimal_response(payload)
    return payload


# 取消下面注释即可实际调用
# result = call_qwen_strategy_probe(SYSTEM_PROMPT, user_prompt)
# result"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# 你也可以先手工粘贴模型输出，单独测试解析和校验

raw_text = json.dumps(example_response, ensure_ascii=False)
parsed = parse_json_response(raw_text)
validate_minimal_response(parsed)
parsed"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 推荐使用方式

建议先从以下状态开始：

- `round 13` 的最终 `1` 冲突状态
- `round 76` 的最终 `1` 冲突状态
- 以及其他最终停在 `1` 冲突的 round

你真正关心的不是 explanation 是否“好听”，而是：

1. AI 给出的第一步是否更接近 `correct child`
2. AI 的整条路径是不是在做“解锁冲突端点颜色空间”
3. `strategy_summary` 和逐步 `reason` 里，是否出现了我们当前 heuristic 没有编码进去的策略线索"""
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
            "version": "3",
        },
    }

    output_path = analysis_dir / NOTEBOOK_NAME
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
