from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path

import pandas as pd


DEFAULT_HEURISTIC_WEIGHTS = {
    "repair": 2.0,
    "opportunity": 1.0,
    "region_preserve": 0.0,
    "color_preserve": 0.0,
}


def default_materials_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "experiment1"
        / "public"
        / "experiment2"
        / "rounds.json"
    )


def load_materials(materials_path: str | Path | None = None) -> dict:
    path = Path(materials_path) if materials_path is not None else default_materials_path()
    return json.loads(path.read_text(encoding="utf-8"))


def build_adjacency_map(map_data: dict) -> dict[int, set[int]]:
    adjacency = {i: set() for i in range(int(map_data["numRegions"]))}
    for a, b in map_data["adjacencyPairs"]:
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return adjacency


def get_conflict_edges(adjacency: dict[int, set[int]], colors: list[int] | tuple[int, ...]) -> list[tuple[int, int]]:
    conflicts: list[tuple[int, int]] = []
    for region in range(len(colors)):
        for neighbor in adjacency[region]:
            if region < neighbor and colors[region] == colors[neighbor]:
                conflicts.append((region, neighbor))
    return conflicts


def get_conflict_regions(conflict_edges: list[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for a, b in conflict_edges:
        out.add(int(a))
        out.add(int(b))
    return out


def is_legal_color(region: int, color: int, adjacency: dict[int, set[int]], colors: list[int] | tuple[int, ...]) -> bool:
    for neighbor in adjacency[region]:
        if colors[neighbor] == color:
            return False
    return True


def legal_recolor_options(region: int, adjacency: dict[int, set[int]], colors: list[int] | tuple[int, ...]) -> list[int]:
    current = colors[region]
    options: list[int] = []
    for new_color in range(4):
        if new_color == current:
            continue
        if is_legal_color(region, new_color, adjacency, colors):
            options.append(new_color)
    return options


def count_legal_recolor_options(
    region: int,
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
) -> int:
    return len(legal_recolor_options(region, adjacency, colors))


def layered_candidate_regions(adjacency: dict[int, set[int]], conflict_regions: set[int]) -> dict[int, list[int]]:
    if not conflict_regions:
        return {}

    layers: dict[int, list[int]] = {}
    seen: set[int] = set(conflict_regions)
    queue = deque((region, 0) for region in sorted(conflict_regions))

    while queue:
        region, depth = queue.popleft()
        layers.setdefault(depth, []).append(region)
        for neighbor in sorted(adjacency[region]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, depth + 1))

    return layers


def collect_legal_actions_by_depth(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
) -> tuple[dict[int, list[dict]], list[tuple[int, int]], set[int], dict[int, list[int]]]:
    conflict_edges = get_conflict_edges(adjacency, colors)
    conflict_regions = get_conflict_regions(conflict_edges)
    layers = layered_candidate_regions(adjacency, conflict_regions)
    legal_by_depth: dict[int, list[dict]] = {}

    for depth in sorted(layers):
        layer_actions: list[dict] = []
        for region in layers[depth]:
            for new_color in legal_recolor_options(region, adjacency, colors):
                layer_actions.append(
                    {
                        "search_depth": depth,
                        "region": region,
                        "old_color": int(colors[region]),
                        "new_color": new_color,
                    }
                )
        if layer_actions:
            legal_by_depth[depth] = layer_actions

    return legal_by_depth, conflict_edges, conflict_regions, layers


def apply_action(colors: list[int] | tuple[int, ...], action: dict) -> tuple[int, ...]:
    next_colors = list(colors)
    next_colors[action["region"]] = action["new_color"]
    return tuple(next_colors)


def _merged_heuristic_weights(heuristic_weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_HEURISTIC_WEIGHTS)
    if heuristic_weights is not None:
        for key, value in heuristic_weights.items():
            if key not in weights:
                raise ValueError(f"Unknown heuristic weight: {key}")
            weights[key] = float(value)
    return weights


def action_heuristic_features(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    action: dict,
) -> dict[str, float]:
    """Experiment-2 action features in the shared heuristic/planning framework.

    The feature families mirror the experiment-1 modeling vocabulary, but are
    adapted to the repair task:
    - repair: immediate goal progress after applying the action
    - opportunity: whether the resulting state lets a conflict region be edited
    - spatial: diagnostic local/conflict-neighborhood continuity
    - neighbor: diagnostic local graph-structure pressure around conflict regions
    - color: diagnostic color-configuration reuse among non-neighboring regions
    - region_preserve: whether the action edits a currently conflicting region
    - color_preserve: environmental edits that preserve/release conflict colors
    """
    conflict_edges_before = get_conflict_edges(adjacency, colors)
    conflict_regions_before = get_conflict_regions(conflict_edges_before)
    layers_before = layered_candidate_regions(adjacency, conflict_regions_before)
    next_state = apply_action(colors, action)
    next_legal_by_depth, next_conflict_edges, conflict_regions_after, _ = collect_legal_actions_by_depth(
        adjacency,
        next_state,
    )
    max_conflict_edges_before = max(1, len(conflict_edges_before))
    repair = float(len(conflict_edges_before) - len(next_conflict_edges)) / max_conflict_edges_before
    next_depth = min(next_legal_by_depth) if next_legal_by_depth else None
    region = int(action["region"])
    new_color = int(action["new_color"])
    non_neighbors = set(range(len(colors))) - set(adjacency[region]) - {region}
    local_conflict_neighborhood = {
        candidate
        for depth, regions in layers_before.items()
        if depth <= 2
        for candidate in regions
    }
    local_nonneighbors = (local_conflict_neighborhood & non_neighbors) - {region}
    nonneighbor_color_matches = sum(1 for other in local_nonneighbors if colors[other] == new_color)
    nonneighbor_color_match_norm = (
        nonneighbor_color_matches / len(local_nonneighbors) if local_nonneighbors else 0.0
    )
    conflict_neighbor_count = sum(1 for neighbor in adjacency[region] if neighbor in conflict_regions_after)
    is_conflict_region = 1.0 if region in conflict_regions_before else 0.0
    direct_conflict_reduction = float(
        sum(1 for edge in conflict_edges_before if region in edge)
        - sum(1 for edge in next_conflict_edges if region in edge)
    )

    conflict_legal_before = sum(count_legal_recolor_options(q, adjacency, colors) for q in conflict_regions_before)
    conflict_legal_after = sum(count_legal_recolor_options(q, adjacency, next_state) for q in conflict_regions_before)
    delta_legal_conflict_colors = float(conflict_legal_after - conflict_legal_before)
    max_conflict_legal_change = max(1, 4 * len(conflict_regions_before))
    delta_legal_conflict_colors_norm = max(0.0, delta_legal_conflict_colors) / max_conflict_legal_change

    same_color_blockers_before = sum(
        1
        for q in conflict_regions_before
        for neighbor in adjacency[q]
        if colors[neighbor] == colors[q]
    )
    same_color_blockers_after = sum(
        1
        for q in conflict_regions_before
        for neighbor in adjacency[q]
        if next_state[neighbor] == next_state[q]
    )
    delta_same_color_blockers_removed = float(same_color_blockers_before - same_color_blockers_after)
    delta_same_color_blockers_removed_norm = max(0.0, delta_same_color_blockers_removed) / max(
        1,
        same_color_blockers_before,
    )
    environment_region = 1.0 if region not in conflict_regions_before and int(action["search_depth"]) <= 2 else 0.0
    region_preserve = is_conflict_region
    color_preserve = (
        0.25 * environment_region
        + 0.25 * delta_legal_conflict_colors_norm
        + 0.25 * delta_same_color_blockers_removed_norm
        + 0.25 * nonneighbor_color_match_norm
    )

    return {
        "repair": repair,
        "repair_raw": -float(len(next_conflict_edges)),
        "repair_delta": float(len(conflict_edges_before) - len(next_conflict_edges)),
        "opportunity": 1.0 if next_depth == 0 else 0.0,
        "spatial": -float(action["search_depth"]),
        "neighbor": float(conflict_neighbor_count),
        "color": float(nonneighbor_color_match_norm),
        "region_preserve": float(region_preserve),
        "color_preserve": float(color_preserve),
        "is_conflict_region": is_conflict_region,
        "direct_conflict_reduction": direct_conflict_reduction,
        "environment_region": environment_region,
        "delta_legal_conflict_colors": delta_legal_conflict_colors,
        "delta_legal_conflict_colors_norm": float(delta_legal_conflict_colors_norm),
        "delta_same_color_blockers_removed": delta_same_color_blockers_removed,
        "delta_same_color_blockers_removed_norm": float(delta_same_color_blockers_removed_norm),
        "local_nonneighbor_color_match": float(nonneighbor_color_matches),
        "local_nonneighbor_color_match_norm": float(nonneighbor_color_match_norm),
        "n_conflict_edges_after": float(len(next_conflict_edges)),
        "next_depth": float(next_depth) if next_depth is not None else math.inf,
    }


def action_heuristic_score(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    action: dict,
    heuristic_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    weights = _merged_heuristic_weights(heuristic_weights)
    features = action_heuristic_features(adjacency, colors, action)
    score = (
        weights["repair"] * features["repair"]
        + weights["opportunity"] * features["opportunity"]
        + weights["region_preserve"] * features["region_preserve"]
        + weights["color_preserve"] * features["color_preserve"]
    )
    return float(score), features


def _heuristic_weights_cache_key(heuristic_weights: dict[str, float] | None = None) -> tuple[tuple[str, float], ...]:
    weights = _merged_heuristic_weights(heuristic_weights)
    return tuple(sorted((key, float(value)) for key, value in weights.items()))


def _state_value_features(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    ordered_actions: list[dict],
    conflict_edges: list[tuple[int, int]],
    conflict_regions: set[int],
) -> dict[str, float]:
    conflict_count = float(len(conflict_edges))
    conflict_region_count = float(len(conflict_regions))
    first_depth = min(int(action["search_depth"]) for action in ordered_actions) if ordered_actions else math.inf

    depth0_actions = [action for action in ordered_actions if int(action["search_depth"]) == 0]
    depth0_action_count = float(len(depth0_actions))
    depth0_region_count = float(len({int(action["region"]) for action in depth0_actions}))

    near_nonconflict_actions = [
        action
        for action in ordered_actions
        if 1 <= int(action["search_depth"]) <= 2 and int(action["region"]) not in conflict_regions
    ]
    useful_nonconflict_access = float(len({int(action["region"]) for action in near_nonconflict_actions}))

    conflict_legal_option_mass = float(
        sum(count_legal_recolor_options(region, adjacency, colors) for region in conflict_regions)
    )
    conflict_legal_option_norm = (
        conflict_legal_option_mass / max(1.0, 4.0 * conflict_region_count) if conflict_region_count > 0 else 0.0
    )

    best_next_conflict_count = conflict_count
    best_conflict_reduction = 0.0
    solve_action_count = 0.0
    for action in ordered_actions:
        next_state = apply_action(colors, action)
        next_conflict_edges = get_conflict_edges(adjacency, next_state)
        next_conflict_count = float(len(next_conflict_edges))
        if next_conflict_count < best_next_conflict_count:
            best_next_conflict_count = next_conflict_count
        best_conflict_reduction = max(best_conflict_reduction, conflict_count - next_conflict_count)
        if next_conflict_count == 0.0:
            solve_action_count += 1.0

    best_conflict_reduction_norm = best_conflict_reduction / max(1.0, conflict_count)
    solve_action_count_norm = solve_action_count / max(1.0, float(len(ordered_actions)))

    normalized_depth0_regions = depth0_region_count / max(1.0, conflict_region_count)
    normalized_nonconflict_access = useful_nonconflict_access / max(1.0, float(len(colors)))

    return {
        "conflict_count": conflict_count,
        "conflict_region_count": conflict_region_count,
        "first_depth": float(first_depth),
        "depth0_action_count": depth0_action_count,
        "depth0_region_count": depth0_region_count,
        "normalized_depth0_regions": float(normalized_depth0_regions),
        "useful_nonconflict_access": useful_nonconflict_access,
        "normalized_nonconflict_access": float(normalized_nonconflict_access),
        "conflict_legal_option_mass": conflict_legal_option_mass,
        "conflict_legal_option_norm": float(conflict_legal_option_norm),
        "best_next_conflict_count": float(best_next_conflict_count),
        "best_conflict_reduction_norm": float(best_conflict_reduction_norm),
        "one_step_solve_available": 1.0 if solve_action_count > 0.0 else 0.0,
        "solve_action_count_norm": float(solve_action_count_norm),
    }


def _conflict_neighbor_color_diversity_norm(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    conflict_regions: set[int],
) -> float:
    if not conflict_regions:
        return 0.0
    total = 0.0
    for region in conflict_regions:
        total += float(len({int(colors[neighbor]) for neighbor in adjacency[region]}))
    return total / max(1.0, 4.0 * float(len(conflict_regions)))


def _conflict_legal_option_mass_for_state(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
) -> tuple[float, set[int], list[tuple[int, int]]]:
    conflict_edges = get_conflict_edges(adjacency, colors)
    conflict_regions = get_conflict_regions(conflict_edges)
    legal_mass = float(
        sum(count_legal_recolor_options(region, adjacency, colors) for region in conflict_regions)
    )
    return legal_mass, conflict_regions, conflict_edges


def _region_neighbor_color_count(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    region: int,
) -> int:
    return len({int(colors[neighbor]) for neighbor in adjacency[region]})


def _prioritized_conflict_endpoint(
    adjacency: dict[int, set[int]],
    conflict_edges: list[tuple[int, int]],
) -> int | None:
    if not conflict_edges:
        return None
    if len(conflict_edges) == 1:
        a, b = conflict_edges[0]
        return min(
            (int(a), len(adjacency[int(a)])),
            (int(b), len(adjacency[int(b)])),
            key=lambda item: (item[1], item[0]),
        )[0]
    conflict_regions = get_conflict_regions(conflict_edges)
    return min(conflict_regions, key=lambda region: (len(adjacency[int(region)]), int(region)))


def _best_helper_recolor_effect(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    target_region: int,
    helper_region: int,
) -> tuple[bool, float, float]:
    baseline_legal = count_legal_recolor_options(target_region, adjacency, colors)
    baseline_neighbor_colors = _region_neighbor_color_count(adjacency, colors, target_region)
    best_legal_gain = 0.0
    best_color_reduction = 0.0
    found = False
    for new_color in legal_recolor_options(helper_region, adjacency, colors):
        next_state = apply_action(
            colors,
            {
                "region": int(helper_region),
                "old_color": int(colors[helper_region]),
                "new_color": int(new_color),
            },
        )
        next_legal = count_legal_recolor_options(target_region, adjacency, next_state)
        next_neighbor_colors = _region_neighbor_color_count(adjacency, next_state, target_region)
        legal_gain = float(max(0, next_legal - baseline_legal))
        color_reduction = float(max(0, baseline_neighbor_colors - next_neighbor_colors))
        if legal_gain > 0.0 or color_reduction > 0.0:
            found = True
            best_legal_gain = max(best_legal_gain, legal_gain)
            best_color_reduction = max(best_color_reduction, color_reduction)
    return found, best_legal_gain, best_color_reduction


def _region_unlock_depth(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    region: int,
    max_depth: int = 3,
) -> int | None:
    """Approximate human-like color-space release depth for one region.

    Depth 0: the region itself is already recolorable.
    Depth 1: one editable neighbor can reduce this region's neighbor-color diversity
             or increase its legal recolor options.
    Depth k>1: some neighbor can itself be unlocked within k-1 steps.
    """
    state0 = tuple(int(c) for c in colors)

    @lru_cache(maxsize=None)
    def search(target_region: int, state_key: tuple[int, ...], depth_remaining: int) -> int | None:
        if count_legal_recolor_options(target_region, adjacency, state_key) > 0:
            return 0
        if depth_remaining == 0:
            return None

        best: int | None = None
        for neighbor in sorted(adjacency[target_region]):
            found, _legal_gain, _color_reduction = _best_helper_recolor_effect(
                adjacency, state_key, target_region, neighbor
            )
            if found:
                candidate = 1
            else:
                subdepth = search(int(neighbor), state_key, depth_remaining - 1)
                candidate = None if subdepth is None else subdepth + 1
            if candidate is not None:
                best = candidate if best is None else min(best, candidate)
        return best

    return search(int(region), state0, int(max_depth))


def near_terminal_expert_features(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
) -> dict[str, float]:
    legal_mass, conflict_regions, conflict_edges = _conflict_legal_option_mass_for_state(adjacency, colors)
    conflict_count = len(conflict_edges)
    if conflict_count == 0:
        return {
            "expert_applicable": 0.0,
            "conflict_legal_mass_norm": 0.0,
            "conflict_neighbor_color_diversity_norm": 0.0,
            "unlock_action_count_norm": 0.0,
            "unlock_distance": math.inf,
            "unlock_reachable_within_2": 0.0,
        }

    ordered_actions, _, _, _ = ordered_legal_actions(adjacency, colors)
    current_legal_mass = legal_mass
    unlock_action_count = 0.0
    for action in ordered_actions:
        if int(action["search_depth"]) > 2:
            continue
        next_state = apply_action(colors, action)
        next_legal_mass, _, _ = _conflict_legal_option_mass_for_state(adjacency, next_state)
        if next_legal_mass > current_legal_mass:
            unlock_action_count += 1.0

    prioritized_endpoint = _prioritized_conflict_endpoint(adjacency, conflict_edges)
    prioritized_neighbor_color_count = (
        float(_region_neighbor_color_count(adjacency, colors, prioritized_endpoint))
        if prioritized_endpoint is not None
        else 0.0
    )
    prioritized_unlock_depth = (
        _region_unlock_depth(adjacency, colors, prioritized_endpoint, max_depth=3)
        if prioritized_endpoint is not None
        else None
    )
    direct_helper_count = 0.0
    best_direct_legal_gain = 0.0
    best_direct_color_reduction = 0.0
    if prioritized_endpoint is not None:
        for neighbor in sorted(adjacency[prioritized_endpoint]):
            found, legal_gain, color_reduction = _best_helper_recolor_effect(
                adjacency, colors, prioritized_endpoint, int(neighbor)
            )
            if found:
                direct_helper_count += 1.0
                best_direct_legal_gain = max(best_direct_legal_gain, legal_gain)
                best_direct_color_reduction = max(best_direct_color_reduction, color_reduction)

    conflict_legal_mass_norm = legal_mass / max(1.0, 4.0 * float(len(conflict_regions)))
    unlock_action_count_norm = unlock_action_count / max(1.0, float(len(ordered_actions)))

    return {
        "expert_applicable": 1.0 if conflict_count == 1 else 0.0,
        "conflict_legal_mass_norm": float(conflict_legal_mass_norm),
        "conflict_neighbor_color_diversity_norm": float(
            _conflict_neighbor_color_diversity_norm(adjacency, colors, conflict_regions)
        ),
        "prioritized_endpoint_degree_norm": (
            float(len(adjacency[prioritized_endpoint])) / max(1.0, float(len(colors)))
            if prioritized_endpoint is not None
            else 0.0
        ),
        "prioritized_endpoint_neighbor_color_count_norm": (
            prioritized_neighbor_color_count / 4.0 if prioritized_endpoint is not None else 0.0
        ),
        "direct_helper_count_norm": (
            direct_helper_count / max(1.0, float(len(adjacency[prioritized_endpoint])))
            if prioritized_endpoint is not None
            else 0.0
        ),
        "best_direct_legal_gain_norm": best_direct_legal_gain / 4.0,
        "best_direct_color_reduction_norm": best_direct_color_reduction / 4.0,
        "unlock_action_count_norm": float(unlock_action_count_norm),
        "best_endpoint_unlock_depth": (
            float(prioritized_unlock_depth) if prioritized_unlock_depth is not None else math.inf
        ),
        "endpoint_unlock_reachable_within_3": 1.0 if prioritized_unlock_depth is not None else 0.0,
    }


def near_terminal_expert_key(
    adjacency: dict[int, set[int]],
    child: "SearchTreeNode",
    adjusted_score: tuple[float, ...],
) -> tuple[float, ...]:
    features = near_terminal_expert_features(adjacency, child.state)
    unlock_distance_score = (
        -features["best_endpoint_unlock_depth"] if math.isfinite(features["best_endpoint_unlock_depth"]) else -999.0
    )
    return (
        features["endpoint_unlock_reachable_within_3"],
        features["conflict_legal_mass_norm"],
        features["direct_helper_count_norm"],
        features["best_direct_legal_gain_norm"],
        features["best_direct_color_reduction_norm"],
        features["unlock_action_count_norm"],
        -features["prioritized_endpoint_neighbor_color_count_norm"],
        -features["prioritized_endpoint_degree_norm"],
        -features["conflict_neighbor_color_diversity_norm"],
        unlock_distance_score,
        *adjusted_score,
    )


def _best_direct_helper_actions(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    target_region: int,
) -> list[tuple[float, dict]]:
    baseline_legal = count_legal_recolor_options(target_region, adjacency, colors)
    baseline_neighbor_colors = _region_neighbor_color_count(adjacency, colors, target_region)
    candidates: list[tuple[float, dict]] = []
    for helper_region in sorted(adjacency[target_region]):
        for new_color in legal_recolor_options(helper_region, adjacency, colors):
            action = {
                "region": int(helper_region),
                "old_color": int(colors[helper_region]),
                "new_color": int(new_color),
            }
            next_state = apply_action(colors, action)
            next_legal = count_legal_recolor_options(target_region, adjacency, next_state)
            next_neighbor_colors = _region_neighbor_color_count(adjacency, next_state, target_region)
            legal_gain = float(max(0, next_legal - baseline_legal))
            color_reduction = float(max(0, baseline_neighbor_colors - next_neighbor_colors))
            if legal_gain > 0.0 or color_reduction > 0.0:
                score = 4.0 * legal_gain + 1.0 * color_reduction
                candidates.append((score, action))
    return sorted(candidates, key=lambda item: (-item[0], item[1]["region"], item[1]["new_color"]))


def _apply_action_sequence(colors: tuple[int, ...], actions: list[dict]) -> tuple[int, ...]:
    state = tuple(int(c) for c in colors)
    for action in actions:
        state = apply_action(state, action)
    return state


def near_terminal_unlock_plan(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    max_depth: int = 3,
) -> list[dict] | None:
    """Programmatic 1-conflict solver based on iterative color-space release.

    Strategy:
    - focus on the conflict endpoint with fewer neighbors
    - try editable neighbors that directly reduce endpoint color diversity or
      increase its legal recolor options
    - if none exist, recursively unlock a neighbor first, then use it as helper
    """
    state0 = tuple(int(c) for c in colors)

    @lru_cache(maxsize=None)
    def search(target_region: int, state_key: tuple[int, ...], depth_remaining: int) -> tuple[tuple[int, int, int], ...] | None:
        if count_legal_recolor_options(target_region, adjacency, state_key) > 0:
            return tuple()
        if depth_remaining == 0:
            return None

        direct_candidates = _best_direct_helper_actions(adjacency, state_key, target_region)
        if direct_candidates:
            best_score, best_action = direct_candidates[0]
            _ = best_score
            return ((int(best_action["region"]), int(best_action["old_color"]), int(best_action["new_color"])),)

        for helper_region in sorted(adjacency[target_region]):
            helper_chain = search(int(helper_region), state_key, depth_remaining - 1)
            if helper_chain is None:
                continue
            helper_state = _apply_action_sequence(
                state_key,
                [
                    {"region": r, "old_color": old, "new_color": new}
                    for r, old, new in helper_chain
                ],
            )
            direct_after_chain = _best_direct_helper_actions(adjacency, helper_state, target_region)
            if direct_after_chain:
                _score, best_action = direct_after_chain[0]
                return helper_chain + (
                    (int(best_action["region"]), int(best_action["old_color"]), int(best_action["new_color"])),
                )
        return None

    legal_mass, conflict_regions, conflict_edges = _conflict_legal_option_mass_for_state(adjacency, state0)
    _ = legal_mass
    target_region = _prioritized_conflict_endpoint(adjacency, conflict_edges)
    if target_region is None:
        return []
    encoded = search(int(target_region), state0, int(max_depth))
    if encoded is None:
        return None
    return [
        {"region": int(region), "old_color": int(old_color), "new_color": int(new_color)}
        for region, old_color, new_color in encoded
    ]


def near_terminal_unlock_search_key(
    adjacency: dict[int, set[int]],
    child: "SearchTreeNode",
    adjusted_score: tuple[float, ...],
    max_depth: int = 3,
) -> tuple[float, ...]:
    plan = near_terminal_unlock_plan(adjacency, child.state, max_depth=max_depth)
    if plan is None:
        return (0.0, -999.0, *near_terminal_expert_key(adjacency, child, adjusted_score))
    return (1.0, -float(len(plan)), *near_terminal_expert_key(adjacency, child, adjusted_score))


def state_heuristic_value(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    heuristic_weights: dict[str, float] | None = None,
    cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
) -> float:
    state_key = tuple(int(c) for c in colors)
    weights_key = _heuristic_weights_cache_key(heuristic_weights)
    cache_key = (state_key, weights_key)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    ordered_actions, conflict_edges, conflict_regions, _ = ordered_legal_actions(adjacency, colors)
    if not conflict_edges:
        value = 0.0
        if cache is not None:
            cache[cache_key] = value
        return value
    if not ordered_actions:
        value = -math.inf
        if cache is not None:
            cache[cache_key] = value
        return value
    features = _state_value_features(
        adjacency=adjacency,
        colors=colors,
        ordered_actions=ordered_actions,
        conflict_edges=conflict_edges,
        conflict_regions=conflict_regions,
    )
    nonconflict_access_weight = 0.75 if features["conflict_count"] > 2.0 else 0.0
    # State-level value: global progress first, flexibility second, direct access as
    # a weak auxiliary term, and an explicit finishing signal for near-terminal states.
    value = (
        -1.25 * features["conflict_count"]
        - 0.5 * features["conflict_region_count"]
        - 0.1 * features["first_depth"]
        + 0.25 * features["normalized_depth0_regions"]
        + nonconflict_access_weight * features["normalized_nonconflict_access"]
        + 0.75 * features["conflict_legal_option_norm"]
        + 0.75 * features["best_conflict_reduction_norm"]
        + 2.5 * features["one_step_solve_available"]
        + 1.0 * features["solve_action_count_norm"]
    )
    if cache is not None:
        cache[cache_key] = value
    return value


def ordered_legal_actions(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
) -> tuple[list[dict], list[tuple[int, int]], set[int], dict[int, list[int]]]:
    legal_by_depth, conflict_edges, conflict_regions, layers = collect_legal_actions_by_depth(adjacency, colors)
    ordered: list[dict] = []
    for depth in sorted(legal_by_depth):
        ordered.extend(
            sorted(
                legal_by_depth[depth],
                key=lambda action: (action["search_depth"], action["region"], action["new_color"]),
            )
        )
    return ordered, conflict_edges, conflict_regions, layers


def state_score(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
) -> tuple[int, int, int, int]:
    legal_by_depth, conflict_edges, _, _ = collect_legal_actions_by_depth(adjacency, colors)
    conflict_count = len(conflict_edges)
    if conflict_count == 0:
        return (1, 0, 1, 0)

    first_depth = min(legal_by_depth) if legal_by_depth else None
    zero_depth_available = 1 if first_depth == 0 else 0
    shallowest_depth = -(first_depth if first_depth is not None else 999)
    return (0, -conflict_count, zero_depth_available, shallowest_depth)


def compose_tree_score(
    base_score: tuple[int, int, int, int],
    distance_to_best_state: int,
    heuristic_value: float = 0.0,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
) -> tuple[float, ...]:
    if tree_score_strategy not in {"task_first", "heuristic_first", "heuristic_only", "value_backup"}:
        raise ValueError(
            "tree_score_strategy must be 'task_first', 'heuristic_first', 'heuristic_only', or 'value_backup'."
        )
    heuristic_component = float(heuristic_eval_weight) * float(heuristic_value)
    solved_component = float(base_score[0])
    if tree_score_strategy == "value_backup":
        return (
            solved_component,
            heuristic_component,
            -float(distance_to_best_state),
        )
    if tree_score_strategy == "heuristic_only":
        return (
            solved_component,
            heuristic_component,
            -float(distance_to_best_state),
        )
    if tree_score_strategy == "heuristic_first":
        return (
            solved_component,
            heuristic_component,
            float(base_score[1]),
            float(base_score[2]),
            float(base_score[3]),
            -float(distance_to_best_state),
        )
    return (*base_score, heuristic_component, -distance_to_best_state)


@dataclass
class SearchTreeNode:
    state: tuple[int, ...]
    action_from_parent: dict | None = None
    parent: "SearchTreeNode | None" = None
    depth: int = 0
    children: list["SearchTreeNode"] = field(default_factory=list)
    expanded: bool = False
    self_score: tuple[int, int, int, int] = (0, -999, 0, -999)
    subtree_score: tuple[float, ...] = (0, -999, 0, -999, 0.0, -999)
    best_child: "SearchTreeNode | None" = None
    best_distance: int = 0
    conflict_edges: list[tuple[int, int]] = field(default_factory=list)
    conflict_regions: set[int] = field(default_factory=set)
    layers: dict[int, list[int]] = field(default_factory=dict)
    visit_count: int = 0
    heuristic_score: float = 0.0

    def initialize(
        self,
        adjacency: dict[int, set[int]],
        heuristic_weights: dict[str, float] | None = None,
        heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
        heuristic_eval_weight: float = 0.0,
        tree_score_strategy: str = "task_first",
    ) -> None:
        legal_by_depth, conflict_edges, conflict_regions, layers = collect_legal_actions_by_depth(adjacency, self.state)
        self.conflict_edges = conflict_edges
        self.conflict_regions = conflict_regions
        self.layers = layers
        self.self_score = state_score(adjacency, self.state)
        self.heuristic_score = state_heuristic_value(
            adjacency,
            self.state,
            heuristic_weights=heuristic_weights,
            cache=heuristic_value_cache,
        )
        self.best_distance = 0
        self.subtree_score = compose_tree_score(
            self.self_score,
            distance_to_best_state=0,
            heuristic_value=self.heuristic_score,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
        )
        self.best_child = None

    def is_terminal(self) -> bool:
        return len(self.conflict_edges) == 0


def choose_best_tree_child(
    node: SearchTreeNode,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> SearchTreeNode | None:
    if not node.children:
        return None
    best_score = max(child.subtree_score for child in node.children)
    best_children = [child for child in node.children if child.subtree_score == best_score]
    if random_tie_break and len(best_children) > 1:
        chooser = rng if rng is not None else random
        return chooser.choice(best_children)
    return best_children[0]


def recompute_tree_values(
    node: SearchTreeNode,
    adjacency: dict[int, set[int]],
    random_tie_break: bool = False,
    rng: random.Random | None = None,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
) -> None:
    for child in node.children:
        recompute_tree_values(
            child,
            adjacency=adjacency,
            random_tie_break=random_tie_break,
            rng=rng,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
        )

    node.heuristic_score = state_heuristic_value(
        adjacency=adjacency,
        colors=node.state,
        heuristic_weights=heuristic_weights,
        cache=heuristic_value_cache,
    )
    self_tree_score = compose_tree_score(
        node.self_score,
        distance_to_best_state=0,
        heuristic_value=node.heuristic_score,
        heuristic_eval_weight=heuristic_eval_weight,
        tree_score_strategy=tree_score_strategy,
    )

    if node.is_terminal():
        node.best_child = None
        node.subtree_score = self_tree_score
        node.best_distance = 0
        return

    if not node.children:
        node.best_child = None
        node.subtree_score = self_tree_score
        node.best_distance = 0
        return

    node.best_child = choose_best_tree_child(node, random_tie_break=random_tie_break, rng=rng)
    if node.best_child is None:
        node.subtree_score = self_tree_score
        node.best_distance = 0
        return

    node.subtree_score = node.best_child.subtree_score
    node.best_distance = node.best_child.best_distance + 1


def backup_tree_value(
    node: SearchTreeNode,
    adjacency: dict[int, set[int]],
    random_tie_break: bool = False,
    rng: random.Random | None = None,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
) -> None:
    """Update one node's backed-up value from its current children."""

    node.heuristic_score = state_heuristic_value(
        adjacency=adjacency,
        colors=node.state,
        heuristic_weights=heuristic_weights,
        cache=heuristic_value_cache,
    )
    self_tree_score = compose_tree_score(
        node.self_score,
        distance_to_best_state=0,
        heuristic_value=node.heuristic_score,
        heuristic_eval_weight=heuristic_eval_weight,
        tree_score_strategy=tree_score_strategy,
    )

    if node.is_terminal() or not node.children:
        node.best_child = None
        node.subtree_score = self_tree_score
        node.best_distance = 0
        return

    node.best_child = choose_best_tree_child(node, random_tie_break=random_tie_break, rng=rng)
    if node.best_child is None:
        node.subtree_score = self_tree_score
        node.best_distance = 0
        return

    node.subtree_score = node.best_child.subtree_score
    node.best_distance = node.best_child.best_distance + 1


def backpropagate_tree_values(
    changed: SearchTreeNode,
    adjacency: dict[int, set[int]],
    random_tie_break: bool = False,
    rng: random.Random | None = None,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
) -> None:
    """Incrementally back up values from a changed node to the root."""

    current: SearchTreeNode | None = changed
    while current is not None:
        backup_tree_value(
            current,
            adjacency=adjacency,
            random_tie_break=random_tie_break,
            rng=rng,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
        )
        current = current.parent


def build_tree_node(
    adjacency: dict[int, set[int]],
    state: tuple[int, ...],
    action_from_parent: dict | None = None,
    parent: SearchTreeNode | None = None,
    depth: int = 0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
) -> SearchTreeNode:
    node = SearchTreeNode(state=state, action_from_parent=action_from_parent, parent=parent, depth=depth)
    node.initialize(
        adjacency,
        heuristic_weights=heuristic_weights,
        heuristic_value_cache=heuristic_value_cache,
        heuristic_eval_weight=heuristic_eval_weight,
        tree_score_strategy=tree_score_strategy,
    )
    return node


def prune_tree_actions(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    actions: list[dict],
    pruning_thresh: float | None = 0.0,
    heuristic_weights: dict[str, float] | None = None,
) -> list[dict]:
    if not actions:
        return []

    scored_actions: list[tuple[float, dict]] = []
    for action in actions:
        heuristic_score, features = action_heuristic_score(
            adjacency,
            colors,
            action,
            heuristic_weights=heuristic_weights,
        )
        action["heuristic_score"] = heuristic_score
        action["heuristic_features"] = features
        scored_actions.append((heuristic_score, action))

    if pruning_thresh is None:
        return [action for _, action in sorted(scored_actions, key=lambda item: item[0], reverse=True)]

    best_score = max(score for score, _ in scored_actions)
    threshold = max(float(pruning_thresh), 0.0)
    kept_actions = [
        action
        for score, action in sorted(scored_actions, key=lambda item: item[0], reverse=True)
        if (best_score - score) <= threshold
    ]
    return kept_actions


def expand_tree_node(
    node: SearchTreeNode,
    adjacency: dict[int, set[int]],
    pruning_thresh: float = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
    force_all_actions: bool = False,
) -> None:
    if node.expanded or node.is_terminal():
        node.expanded = True
        return

    ordered_actions, _, _, _ = ordered_legal_actions(adjacency, node.state)
    if force_all_actions:
        scored_actions: list[tuple[float, dict]] = []
        for action in ordered_actions:
            heuristic_score, features = action_heuristic_score(
                adjacency,
                node.state,
                action,
                heuristic_weights=heuristic_weights,
            )
            action["heuristic_score"] = heuristic_score
            action["heuristic_features"] = features
            scored_actions.append((heuristic_score, action))
        pruned_actions = [action for _, action in sorted(scored_actions, key=lambda item: item[0], reverse=True)]
    else:
        pruned_actions = prune_tree_actions(
            adjacency,
            node.state,
            ordered_actions,
            pruning_thresh=pruning_thresh,
            heuristic_weights=heuristic_weights,
        )
    for action in pruned_actions:
        child_state = apply_action(node.state, action)
        child = build_tree_node(
            adjacency,
            child_state,
            action_from_parent=action,
            parent=node,
            depth=node.depth + 1,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
        )
        node.children.append(child)
    node.expanded = True


def collect_expandable_frontier_nodes(node: SearchTreeNode, max_depth: int) -> list[SearchTreeNode]:
    frontier: list[SearchTreeNode] = []

    def walk(current: SearchTreeNode) -> None:
        if current.is_terminal() or current.depth >= max_depth:
            return
        if not current.expanded:
            frontier.append(current)
            return
        for child in current.children:
            walk(child)

    walk(node)
    return frontier


def select_frontier_node(
    node: SearchTreeNode,
    max_depth: int,
    heuristic_frontier_weight: float = 0.0,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> SearchTreeNode:
    frontier = collect_expandable_frontier_nodes(node, max_depth=max_depth)
    if not frontier:
        return node

    def frontier_key(candidate: SearchTreeNode) -> tuple[float, ...]:
        heuristic_component = float(heuristic_frontier_weight) * float(candidate.heuristic_score)
        return (*candidate.self_score, heuristic_component, -candidate.depth)

    best_key = max(frontier_key(candidate) for candidate in frontier)
    best_nodes = [candidate for candidate in frontier if frontier_key(candidate) == best_key]
    if random_tie_break and len(best_nodes) > 1:
        chooser = rng if rng is not None else random
        return chooser.choice(best_nodes)
    return sorted(
        best_nodes,
        key=lambda candidate: (
            candidate.visit_count,
            candidate.depth,
            candidate.action_from_parent["search_depth"] if candidate.action_from_parent is not None else -1,
            candidate.action_from_parent["region"] if candidate.action_from_parent is not None else -1,
            candidate.action_from_parent["new_color"] if candidate.action_from_parent is not None else -1,
        ),
    )[0]


def select_best_path_leaf(
    node: SearchTreeNode,
    max_depth: int,
) -> SearchTreeNode:
    current = node
    while True:
        if current.is_terminal() or current.depth >= max_depth:
            return current
        if not current.expanded or not current.children or current.best_child is None:
            return current
        current = current.best_child


def reset_subtree_depths(node: SearchTreeNode, depth: int = 0) -> None:
    node.depth = depth
    for child in node.children:
        child.parent = node
        reset_subtree_depths(child, depth + 1)


def iterations_from_gamma(
    gamma: float,
    min_iterations: int = 1,
    max_iterations: int = 500,
) -> int:
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    n_iterations = int(math.floor(1.0 / float(gamma))) + 1
    return max(int(min_iterations), min(int(max_iterations), n_iterations))


def action_key(action: dict) -> tuple[int, int]:
    return int(action["region"]), int(action["new_color"])


def history_loop_penalty(
    child_state: tuple[int, ...],
    history_states: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None = None,
    window: int = 6,
) -> float:
    if not history_states:
        return 0.0
    recent = list(history_states[-max(int(window), 1) :])
    penalty = float(sum(1 for state in recent if tuple(state) == tuple(child_state)))
    if len(history_states) >= 2 and tuple(history_states[-2]) == tuple(child_state):
        penalty += 2.0
    return penalty


def apply_history_penalty_to_score(
    score: tuple[float, ...],
    penalty: float,
    weight: float,
) -> tuple[float, ...]:
    if not score or penalty <= 0.0 or weight <= 0.0:
        return score
    adjusted = list(score)
    if len(adjusted) >= 2:
        adjusted[1] = float(adjusted[1]) - float(weight) * float(penalty)
    else:
        adjusted[0] = float(adjusted[0]) - float(weight) * float(penalty)
    return tuple(adjusted)


def tree_policy_from_root(
    root: SearchTreeNode,
    adjacency: dict[int, set[int]],
    max_depth: int,
    n_iterations: int = 20,
    pruning_thresh: float = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_eval_weight: float = 0.0,
    heuristic_frontier_weight: float = 0.0,
    force_expand_root: bool = False,
    frontier_strategy: str = "global_frontier",
    tree_score_strategy: str = "task_first",
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[SearchTreeNode], dict[tuple[int, int], float], list[dict], list[tuple[int, int]], set[int], dict[int, list[int]], SearchTreeNode]:
    if root.depth != 0:
        raise ValueError("root node must have depth 0")
    if frontier_strategy not in {"global_frontier", "best_path"}:
        raise ValueError("frontier_strategy must be 'global_frontier' or 'best_path'.")
    if heuristic_value_cache is None:
        heuristic_value_cache = {}

    if force_expand_root and not root.expanded and not root.is_terminal():
        root.visit_count += 1
        expand_tree_node(
            root,
            adjacency,
            pruning_thresh=pruning_thresh,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
            force_all_actions=True,
        )
        recompute_tree_values(
            root,
            adjacency=adjacency,
            random_tie_break=random_tie_break,
            rng=rng,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
        )

    for _ in range(n_iterations):
        if frontier_strategy == "best_path":
            frontier = select_best_path_leaf(root, max_depth=max_depth)
        else:
            frontier = select_frontier_node(
                root,
                max_depth=max_depth,
                heuristic_frontier_weight=heuristic_frontier_weight,
                random_tie_break=random_tie_break,
                rng=rng,
            )
        frontier.visit_count += 1
        if frontier.depth >= max_depth or frontier.is_terminal():
            break
        expand_tree_node(
            frontier,
            adjacency,
            pruning_thresh=pruning_thresh,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
            force_all_actions=False,
        )
        recompute_tree_values(
            root,
            adjacency=adjacency,
            random_tie_break=random_tie_break,
            rng=rng,
            heuristic_weights=heuristic_weights,
            heuristic_value_cache=heuristic_value_cache,
            heuristic_eval_weight=heuristic_eval_weight,
            tree_score_strategy=tree_score_strategy,
        )

    if root.best_child is None:
        return [], {}, [], root.conflict_edges, root.conflict_regions, root.layers, root

    best_score = root.best_child.subtree_score
    best_children = [child for child in root.children if child.subtree_score == best_score]
    tree_action_probs = {
        action_key(child.action_from_parent): 1.0 / len(best_children) for child in best_children
    }

    evaluated_actions = []
    for idx, child in enumerate(root.children):
        action = child.action_from_parent
        next_legal_by_depth, next_conflict_edges, _, _ = collect_legal_actions_by_depth(adjacency, child.state)
        next_depth = min(next_legal_by_depth) if next_legal_by_depth else None
        evaluated_actions.append(
            {
                **action,
                "candidate_index": idx,
                "score": child.subtree_score,
                "heuristic_score": action.get("heuristic_score"),
                "heuristic_features": action.get("heuristic_features"),
                "n_conflict_edges_after": len(next_conflict_edges),
                "conflict_delta": len(next_conflict_edges) - len(root.conflict_edges),
                "next_depth": next_depth,
                "solves_after_one_move": len(next_conflict_edges) == 0,
                "opens_conflict_move_next": next_depth == 0,
                "revisits_known_state": False,
                "found_solution_within_depth": child.subtree_score[0] == 1,
            }
        )

    return best_children, tree_action_probs, evaluated_actions, root.conflict_edges, root.conflict_regions, root.layers, root


def tree_choice_from_root(
    root: SearchTreeNode,
    adjacency: dict[int, set[int]],
    max_depth: int,
    n_iterations: int = 20,
    pruning_thresh: float = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_eval_weight: float = 0.0,
    heuristic_frontier_weight: float = 0.0,
    force_expand_root: bool = False,
    frontier_strategy: str = "global_frontier",
    tree_score_strategy: str = "task_first",
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] | None = None,
    history_states: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None = None,
    history_penalty_weight: float = 0.0,
    history_penalty_window: int = 6,
    lapse_rate: float = 0.0,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> tuple[dict | None, dict[int, float], list[dict], list[tuple[int, int]], set[int], dict[int, list[int]], SearchTreeNode]:
    best_children, tree_action_probs, evaluated_actions, conflict_edges, conflict_regions, layers, root = tree_policy_from_root(
        root,
        adjacency,
        max_depth=max_depth,
        n_iterations=n_iterations,
        pruning_thresh=pruning_thresh,
        heuristic_weights=heuristic_weights,
        heuristic_eval_weight=heuristic_eval_weight,
        heuristic_frontier_weight=heuristic_frontier_weight,
        force_expand_root=force_expand_root,
        frontier_strategy=frontier_strategy,
        tree_score_strategy=tree_score_strategy,
        heuristic_value_cache=heuristic_value_cache,
        random_tie_break=random_tie_break,
        rng=rng,
    )
    if not best_children:
        return None, {}, [], conflict_edges, conflict_regions, layers, root

    child_penalties = {
        id(child): history_loop_penalty(
            child.state,
            history_states=history_states,
            window=history_penalty_window,
        )
        for child in root.children
    }
    child_adjusted_scores = {
        id(child): apply_history_penalty_to_score(
            child.subtree_score,
            child_penalties[id(child)],
            history_penalty_weight,
        )
        for child in root.children
    }
    best_adjusted_score = max(child_adjusted_scores.values())
    best_children = [child for child in root.children if child_adjusted_scores[id(child)] == best_adjusted_score]

    # Finishing override: once the current state is already low-conflict, prefer
    # children that immediately terminate the puzzle over merely low-conflict continuations.
    if len(conflict_edges) <= 2:
        one_step_solvers = [child for child in root.children if len(child.conflict_edges) == 0]
        if one_step_solvers:
            best_children = one_step_solvers
        elif len(conflict_edges) == 1:
            expert_keys = {
                id(child): near_terminal_expert_key(
                    adjacency,
                    child,
                    child_adjusted_scores[id(child)],
                )
                for child in root.children
            }
            unlock_keys = {
                id(child): near_terminal_unlock_search_key(
                    adjacency,
                    child,
                    child_adjusted_scores[id(child)],
                    max_depth=3,
                )
                for child in root.children
            }
            best_unlock_key = max(unlock_keys.values())
            best_children = [child for child in root.children if unlock_keys[id(child)] == best_unlock_key]

    counts: dict[int, int] = {}
    for child in best_children:
        region = child.action_from_parent["region"]
        counts[region] = counts.get(region, 0) + 1
    total = sum(counts.values())
    probs = {region: count / total for region, count in counts.items()}

    chooser = rng if rng is not None else random
    lapse_applied = False
    if root.children and lapse_rate > 0.0 and chooser.random() < lapse_rate:
        selected_child = chooser.choice(root.children)
        lapse_applied = True
    else:
        if random_tie_break and len(best_children) > 1:
            selected_child = chooser.choice(best_children)
        else:
            selected_child = sorted(
                best_children,
                key=lambda child: (
                    child.visit_count,
                    -(float(child.action_from_parent.get("heuristic_score", 0.0)) if child.action_from_parent is not None else 0.0),
                    child.action_from_parent["search_depth"],
                    child.action_from_parent["region"],
                    child.action_from_parent["new_color"],
                ),
            )[0]

    for action_info in evaluated_actions:
        matching_child = next(
            (
                child
                for child in root.children
                if action_key(child.action_from_parent) == action_key(action_info)
            ),
            None,
        )
        if matching_child is not None:
            action_info["adjusted_score"] = child_adjusted_scores[id(matching_child)]
            action_info["revisits_known_state"] = bool(child_penalties[id(matching_child)] > 0.0)
            action_info["history_penalty"] = float(child_penalties[id(matching_child)])
        action_info["selected_by_lapse"] = lapse_applied and action_key(action_info) == action_key(selected_child.action_from_parent)

    selected = {
        **selected_child.action_from_parent,
        "candidate_index": -1,
        "score": selected_child.subtree_score,
        "adjusted_score": child_adjusted_scores[id(selected_child)],
        "heuristic_score": selected_child.action_from_parent.get("heuristic_score"),
        "heuristic_features": selected_child.action_from_parent.get("heuristic_features"),
        "n_conflict_edges_after": len(selected_child.conflict_edges),
        "conflict_delta": len(selected_child.conflict_edges) - len(conflict_edges),
        "next_depth": min(selected_child.layers) if selected_child.layers else None,
        "solves_after_one_move": len(selected_child.conflict_edges) == 0,
        "opens_conflict_move_next": (min(selected_child.layers) == 0) if selected_child.layers else False,
        "revisits_known_state": bool(child_penalties[id(selected_child)] > 0.0),
        "history_penalty": float(child_penalties[id(selected_child)]),
        "found_solution_within_depth": selected_child.subtree_score[0] == 1,
        "selected_by_lapse": lapse_applied,
    }
    return selected, probs, evaluated_actions, conflict_edges, conflict_regions, layers, root


def find_solution_path(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    depth_limit: int,
) -> list[dict] | None:
    state = tuple(int(c) for c in colors)

    @lru_cache(maxsize=None)
    def search(state_key: tuple[int, ...], depth_remaining: int) -> tuple[tuple[int, int, int, int], ...] | None:
        conflict_edges = get_conflict_edges(adjacency, state_key)
        if not conflict_edges:
            return tuple()
        if depth_remaining == 0:
            return None

        ordered_actions, _, _, _ = ordered_legal_actions(adjacency, state_key)
        for action in ordered_actions:
            next_state = apply_action(state_key, action)
            suffix = search(next_state, depth_remaining - 1)
            if suffix is not None:
                step = (
                    int(action["search_depth"]),
                    int(action["region"]),
                    int(action["old_color"]),
                    int(action["new_color"]),
                )
                return (step,) + suffix
        return None

    encoded = search(state, depth_limit)
    if encoded is None:
        return None
    return [
        {
            "search_depth": int(search_depth),
            "region": int(region),
            "old_color": int(old_color),
            "new_color": int(new_color),
        }
        for search_depth, region, old_color, new_color in encoded
    ]


def planning_agent_choice(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    max_depth: int = 3,
    forbidden_states: set[tuple[int, ...]] | None = None,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> tuple[dict | None, dict[int, float], list[dict], list[tuple[int, int]], set[int], dict[int, list[int]]]:
    color_tuple = tuple(int(c) for c in colors)
    _ = forbidden_states

    @lru_cache(maxsize=None)
    def best_future_score(state: tuple[int, ...], depth_remaining: int) -> tuple[int, int, int, int]:
        legal_by_depth, conflict_edges, _, _ = collect_legal_actions_by_depth(adjacency, state)
        conflict_count = len(conflict_edges)
        if conflict_count == 0:
            return (1, 0, 1, 0)

        first_depth = min(legal_by_depth) if legal_by_depth else None
        zero_depth_available = 1 if first_depth == 0 else 0
        shallowest_depth = -(first_depth if first_depth is not None else 999)

        base_score = (0, -conflict_count, zero_depth_available, shallowest_depth)
        if depth_remaining == 0 or not legal_by_depth:
            return base_score

        candidate_scores = [base_score]
        current_depth = min(legal_by_depth)
        for action in legal_by_depth[current_depth]:
            next_state = apply_action(state, action)
            candidate_scores.append(best_future_score(next_state, depth_remaining - 1))
        return max(candidate_scores)

    legal_by_depth, conflict_edges, conflict_regions, layers = collect_legal_actions_by_depth(adjacency, color_tuple)
    if not legal_by_depth:
        return None, {}, [], conflict_edges, conflict_regions, layers

    solution_path = find_solution_path(adjacency, color_tuple, max_depth)
    if solution_path is not None and solution_path:
        selected = solution_path[0]
        next_state = apply_action(color_tuple, selected)
        next_legal_by_depth, next_conflict_edges, _, _ = collect_legal_actions_by_depth(adjacency, next_state)
        next_depth = min(next_legal_by_depth) if next_legal_by_depth else None
        selected_with_meta = {
            **selected,
            "candidate_index": 0,
            "score": (1, -len(next_conflict_edges), 1 if next_depth == 0 else 0, -(next_depth if next_depth is not None else 999)),
            "n_conflict_edges_after": len(next_conflict_edges),
            "conflict_delta": len(next_conflict_edges) - len(conflict_edges),
            "next_depth": next_depth,
            "solves_after_one_move": len(next_conflict_edges) == 0,
            "opens_conflict_move_next": next_depth == 0,
            "revisits_known_state": False,
            "found_solution_within_depth": True,
        }
        probs = {selected["region"]: 1.0}
        return (
            selected_with_meta,
            probs,
            [selected_with_meta],
            conflict_edges,
            conflict_regions,
            layers,
        )

    first_depth = min(legal_by_depth)
    first_actions = legal_by_depth[first_depth]
    evaluated_actions: list[dict] = []

    for idx, action in enumerate(first_actions):
        next_state = apply_action(color_tuple, action)
        future_score = best_future_score(next_state, max_depth - 1)
        next_legal_by_depth, next_conflict_edges, _, _ = collect_legal_actions_by_depth(adjacency, next_state)
        next_depth = min(next_legal_by_depth) if next_legal_by_depth else None
        evaluated_actions.append(
            {
                **action,
                "candidate_index": idx,
                "score": future_score,
                "n_conflict_edges_after": len(next_conflict_edges),
                "conflict_delta": len(next_conflict_edges) - len(conflict_edges),
                "next_depth": next_depth,
                "solves_after_one_move": len(next_conflict_edges) == 0,
                "opens_conflict_move_next": next_depth == 0,
                "revisits_known_state": False,
                "found_solution_within_depth": False,
            }
        )

    best_score = max(action["score"] for action in evaluated_actions)
    best_actions = [action for action in evaluated_actions if action["score"] == best_score]
    if random_tie_break:
        chooser = rng if rng is not None else random
        selected = chooser.choice(best_actions)
    else:
        selected = sorted(best_actions, key=lambda x: (x["region"], x["new_color"]))[0]

    counts: dict[int, int] = {}
    for action in best_actions:
        counts[action["region"]] = counts.get(action["region"], 0) + 1
    total = sum(counts.values())
    probs = {region: count / total for region, count in counts.items()}
    return selected, probs, evaluated_actions, conflict_edges, conflict_regions, layers


def first_legal_recolor_action(
    adjacency: dict[int, set[int]],
    colors: list[int] | tuple[int, ...],
    max_depth: int = 3,
    forbidden_states: set[tuple[int, ...]] | None = None,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> dict | None:
    selected, probs, evaluated_actions, conflict_edges, conflict_regions, layers = planning_agent_choice(
        adjacency,
        colors,
        max_depth=max_depth,
        forbidden_states=forbidden_states,
        random_tie_break=random_tie_break,
        rng=rng,
    )
    if selected is None:
        return None

    return {
        "search_depth": selected["search_depth"],
        "region": selected["region"],
        "old_color": selected["old_color"],
        "new_color": selected["new_color"],
        "n_conflict_edges_before": len(conflict_edges),
        "conflict_regions_before": sorted(conflict_regions),
        "region_probabilities": probs,
        "n_legal_actions_first_depth": len(evaluated_actions),
        "n_best_actions": sum(1 for x in evaluated_actions if x["score"] == selected["score"]),
        "selected_next_depth": selected["next_depth"],
        "selected_score": tuple(int(x) for x in selected["score"]),
        "selected_opens_conflict_move_next": bool(selected["opens_conflict_move_next"]),
        "selected_solves_after_one_move": bool(selected["solves_after_one_move"]),
        "selected_found_solution_within_depth": bool(selected["found_solution_within_depth"]),
        "selected_revisits_known_state": bool(selected["revisits_known_state"]),
        "layer_sizes": {depth: len(regions) for depth, regions in layers.items()},
        "candidate_actions": [
            {
                "region": int(x["region"]),
                "old_color": int(x["old_color"]),
                "new_color": int(x["new_color"]),
                "search_depth": int(x["search_depth"]),
                "n_conflict_edges_after": int(x["n_conflict_edges_after"]),
                "conflict_delta": int(x["conflict_delta"]),
                "next_depth": None if x["next_depth"] is None else int(x["next_depth"]),
                "score": tuple(int(v) for v in x["score"]),
                "opens_conflict_move_next": bool(x["opens_conflict_move_next"]),
                "solves_after_one_move": bool(x["solves_after_one_move"]),
                "found_solution_within_depth": bool(x["found_solution_within_depth"]),
                "revisits_known_state": bool(x["revisits_known_state"]),
            }
            for x in evaluated_actions
        ],
    }


def run_agent_on_round(
    round_data: dict,
    max_steps: int = 500,
    max_depth: int = 3,
    stuck_threshold: int = 3,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[int]]:
    adjacency = build_adjacency_map(round_data["mapData"])
    colors = [int(c) for c in round_data["initialColors"]]
    trajectory: list[dict] = []
    current_depth_limit = 1
    stagnant_steps = 0

    for step_index in range(max_steps):
        conflict_edges_before = get_conflict_edges(adjacency, colors)
        if not conflict_edges_before:
            break
        depth_used = current_depth_limit

        action = first_legal_recolor_action(
            adjacency,
            colors,
            max_depth=depth_used,
            random_tie_break=random_tie_break,
            rng=rng,
        )
        if action is None:
            trajectory.append(
                {
                    "agent_step": step_index,
                    "terminated": True,
                    "reason": "no_legal_action_found",
                    "n_conflict_edges_before": len(conflict_edges_before),
                }
            )
            break

        colors[action["region"]] = action["new_color"]
        conflict_edges_after = get_conflict_edges(adjacency, colors)
        improved = len(conflict_edges_after) < len(conflict_edges_before)
        if improved:
            stagnant_steps = 0
            current_depth_limit = 1
        else:
            stagnant_steps += 1
            if stagnant_steps >= stuck_threshold:
                current_depth_limit = min(max_depth, current_depth_limit + 1)
                stagnant_steps = 0

        trajectory.append(
            {
                "agent_step": step_index,
                "terminated": False,
                "reason": "",
                "search_depth": action["search_depth"],
                "region": action["region"],
                "old_color": action["old_color"],
                "new_color": action["new_color"],
                "n_conflict_edges_before": action["n_conflict_edges_before"],
                "n_conflict_edges_after": len(conflict_edges_after),
                "conflict_delta": len(conflict_edges_after) - action["n_conflict_edges_before"],
                "conflict_regions_before": " ".join(str(x + 1) for x in action["conflict_regions_before"]),
                "selected_found_solution_within_depth": action["selected_found_solution_within_depth"],
                "selected_revisits_known_state": action["selected_revisits_known_state"],
                "planning_depth_used": depth_used,
                "improved_conflicts": improved,
                "stagnant_steps_after_move": stagnant_steps,
            }
        )

    return trajectory, colors


def trace_agent_on_round(
    round_data: dict,
    max_steps: int = 200,
    stop_on_cycle: bool = True,
    max_depth: int = 3,
    stuck_threshold: int = 3,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    adjacency = build_adjacency_map(round_data["mapData"])
    colors = [int(c) for c in round_data["initialColors"]]
    visited: set[tuple[int, ...]] = set()
    trace: list[dict] = []
    current_depth_limit = 1
    stagnant_steps = 0

    for step_index in range(max_steps):
        state_key = tuple(colors)
        if stop_on_cycle and state_key in visited:
            trace.append(
                {
                    "agent_step": step_index,
                    "status": "cycle_detected",
                    "colors_before": list(colors),
                }
            )
            break
        visited.add(state_key)

        selected, probs, evaluated_actions, conflict_edges, conflict_regions, layers = planning_agent_choice(
            adjacency,
            colors,
            max_depth=current_depth_limit,
            random_tie_break=random_tie_break,
            rng=rng,
        )
        if not conflict_edges:
            trace.append(
                {
                    "agent_step": step_index,
                    "status": "solved",
                    "colors_before": list(colors),
                }
            )
            break
        if selected is None:
            trace.append(
                {
                    "agent_step": step_index,
                    "status": "no_legal_action_found",
                    "colors_before": list(colors),
                }
            )
            break

        colors_before = list(colors)
        colors[selected["region"]] = selected["new_color"]
        colors_after = list(colors)
        conflict_edges_after = get_conflict_edges(adjacency, colors_after)
        improved = len(conflict_edges_after) < len(conflict_edges)
        if improved:
            stagnant_steps = 0
            next_depth_limit = 1
        else:
            stagnant_steps += 1
            next_depth_limit = current_depth_limit
            if stagnant_steps >= stuck_threshold:
                next_depth_limit = min(max_depth, current_depth_limit + 1)
                stagnant_steps = 0

        trace.append(
            {
                "agent_step": step_index,
                "status": "action",
                "planning_depth_used": current_depth_limit,
                "search_depth": selected["search_depth"],
                "region": selected["region"],
                "old_color": selected["old_color"],
                "new_color": selected["new_color"],
                "colors_before": colors_before,
                "colors_after": colors_after,
                "conflict_edges_before": [tuple(x) for x in conflict_edges],
                "conflict_edges_after": [tuple(x) for x in conflict_edges_after],
                "conflict_regions_before": sorted(conflict_regions),
                "region_probabilities": probs,
                "n_legal_actions_first_depth": len(evaluated_actions),
                "n_best_actions": sum(1 for x in evaluated_actions if x["score"] == selected["score"]),
                "selected_next_depth": selected["next_depth"],
                "selected_score": tuple(int(v) for v in selected["score"]),
                "selected_opens_conflict_move_next": bool(selected["opens_conflict_move_next"]),
                "selected_solves_after_one_move": bool(selected["solves_after_one_move"]),
                "selected_found_solution_within_depth": bool(selected["found_solution_within_depth"]),
                "selected_revisits_known_state": bool(selected["revisits_known_state"]),
                "improved_conflicts": improved,
                "stagnant_steps_after_move": stagnant_steps,
                "planning_depth_next": next_depth_limit,
                "candidate_actions": [
                    {
                        "region": int(x["region"]),
                        "old_color": int(x["old_color"]),
                        "new_color": int(x["new_color"]),
                        "search_depth": int(x["search_depth"]),
                        "n_conflict_edges_after": int(x["n_conflict_edges_after"]),
                        "conflict_delta": int(x["conflict_delta"]),
                        "next_depth": None if x["next_depth"] is None else int(x["next_depth"]),
                        "score": tuple(int(v) for v in x["score"]),
                        "opens_conflict_move_next": bool(x["opens_conflict_move_next"]),
                        "solves_after_one_move": bool(x["solves_after_one_move"]),
                        "found_solution_within_depth": bool(x["found_solution_within_depth"]),
                        "revisits_known_state": bool(x["revisits_known_state"]),
                    }
                    for x in evaluated_actions
                ],
                "layer_sizes": {depth: len(regions) for depth, regions in layers.items()},
            }
        )
        current_depth_limit = next_depth_limit

    return trace


def run_agent_on_all_rounds(
    materials_path: str | Path | None = None,
    max_depth: int = 3,
    stuck_threshold: int = 3,
    random_tie_break: bool = False,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    materials = load_materials(materials_path)
    round_rows: list[dict] = []
    step_rows: list[dict] = []
    rng = random.Random(random_seed) if random_seed is not None else None

    for round_index, round_data in enumerate(materials["rounds"], start=1):
        adjacency = build_adjacency_map(round_data["mapData"])
        initial_colors = [int(c) for c in round_data["initialColors"]]
        initial_conflicts = get_conflict_edges(adjacency, initial_colors)
        trajectory, final_colors = run_agent_on_round(
            round_data,
            max_depth=max_depth,
            stuck_threshold=stuck_threshold,
            random_tie_break=random_tie_break,
            rng=rng,
        )
        final_conflicts = get_conflict_edges(adjacency, final_colors)

        round_rows.append(
            {
                "round": round_index,
                "condition_type": round_data.get("conditionType", ""),
                "initial_conflict_edges": len(initial_conflicts),
                "final_conflict_edges": len(final_conflicts),
                "n_agent_steps": sum(1 for row in trajectory if not row.get("terminated", False)),
                "solved": len(final_conflicts) == 0,
                "max_depth": max_depth,
                "stuck_threshold": stuck_threshold,
                "random_tie_break": random_tie_break,
                "random_seed": random_seed,
            }
        )

        for row in trajectory:
            step_rows.append(
                {
                    "round": round_index,
                    "condition_type": round_data.get("conditionType", ""),
                    **row,
                }
            )

    return pd.DataFrame(round_rows), pd.DataFrame(step_rows)


def run_tree_agent_on_round(
    round_data: dict,
    max_steps: int = 500,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_eval_weight: float = 0.0,
    heuristic_frontier_weight: float = 0.0,
    force_expand_root: bool = False,
    frontier_strategy: str = "global_frontier",
    tree_score_strategy: str = "task_first",
    history_penalty_weight: float = 0.0,
    history_penalty_window: int = 6,
    lapse_rate: float = 0.0,
    gamma: float | None = None,
    near_terminal_conflict_threshold: int | None = None,
    near_terminal_max_depth: int | None = None,
    near_terminal_n_iterations: int | None = None,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[int]]:
    adjacency = build_adjacency_map(round_data["mapData"])
    colors = tuple(int(c) for c in round_data["initialColors"])
    trajectory: list[dict] = []
    history_states: list[tuple[int, ...]] = [colors]
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] = {}
    root = build_tree_node(
        adjacency,
        colors,
        heuristic_weights=heuristic_weights,
        heuristic_value_cache=heuristic_value_cache,
        heuristic_eval_weight=heuristic_eval_weight,
        tree_score_strategy=tree_score_strategy,
    )
    effective_n_iterations = iterations_from_gamma(gamma) if gamma is not None else int(n_iterations)

    for step_index in range(max_steps):
        if root.is_terminal():
            break

        current_max_depth = int(max_depth)
        current_n_iterations = int(effective_n_iterations)
        if (
            near_terminal_conflict_threshold is not None
            and len(root.conflict_edges) <= int(near_terminal_conflict_threshold)
        ):
            if near_terminal_max_depth is not None:
                current_max_depth = max(current_max_depth, int(near_terminal_max_depth))
            if near_terminal_n_iterations is not None:
                current_n_iterations = max(current_n_iterations, int(near_terminal_n_iterations))

        selected, probs, evaluated_actions, conflict_edges, conflict_regions, layers, root = tree_choice_from_root(
            root,
            adjacency,
            max_depth=current_max_depth,
            n_iterations=current_n_iterations,
            pruning_thresh=pruning_thresh,
            heuristic_weights=heuristic_weights,
            heuristic_eval_weight=heuristic_eval_weight,
            heuristic_frontier_weight=heuristic_frontier_weight,
            force_expand_root=force_expand_root,
            frontier_strategy=frontier_strategy,
            tree_score_strategy=tree_score_strategy,
            heuristic_value_cache=heuristic_value_cache,
            history_states=history_states,
            history_penalty_weight=history_penalty_weight,
            history_penalty_window=history_penalty_window,
            lapse_rate=lapse_rate,
            random_tie_break=random_tie_break,
            rng=rng,
        )
        if selected is None:
            trajectory.append(
                {
                    "agent_step": step_index,
                    "terminated": True,
                    "reason": "no_tree_action_found",
                    "n_conflict_edges_before": len(root.conflict_edges),
                }
            )
            break

        chosen_child = None
        for child in root.children:
            action = child.action_from_parent
            if action["region"] == selected["region"] and action["new_color"] == selected["new_color"]:
                chosen_child = child
                break
        if chosen_child is None:
            trajectory.append(
                {
                    "agent_step": step_index,
                    "terminated": True,
                    "reason": "selected_child_missing",
                    "n_conflict_edges_before": len(root.conflict_edges),
                }
            )
            break

        conflict_edges_after = chosen_child.conflict_edges
        trajectory.append(
            {
                "agent_step": step_index,
                "terminated": False,
                "reason": "",
                "search_depth": selected["search_depth"],
                "region": selected["region"],
                "old_color": selected["old_color"],
                "new_color": selected["new_color"],
                "n_conflict_edges_before": len(conflict_edges),
                "n_conflict_edges_after": len(conflict_edges_after),
                "conflict_delta": len(conflict_edges_after) - len(conflict_edges),
                "conflict_regions_before": " ".join(str(x + 1) for x in sorted(conflict_regions)),
                "selected_found_solution_within_depth": bool(selected["found_solution_within_depth"]),
                "planning_depth_used": current_max_depth,
                "tree_iterations_used": current_n_iterations,
                "pruning_thresh": pruning_thresh,
                "lapse_rate": lapse_rate,
                "gamma": gamma,
                "heuristic_weights": _merged_heuristic_weights(heuristic_weights),
                "heuristic_eval_weight": heuristic_eval_weight,
                "heuristic_frontier_weight": heuristic_frontier_weight,
                "force_expand_root": force_expand_root,
                "frontier_strategy": frontier_strategy,
                "tree_score_strategy": tree_score_strategy,
                "history_penalty_weight": history_penalty_weight,
                "selected_heuristic_score": selected.get("heuristic_score"),
                "selected_by_lapse": bool(selected["selected_by_lapse"]),
                "selected_history_penalty": float(selected.get("history_penalty", 0.0)),
                "expanded_root_children": len(root.children),
                "improved_conflicts": len(conflict_edges_after) < len(conflict_edges),
            }
        )

        root = chosen_child
        root.parent = None
        reset_subtree_depths(root, depth=0)
        history_states.append(tuple(root.state))

    return trajectory, list(root.state)


def _reconstruct_bfs_action_path(
    node_id: int,
    parents: dict[int, int | None],
    actions: dict[int, dict | None],
) -> list[dict]:
    path: list[dict] = []
    current: int | None = node_id
    while current is not None:
        action = actions[current]
        if action is not None:
            path.append(action)
        current = parents[current]
    path.reverse()
    return path


def run_bfs_tree_agent_on_round(
    round_data: dict,
    max_steps: int = 500,
    max_depth: int = 8,
    max_expansions: int = 5000,
    n_iterations: int | None = 20,
    pruning_thresh: float | None = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_eval_weight: float = 0.0,
    tree_score_strategy: str = "task_first",
    gamma: float | None = None,
    lapse_rate: float = 0.0,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[int]]:
    """Solve a round with fourinarow-style BFS tree expansion.

    This variant uses the same legal recolor action generator as the tree agent,
    including heuristic action pruning. The expansion loop matches the
    fourinarow BFS pattern: expand the current best leaf, back up values to the
    root, then follow the root's best pointers to choose the next leaf. gamma
    controls the total expansion budget as int(1 / gamma) + 1, matching
    fourinarow BFS. lapse_rate is applied before each search as a random-action
    lapse.
    """

    chooser = rng if rng is not None else random
    adjacency = build_adjacency_map(round_data["mapData"])
    current_state = tuple(int(c) for c in round_data["initialColors"])
    trajectory: list[dict] = []
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] = {}

    effective_max_expansions = iterations_from_gamma(
        gamma,
        max_iterations=int(max_expansions),
    ) if gamma is not None else int(max_expansions)

    def count_subtree_nodes(node: SearchTreeNode) -> int:
        return 1 + sum(count_subtree_nodes(child) for child in node.children)

    for step_index in range(int(max_steps)):
        conflict_edges_before = get_conflict_edges(adjacency, current_state)
        conflict_regions_before = get_conflict_regions(conflict_edges_before)
        if not conflict_edges_before:
            break

        if lapse_rate > 0.0 and chooser.random() < lapse_rate:
            ordered_actions, _, _, _ = ordered_legal_actions(adjacency, current_state)
            if not ordered_actions:
                trajectory.append(
                    {
                        "agent_step": step_index,
                        "terminated": True,
                        "reason": "no_bfs_action_found",
                        "n_conflict_edges_before": len(conflict_edges_before),
                    }
                )
                break
            action = chooser.choice(ordered_actions)
            iterations_used = 0
            root_children = 0
            tree_nodes = 1
            found_solution_within_depth = False
            selected_by_lapse = True
        else:
            root = build_tree_node(
                adjacency,
                current_state,
                heuristic_weights=heuristic_weights,
                heuristic_value_cache=heuristic_value_cache,
                heuristic_eval_weight=heuristic_eval_weight,
                tree_score_strategy=tree_score_strategy,
            )
            iterations_used = 0
            for iteration in range(effective_max_expansions):
                frontier = select_best_path_leaf(root, max_depth=int(max_depth))
                if frontier.depth >= int(max_depth) or frontier.is_terminal():
                    break
                expand_tree_node(
                    frontier,
                    adjacency,
                    pruning_thresh=pruning_thresh,
                    heuristic_weights=heuristic_weights,
                    heuristic_value_cache=heuristic_value_cache,
                    heuristic_eval_weight=heuristic_eval_weight,
                    tree_score_strategy=tree_score_strategy,
                    force_all_actions=False,
                )
                backpropagate_tree_values(
                    frontier,
                    adjacency=adjacency,
                    random_tie_break=random_tie_break,
                    rng=chooser,
                    heuristic_weights=heuristic_weights,
                    heuristic_value_cache=heuristic_value_cache,
                    heuristic_eval_weight=heuristic_eval_weight,
                    tree_score_strategy=tree_score_strategy,
                )
                iterations_used = iteration + 1
                if root.is_terminal():
                    break

            selected_child = root.best_child
            if selected_child is None or selected_child.action_from_parent is None:
                trajectory.append(
                    {
                        "agent_step": step_index,
                        "terminated": True,
                        "reason": "no_bfs_action_found",
                        "n_conflict_edges_before": len(conflict_edges_before),
                        "bfs_expansions": iterations_used,
                        "bfs_max_expansions": effective_max_expansions,
                        "bfs_visited_states": count_subtree_nodes(root),
                        "bfs_solved_within_depth": False,
                    }
                )
                break
            action = selected_child.action_from_parent
            root_children = len(root.children)
            tree_nodes = count_subtree_nodes(root)
            found_solution_within_depth = bool(selected_child.subtree_score[0] == 1)
            selected_by_lapse = False

        next_state = apply_action(current_state, action)
        conflict_edges_after = get_conflict_edges(adjacency, next_state)
        trajectory.append(
            {
                "agent_step": step_index,
                "terminated": False,
                "reason": "",
                "search_depth": action["search_depth"],
                "region": action["region"],
                "old_color": action["old_color"],
                "new_color": action["new_color"],
                "n_conflict_edges_before": len(conflict_edges_before),
                "n_conflict_edges_after": len(conflict_edges_after),
                "conflict_delta": len(conflict_edges_after) - len(conflict_edges_before),
                "conflict_regions_before": " ".join(str(x + 1) for x in sorted(conflict_regions_before)),
                "selected_found_solution_within_depth": found_solution_within_depth,
                "planning_depth_used": max_depth,
                "tree_iterations_used": iterations_used,
                "pruning_thresh": pruning_thresh,
                "lapse_rate": lapse_rate,
                "gamma": gamma,
                "heuristic_weights": _merged_heuristic_weights(heuristic_weights),
                "heuristic_eval_weight": heuristic_eval_weight,
                "heuristic_frontier_weight": 0.0,
                "force_expand_root": False,
                "frontier_strategy": "fourinarow_best_leaf",
                "tree_score_strategy": tree_score_strategy,
                "history_penalty_weight": 0.0,
                "selected_heuristic_score": action.get("heuristic_score"),
                "selected_by_lapse": selected_by_lapse,
                "selected_history_penalty": 0.0,
                "expanded_root_children": root_children,
                "improved_conflicts": len(conflict_edges_after) < len(conflict_edges_before),
                "bfs_expansions": iterations_used,
                "bfs_max_expansions": effective_max_expansions,
                "bfs_visited_states": tree_nodes,
                "bfs_solved_within_depth": found_solution_within_depth,
            }
        )
        current_state = next_state

    if not trajectory:
        trajectory.append(
            {
                "agent_step": 0,
                "terminated": True,
                "reason": "no_bfs_solution_found",
                "n_conflict_edges_before": len(get_conflict_edges(adjacency, current_state)),
                "bfs_expansions": 0,
                "bfs_max_expansions": effective_max_expansions,
                "bfs_visited_states": 1,
                "bfs_solved_within_depth": False,
            }
        )

    return trajectory, list(current_state)


def trace_tree_agent_on_round(
    round_data: dict,
    max_steps: int = 200,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_eval_weight: float = 0.0,
    heuristic_frontier_weight: float = 0.0,
    frontier_strategy: str = "global_frontier",
    tree_score_strategy: str = "task_first",
    history_penalty_weight: float = 0.0,
    history_penalty_window: int = 6,
    lapse_rate: float = 0.0,
    gamma: float | None = None,
    near_terminal_conflict_threshold: int | None = None,
    near_terminal_max_depth: int | None = None,
    near_terminal_n_iterations: int | None = None,
    random_tie_break: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    adjacency = build_adjacency_map(round_data["mapData"])
    heuristic_value_cache: dict[tuple[tuple[int, ...], tuple[tuple[str, float], ...]], float] = {}
    history_states: list[tuple[int, ...]] = [tuple(int(c) for c in round_data["initialColors"])]
    root = build_tree_node(
        adjacency,
        tuple(int(c) for c in round_data["initialColors"]),
        heuristic_weights=heuristic_weights,
        heuristic_value_cache=heuristic_value_cache,
        heuristic_eval_weight=heuristic_eval_weight,
        tree_score_strategy=tree_score_strategy,
    )
    trace: list[dict] = []
    effective_n_iterations = iterations_from_gamma(gamma) if gamma is not None else int(n_iterations)

    for step_index in range(max_steps):
        if root.is_terminal():
            trace.append(
                {
                    "agent_step": step_index,
                    "status": "solved",
                    "colors_before": list(root.state),
                }
            )
            break

        current_max_depth = int(max_depth)
        current_n_iterations = int(effective_n_iterations)
        if (
            near_terminal_conflict_threshold is not None
            and len(root.conflict_edges) <= int(near_terminal_conflict_threshold)
        ):
            if near_terminal_max_depth is not None:
                current_max_depth = max(current_max_depth, int(near_terminal_max_depth))
            if near_terminal_n_iterations is not None:
                current_n_iterations = max(current_n_iterations, int(near_terminal_n_iterations))

        selected, probs, evaluated_actions, conflict_edges, conflict_regions, layers, _ = tree_choice_from_root(
            root,
            adjacency,
            max_depth=current_max_depth,
            n_iterations=current_n_iterations,
            pruning_thresh=pruning_thresh,
            heuristic_weights=heuristic_weights,
            heuristic_eval_weight=heuristic_eval_weight,
            heuristic_frontier_weight=heuristic_frontier_weight,
            frontier_strategy=frontier_strategy,
            tree_score_strategy=tree_score_strategy,
            heuristic_value_cache=heuristic_value_cache,
            history_states=history_states,
            history_penalty_weight=history_penalty_weight,
            history_penalty_window=history_penalty_window,
            lapse_rate=lapse_rate,
            random_tie_break=random_tie_break,
            rng=rng,
        )
        if selected is None:
            trace.append(
                {
                    "agent_step": step_index,
                    "status": "no_tree_action_found",
                    "colors_before": list(root.state),
                }
            )
            break

        chosen_child = None
        for child in root.children:
            action = child.action_from_parent
            if action["region"] == selected["region"] and action["new_color"] == selected["new_color"]:
                chosen_child = child
                break
        if chosen_child is None:
            trace.append(
                {
                    "agent_step": step_index,
                    "status": "selected_child_missing",
                    "colors_before": list(root.state),
                }
            )
            break

        trace.append(
            {
                "agent_step": step_index,
                "status": "action",
                "planning_depth_used": current_max_depth,
                "tree_iterations_used": current_n_iterations,
                "pruning_thresh": pruning_thresh,
                "lapse_rate": lapse_rate,
                "gamma": gamma,
                "heuristic_weights": _merged_heuristic_weights(heuristic_weights),
                "heuristic_eval_weight": heuristic_eval_weight,
                "heuristic_frontier_weight": heuristic_frontier_weight,
                "frontier_strategy": frontier_strategy,
                "tree_score_strategy": tree_score_strategy,
                "history_penalty_weight": history_penalty_weight,
                "selected_by_lapse": bool(selected["selected_by_lapse"]),
                "selected_heuristic_score": selected.get("heuristic_score"),
                "selected_history_penalty": float(selected.get("history_penalty", 0.0)),
                "search_depth": selected["search_depth"],
                "region": selected["region"],
                "old_color": selected["old_color"],
                "new_color": selected["new_color"],
                "colors_before": list(root.state),
                "colors_after": list(chosen_child.state),
                "conflict_edges_before": [tuple(x) for x in conflict_edges],
                "conflict_edges_after": [tuple(x) for x in chosen_child.conflict_edges],
                "conflict_regions_before": sorted(conflict_regions),
                "region_probabilities": probs,
                "n_legal_actions_first_depth": len(evaluated_actions),
                "n_best_actions": sum(1 for x in evaluated_actions if x["score"] == selected["score"]),
                "selected_next_depth": selected["next_depth"],
                "selected_score": tuple(float(v) for v in selected["score"]),
                "selected_found_solution_within_depth": bool(selected["found_solution_within_depth"]),
                "candidate_actions": [
                    {
                        "region": int(x["region"]),
                        "old_color": int(x["old_color"]),
                        "new_color": int(x["new_color"]),
                        "search_depth": int(x["search_depth"]),
                        "n_conflict_edges_after": int(x["n_conflict_edges_after"]),
                        "conflict_delta": int(x["conflict_delta"]),
                        "next_depth": None if x["next_depth"] is None else int(x["next_depth"]),
                        "score": tuple(float(v) for v in x["score"]),
                        "heuristic_score": None
                        if x.get("heuristic_score") is None
                        else float(x["heuristic_score"]),
                        "heuristic_features": x.get("heuristic_features"),
                        "opens_conflict_move_next": bool(x["opens_conflict_move_next"]),
                        "solves_after_one_move": bool(x["solves_after_one_move"]),
                        "found_solution_within_depth": bool(x["found_solution_within_depth"]),
                    }
                    for x in evaluated_actions
                ],
                "layer_sizes": {depth: len(regions) for depth, regions in layers.items()},
            }
        )

        root = chosen_child
        root.parent = None
        reset_subtree_depths(root, depth=0)
        history_states.append(tuple(root.state))

    return trace


def run_tree_agent_on_all_rounds(
    materials_path: str | Path | None = None,
    max_steps: int = 500,
    max_depth: int = 4,
    n_iterations: int = 20,
    pruning_thresh: float = 0.0,
    heuristic_weights: dict[str, float] | None = None,
    heuristic_eval_weight: float = 0.0,
    heuristic_frontier_weight: float = 0.0,
    force_expand_root: bool = False,
    frontier_strategy: str = "global_frontier",
    tree_score_strategy: str = "task_first",
    lapse_rate: float = 0.0,
    gamma: float | None = None,
    random_tie_break: bool = False,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    materials = load_materials(materials_path)
    round_rows: list[dict] = []
    step_rows: list[dict] = []
    rng = random.Random(random_seed) if random_seed is not None else None

    for round_index, round_data in enumerate(materials["rounds"], start=1):
        adjacency = build_adjacency_map(round_data["mapData"])
        initial_colors = [int(c) for c in round_data["initialColors"]]
        initial_conflicts = get_conflict_edges(adjacency, initial_colors)
        trajectory, final_colors = run_tree_agent_on_round(
            round_data,
            max_steps=max_steps,
            max_depth=max_depth,
            n_iterations=n_iterations,
            pruning_thresh=pruning_thresh,
            heuristic_weights=heuristic_weights,
            heuristic_eval_weight=heuristic_eval_weight,
            heuristic_frontier_weight=heuristic_frontier_weight,
            force_expand_root=force_expand_root,
            frontier_strategy=frontier_strategy,
            tree_score_strategy=tree_score_strategy,
            lapse_rate=lapse_rate,
            gamma=gamma,
            random_tie_break=random_tie_break,
            rng=rng,
        )
        final_conflicts = get_conflict_edges(adjacency, final_colors)
        round_rows.append(
            {
                "round": round_index,
                "condition_type": round_data.get("conditionType", ""),
                "initial_conflict_edges": len(initial_conflicts),
                "final_conflict_edges": len(final_conflicts),
                "n_agent_steps": sum(1 for row in trajectory if not row.get("terminated", False)),
                "solved": len(final_conflicts) == 0,
                "max_depth": max_depth,
                "n_iterations": iterations_from_gamma(gamma) if gamma is not None else n_iterations,
                "pruning_thresh": pruning_thresh,
                "lapse_rate": lapse_rate,
                "gamma": gamma,
                "heuristic_weights": _merged_heuristic_weights(heuristic_weights),
                "heuristic_eval_weight": heuristic_eval_weight,
                "heuristic_frontier_weight": heuristic_frontier_weight,
                "force_expand_root": force_expand_root,
                "frontier_strategy": frontier_strategy,
                "tree_score_strategy": tree_score_strategy,
                "random_tie_break": random_tie_break,
                "random_seed": random_seed,
            }
        )
        for row in trajectory:
            step_rows.append(
                {
                    "round": round_index,
                    "condition_type": round_data.get("conditionType", ""),
                    **row,
                }
            )

    return pd.DataFrame(round_rows), pd.DataFrame(step_rows)
