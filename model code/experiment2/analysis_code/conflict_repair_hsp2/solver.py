"""Conflict repair solver with Min-Conflicts, EFLOP, and HSP2 planning.

This module implements a repair-oriented map recoloring solver. The solver is
not designed to guarantee global optimality. Instead it combines:

1. Min-Conflicts for fast local descent on obvious conflicts.
2. EFLOP for local chain perturbations when the search gets stuck.
3. HSP2-style weighted best-first planning with a relaxed repair heuristic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import math
import random
from typing import Deque, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


State = Tuple[int, ...]
Action = Tuple[int, int, int]
AdjList = Dict[int, List[int]]
CandidateReason = Dict[int, str]

LARGE_COST = 10


@dataclass
class Node:
    """Planning node used by HSP2 weighted best-first search."""

    state: State
    g: int
    h: int
    f: float
    parent: Optional["Node"]
    action: Optional[Action]
    depth: int


def state_to_key(state: Sequence[int]) -> State:
    """Return an immutable key for a color assignment state."""

    return tuple(state)


def get_conflict_edges(state: Sequence[int], adj_list: AdjList) -> Set[Tuple[int, int]]:
    """Return all undirected conflict edges in the current coloring state."""

    edges: Set[Tuple[int, int]] = set()
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            if node < neighbor and state[node] == state[neighbor]:
                edges.add((node, neighbor))
    return edges


def get_conflict_nodes(state: Sequence[int], adj_list: AdjList) -> Set[int]:
    """Return all nodes participating in at least one conflict edge."""

    conflict_nodes: Set[int] = set()
    for u, v in get_conflict_edges(state, adj_list):
        conflict_nodes.add(u)
        conflict_nodes.add(v)
    return conflict_nodes


def count_conflicts(state: Sequence[int], adj_list: AdjList) -> int:
    """Return the number of conflict edges in the current state."""

    return len(get_conflict_edges(state, adj_list))


def count_node_conflicts(state: Sequence[int], node: int, adj_list: AdjList) -> int:
    """Return the number of conflicts between a node and its neighbors."""

    node_color = state[node]
    return sum(1 for neighbor in adj_list[node] if state[neighbor] == node_color)


def apply_action(state: Sequence[int], action: Action) -> State:
    """Return a new state after applying an action without mutating the input."""

    node_id, old_color, new_color = action
    if state[node_id] != old_color:
        raise ValueError(
            f"Action old_color mismatch at node {node_id}: "
            f"state has {state[node_id]}, action expected {old_color}."
        )
    next_state = list(state)
    next_state[node_id] = new_color
    return tuple(next_state)


def is_solved(state: Sequence[int], adj_list: AdjList) -> bool:
    """Return True if the state contains no conflicts."""

    return count_conflicts(state, adj_list) == 0


def legal_colors_for_node(
    state: Sequence[int],
    node: int,
    colors: Sequence[int],
    adj_list: AdjList,
) -> List[int]:
    """Return all colors that would leave the node conflict-free with neighbors."""

    legal: List[int] = []
    for color in colors:
        if all(state[neighbor] != color for neighbor in adj_list[node]):
            legal.append(color)
    return legal


def has_legal_recolor(
    state: Sequence[int],
    node: int,
    colors: Sequence[int],
    adj_list: AdjList,
) -> bool:
    """Return True if the node has a legal recolor different from its current color."""

    current_color = state[node]
    return any(color != current_color for color in legal_colors_for_node(state, node, colors, adj_list))


def _random_choice(rng: Optional[random.Random], items: Sequence[int]) -> int:
    """Choose one item using the provided RNG when available."""

    chooser = rng if rng is not None else random
    return chooser.choice(list(items))


def run_min_conflicts(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    max_steps: int = 200,
    stuck_window: int = 20,
    rng: Optional[random.Random] = None,
) -> Tuple[State, List[Action], bool, bool, List[dict]]:
    """Run a Min-Conflicts local repair pass on the current state."""

    current_state = state_to_key(state)
    action_path: List[Action] = []
    logs: List[dict] = []

    best_state = current_state
    best_conflicts = count_conflicts(current_state, adj_list)
    best_action_path: List[Action] = []

    steps_since_improvement = 0
    history: List[int] = []

    for step in range(max_steps):
        before_conflicts = count_conflicts(current_state, adj_list)
        if before_conflicts == 0:
            return current_state, action_path, False, True, logs

        if before_conflicts < best_conflicts:
            best_state = current_state
            best_conflicts = before_conflicts
            best_action_path = list(action_path)
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        conflict_nodes = sorted(get_conflict_nodes(current_state, adj_list))
        if not conflict_nodes:
            return current_state, action_path, False, True, logs

        legal_conflict_nodes = [
            node
            for node in conflict_nodes
            if any(
                color != current_state[node]
                for color in legal_colors_for_node(current_state, node, colors, adj_list)
            )
        ]
        if not legal_conflict_nodes:
            logs.append(
                {
                    "step": step,
                    "module": "min_conflicts",
                    "event": "stuck_no_legal_conflict_recolor",
                    "before_conflicts": before_conflicts,
                }
            )
            return best_state, best_action_path, True, False, logs

        node = _random_choice(rng, legal_conflict_nodes)
        old_color = current_state[node]

        best_colors: List[int] = []
        best_child_conflicts: Optional[int] = None
        for color in legal_colors_for_node(current_state, node, colors, adj_list):
            if color == old_color:
                continue
            child = apply_action(current_state, (node, old_color, color))
            child_conflicts = count_conflicts(child, adj_list)
            if best_child_conflicts is None or child_conflicts < best_child_conflicts:
                best_child_conflicts = child_conflicts
                best_colors = [color]
            elif child_conflicts == best_child_conflicts:
                best_colors.append(color)

        if not best_colors:
            return best_state, best_action_path, True, False, logs

        chosen_color = _random_choice(rng, best_colors)
        action = (node, old_color, chosen_color)
        current_state = apply_action(current_state, action)
        action_path.append(action)

        after_conflicts = count_conflicts(current_state, adj_list)
        logs.append(
            {
                "step": step,
                "module": "min_conflicts",
                "action": action,
                "before_conflicts": before_conflicts,
                "after_conflicts": after_conflicts,
            }
        )

        history.append(after_conflicts)
        if len(history) > stuck_window:
            history = history[-stuck_window:]

        if after_conflicts < best_conflicts:
            best_state = current_state
            best_conflicts = after_conflicts
            best_action_path = list(action_path)
            steps_since_improvement = 0

        stuck = steps_since_improvement >= stuck_window or (
            len(history) >= stuck_window and min(history) >= best_conflicts
        )
        if stuck:
            return best_state, best_action_path, True, False, logs

    return best_state, best_action_path, True, is_solved(best_state, adj_list), logs


def run_eflop(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    max_depth: int = 5,
    max_visits_per_node: int = 2,
    rng: Optional[random.Random] = None,
) -> Tuple[State, List[Action], List[dict]]:
    """Run EFLOP local chain perturbation from a stuck repair state."""

    current_state = state_to_key(state)
    action_path: List[Action] = []
    logs: List[dict] = []

    conflict_nodes = sorted(get_conflict_nodes(current_state, adj_list))
    if not conflict_nodes:
        return current_state, action_path, logs

    v_start = _random_choice(rng, conflict_nodes)
    old_color = current_state[v_start]
    candidate_colors = [color for color in colors if color != old_color]
    if not candidate_colors:
        return current_state, action_path, logs

    seed_color = _random_choice(rng, candidate_colors)
    seed_action = (v_start, old_color, seed_color)
    before_conflicts = count_conflicts(current_state, adj_list)
    current_state = apply_action(current_state, seed_action)
    after_conflicts = count_conflicts(current_state, adj_list)
    action_path.append(seed_action)
    logs.append(
        {
            "module": "eflop_seed",
            "action": seed_action,
            "before_conflicts": before_conflicts,
            "after_conflicts": after_conflicts,
        }
    )

    queue: Deque[Tuple[int, int, int]] = deque()
    for neighbor in adj_list[v_start]:
        if current_state[neighbor] == current_state[v_start]:
            queue.append((neighbor, v_start, 1))

    visited_count: DefaultDict[int, int] = defaultdict(int)

    while queue:
        node, source, depth = queue.popleft()
        if depth > max_depth:
            continue
        if visited_count[node] >= max_visits_per_node:
            continue
        visited_count[node] += 1

        old_color = current_state[node]
        best_colors: List[int] = []
        best_score: Optional[int] = None

        for color in colors:
            if color == old_color:
                continue
            child = apply_action(current_state, (node, old_color, color))
            local_conflicts = sum(
                1
                for neighbor in adj_list[node]
                if neighbor != source and child[neighbor] == child[node]
            )
            if best_score is None or local_conflicts < best_score:
                best_score = local_conflicts
                best_colors = [color]
            elif local_conflicts == best_score:
                best_colors.append(color)

        if not best_colors:
            continue

        chosen_color = _random_choice(rng, best_colors)
        action = (node, old_color, chosen_color)
        before_conflicts = count_conflicts(current_state, adj_list)
        current_state = apply_action(current_state, action)
        after_conflicts = count_conflicts(current_state, adj_list)
        action_path.append(action)
        logs.append(
            {
                "module": "eflop_propagation",
                "depth": depth,
                "source": source,
                "action": action,
                "before_conflicts": before_conflicts,
                "after_conflicts": after_conflicts,
            }
        )

        for neighbor in adj_list[node]:
            if neighbor == source:
                continue
            if current_state[neighbor] == current_state[node]:
                queue.append((neighbor, node, depth + 1))

    return current_state, action_path, logs


def _blockers_for_color(state: Sequence[int], node: int, color: int, adj_list: AdjList) -> List[int]:
    """Return neighbors currently blocking a node from taking a target color."""

    return [neighbor for neighbor in adj_list[node] if state[neighbor] == color]


def _dynamic_neighbor_cost(
    state: Sequence[int],
    node: int,
    colors: Sequence[int],
    adj_list: AdjList,
    depth_limit: int = 3,
    cache: Optional[Dict[Tuple[State, int, int], int]] = None,
    active: Optional[Set[Tuple[int, int]]] = None,
) -> int:
    """Estimate the local unlock cost needed to move a blocker node away."""

    state_key = state_to_key(state)
    cache_key = (state_key, node, depth_limit)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if has_legal_recolor(state, node, colors, adj_list):
        if cache is not None:
            cache[cache_key] = 1
        return 1

    if depth_limit <= 0:
        if cache is not None:
            cache[cache_key] = LARGE_COST
        return LARGE_COST

    if active is None:
        active = set()
    active_key = (node, depth_limit)
    if active_key in active:
        return LARGE_COST
    active.add(active_key)

    current_color = state[node]
    best_cost = LARGE_COST
    for color in colors:
        if color == current_color:
            continue
        blockers = _blockers_for_color(state, node, color, adj_list)
        if not blockers:
            best_cost = min(best_cost, 1)
            continue
        blocker_sum = 0
        for blocker in blockers:
            blocker_sum += _dynamic_neighbor_cost(
                state,
                blocker,
                colors,
                adj_list,
                depth_limit=depth_limit - 1,
                cache=cache,
                active=active,
            )
        best_cost = min(best_cost, 1 + blocker_sum)

    active.remove(active_key)
    if cache is not None:
        cache[cache_key] = best_cost
    return best_cost


def _node_repair_cost(
    state: Sequence[int],
    node: int,
    colors: Sequence[int],
    adj_list: AdjList,
    unlock_depth_limit: int = 3,
    cache: Optional[Dict[Tuple[State, int, int], int]] = None,
) -> int:
    """Estimate the relaxed cost to make a conflict node repairable."""

    current_color = state[node]
    best_cost = LARGE_COST
    for color in colors:
        if color == current_color:
            continue
        blockers = _blockers_for_color(state, node, color, adj_list)
        if not blockers:
            best_cost = min(best_cost, 1)
            continue
        blocker_cost = sum(
            _dynamic_neighbor_cost(
                state,
                blocker,
                colors,
                adj_list,
                depth_limit=unlock_depth_limit,
                cache=cache,
            )
            for blocker in blockers
        )
        best_cost = min(best_cost, 1 + blocker_cost)
    return best_cost


def calculate_h_add_relaxed(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    unlock_depth_limit: int = 3,
) -> int:
    """Estimate remaining repair cost by summing relaxed node repair costs."""

    conflict_nodes = get_conflict_nodes(state, adj_list)
    if not conflict_nodes:
        return 0
    cache: Dict[Tuple[State, int, int], int] = {}
    return sum(
        _node_repair_cost(state, node, colors, adj_list, unlock_depth_limit, cache)
        for node in conflict_nodes
    )


def calculate_h_conflict_count(state: Sequence[int], adj_list: AdjList) -> int:
    """Return the raw number of conflict edges as a baseline heuristic."""

    return count_conflicts(state, adj_list)


def calculate_h_max_relaxed(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    unlock_depth_limit: int = 3,
) -> int:
    """Estimate remaining repair cost by the hardest single conflict node."""

    conflict_nodes = get_conflict_nodes(state, adj_list)
    if not conflict_nodes:
        return 0
    cache: Dict[Tuple[State, int, int], int] = {}
    return max(
        _node_repair_cost(state, node, colors, adj_list, unlock_depth_limit, cache)
        for node in conflict_nodes
    )


def get_dependency_candidate_nodes(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    dependency_depth: int = 3,
) -> Tuple[Set[int], CandidateReason]:
    """Expand candidate nodes from conflicts through recursively blocking neighbors."""

    conflict_nodes = get_conflict_nodes(state, adj_list)
    candidate_nodes = set(conflict_nodes)
    candidate_reason: CandidateReason = {node: "conflict" for node in conflict_nodes}
    frontier = set(conflict_nodes)

    for _ in range(dependency_depth):
        new_frontier: Set[int] = set()
        for node in frontier:
            if has_legal_recolor(state, node, colors, adj_list):
                continue
            current_color = state[node]
            for color in colors:
                if color == current_color:
                    continue
                for blocker in _blockers_for_color(state, node, color, adj_list):
                    if blocker not in candidate_nodes:
                        candidate_nodes.add(blocker)
                        candidate_reason[blocker] = "blocker"
                        new_frontier.add(blocker)
        frontier = new_frontier
        if not frontier:
            break

    if candidate_nodes == conflict_nodes:
        fallback_nodes = set(conflict_nodes)
        for node in conflict_nodes:
            for neighbor in adj_list[node]:
                if neighbor not in candidate_nodes:
                    candidate_nodes.add(neighbor)
                    candidate_reason[neighbor] = "fallback_neighbor"
                    fallback_nodes.add(neighbor)
        if fallback_nodes - conflict_nodes:
            candidate_reason.setdefault(-1, "fallback_used")

    return candidate_nodes, candidate_reason


def generate_relevant_actions(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    candidate_nodes: Iterable[int],
    allowed_worsening: int = 1,
    candidate_reason: Optional[CandidateReason] = None,
) -> List[Action]:
    """Generate actions that are locally relevant to conflict repair."""

    conflict_nodes = get_conflict_nodes(state, adj_list)
    current_conflicts = count_conflicts(state, adj_list)
    legal_color_counts = {
        node: len(legal_colors_for_node(state, node, colors, adj_list))
        for node in conflict_nodes
    }

    actions: List[Action] = []
    seen: Set[Action] = set()
    for node in candidate_nodes:
        old_color = state[node]
        for color in colors:
            if color == old_color:
                continue
            action = (node, old_color, color)
            child = apply_action(state, action)
            child_conflicts = count_conflicts(child, adj_list)

            increases_legal_options = any(
                len(legal_colors_for_node(child, conflict_node, colors, adj_list))
                > legal_color_counts[conflict_node]
                for conflict_node in conflict_nodes
            )

            relevant = (
                node in conflict_nodes
                or child_conflicts < current_conflicts
                or child_conflicts <= current_conflicts + allowed_worsening
                or increases_legal_options
                or (
                    candidate_reason is not None
                    and candidate_reason.get(node) in {"blocker", "fallback_neighbor"}
                )
            )
            if relevant and action not in seen:
                actions.append(action)
                seen.add(action)
    return actions


def _compute_planner_heuristic(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    heuristic: str,
    unlock_depth_limit: int = 3,
) -> int:
    """Dispatch planner heuristic calculation."""

    if heuristic == "h_add":
        return calculate_h_add_relaxed(state, colors, adj_list, unlock_depth_limit)
    if heuristic == "h_max":
        return calculate_h_max_relaxed(state, colors, adj_list, unlock_depth_limit)
    if heuristic == "conflict_count":
        return calculate_h_conflict_count(state, adj_list)
    raise ValueError(f"Unknown heuristic: {heuristic}")


def reconstruct_action_path(node: Optional[Node]) -> List[Action]:
    """Reconstruct the action path from the root to the given node."""

    if node is None:
        return []
    path: List[Action] = []
    current = node
    while current is not None and current.action is not None:
        path.append(current.action)
        current = current.parent
    path.reverse()
    return path


def search_iterations_from_gamma(
    gamma: Optional[float],
    fallback: int,
    min_iterations: int = 1,
    max_iterations: int = 5000,
) -> int:
    """Map gamma to a search-iteration budget, following fourinarow BFS."""

    if gamma is None:
        return int(fallback)
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    iterations = int(1.0 / float(gamma)) + 1
    return max(int(min_iterations), min(int(max_iterations), iterations))


def run_hsp2_weighted_planner(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    max_expansions: int = 5000,
    max_depth: int = 8,
    W: float = 5.0,
    dependency_depth: int = 3,
    early_stop: bool = True,
    improvement_threshold: int = 1,
    heuristic: str = "h_add",
    rng: Optional[random.Random] = None,
) -> Tuple[List[Action], State, bool, List[dict]]:
    """Run weighted best-first repair planning from the current state.

    The planner now prioritizes finding a fully solved path within its bounded
    search budget. If no solved node is found, it falls back to the best
    partially improved node discovered during search.
    """

    del rng  # Deterministic ordering is preferred in the planner.

    root_state = state_to_key(state)
    root_conflicts = count_conflicts(root_state, adj_list)
    h_initial = _compute_planner_heuristic(
        root_state, colors, adj_list, heuristic, dependency_depth
    )
    root = Node(
        state=root_state,
        g=0,
        h=h_initial,
        f=W * h_initial,
        parent=None,
        action=None,
        depth=0,
    )

    open_list: List[Tuple[float, int, Node]] = []
    push_counter = 0
    heapq.heappush(open_list, (root.f, push_counter, root))
    closed_set: Set[State] = set()
    best_node = root
    expansions = 0
    logs: List[dict] = [
        {
            "module": "hsp2",
            "event": "start",
            "root_conflicts": root_conflicts,
            "root_h": h_initial,
            "max_depth": max_depth,
            "dependency_depth": dependency_depth,
        }
    ]

    best_partial_node: Optional[Node] = None

    while open_list and expansions < max_expansions:
        _, _, current = heapq.heappop(open_list)
        state_key = state_to_key(current.state)
        if state_key in closed_set:
            continue
        closed_set.add(state_key)
        expansions += 1

        current_conflicts = count_conflicts(current.state, adj_list)
        logs.append(
            {
                "module": "hsp2",
                "event": "expand",
                "expansion": expansions,
                "depth": current.depth,
                "g": current.g,
                "h": current.h,
                "f": current.f,
                "conflicts": current_conflicts,
            }
        )

        if is_solved(current.state, adj_list):
            logs.append(
                {
                    "module": "hsp2",
                    "event": "solved",
                    "expansions": expansions,
                }
            )
            return reconstruct_action_path(current), current.state, True, logs

        current_conflicts = count_conflicts(current.state, adj_list)
        best_partial_conflicts = (
            count_conflicts(best_partial_node.state, adj_list)
            if best_partial_node is not None
            else None
        )
        if best_partial_node is None or (
            current_conflicts,
            current.h,
            current.g,
        ) < (
            best_partial_conflicts,
            best_partial_node.h,
            best_partial_node.g,
        ):
            best_partial_node = current

        if current.depth >= max_depth:
            continue

        candidate_nodes, candidate_reason = get_dependency_candidate_nodes(
            current.state,
            colors,
            adj_list,
            dependency_depth=dependency_depth,
        )
        actions = generate_relevant_actions(
            current.state,
            colors,
            adj_list,
            candidate_nodes,
            candidate_reason=candidate_reason,
        )
        logs.append(
            {
                "module": "hsp2",
                "event": "generate_actions",
                "depth": current.depth,
                "candidate_nodes": len(candidate_nodes),
                "actions": len(actions),
            }
        )

        for action in actions:
            child_state = apply_action(current.state, action)
            child_key = state_to_key(child_state)
            if child_key in closed_set:
                continue

            g_child = current.g + 1
            h_child = _compute_planner_heuristic(
                child_state,
                colors,
                adj_list,
                heuristic,
                dependency_depth,
            )
            f_child = g_child + W * h_child
            child = Node(
                state=child_state,
                g=g_child,
                h=h_child,
                f=f_child,
                parent=current,
                action=action,
                depth=current.depth + 1,
            )

            child_conflicts = count_conflicts(child_state, adj_list)
            best_conflicts = count_conflicts(best_node.state, adj_list)
            if (child.h, child_conflicts, child.g) < (best_node.h, best_conflicts, best_node.g):
                best_node = child

            if early_stop and best_partial_node is not None:
                best_partial_conflicts = count_conflicts(best_partial_node.state, adj_list)
                if (
                    best_partial_node.depth >= max_depth
                    and (
                        best_partial_node.h <= h_initial - improvement_threshold
                        or best_partial_conflicts < root_conflicts
                    )
                ):
                    logs.append(
                        {
                            "module": "hsp2",
                            "event": "early_stop_partial",
                            "depth": best_partial_node.depth,
                            "best_partial_h": best_partial_node.h,
                            "best_partial_conflicts": best_partial_conflicts,
                            "expansions": expansions,
                        }
                    )
                    return (
                        reconstruct_action_path(best_partial_node),
                        best_partial_node.state,
                        False,
                        logs,
                    )

            push_counter += 1
            heapq.heappush(open_list, (child.f, push_counter, child))

    if best_partial_node is not None and best_partial_node is not best_node:
        best_node = best_partial_node

    logs.append(
        {
            "module": "hsp2",
            "event": "budget_exhausted",
            "expansions": expansions,
            "max_expansions": effective_max_expansions,
            "best_h": best_node.h,
            "best_conflicts": count_conflicts(best_node.state, adj_list),
        }
    )
    return reconstruct_action_path(best_node), best_node.state, False, logs


def run_hsp2_level_order_planner(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    max_expansions: int = 5000,
    max_depth: int = 8,
    dependency_depth: int = 3,
    early_stop: bool = False,
    improvement_threshold: int = 1,
    heuristic: str = "h_add",
    sort_children_by_heuristic: bool = True,
    pruning_thresh: Optional[float] = None,
    gamma: Optional[float] = None,
    lapse_rate: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Action], State, bool, List[dict]]:
    """Run HSP2 candidate repair with strict level-order expansion.

    This variant keeps the HSP2 candidate/action generator and relaxed repair
    heuristic, but replaces the weighted priority queue with a level frontier.
    The resulting expansion order is breadth-first by level: all retained
    depth-k nodes are expanded before any depth-(k+1) node. pruning_thresh
    keeps only states whose relaxed heuristic is within threshold of the best
    next-depth state. gamma controls the total expansion budget as int(1 /
    gamma) + 1, matching the fourinarow BFS convention. lapse_rate is applied
    before search as a random-action lapse.
    """

    chooser = rng if rng is not None else random
    effective_max_expansions = search_iterations_from_gamma(gamma, max_expansions)

    root_state = state_to_key(state)
    root_conflicts = count_conflicts(root_state, adj_list)
    if root_conflicts > 0 and lapse_rate > 0.0 and chooser.random() < lapse_rate:
        candidate_nodes, candidate_reason = get_dependency_candidate_nodes(
            root_state,
            colors,
            adj_list,
            dependency_depth=dependency_depth,
        )
        lapse_actions = generate_relevant_actions(
            root_state,
            colors,
            adj_list,
            candidate_nodes,
            candidate_reason=candidate_reason,
        )
        if lapse_actions:
            action = chooser.choice(lapse_actions)
            next_state = apply_action(root_state, action)
            return (
                [action],
                next_state,
                is_solved(next_state, adj_list),
                [
                    {
                        "module": "hsp2_level_order",
                        "event": "lapse_random_action",
                        "action": action,
                        "before_conflicts": root_conflicts,
                        "after_conflicts": count_conflicts(next_state, adj_list),
                        "lapse_rate": lapse_rate,
                    }
                ],
            )

    h_initial = _compute_planner_heuristic(
        root_state, colors, adj_list, heuristic, dependency_depth
    )
    root = Node(
        state=root_state,
        g=0,
        h=h_initial,
        f=float(h_initial),
        parent=None,
        action=None,
        depth=0,
    )

    frontier: List[Node] = [root]
    closed_set: Set[State] = set()
    best_node = root
    expansions = 0
    logs: List[dict] = [
        {
            "module": "hsp2_level_order",
            "event": "start",
            "root_conflicts": root_conflicts,
            "root_h": h_initial,
            "max_depth": max_depth,
            "max_expansions": effective_max_expansions,
            "dependency_depth": dependency_depth,
            "sort_children_by_heuristic": sort_children_by_heuristic,
            "pruning_thresh": pruning_thresh,
            "gamma": gamma,
            "lapse_rate": lapse_rate,
        }
    ]

    best_partial_node: Optional[Node] = None

    def node_rank(node: Node) -> tuple[int, int, int, Action]:
        return (
            node.h,
            count_conflicts(node.state, adj_list),
            node.g,
            node.action or (-1, -1, -1),
        )

    while frontier and expansions < effective_max_expansions:
        next_candidates: Dict[State, Node] = {}
        current_depth = frontier[0].depth

        for current in frontier:
            if expansions >= effective_max_expansions:
                break

            state_key = state_to_key(current.state)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            expansions += 1

            current_conflicts = count_conflicts(current.state, adj_list)
            logs.append(
                {
                    "module": "hsp2_level_order",
                    "event": "expand",
                    "expansion": expansions,
                    "depth": current.depth,
                    "g": current.g,
                    "h": current.h,
                    "f": current.f,
                    "conflicts": current_conflicts,
                }
            )

            if is_solved(current.state, adj_list):
                logs.append(
                    {
                        "module": "hsp2_level_order",
                        "event": "solved",
                        "expansions": expansions,
                    }
                )
                return reconstruct_action_path(current), current.state, True, logs

            best_partial_conflicts = (
                count_conflicts(best_partial_node.state, adj_list)
                if best_partial_node is not None
                else None
            )
            if best_partial_node is None or (
                current_conflicts,
                current.h,
                current.g,
            ) < (
                best_partial_conflicts,
                best_partial_node.h,
                best_partial_node.g,
            ):
                best_partial_node = current

            best_conflicts = count_conflicts(best_node.state, adj_list)
            if (current.h, current_conflicts, current.g) < (best_node.h, best_conflicts, best_node.g):
                best_node = current

            if current.depth >= max_depth:
                if early_stop and (
                    current.h <= h_initial - improvement_threshold
                    or current_conflicts < root_conflicts
                ):
                    logs.append(
                        {
                            "module": "hsp2_level_order",
                            "event": "early_stop_partial",
                            "depth": current.depth,
                            "best_partial_h": current.h,
                            "best_partial_conflicts": current_conflicts,
                            "expansions": expansions,
                        }
                    )
                    return reconstruct_action_path(current), current.state, False, logs
                continue

            candidate_nodes, candidate_reason = get_dependency_candidate_nodes(
                current.state,
                colors,
                adj_list,
                dependency_depth=dependency_depth,
            )
            actions = generate_relevant_actions(
                current.state,
                colors,
                adj_list,
                candidate_nodes,
                candidate_reason=candidate_reason,
            )
            logs.append(
                {
                    "module": "hsp2_level_order",
                    "event": "generate_actions",
                    "depth": current.depth,
                    "candidate_nodes": len(candidate_nodes),
                    "actions": len(actions),
                }
            )

            for action in actions:
                child_state = apply_action(current.state, action)
                child_key = state_to_key(child_state)
                if child_key in closed_set:
                    continue

                g_child = current.g + 1
                h_child = _compute_planner_heuristic(
                    child_state,
                    colors,
                    adj_list,
                    heuristic,
                    dependency_depth,
                )
                child = Node(
                    state=child_state,
                    g=g_child,
                    h=h_child,
                    f=float(g_child + h_child),
                    parent=current,
                    action=action,
                    depth=current.depth + 1,
                )

                existing = next_candidates.get(child_key)
                if existing is None or node_rank(child) < node_rank(existing):
                    next_candidates[child_key] = child

        children = list(next_candidates.values())
        if sort_children_by_heuristic:
            children.sort(key=node_rank)

        before_score_prune = len(children)
        if pruning_thresh is not None and pruning_thresh >= 0 and children:
            best_h = min(child.h for child in children)
            children = [
                child
                for child in children
                if (float(child.h) - float(best_h)) <= float(pruning_thresh)
            ]
            if before_score_prune > len(children):
                logs.append(
                    {
                        "module": "hsp2_level_order",
                        "event": "score_prune",
                        "from_depth": current_depth,
                        "to_depth": current_depth + 1,
                        "kept": len(children),
                        "pruned": before_score_prune - len(children),
                        "best_h": best_h,
                        "pruning_thresh": pruning_thresh,
                    }
                )

        frontier = children

    if best_partial_node is not None and best_partial_node is not best_node:
        best_node = best_partial_node

    logs.append(
        {
            "module": "hsp2_level_order",
            "event": "budget_exhausted",
            "expansions": expansions,
            "best_h": best_node.h,
            "best_conflicts": count_conflicts(best_node.state, adj_list),
        }
    )
    return reconstruct_action_path(best_node), best_node.state, False, logs


def run_hsp2_planner(
    state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    max_expansions: int = 5000,
    max_depth: int = 8,
    W: float = 5.0,
    dependency_depth: int = 3,
    early_stop: bool = False,
    improvement_threshold: int = 1,
    heuristic: str = "h_add",
    rng: Optional[random.Random] = None,
    sort_children_by_heuristic: bool = True,
    pruning_thresh: Optional[float] = None,
    gamma: Optional[float] = None,
    lapse_rate: float = 0.0,
) -> Tuple[List[Action], State, bool, List[dict]]:
    """Run the default HSP2 repair planner.

    The default planner is the level-order HSP2 variant without beam pruning.
    W is accepted for backward compatibility with the old weighted best-first
    signature, but it is not used by level-order expansion.
    """

    del W
    return run_hsp2_level_order_planner(
        state,
        colors,
        adj_list,
        max_expansions=max_expansions,
        max_depth=max_depth,
        dependency_depth=dependency_depth,
        early_stop=early_stop,
        improvement_threshold=improvement_threshold,
        heuristic=heuristic,
        sort_children_by_heuristic=sort_children_by_heuristic,
        pruning_thresh=pruning_thresh,
        gamma=gamma,
        lapse_rate=lapse_rate,
        rng=rng,
    )


def solve_map_coloring_repair(
    initial_state: Sequence[int],
    colors: Sequence[int],
    adj_list: AdjList,
    max_outer_loops: int = 50,
    max_min_conflicts_steps: int = 200,
    max_eflop_retries: int = 5,
    rng_seed: int = 0,
) -> Tuple[State, List[Action], bool, List[dict]]:
    """Run the full conflict repair system until solved or budget exhausted."""

    current_state = state_to_key(initial_state)
    full_action_path: List[Action] = []
    logs: List[dict] = []
    rng = random.Random(rng_seed)

    best_global_state = current_state
    best_global_conflicts = count_conflicts(current_state, adj_list)
    best_global_action_path: List[Action] = []

    for outer in range(max_outer_loops):
        current_conflicts = count_conflicts(current_state, adj_list)
        logs.append(
            {
                "module": "outer_loop",
                "outer": outer,
                "state_conflicts": current_conflicts,
            }
        )
        if is_solved(current_state, adj_list):
            return current_state, full_action_path, True, logs

        mc_state, mc_actions, stuck, solved, mc_logs = run_min_conflicts(
            current_state,
            colors,
            adj_list,
            max_steps=max_min_conflicts_steps,
            rng=rng,
        )
        current_state = mc_state
        full_action_path.extend(mc_actions)
        logs.extend(mc_logs)

        current_conflicts = count_conflicts(current_state, adj_list)
        if current_conflicts < best_global_conflicts:
            best_global_state = current_state
            best_global_conflicts = current_conflicts
            best_global_action_path = list(full_action_path)

        if solved or is_solved(current_state, adj_list):
            return current_state, full_action_path, True, logs

        action_plan, planned_state, hsp_solved, hsp_logs = run_hsp2_planner(
            current_state,
            colors,
            adj_list,
            rng=rng,
        )
        logs.extend(hsp_logs)
        if action_plan:
            current_state = planned_state
            full_action_path.extend(action_plan)
            planned_conflicts = count_conflicts(current_state, adj_list)
            if planned_conflicts < best_global_conflicts:
                best_global_state = current_state
                best_global_conflicts = planned_conflicts
                best_global_action_path = list(full_action_path)
        else:
            logs.append(
                {
                    "module": "hsp2",
                    "event": "failed_to_improve",
                    "outer": outer,
                }
            )
            break

        if hsp_solved or is_solved(current_state, adj_list):
            return current_state, full_action_path, True, logs

        success_eflop = False
        for retry in range(max_eflop_retries):
            temp_state, ef_actions, ef_logs = run_eflop(
                current_state,
                colors,
                adj_list,
                rng=rng,
            )
            temp_state2, mc_actions2, stuck2, solved2, mc_logs2 = run_min_conflicts(
                temp_state,
                colors,
                adj_list,
                max_steps=max_min_conflicts_steps,
                rng=rng,
            )
            temp_conflicts = count_conflicts(temp_state2, adj_list)
            current_conflicts = count_conflicts(current_state, adj_list)

            logs.append(
                {
                    "module": "eflop_retry",
                    "outer": outer,
                    "retry": retry,
                    "accepted": temp_conflicts < current_conflicts or solved2,
                    "before_conflicts": current_conflicts,
                    "after_conflicts": temp_conflicts,
                    "mc_stuck": stuck2,
                }
            )

            if temp_conflicts < current_conflicts or solved2:
                current_state = temp_state2
                full_action_path.extend(ef_actions)
                full_action_path.extend(mc_actions2)
                logs.extend(ef_logs)
                logs.extend(mc_logs2)
                success_eflop = True

                if temp_conflicts < best_global_conflicts:
                    best_global_state = current_state
                    best_global_conflicts = temp_conflicts
                    best_global_action_path = list(full_action_path)

                if solved2 or is_solved(current_state, adj_list):
                    return current_state, full_action_path, True, logs
                break

        if success_eflop:
            continue

        if not stuck and not mc_actions and not action_plan:
            break

    return best_global_state, best_global_action_path, False, logs


def _demo_graph() -> Tuple[State, List[int], AdjList]:
    """Build a small demo graph with several initial conflicts."""

    colors = [0, 1, 2, 3]
    adj_list = {
        0: [1, 5],
        1: [0, 2, 4],
        2: [1, 3],
        3: [2, 4],
        4: [1, 3, 5],
        5: [0, 4],
    }
    initial_state = (0, 0, 1, 1, 2, 2)
    return initial_state, colors, adj_list


def demo() -> None:
    """Run a small end-to-end demo of the conflict repair system."""

    initial_state, colors, adj_list = _demo_graph()
    final_state, action_path, success, _ = solve_map_coloring_repair(
        initial_state,
        colors,
        adj_list,
        rng_seed=0,
    )

    print("Initial state:", initial_state)
    current_state = initial_state
    print("Initial conflicts:", count_conflicts(current_state, adj_list))
    for step, action in enumerate(action_path, start=1):
        current_state = apply_action(current_state, action)
        print(
            f"Step {step}: action={action}, "
            f"conflicts={count_conflicts(current_state, adj_list)}"
        )
    print("Final state:", final_state)
    print("Success:", success)


if __name__ == "__main__":
    demo()
