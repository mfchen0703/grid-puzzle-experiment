"""
Generate many random experiment-2-style maps, solve their optimal recoloring
length, and sort candidates by difficulty.

The optimal length is the minimum number of regions whose colors must change
from the initial state to reach any legal 4-coloring.

Outputs:
    experiment2/generated_map_search/generated_maps_sorted.csv
    experiment2/generated_map_search/generated_maps_sorted.json
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
from pathlib import Path
from time import perf_counter

from generate_rounds_json import (
    COLORS,
    NUM_REGIONS,
    build_conflict_start_state,
    generate_map_data,
    get_conflict_edges,
    get_conflict_regions,
    mulberry32,
    shuffle,
    solve_coloring,
)


class SeedTimeout(RuntimeError):
    pass


def _handle_timeout(signum, frame) -> None:
    raise SeedTimeout("seed attempt exceeded per-seed timeout")


class seed_time_limit:
    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.previous_handler = None

    def __enter__(self):
        if self.seconds is None or self.seconds <= 0:
            return
        self.previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))

    def __exit__(self, exc_type, exc, tb):
        if self.seconds is None or self.seconds <= 0:
            return False
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def build_adjacency_list(adjacency: dict[int, set[int]]) -> list[list[int]]:
    return [sorted(adjacency[idx]) for idx in range(len(adjacency))]


def adjacent_nonconflict_regions(adjacency: list[list[int]], conflict_regions: set[int]) -> set[int]:
    out: set[int] = set()
    for region in conflict_regions:
        for neighbor in adjacency[region]:
            if neighbor not in conflict_regions:
                out.add(neighbor)
    return out


def choose_variable(
    assignment: list[int],
    domains: list[list[int]],
    adjacency: list[list[int]],
) -> int | None:
    best_region: int | None = None
    best_key: tuple[int, int] | None = None
    for region, value in enumerate(assignment):
        if value != -1:
            continue
        legal_count = 0
        for color in domains[region]:
            if all(assignment[neighbor] != color for neighbor in adjacency[region]):
                legal_count += 1
        if legal_count == 0:
            return region
        key = (legal_count, -len(adjacency[region]))
        if best_key is None or key < best_key:
            best_key = key
            best_region = region
    return best_region


def solve_with_budget(
    adjacency: list[list[int]],
    initial_colors: tuple[int, ...],
    budget: int,
    *,
    collect_solutions: bool,
    solution_limit: int,
) -> dict:
    n_regions = len(initial_colors)
    assignment = [-1] * n_regions
    domains = [
        [initial_colors[region]]
        + [color for color in range(len(COLORS)) if color != initial_colors[region]]
        for region in range(n_regions)
    ]
    solutions: list[tuple[int, ...]] = []
    nodes = 0
    capped = False

    def dfs(changed_count: int) -> bool:
        nonlocal nodes, capped
        nodes += 1
        if changed_count > budget:
            return False

        region = choose_variable(assignment, domains, adjacency)
        if region is None:
            solution = tuple(assignment)
            if collect_solutions:
                if len(solutions) < solution_limit:
                    solutions.append(solution)
                else:
                    capped = True
                    return True
            return True

        found = False
        for color in domains[region]:
            if color != initial_colors[region] and changed_count >= budget:
                continue
            if any(assignment[neighbor] == color for neighbor in adjacency[region]):
                continue
            assignment[region] = color
            child_found = dfs(changed_count + int(color != initial_colors[region]))
            assignment[region] = -1
            if child_found:
                found = True
                if not collect_solutions or capped:
                    return True
        return found

    exists = dfs(0)
    return {
        "exists": exists,
        "solutions": solutions,
        "nodes": nodes,
        "solution_collection_capped": capped,
    }


def minimize_planning_requirement(
    adjacency: list[list[int]],
    initial_colors: tuple[int, ...],
    budget: int,
    planning_regions: set[int],
) -> dict:
    n_regions = len(initial_colors)
    assignment = [-1] * n_regions
    domains = [
        [initial_colors[region]]
        + [color for color in range(len(COLORS)) if color != initial_colors[region]]
        for region in range(n_regions)
    ]
    nodes = 0
    best_planning: int | None = None
    best_solution: tuple[int, ...] | None = None

    def dfs(changed_count: int, planning_count: int) -> None:
        nonlocal nodes, best_planning, best_solution
        nodes += 1
        if changed_count > budget:
            return
        if best_planning is not None and planning_count >= best_planning:
            return

        region = choose_variable(assignment, domains, adjacency)
        if region is None:
            best_planning = planning_count
            best_solution = tuple(assignment)
            return

        for color in domains[region]:
            changed = color != initial_colors[region]
            if changed and changed_count >= budget:
                continue
            if any(assignment[neighbor] == color for neighbor in adjacency[region]):
                continue
            assignment[region] = color
            dfs(
                changed_count + int(changed),
                planning_count + int(changed and region in planning_regions),
            )
            assignment[region] = -1
            if best_planning == 0:
                return

    dfs(0, 0)

    example_solution = []
    if best_solution is not None:
        example_solution = [
            {"region": region, "from": int(initial_colors[region]), "to": int(best_solution[region])}
            for region in range(n_regions)
            if initial_colors[region] != best_solution[region]
        ]

    return {
        "min_planning_requirement": best_planning,
        "example_optimal_solution": example_solution,
        "nodes": nodes,
    }


def find_optimal_solution_stats(
    adjacency: list[list[int]],
    initial_colors: list[int],
    *,
    max_depth: int,
    solution_limit: int,
) -> dict:
    initial = tuple(int(color) for color in initial_colors)
    total_nodes = 0

    for budget in range(max_depth + 1):
        existence = solve_with_budget(
            adjacency,
            initial,
            budget,
            collect_solutions=False,
            solution_limit=1,
        )
        total_nodes += int(existence["nodes"])
        if not existence["exists"]:
            continue

        conflict_regions = get_conflict_regions(get_conflict_edges(
            {idx: set(neighbors) for idx, neighbors in enumerate(adjacency)},
            initial,
        ))
        adjacent_nonconflict = adjacent_nonconflict_regions(adjacency, conflict_regions)
        planning = minimize_planning_requirement(adjacency, initial, budget, adjacent_nonconflict)
        total_nodes += int(planning["nodes"])

        return {
            "optimal_length": budget,
            "solved_within_bound": True,
            "n_collected_optimal_solutions": 1 if planning["example_optimal_solution"] else 0,
            "solution_collection_capped": False,
            "min_adjacent_nonconflict_changes_collected": planning["min_planning_requirement"],
            "example_optimal_solution": planning["example_optimal_solution"],
            "search_nodes": total_nodes,
        }

    return {
        "optimal_length": None,
        "solved_within_bound": False,
        "n_collected_optimal_solutions": 0,
        "solution_collection_capped": False,
        "min_adjacent_nonconflict_changes_collected": None,
        "example_optimal_solution": [],
        "search_nodes": total_nodes,
    }


def classify_difficulty(optimal_length: int | None, planning_requirement: int | None) -> str:
    if optimal_length is None or planning_requirement is None:
        return "above_bound"
    if 1 <= optimal_length <= 3 and planning_requirement == 0:
        return "easy"
    if 3 <= optimal_length <= 5 and planning_requirement == 1:
        return "medium"
    if 4 <= optimal_length <= 6 and planning_requirement >= 2:
        return "hard"
    return "other"


def difficulty_score(optimal_length: int | None, planning_requirement: int | None, max_depth: int) -> float:
    if optimal_length is None:
        return float(max_depth + 1)
    planning_bonus = 0.0 if planning_requirement is None else min(float(planning_requirement), 9.0) / 10.0
    return float(optimal_length) + planning_bonus


def changed_regions(initial_colors: list[int], solved_colors: list[int]) -> set[int]:
    return {
        idx
        for idx, (initial, solved) in enumerate(zip(initial_colors, solved_colors))
        if int(initial) != int(solved)
    }


def neighbor_color_options(region: int, adjacency: dict[int, set[int]], colors: list[int]) -> list[int]:
    return sorted({int(colors[neighbor]) for neighbor in adjacency[region] if int(colors[neighbor]) != int(colors[region])})


def non_current_colors(color: int) -> list[int]:
    return [candidate for candidate in range(len(COLORS)) if candidate != int(color)]


def build_easy_start_state(adjacency: dict[int, set[int]], solved_colors: list[int], random) -> tuple[list[int], list[list[int]]]:
    region_ids = list(range(len(solved_colors)))
    for _ in range(2000):
        candidate = list(solved_colors)
        change_count = 1 + int(random() * 3)
        for region in shuffle(region_ids, random)[:change_count]:
            options = neighbor_color_options(region, adjacency, candidate)
            if not options:
                options = non_current_colors(candidate[region])
            candidate[region] = shuffle(options, random)[0]

        conflict_edges = get_conflict_edges(adjacency, candidate)
        if not conflict_edges:
            continue
        conflict_regions = get_conflict_regions(conflict_edges)
        if changed_regions(candidate, solved_colors).issubset(conflict_regions):
            return candidate, conflict_edges
    raise RuntimeError("Failed to build an easy conflict-only start state.")


def legal_nonconflicting_alternatives(region: int, adjacency: dict[int, set[int]], colors: list[int]) -> list[int]:
    options = []
    for color in non_current_colors(colors[region]):
        if all(colors[neighbor] != color for neighbor in adjacency[region]):
            options.append(color)
    return options


def build_broad_start_state(
    adjacency: dict[int, set[int]],
    solved_colors: list[int],
    random,
    *,
    min_changes: int,
    max_changes: int,
    force_adjacent_clean_pair: bool,
) -> tuple[list[int], list[list[int]]]:
    region_ids = list(range(len(solved_colors)))
    adjacency_edges = [(a, b) for a in region_ids for b in adjacency[a] if a < b]

    for _ in range(12000):
        candidate = list(solved_colors)
        selected: list[int] = []

        if force_adjacent_clean_pair and adjacency_edges:
            core_a, core_b = shuffle(adjacency_edges, random)[0]
            options_a = legal_nonconflicting_alternatives(core_a, adjacency, candidate)
            if not options_a:
                continue
            candidate[core_a] = shuffle(options_a, random)[0]
            options_b = legal_nonconflicting_alternatives(core_b, adjacency, candidate)
            if not options_b:
                continue
            candidate[core_b] = shuffle(options_b, random)[0]
            selected.extend([core_a, core_b])

        change_count = min_changes + int(random() * (max_changes - min_changes + 1))
        extra_count = max(0, change_count - len(selected))
        selected_set = set(selected)
        extras = [region for region in shuffle(region_ids, random) if region not in selected_set][:extra_count]
        selected.extend(extras)

        for region in extras:
            if random() < 0.75:
                options = neighbor_color_options(region, adjacency, candidate)
            else:
                options = legal_nonconflicting_alternatives(region, adjacency, candidate)
            if not options:
                options = non_current_colors(candidate[region])
            candidate[region] = shuffle(options, random)[0]

        conflict_edges = get_conflict_edges(adjacency, candidate)
        if len(conflict_edges) < 1:
            continue
        if len(changed_regions(candidate, solved_colors)) < min_changes:
            continue
        return candidate, conflict_edges

    raise RuntimeError("Failed to build a broad random start state.")


def build_uniform_perturb_start_state(
    adjacency: dict[int, set[int]],
    solved_colors: list[int],
    random,
    *,
    min_changes: int,
    max_changes: int,
) -> tuple[list[int], list[list[int]]]:
    region_ids = list(range(len(solved_colors)))
    for _ in range(4000):
        candidate = list(solved_colors)
        change_count = min_changes + int(random() * (max_changes - min_changes + 1))
        for region in shuffle(region_ids, random)[:change_count]:
            candidate[region] = shuffle(non_current_colors(candidate[region]), random)[0]
        conflict_edges = get_conflict_edges(adjacency, candidate)
        if conflict_edges:
            return candidate, conflict_edges
    raise RuntimeError("Failed to build a uniform perturbation start state.")


def build_candidate(seed: int, condition_type: str, generation_mode: str = "legacy") -> dict:
    random = mulberry32(seed)
    map_data, adjacency = generate_map_data(NUM_REGIONS, random)
    solved_colors = solve_coloring(adjacency, random)
    actual_generation_mode = generation_mode
    if generation_mode == "mixed_perturb":
        variant = seed % 5
        if variant == 0:
            actual_generation_mode = "easy_perturb"
        elif variant == 1:
            actual_generation_mode = "uniform_3_4"
        elif variant == 2:
            actual_generation_mode = "uniform_5_6"
        elif variant == 3:
            actual_generation_mode = "uniform_7_8"
        else:
            actual_generation_mode = "hard_perturb"

    if actual_generation_mode == "easy_perturb":
        initial_colors, conflict_edges = build_easy_start_state(adjacency, solved_colors, random)
    elif actual_generation_mode == "uniform_3_4":
        initial_colors, conflict_edges = build_uniform_perturb_start_state(
            adjacency,
            solved_colors,
            random,
            min_changes=3,
            max_changes=4,
        )
    elif actual_generation_mode == "uniform_5_6":
        initial_colors, conflict_edges = build_uniform_perturb_start_state(
            adjacency,
            solved_colors,
            random,
            min_changes=5,
            max_changes=6,
        )
    elif actual_generation_mode == "uniform_7_8":
        initial_colors, conflict_edges = build_uniform_perturb_start_state(
            adjacency,
            solved_colors,
            random,
            min_changes=7,
            max_changes=8,
        )
    elif actual_generation_mode == "broad_perturb":
        initial_colors, conflict_edges = build_broad_start_state(
            adjacency,
            solved_colors,
            random,
            min_changes=1,
            max_changes=6,
            force_adjacent_clean_pair=False,
        )
    elif actual_generation_mode == "hard_perturb":
        initial_colors, conflict_edges = build_broad_start_state(
            adjacency,
            solved_colors,
            random,
            min_changes=4,
            max_changes=6,
            force_adjacent_clean_pair=True,
        )
    else:
        initial_colors, conflict_edges = build_conflict_start_state(
            adjacency,
            solved_colors,
            random,
            condition_type=condition_type,
        )
    return {
        "seed": seed,
        "conditionType": condition_type,
        "generationMode": actual_generation_mode,
        "mapData": map_data,
        "initialColors": initial_colors,
        "solvedColors": solved_colors,
        "conflictEdges": conflict_edges,
        "_adjacency": adjacency,
    }


def search_candidates(
    *,
    n_candidates: int,
    start_seed: int,
    max_depth: int,
    solution_limit: int,
    sort_by: str,
    max_seed_attempts: int | None,
    per_seed_timeout: float | None,
    generation_mode: str,
    target_planning_requirement: int | None,
) -> list[dict]:
    rows: list[dict] = []
    attempts = 0
    skipped = 0
    attempt_limit = max_seed_attempts if max_seed_attempts is not None else max(n_candidates * 5, n_candidates + 100)

    while len(rows) < n_candidates and attempts < attempt_limit:
        seed = start_seed + attempts
        attempts += 1
        condition_type = (
            "requires_nonconflict_region"
            if attempts % 2 == 1
            else "requires_adjacent_nonconflict_pair"
        )
        started = perf_counter()
        try:
            with seed_time_limit(per_seed_timeout):
                candidate = build_candidate(seed, condition_type, generation_mode=generation_mode)
                adjacency = build_adjacency_list(candidate.pop("_adjacency"))
                stats = find_optimal_solution_stats(
                    adjacency,
                    candidate["initialColors"],
                    max_depth=max_depth,
                    solution_limit=solution_limit,
                )
        except (RuntimeError, SeedTimeout) as exc:
            skipped += 1
            print(f"skip seed={seed} condition={condition_type}: {exc}", flush=True)
            continue

        conflict_regions = get_conflict_regions(candidate["conflictEdges"])
        adjacent_nonconflict = adjacent_nonconflict_regions(adjacency, conflict_regions)
        planning_requirement = stats["min_adjacent_nonconflict_changes_collected"]
        difficulty = classify_difficulty(stats["optimal_length"], planning_requirement)
        if target_planning_requirement is not None and planning_requirement != target_planning_requirement:
            skipped += 1
            print(
                f"reject seed={seed} planning={planning_requirement} "
                f"target_planning={target_planning_requirement} optimal={stats['optimal_length']}",
                flush=True,
            )
            continue

        metadata = {
            "candidateIndex": len(rows) + 1,
            "seedAttemptIndex": attempts,
            "seed": seed,
            "generationMode": generation_mode,
            "difficultyLabel": difficulty,
            "difficultyScore": difficulty_score(stats["optimal_length"], planning_requirement, max_depth),
            "optimalLength": stats["optimal_length"],
            "solvedWithinBound": stats["solved_within_bound"],
            "planningRequirementCollected": planning_requirement,
            "nCollectedOptimalSolutions": stats["n_collected_optimal_solutions"],
            "solutionCollectionCapped": stats["solution_collection_capped"],
            "searchNodes": stats["search_nodes"],
            "elapsedSeconds": round(perf_counter() - started, 4),
            "initialConflictRegions": sorted(conflict_regions),
            "adjacentNonconflictRegions": sorted(adjacent_nonconflict),
            "exampleOptimalSolution": stats["example_optimal_solution"],
        }
        candidate["metadata"] = metadata
        rows.append(candidate)
        print(
            f"{len(rows):>4}/{n_candidates} seed={seed} "
            f"optimal={stats['optimal_length']} planning={planning_requirement} "
            f"label={difficulty} time={metadata['elapsedSeconds']}s",
            flush=True,
        )

    if len(rows) < n_candidates:
        raise RuntimeError(
            f"Only generated {len(rows)} successful candidates after {attempts} seed attempts "
            f"({skipped} skipped). Increase --max-seed-attempts or lower --n-candidates."
        )

    if skipped:
        print(f"Skipped {skipped} failed seeds while collecting {len(rows)} candidates.", flush=True)

    label_order = {"easy": 0, "medium": 1, "hard": 2, "other": 3, "above_bound": 4}

    def length_key(row: dict) -> tuple:
        metadata = row["metadata"]
        return (
            metadata["optimalLength"] is None,
            metadata["optimalLength"] if metadata["optimalLength"] is not None else 999,
            metadata["planningRequirementCollected"]
            if metadata["planningRequirementCollected"] is not None
            else 999,
            metadata["seed"],
        )

    def difficulty_key(row: dict) -> tuple:
        metadata = row["metadata"]
        return (
            metadata["difficultyScore"],
            metadata["optimalLength"] is None,
            metadata["planningRequirementCollected"]
            if metadata["planningRequirementCollected"] is not None
            else 999,
            metadata["seed"],
        )

    def label_key(row: dict) -> tuple:
        metadata = row["metadata"]
        return (
            label_order.get(metadata["difficultyLabel"], 99),
            metadata["optimalLength"] is None,
            metadata["optimalLength"] if metadata["optimalLength"] is not None else 999,
            metadata["planningRequirementCollected"]
            if metadata["planningRequirementCollected"] is not None
            else 999,
            metadata["seed"],
        )

    if sort_by == "label":
        sort_key = label_key
    elif sort_by == "difficulty":
        sort_key = difficulty_key
    else:
        sort_key = length_key
    return sorted(rows, key=sort_key)


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "generated_maps_sorted.json"
    csv_path = output_dir / "generated_maps_sorted.csv"

    payload = {
        "rows": 12,
        "cols": 20,
        "numRegions": NUM_REGIONS,
        "colors": COLORS,
        "rounds": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "candidateIndex",
        "seedAttemptIndex",
        "seed",
        "targetLabel",
        "difficultyLabel",
        "difficultyScore",
        "conditionType",
        "generationMode",
        "optimalLength",
        "solvedWithinBound",
        "planningRequirementCollected",
        "nCollectedOptimalSolutions",
        "solutionCollectionCapped",
        "initialConflictEdges",
        "initialConflictRegions",
        "adjacentNonconflictRegions",
        "adjacencyPairs",
        "searchNodes",
        "elapsedSeconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metadata = row["metadata"]
            writer.writerow(
                {
                    "candidateIndex": metadata["candidateIndex"],
                    "seedAttemptIndex": metadata["seedAttemptIndex"],
                    "seed": metadata["seed"],
                    "targetLabel": metadata.get("targetLabel", ""),
                    "difficultyLabel": metadata["difficultyLabel"],
                    "difficultyScore": metadata["difficultyScore"],
                    "conditionType": row["conditionType"],
                    "generationMode": row.get("generationMode", ""),
                    "optimalLength": metadata["optimalLength"],
                    "solvedWithinBound": metadata["solvedWithinBound"],
                    "planningRequirementCollected": metadata["planningRequirementCollected"],
                    "nCollectedOptimalSolutions": metadata["nCollectedOptimalSolutions"],
                    "solutionCollectionCapped": metadata["solutionCollectionCapped"],
                    "initialConflictEdges": len(row["conflictEdges"]),
                    "initialConflictRegions": len(metadata["initialConflictRegions"]),
                    "adjacentNonconflictRegions": len(metadata["adjacentNonconflictRegions"]),
                    "adjacencyPairs": len(row["mapData"]["adjacencyPairs"]),
                    "searchNodes": metadata["searchNodes"],
                    "elapsedSeconds": metadata["elapsedSeconds"],
                }
            )

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


def parse_target_label_counts(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    counts: dict[str, int] = {}
    for chunk in raw.split(","):
        label, value = chunk.split("=", 1)
        label = label.strip()
        if label not in {"easy", "medium", "hard", "other", "above_bound"}:
            raise ValueError(f"Unknown difficulty label in --target-label-counts: {label}")
        counts[label] = int(value)
    return counts


def mode_for_target(label: str, attempt_number: int) -> str:
    if label == "easy":
        return "easy_perturb" if attempt_number % 3 else "broad_perturb"
    if label == "hard":
        return "hard_perturb" if attempt_number % 2 else "legacy"
    if label == "medium":
        return "legacy" if attempt_number % 2 else "broad_perturb"
    return "broad_perturb"


def next_target_label(target_counts: dict[str, int], accepted_counts: dict[str, int]) -> str:
    labels = [label for label, target in target_counts.items() if accepted_counts.get(label, 0) < target]
    return min(labels, key=lambda label: accepted_counts.get(label, 0) / max(1, target_counts[label]))


def search_targeted_candidates(
    *,
    target_counts: dict[str, int],
    start_seed: int,
    max_depth: int,
    solution_limit: int,
    sort_by: str,
    max_seed_attempts: int | None,
    per_seed_timeout: float | None,
) -> list[dict]:
    rows: list[dict] = []
    accepted_counts = {label: 0 for label in target_counts}
    attempts = 0
    skipped = 0
    rejected = 0
    total_needed = sum(target_counts.values())
    attempt_limit = max_seed_attempts if max_seed_attempts is not None else max(total_needed * 30, total_needed + 300)

    while len(rows) < total_needed and attempts < attempt_limit:
        target_label = next_target_label(target_counts, accepted_counts)
        seed = start_seed + attempts
        attempts += 1
        condition_type = (
            "requires_nonconflict_region"
            if attempts % 2 == 1
            else "requires_adjacent_nonconflict_pair"
        )
        generation_mode = mode_for_target(target_label, attempts)
        started = perf_counter()
        try:
            with seed_time_limit(per_seed_timeout):
                candidate = build_candidate(seed, condition_type, generation_mode=generation_mode)
                adjacency = build_adjacency_list(candidate.pop("_adjacency"))
                stats = find_optimal_solution_stats(
                    adjacency,
                    candidate["initialColors"],
                    max_depth=max_depth,
                    solution_limit=solution_limit,
                )
        except (RuntimeError, SeedTimeout) as exc:
            skipped += 1
            print(f"skip seed={seed} target={target_label} mode={generation_mode}: {exc}", flush=True)
            continue

        conflict_regions = get_conflict_regions(candidate["conflictEdges"])
        adjacent_nonconflict = adjacent_nonconflict_regions(adjacency, conflict_regions)
        planning_requirement = stats["min_adjacent_nonconflict_changes_collected"]
        difficulty = classify_difficulty(stats["optimal_length"], planning_requirement)

        if accepted_counts.get(difficulty, 0) >= target_counts.get(difficulty, 0):
            rejected += 1
            print(
                f"reject seed={seed} target={target_label} got={difficulty} "
                f"optimal={stats['optimal_length']} planning={planning_requirement}",
                flush=True,
            )
            continue

        accepted_counts[difficulty] = accepted_counts.get(difficulty, 0) + 1
        metadata = {
            "candidateIndex": len(rows) + 1,
            "seedAttemptIndex": attempts,
            "seed": seed,
            "targetLabel": target_label,
            "generationMode": generation_mode,
            "difficultyLabel": difficulty,
            "difficultyScore": difficulty_score(stats["optimal_length"], planning_requirement, max_depth),
            "optimalLength": stats["optimal_length"],
            "solvedWithinBound": stats["solved_within_bound"],
            "planningRequirementCollected": planning_requirement,
            "nCollectedOptimalSolutions": stats["n_collected_optimal_solutions"],
            "solutionCollectionCapped": stats["solution_collection_capped"],
            "searchNodes": stats["search_nodes"],
            "elapsedSeconds": round(perf_counter() - started, 4),
            "initialConflictRegions": sorted(conflict_regions),
            "adjacentNonconflictRegions": sorted(adjacent_nonconflict),
            "exampleOptimalSolution": stats["example_optimal_solution"],
        }
        candidate["metadata"] = metadata
        rows.append(candidate)
        print(
            f"{len(rows):>4}/{total_needed} seed={seed} target={target_label} "
            f"mode={generation_mode} label={difficulty} optimal={stats['optimal_length']} "
            f"planning={planning_requirement} counts={accepted_counts}",
            flush=True,
        )

    if len(rows) < total_needed:
        raise RuntimeError(
            f"Only generated {len(rows)}/{total_needed} targeted candidates after {attempts} attempts "
            f"({skipped} skipped, {rejected} rejected). Increase --max-seed-attempts or relax targets."
        )

    if skipped or rejected:
        print(f"Skipped {skipped} failed seeds and rejected {rejected} off-target candidates.", flush=True)

    label_order = {"easy": 0, "medium": 1, "hard": 2, "other": 3, "above_bound": 4}

    def label_key(row: dict) -> tuple:
        metadata = row["metadata"]
        return (
            label_order.get(metadata["difficultyLabel"], 99),
            metadata["optimalLength"] is None,
            metadata["optimalLength"] if metadata["optimalLength"] is not None else 999,
            metadata["planningRequirementCollected"]
            if metadata["planningRequirementCollected"] is not None
            else 999,
            metadata["seed"],
        )

    def length_key(row: dict) -> tuple:
        metadata = row["metadata"]
        return (
            metadata["optimalLength"] is None,
            metadata["optimalLength"] if metadata["optimalLength"] is not None else 999,
            metadata["planningRequirementCollected"]
            if metadata["planningRequirementCollected"] is not None
            else 999,
            metadata["seed"],
        )

    if sort_by == "label":
        sort_key = label_key
    elif sort_by == "difficulty":
        sort_key = length_key
    else:
        sort_key = length_key
    return sorted(rows, key=sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random maps, solve optimal recoloring length, and sort by optimal length."
    )
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--solution-limit", type=int, default=200)
    parser.add_argument(
        "--sort-by",
        choices=["length", "label", "difficulty"],
        default="length",
        help="Sort output rows by optimal length, difficulty label, or continuous difficulty score.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["legacy", "easy_perturb", "broad_perturb", "hard_perturb", "mixed_perturb"],
        default="legacy",
        help="Initial-state generator to use when --target-label-counts is not set.",
    )
    parser.add_argument(
        "--target-label-counts",
        default=None,
        help="Optional quota string such as easy=20,medium=40,hard=20. Candidates are accepted only if solver labels match an unfilled quota.",
    )
    parser.add_argument(
        "--target-planning-requirement",
        type=int,
        default=None,
        help="Optional exact planningRequirementCollected value to retain, e.g. 1 keeps only maps whose optimal solution must change one adjacent non-conflict region.",
    )
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=None,
        help="Maximum seed attempts before giving up. Defaults to max(n_candidates*5, n_candidates+100).",
    )
    parser.add_argument(
        "--per-seed-timeout",
        type=float,
        default=20.0,
        help="Skip a seed attempt if generation and search take longer than this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_map_search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_counts = parse_target_label_counts(args.target_label_counts)
    if target_counts:
        rows = search_targeted_candidates(
            target_counts=target_counts,
            start_seed=args.start_seed,
            max_depth=args.max_depth,
            solution_limit=args.solution_limit,
            sort_by=args.sort_by,
            max_seed_attempts=args.max_seed_attempts,
            per_seed_timeout=args.per_seed_timeout,
        )
    else:
        rows = search_candidates(
            n_candidates=args.n_candidates,
            start_seed=args.start_seed,
            max_depth=args.max_depth,
            solution_limit=args.solution_limit,
            sort_by=args.sort_by,
            max_seed_attempts=args.max_seed_attempts,
            per_seed_timeout=args.per_seed_timeout,
            generation_mode=args.generation_mode,
            target_planning_requirement=args.target_planning_requirement,
        )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
