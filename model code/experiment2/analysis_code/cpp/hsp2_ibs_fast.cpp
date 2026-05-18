#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using State = std::vector<int>;

struct Action {
    int node = -1;
    int old_color = -1;
    int new_color = -1;
};

struct RoundData {
    int n_regions = 0;
    std::vector<std::vector<int>> adj;
    State initial_state;
};

struct ObsTask {
    int obs_index = 0;
    int round_id = 0;
    State state;
    int target_region = -1;
    int target_new_color = -1;
};

struct PlannerNode {
    State state;
    int g = 0;
    int h = 0;
    int depth = 0;
    int parent = -1;
    Action action;
    bool has_action = false;
};

static constexpr int LARGE_COST = 10;
static constexpr int N_COLORS = 4;

static std::string state_key(const State& state) {
    std::string key;
    key.reserve(state.size());
    for (int color : state) {
        key.push_back(static_cast<char>('0' + color));
    }
    return key;
}

static std::string state_text(const State& state) {
    std::ostringstream out;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i) out << ' ';
        out << state[i];
    }
    return out.str();
}

static int count_conflicts(const State& state, const RoundData& round) {
    int conflicts = 0;
    for (int u = 0; u < round.n_regions; ++u) {
        for (int v : round.adj[u]) {
            if (u < v && state[u] == state[v]) {
                ++conflicts;
            }
        }
    }
    return conflicts;
}

static std::vector<int> conflict_nodes(const State& state, const RoundData& round) {
    std::vector<char> seen(round.n_regions, 0);
    for (int u = 0; u < round.n_regions; ++u) {
        for (int v : round.adj[u]) {
            if (u < v && state[u] == state[v]) {
                seen[u] = 1;
                seen[v] = 1;
            }
        }
    }
    std::vector<int> nodes;
    for (int i = 0; i < round.n_regions; ++i) {
        if (seen[i]) nodes.push_back(i);
    }
    return nodes;
}

static State apply_action(const State& state, const Action& action) {
    State next = state;
    next[action.node] = action.new_color;
    return next;
}

static bool legal_color_for_node(
    const State& state,
    int node,
    int color,
    const RoundData& round
) {
    for (int neighbor : round.adj[node]) {
        if (state[neighbor] == color) return false;
    }
    return true;
}

static int legal_color_count(const State& state, int node, const RoundData& round) {
    int count = 0;
    for (int color = 0; color < N_COLORS; ++color) {
        if (legal_color_for_node(state, node, color, round)) ++count;
    }
    return count;
}

static bool has_legal_recolor(const State& state, int node, const RoundData& round) {
    const int old_color = state[node];
    for (int color = 0; color < N_COLORS; ++color) {
        if (color != old_color && legal_color_for_node(state, node, color, round)) {
            return true;
        }
    }
    return false;
}

static std::vector<int> blockers_for_color(
    const State& state,
    int node,
    int color,
    const RoundData& round
) {
    std::vector<int> blockers;
    for (int neighbor : round.adj[node]) {
        if (state[neighbor] == color) blockers.push_back(neighbor);
    }
    return blockers;
}

static int dynamic_neighbor_cost(
    const State& state,
    int node,
    const RoundData& round,
    int depth_limit,
    std::unordered_map<std::string, int>& cache,
    std::unordered_set<std::string>& active
) {
    std::string cache_key = state_key(state) + "|" + std::to_string(node) + "|" + std::to_string(depth_limit);
    auto cached = cache.find(cache_key);
    if (cached != cache.end()) return cached->second;

    if (has_legal_recolor(state, node, round)) {
        cache[cache_key] = 1;
        return 1;
    }
    if (depth_limit <= 0) {
        cache[cache_key] = LARGE_COST;
        return LARGE_COST;
    }

    std::string active_key = std::to_string(node) + "|" + std::to_string(depth_limit);
    if (active.find(active_key) != active.end()) return LARGE_COST;
    active.insert(active_key);

    const int current_color = state[node];
    int best_cost = LARGE_COST;
    for (int color = 0; color < N_COLORS; ++color) {
        if (color == current_color) continue;
        auto blockers = blockers_for_color(state, node, color, round);
        if (blockers.empty()) {
            best_cost = std::min(best_cost, 1);
            continue;
        }
        int blocker_sum = 0;
        for (int blocker : blockers) {
            blocker_sum += dynamic_neighbor_cost(
                state,
                blocker,
                round,
                depth_limit - 1,
                cache,
                active
            );
        }
        best_cost = std::min(best_cost, 1 + blocker_sum);
    }

    active.erase(active_key);
    cache[cache_key] = best_cost;
    return best_cost;
}

static int node_repair_cost(
    const State& state,
    int node,
    const RoundData& round,
    int unlock_depth_limit,
    std::unordered_map<std::string, int>& cache
) {
    const int current_color = state[node];
    int best_cost = LARGE_COST;
    for (int color = 0; color < N_COLORS; ++color) {
        if (color == current_color) continue;
        auto blockers = blockers_for_color(state, node, color, round);
        if (blockers.empty()) {
            best_cost = std::min(best_cost, 1);
            continue;
        }
        int blocker_cost = 0;
        for (int blocker : blockers) {
            std::unordered_set<std::string> active;
            blocker_cost += dynamic_neighbor_cost(
                state,
                blocker,
                round,
                unlock_depth_limit,
                cache,
                active
            );
        }
        best_cost = std::min(best_cost, 1 + blocker_cost);
    }
    return best_cost;
}

static int h_add_relaxed(const State& state, const RoundData& round, int unlock_depth_limit) {
    auto conflicts = conflict_nodes(state, round);
    if (conflicts.empty()) return 0;
    std::unordered_map<std::string, int> cache;
    int total = 0;
    for (int node : conflicts) {
        total += node_repair_cost(state, node, round, unlock_depth_limit, cache);
    }
    return total;
}

static std::pair<std::vector<int>, std::vector<int>> dependency_candidate_nodes(
    const State& state,
    const RoundData& round,
    int dependency_depth
) {
    auto conflicts = conflict_nodes(state, round);
    std::vector<char> candidate(round.n_regions, 0);
    std::vector<int> reason(round.n_regions, 0);  // 1 conflict, 2 blocker, 3 fallback_neighbor
    std::vector<int> frontier = conflicts;
    for (int node : conflicts) {
        candidate[node] = 1;
        reason[node] = 1;
    }

    for (int depth = 0; depth < dependency_depth; ++depth) {
        std::vector<int> new_frontier;
        for (int node : frontier) {
            if (has_legal_recolor(state, node, round)) continue;
            const int current_color = state[node];
            for (int color = 0; color < N_COLORS; ++color) {
                if (color == current_color) continue;
                for (int blocker : blockers_for_color(state, node, color, round)) {
                    if (!candidate[blocker]) {
                        candidate[blocker] = 1;
                        reason[blocker] = 2;
                        new_frontier.push_back(blocker);
                    }
                }
            }
        }
        frontier.swap(new_frontier);
        if (frontier.empty()) break;
    }

    bool only_conflicts = true;
    for (int i = 0; i < round.n_regions; ++i) {
        if (candidate[i] && reason[i] != 1) {
            only_conflicts = false;
            break;
        }
    }
    if (only_conflicts) {
        for (int node : conflicts) {
            for (int neighbor : round.adj[node]) {
                if (!candidate[neighbor]) {
                    candidate[neighbor] = 1;
                    reason[neighbor] = 3;
                }
            }
        }
    }

    std::vector<int> nodes;
    for (int i = 0; i < round.n_regions; ++i) {
        if (candidate[i]) nodes.push_back(i);
    }
    return {nodes, reason};
}

static std::vector<Action> generate_relevant_actions(
    const State& state,
    const RoundData& round,
    const std::vector<int>& candidate_nodes,
    const std::vector<int>& candidate_reason,
    int allowed_worsening = 1
) {
    auto conflicts = conflict_nodes(state, round);
    std::vector<char> is_conflict(round.n_regions, 0);
    for (int node : conflicts) is_conflict[node] = 1;

    const int current_conflicts = count_conflicts(state, round);
    std::vector<int> legal_counts(round.n_regions, 0);
    for (int node : conflicts) {
        legal_counts[node] = legal_color_count(state, node, round);
    }

    std::vector<Action> actions;
    std::unordered_set<std::string> seen;
    for (int node : candidate_nodes) {
        const int old_color = state[node];
        for (int color = 0; color < N_COLORS; ++color) {
            if (color == old_color) continue;
            Action action{node, old_color, color};
            State child = apply_action(state, action);
            const int child_conflicts = count_conflicts(child, round);
            bool increases_legal = false;
            for (int conflict_node : conflicts) {
                if (legal_color_count(child, conflict_node, round) > legal_counts[conflict_node]) {
                    increases_legal = true;
                    break;
                }
            }
            const int reason = candidate_reason[node];
            bool relevant =
                is_conflict[node] ||
                child_conflicts < current_conflicts ||
                child_conflicts <= current_conflicts + allowed_worsening ||
                increases_legal ||
                reason == 2 ||
                reason == 3;
            if (!relevant) continue;
            std::string key = std::to_string(action.node) + "," +
                std::to_string(action.old_color) + "," +
                std::to_string(action.new_color);
            if (seen.insert(key).second) actions.push_back(action);
        }
    }
    return actions;
}

static int effective_expansions(double gamma, int fallback) {
    if (gamma <= 0.0) return fallback;
    int iterations = static_cast<int>(1.0 / gamma) + 1;
    return std::max(1, std::min(5000, iterations));
}

static bool action_less(const Action& a, const Action& b) {
    if (a.node != b.node) return a.node < b.node;
    if (a.old_color != b.old_color) return a.old_color < b.old_color;
    return a.new_color < b.new_color;
}

static bool node_rank_less(const PlannerNode& a, const PlannerNode& b, const RoundData& round) {
    if (a.h != b.h) return a.h < b.h;
    int ca = count_conflicts(a.state, round);
    int cb = count_conflicts(b.state, round);
    if (ca != cb) return ca < cb;
    if (a.g != b.g) return a.g < b.g;
    if (a.has_action != b.has_action) return !a.has_action;
    if (!a.has_action) return false;
    return action_less(a.action, b.action);
}

static std::vector<Action> reconstruct_path(const std::vector<PlannerNode>& nodes, int node_index) {
    std::vector<Action> path;
    int current = node_index;
    while (current >= 0 && nodes[current].has_action) {
        path.push_back(nodes[current].action);
        current = nodes[current].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

static Action sample_hsp2_next_action(
    const RoundData& round,
    const State& root_state,
    double pruning_thresh,
    double gamma,
    double lapse_rate,
    int max_depth,
    int max_expansions,
    std::mt19937& rng,
    bool& has_action
) {
    has_action = false;
    const int root_conflicts = count_conflicts(root_state, round);
    if (root_conflicts > 0 && lapse_rate > 0.0) {
        std::uniform_real_distribution<double> dist01(0.0, 1.0);
        if (dist01(rng) < lapse_rate) {
            auto [candidate_nodes, candidate_reason] = dependency_candidate_nodes(root_state, round, 3);
            auto actions = generate_relevant_actions(root_state, round, candidate_nodes, candidate_reason);
            if (!actions.empty()) {
                std::uniform_int_distribution<int> pick(0, static_cast<int>(actions.size()) - 1);
                has_action = true;
                return actions[pick(rng)];
            }
        }
    }

    const int max_iters = effective_expansions(gamma, max_expansions);
    std::vector<PlannerNode> nodes;
    nodes.reserve(1024);
    PlannerNode root;
    root.state = root_state;
    root.g = 0;
    root.h = h_add_relaxed(root_state, round, 3);
    root.depth = 0;
    nodes.push_back(root);

    std::vector<int> frontier{0};
    std::unordered_set<std::string> closed;
    int best_node = 0;
    int best_partial = -1;
    int expansions = 0;

    while (!frontier.empty() && expansions < max_iters) {
        std::unordered_map<std::string, int> next_candidates;
        int current_depth = nodes[frontier[0]].depth;
        (void)current_depth;

        for (int current_idx : frontier) {
            if (expansions >= max_iters) break;
            PlannerNode& current = nodes[current_idx];
            std::string key = state_key(current.state);
            if (closed.find(key) != closed.end()) continue;
            closed.insert(key);
            ++expansions;

            const int current_conflicts = count_conflicts(current.state, round);
            if (current_conflicts == 0) {
                auto path = reconstruct_path(nodes, current_idx);
                if (!path.empty()) {
                    has_action = true;
                    return path.front();
                }
                return Action{};
            }

            if (
                best_partial < 0 ||
                std::make_tuple(current_conflicts, current.h, current.g) <
                    std::make_tuple(
                        count_conflicts(nodes[best_partial].state, round),
                        nodes[best_partial].h,
                        nodes[best_partial].g
                    )
            ) {
                best_partial = current_idx;
            }

            if (node_rank_less(current, nodes[best_node], round)) {
                best_node = current_idx;
            }

            if (current.depth >= max_depth) continue;

            State current_state_copy = current.state;
            int current_g = current.g;
            int current_node_depth = current.depth;
            auto [candidate_nodes, candidate_reason] = dependency_candidate_nodes(current_state_copy, round, 3);
            auto actions = generate_relevant_actions(current_state_copy, round, candidate_nodes, candidate_reason);

            for (const Action& action : actions) {
                State child_state = apply_action(current_state_copy, action);
                std::string child_key = state_key(child_state);
                if (closed.find(child_key) != closed.end()) continue;

                PlannerNode child;
                child.state = std::move(child_state);
                child.g = current_g + 1;
                child.h = h_add_relaxed(child.state, round, 3);
                child.depth = current_node_depth + 1;
                child.parent = current_idx;
                child.action = action;
                child.has_action = true;
                nodes.push_back(std::move(child));
                int child_idx = static_cast<int>(nodes.size()) - 1;

                auto existing = next_candidates.find(child_key);
                if (existing == next_candidates.end() ||
                    node_rank_less(nodes[child_idx], nodes[existing->second], round)) {
                    next_candidates[child_key] = child_idx;
                }
            }
        }

        std::vector<int> children;
        children.reserve(next_candidates.size());
        for (const auto& item : next_candidates) children.push_back(item.second);

        std::sort(children.begin(), children.end(), [&](int lhs, int rhs) {
            return node_rank_less(nodes[lhs], nodes[rhs], round);
        });

        if (pruning_thresh >= 0.0 && !children.empty()) {
            int best_h = std::numeric_limits<int>::max();
            for (int idx : children) best_h = std::min(best_h, nodes[idx].h);
            std::vector<int> kept;
            kept.reserve(children.size());
            for (int idx : children) {
                if (static_cast<double>(nodes[idx].h - best_h) <= pruning_thresh) {
                    kept.push_back(idx);
                }
            }
            children.swap(kept);
        }
        frontier.swap(children);
    }

    if (best_partial >= 0 && best_partial != best_node) best_node = best_partial;
    auto path = reconstruct_path(nodes, best_node);
    if (!path.empty()) {
        has_action = true;
        return path.front();
    }
    return Action{};
}

static double ibs_action_nll(
    const ObsTask& task,
    const std::vector<RoundData>& rounds,
    double pruning_thresh,
    double gamma,
    double lapse_rate,
    int ibs_samples,
    int base_seed,
    int max_depth,
    int max_expansions,
    int max_tries
) {
    double nll = 0.0;
    int times_left = ibs_samples;
    int tries = 0;
    const RoundData& round = rounds.at(task.round_id);
    while (times_left > 0 && tries < max_tries) {
        ++tries;
        int seed = base_seed + task.obs_index * 1000003 + tries;
        std::mt19937 rng(static_cast<unsigned int>(seed));
        bool has_action = false;
        Action sampled = sample_hsp2_next_action(
            round,
            task.state,
            pruning_thresh,
            gamma,
            lapse_rate,
            max_depth,
            max_expansions,
            rng,
            has_action
        );
        if (has_action &&
            sampled.node == task.target_region &&
            sampled.new_color == task.target_new_color) {
            --times_left;
        } else {
            nll += 1.0 / (static_cast<double>(tries) * static_cast<double>(ibs_samples));
        }
    }
    if (times_left > 0) nll += static_cast<double>(times_left) * 3.5;
    return nll;
}

static double exact_hsp2_action_nll(
    const ObsTask& task,
    const std::vector<RoundData>& rounds,
    double pruning_thresh,
    double gamma,
    double lapse_rate,
    int max_depth,
    int max_expansions
) {
    const RoundData& round = rounds.at(task.round_id);
    double probability = 0.0;

    bool has_planner_action = false;
    std::mt19937 dummy_rng(1);
    Action planner_action = sample_hsp2_next_action(
        round,
        task.state,
        pruning_thresh,
        gamma,
        0.0,
        max_depth,
        max_expansions,
        dummy_rng,
        has_planner_action
    );
    if (has_planner_action &&
        planner_action.node == task.target_region &&
        planner_action.new_color == task.target_new_color) {
        probability += 1.0 - lapse_rate;
    }

    if (lapse_rate > 0.0 && count_conflicts(task.state, round) > 0) {
        auto [candidate_nodes, candidate_reason] = dependency_candidate_nodes(task.state, round, 3);
        auto lapse_actions = generate_relevant_actions(task.state, round, candidate_nodes, candidate_reason);
        if (lapse_actions.empty()) {
            if (has_planner_action &&
                planner_action.node == task.target_region &&
                planner_action.new_color == task.target_new_color) {
                probability = 1.0;
            }
        } else {
            int matches = 0;
            for (const Action& action : lapse_actions) {
                if (action.node == task.target_region &&
                    action.new_color == task.target_new_color) {
                    ++matches;
                }
            }
            probability += lapse_rate * static_cast<double>(matches) /
                static_cast<double>(lapse_actions.size());
        }
    }

    if (probability <= 0.0) {
        return 50.0;
    }
    return -std::log(probability);
}

static void read_tasks(
    const std::string& path,
    std::vector<RoundData>& rounds,
    std::vector<ObsTask>& tasks
) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open task file: " + path);
    std::string tag;
    int n_rounds = 0;
    in >> tag >> n_rounds;
    if (tag != "ROUNDS") throw std::runtime_error("Expected ROUNDS header.");
    rounds.assign(1, RoundData{});
    for (int r = 0; r < n_rounds; ++r) {
        int round_id = 0, n_regions = 0, n_edges = 0;
        in >> tag >> round_id >> n_regions >> n_edges;
        if (tag != "ROUND") throw std::runtime_error("Expected ROUND row.");
        if (round_id >= static_cast<int>(rounds.size())) {
            rounds.resize(round_id + 1);
        }
        rounds[round_id].n_regions = n_regions;
        rounds[round_id].adj.assign(n_regions, {});
        for (int e = 0; e < n_edges; ++e) {
            int u = 0, v = 0;
            in >> tag >> u >> v;
            if (tag != "EDGE") throw std::runtime_error("Expected EDGE row.");
            rounds[round_id].adj[u].push_back(v);
            rounds[round_id].adj[v].push_back(u);
        }
        for (auto& neighbors : rounds[round_id].adj) {
            std::sort(neighbors.begin(), neighbors.end());
        }
    }
    int n_obs = 0;
    in >> tag >> n_obs;
    if (tag != "OBS") throw std::runtime_error("Expected OBS header.");
    tasks.reserve(n_obs);
    for (int i = 0; i < n_obs; ++i) {
        ObsTask task;
        int state_len = 0;
        in >> tag >> task.obs_index >> task.round_id >> state_len;
        if (tag != "OBSROW") throw std::runtime_error("Expected OBSROW row.");
        task.state.resize(state_len);
        for (int j = 0; j < state_len; ++j) in >> task.state[j];
        in >> task.target_region >> task.target_new_color;
        tasks.push_back(std::move(task));
    }
}

static std::vector<RoundData> read_rounds_for_simulation(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open rounds file: " + path);
    std::string tag;
    int n_rounds = 0;
    in >> tag >> n_rounds;
    if (tag != "ROUNDS") throw std::runtime_error("Expected ROUNDS header.");

    std::vector<RoundData> rounds;
    rounds.reserve(n_rounds);
    for (int r = 0; r < n_rounds; ++r) {
        int round_id = 0, n_regions = 0, n_edges = 0;
        in >> tag >> round_id >> n_regions >> n_edges;
        if (tag != "ROUND") throw std::runtime_error("Expected ROUND row.");
        RoundData round;
        round.n_regions = n_regions;
        round.adj.assign(n_regions, {});
        round.initial_state.assign(n_regions, 0);

        in >> tag;
        if (tag != "INIT") throw std::runtime_error("Expected INIT row.");
        for (int i = 0; i < n_regions; ++i) {
            in >> round.initial_state[i];
        }
        for (int e = 0; e < n_edges; ++e) {
            int u = 0, v = 0;
            in >> tag >> u >> v;
            if (tag != "EDGE") throw std::runtime_error("Expected EDGE row.");
            round.adj[u].push_back(v);
            round.adj[v].push_back(u);
        }
        for (auto& neighbors : round.adj) {
            std::sort(neighbors.begin(), neighbors.end());
        }
        rounds.push_back(std::move(round));
    }
    return rounds;
}

static int run_simulate_mode(int argc, char** argv) {
    if (argc != 11) {
        std::cerr
            << "Usage: hsp2_ibs_fast --simulate ROUNDS_FILE pruning gamma lapse "
            << "max_steps max_depth max_expansions seed round_limit\n";
        return 2;
    }

    const std::string rounds_path = argv[2];
    const double pruning_thresh = std::stod(argv[3]);
    const double gamma = std::stod(argv[4]);
    const double lapse_rate = std::stod(argv[5]);
    const int max_steps = std::stoi(argv[6]);
    const int max_depth = std::stoi(argv[7]);
    const int max_expansions = std::stoi(argv[8]);
    const int seed = std::stoi(argv[9]);
    const int round_limit = std::stoi(argv[10]);

    auto rounds = read_rounds_for_simulation(rounds_path);
    if (round_limit > 0 && round_limit < static_cast<int>(rounds.size())) {
        rounds.resize(round_limit);
    }
    std::mt19937 rng(static_cast<unsigned int>(seed));

    std::cout
        << "agent,round,agent_step,module,state_before,region,old_color,new_color,"
        << "n_conflict_edges_before,n_conflict_edges_after,final_conflicts,success,random_seed\n";
    for (size_t round_offset = 0; round_offset < rounds.size(); ++round_offset) {
        const RoundData& round = rounds[round_offset];
        const int round_id = static_cast<int>(round_offset) + 1;
        State state = round.initial_state;
        std::vector<Action> actions;
        std::vector<State> states_before;
        for (int step = 0; step < max_steps; ++step) {
            if (count_conflicts(state, round) == 0) break;
            bool has_action = false;
            Action action = sample_hsp2_next_action(
                round,
                state,
                pruning_thresh,
                gamma,
                lapse_rate,
                max_depth,
                max_expansions,
                rng,
                has_action
            );
            if (!has_action) break;
            states_before.push_back(state);
            actions.push_back(action);
            state = apply_action(state, action);
        }
        const int final_conflicts = count_conflicts(state, round);
        for (size_t step = 0; step < actions.size(); ++step) {
            const State& before_state = states_before[step];
            State after_state = apply_action(before_state, actions[step]);
            std::cout
                << "hsp2,"
                << round_id << ','
                << step << ','
                << "hsp2,"
                << '"' << state_text(before_state) << '"' << ','
                << actions[step].node << ','
                << actions[step].old_color << ','
                << actions[step].new_color << ','
                << count_conflicts(before_state, round) << ','
                << count_conflicts(after_state, round) << ','
                << final_conflicts << ','
                << (final_conflicts == 0 ? 1 : 0) << ','
                << seed << '\n';
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--simulate") {
        try {
            return run_simulate_mode(argc, argv);
        } catch (const std::exception& exc) {
            std::cerr << "hsp2_ibs_fast --simulate error: " << exc.what() << "\n";
            return 1;
        }
    }

    if (argc != 11 && argc != 12) {
        std::cerr
            << "Usage: hsp2_ibs_fast TASKS pruning gamma lapse ibs_samples base_seed "
            << "max_depth max_expansions max_tries n_workers [ibs|exact]\n"
            << "   or: hsp2_ibs_fast --simulate ROUNDS_FILE pruning gamma lapse "
            << "max_steps max_depth max_expansions seed round_limit\n";
        return 2;
    }

    try {
        const std::string task_path = argv[1];
        const double pruning_thresh = std::stod(argv[2]);
        const double gamma = std::stod(argv[3]);
        const double lapse_rate = std::stod(argv[4]);
        const int ibs_samples = std::stoi(argv[5]);
        const int base_seed = std::stoi(argv[6]);
        const int max_depth = std::stoi(argv[7]);
        const int max_expansions = std::stoi(argv[8]);
        const int max_tries = std::stoi(argv[9]);
        const int n_workers = std::max(1, std::stoi(argv[10]));
        const std::string mode = argc == 12 ? argv[11] : "ibs";

        std::vector<RoundData> rounds;
        std::vector<ObsTask> tasks;
        read_tasks(task_path, rounds, tasks);

        std::vector<double> results(tasks.size(), 0.0);
        int workers = std::min<int>(n_workers, std::max<int>(1, tasks.size()));
        std::vector<std::future<void>> futures;
        futures.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            futures.push_back(std::async(std::launch::async, [&, worker]() {
                for (size_t i = worker; i < tasks.size(); i += workers) {
                    if (mode == "exact") {
                        results[i] = exact_hsp2_action_nll(
                            tasks[i],
                            rounds,
                            pruning_thresh,
                            gamma,
                            lapse_rate,
                            max_depth,
                            max_expansions
                        );
                    } else {
                        results[i] = ibs_action_nll(
                            tasks[i],
                            rounds,
                            pruning_thresh,
                            gamma,
                            lapse_rate,
                            ibs_samples,
                            base_seed,
                            max_depth,
                            max_expansions,
                            max_tries
                        );
                    }
                }
            }));
        }
        for (auto& f : futures) f.get();

        for (size_t i = 0; i < tasks.size(); ++i) {
            std::cout << tasks[i].obs_index << " " << results[i] << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "hsp2_ibs_fast error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
