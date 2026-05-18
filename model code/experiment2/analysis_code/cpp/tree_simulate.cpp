#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using State = std::vector<int>;

struct Action {
    int search_depth = 0;
    int region = -1;
    int old_color = -1;
    int new_color = -1;
    double heuristic_score = 0.0;
};

struct RoundData {
    int round_id = 0;
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

struct Score4 {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
};

struct TreeScore {
    std::vector<double> values;
};

struct Node {
    State state;
    Action action;
    bool has_action = false;
    int parent = -1;
    int depth = 0;
    std::vector<int> children;
    bool expanded = false;
    Score4 self_score;
    TreeScore subtree_score;
    int best_child = -1;
    int best_distance = 0;
    int conflict_count = 0;
};

static std::string state_text(const State& state) {
    std::ostringstream out;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i) out << ' ';
        out << state[i];
    }
    return out.str();
}

static std::vector<std::pair<int, int>> conflict_edges(const RoundData& round, const State& state) {
    std::vector<std::pair<int, int>> edges;
    for (int u = 0; u < round.n_regions; ++u) {
        for (int v : round.adj[u]) {
            if (u < v && state[u] == state[v]) {
                edges.emplace_back(u, v);
            }
        }
    }
    return edges;
}

static std::vector<int> conflict_regions_from_edges(
    int n_regions,
    const std::vector<std::pair<int, int>>& edges
) {
    std::vector<char> seen(n_regions, 0);
    for (auto [u, v] : edges) {
        seen[u] = 1;
        seen[v] = 1;
    }
    std::vector<int> regions;
    for (int i = 0; i < n_regions; ++i) {
        if (seen[i]) regions.push_back(i);
    }
    return regions;
}

static bool is_legal_color(const RoundData& round, const State& state, int region, int color) {
    for (int neighbor : round.adj[region]) {
        if (state[neighbor] == color) return false;
    }
    return true;
}

static std::vector<int> legal_recolor_options(
    const RoundData& round,
    const State& state,
    int region
) {
    std::vector<int> options;
    int current = state[region];
    for (int color = 0; color < 4; ++color) {
        if (color == current) continue;
        if (is_legal_color(round, state, region, color)) options.push_back(color);
    }
    return options;
}

static int count_legal_recolor_options(
    const RoundData& round,
    const State& state,
    int region
) {
    return static_cast<int>(legal_recolor_options(round, state, region).size());
}

static State apply_action(const State& state, const Action& action) {
    State next = state;
    next[action.region] = action.new_color;
    return next;
}

static std::vector<std::vector<int>> layered_candidate_regions(
    const RoundData& round,
    const std::vector<int>& conflict_regions
) {
    std::vector<std::vector<int>> layers;
    if (conflict_regions.empty()) return layers;

    std::vector<char> seen(round.n_regions, 0);
    std::deque<std::pair<int, int>> queue;
    for (int region : conflict_regions) {
        seen[region] = 1;
        queue.emplace_back(region, 0);
    }

    while (!queue.empty()) {
        auto [region, depth] = queue.front();
        queue.pop_front();
        if (depth >= static_cast<int>(layers.size())) layers.resize(depth + 1);
        layers[depth].push_back(region);

        for (int neighbor : round.adj[region]) {
            if (seen[neighbor]) continue;
            seen[neighbor] = 1;
            queue.emplace_back(neighbor, depth + 1);
        }
    }
    return layers;
}

static std::vector<Action> ordered_legal_actions(
    const RoundData& round,
    const State& state,
    std::vector<std::pair<int, int>>* out_conflict_edges = nullptr,
    std::vector<int>* out_conflict_regions = nullptr
) {
    auto edges = conflict_edges(round, state);
    auto regions = conflict_regions_from_edges(round.n_regions, edges);
    auto layers = layered_candidate_regions(round, regions);

    std::vector<Action> actions;
    for (int depth = 0; depth < static_cast<int>(layers.size()); ++depth) {
        std::vector<Action> layer_actions;
        for (int region : layers[depth]) {
            for (int new_color : legal_recolor_options(round, state, region)) {
                layer_actions.push_back(Action{
                    depth,
                    region,
                    state[region],
                    new_color,
                    0.0,
                });
            }
        }
        std::sort(layer_actions.begin(), layer_actions.end(), [](const Action& lhs, const Action& rhs) {
            return std::tie(lhs.search_depth, lhs.region, lhs.new_color) <
                std::tie(rhs.search_depth, rhs.region, rhs.new_color);
        });
        actions.insert(actions.end(), layer_actions.begin(), layer_actions.end());
    }

    if (out_conflict_edges) *out_conflict_edges = std::move(edges);
    if (out_conflict_regions) *out_conflict_regions = std::move(regions);
    return actions;
}

static Score4 state_score(const RoundData& round, const State& state) {
    std::vector<std::pair<int, int>> edges;
    auto actions = ordered_legal_actions(round, state, &edges, nullptr);
    int conflict_count = static_cast<int>(edges.size());
    if (conflict_count == 0) return Score4{1.0, 0.0, 1.0, 0.0};
    int first_depth = 999;
    if (!actions.empty()) first_depth = actions.front().search_depth;
    int zero_depth_available = first_depth == 0 ? 1 : 0;
    return Score4{0.0, -static_cast<double>(conflict_count), static_cast<double>(zero_depth_available), -static_cast<double>(first_depth)};
}

static TreeScore compose_tree_score(const Score4& base, int distance_to_best_state) {
    return TreeScore{{base.a, base.b, base.c, base.d, 0.0, -static_cast<double>(distance_to_best_state)}};
}

static bool score_less(const TreeScore& lhs, const TreeScore& rhs) {
    return std::lexicographical_compare(
        lhs.values.begin(), lhs.values.end(),
        rhs.values.begin(), rhs.values.end()
    );
}

static bool score_equal(const TreeScore& lhs, const TreeScore& rhs) {
    return lhs.values == rhs.values;
}

static double action_heuristic_score(const RoundData& round, const State& state, const Action& action) {
    auto before_edges = conflict_edges(round, state);
    State next = apply_action(state, action);
    auto after_edges = conflict_edges(round, next);
    double repair = static_cast<double>(before_edges.size()) - static_cast<double>(after_edges.size());
    repair /= std::max<size_t>(1, before_edges.size());

    auto next_actions = ordered_legal_actions(round, next);
    double opportunity = 0.0;
    if (!next_actions.empty() && next_actions.front().search_depth == 0) {
        opportunity = 1.0;
    }
    return 2.0 * repair + opportunity;
}

static std::vector<Action> prune_tree_actions(
    const RoundData& round,
    const State& state,
    const std::vector<Action>& actions,
    double pruning_thresh
) {
    if (actions.empty()) return {};
    std::vector<Action> scored = actions;
    for (auto& action : scored) {
        action.heuristic_score = action_heuristic_score(round, state, action);
    }
    std::stable_sort(scored.begin(), scored.end(), [](const Action& lhs, const Action& rhs) {
        return lhs.heuristic_score > rhs.heuristic_score;
    });

    if (pruning_thresh < 0.0) return scored;

    const double best = scored.front().heuristic_score;
    const double threshold = std::max(0.0, pruning_thresh);
    std::vector<Action> kept;
    for (const Action& action : scored) {
        if ((best - action.heuristic_score) <= threshold + 1e-12) {
            kept.push_back(action);
        }
    }
    return kept;
}

static int make_node(
    std::vector<Node>& nodes,
    const RoundData& round,
    const State& state,
    int parent,
    int depth,
    const Action* action
) {
    Node node;
    node.state = state;
    node.parent = parent;
    node.depth = depth;
    if (action) {
        node.action = *action;
        node.has_action = true;
    }
    auto edges = conflict_edges(round, state);
    node.conflict_count = static_cast<int>(edges.size());
    node.self_score = state_score(round, state);
    node.subtree_score = compose_tree_score(node.self_score, 0);
    nodes.push_back(std::move(node));
    return static_cast<int>(nodes.size()) - 1;
}

static int choose_best_tree_child(
    std::vector<Node>& nodes,
    int node_idx,
    bool random_tie_break,
    std::mt19937& rng
) {
    Node& node = nodes[node_idx];
    if (node.children.empty()) return -1;
    int best = node.children.front();
    for (int child : node.children) {
        if (score_less(nodes[best].subtree_score, nodes[child].subtree_score)) {
            best = child;
        }
    }
    if (!random_tie_break) return best;

    std::vector<int> tied;
    for (int child : node.children) {
        if (score_equal(nodes[child].subtree_score, nodes[best].subtree_score)) {
            tied.push_back(child);
        }
    }
    if (tied.size() <= 1) return best;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(tied.size()) - 1);
    return tied[pick(rng)];
}

static void backup_tree_value(
    std::vector<Node>& nodes,
    int node_idx,
    bool random_tie_break,
    std::mt19937& rng
) {
    Node& node = nodes[node_idx];
    TreeScore self_score = compose_tree_score(node.self_score, 0);
    if (node.conflict_count == 0 || node.children.empty()) {
        node.best_child = -1;
        node.subtree_score = self_score;
        node.best_distance = 0;
        return;
    }
    int best_child = choose_best_tree_child(nodes, node_idx, random_tie_break, rng);
    if (best_child < 0) {
        node.best_child = -1;
        node.subtree_score = self_score;
        node.best_distance = 0;
        return;
    }
    node.best_child = best_child;
    node.subtree_score = nodes[best_child].subtree_score;
    node.best_distance = nodes[best_child].best_distance + 1;
}

static void backpropagate_tree_values(
    std::vector<Node>& nodes,
    int changed_idx,
    bool random_tie_break,
    std::mt19937& rng
) {
    int current = changed_idx;
    while (current >= 0) {
        backup_tree_value(nodes, current, random_tie_break, rng);
        current = nodes[current].parent;
    }
}

static void expand_tree_node(
    std::vector<Node>& nodes,
    int node_idx,
    const RoundData& round,
    double pruning_thresh
) {
    Node& node = nodes[node_idx];
    if (node.expanded || node.conflict_count == 0) {
        node.expanded = true;
        return;
    }
    State state_copy = node.state;
    auto ordered = ordered_legal_actions(round, state_copy);
    auto pruned = prune_tree_actions(round, state_copy, ordered, pruning_thresh);
    node.expanded = true;

    for (const Action& action : pruned) {
        State child_state = apply_action(state_copy, action);
        int child_idx = make_node(nodes, round, child_state, node_idx, nodes[node_idx].depth + 1, &action);
        nodes[node_idx].children.push_back(child_idx);
    }
}

static int select_best_path_leaf(const std::vector<Node>& nodes, int root_idx, int max_depth) {
    int current = root_idx;
    while (true) {
        const Node& node = nodes[current];
        if (node.conflict_count == 0 || node.depth >= max_depth) return current;
        if (!node.expanded || node.children.empty() || node.best_child < 0) return current;
        current = node.best_child;
    }
}

static int effective_expansions(double gamma, int fallback) {
    if (gamma <= 0.0) return fallback;
    int iterations = static_cast<int>(std::floor(1.0 / gamma)) + 1;
    return std::max(1, std::min(fallback, iterations));
}

static Action sample_tree_next_action(
    const RoundData& round,
    const State& state,
    double pruning_thresh,
    double gamma,
    double lapse_rate,
    int max_depth,
    int max_expansions,
    bool random_tie_break,
    std::mt19937& rng,
    bool& has_selected
) {
    has_selected = false;
    if (conflict_edges(round, state).empty()) return Action{};

    int expansion_budget = effective_expansions(gamma, max_expansions);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    if (lapse_rate > 0.0 && dist01(rng) < lapse_rate) {
        auto actions = ordered_legal_actions(round, state);
        if (actions.empty()) return Action{};
        std::uniform_int_distribution<int> pick(0, static_cast<int>(actions.size()) - 1);
        has_selected = true;
        return actions[pick(rng)];
    }

    std::vector<Node> nodes;
    nodes.reserve(4096);
    int root_idx = make_node(nodes, round, state, -1, 0, nullptr);
    for (int iter = 0; iter < expansion_budget; ++iter) {
        int frontier = select_best_path_leaf(nodes, root_idx, max_depth);
        if (nodes[frontier].depth >= max_depth || nodes[frontier].conflict_count == 0) break;
        expand_tree_node(nodes, frontier, round, pruning_thresh);
        backpropagate_tree_values(nodes, frontier, random_tie_break, rng);
        if (nodes[root_idx].conflict_count == 0) break;
    }
    int best_child = nodes[root_idx].best_child;
    if (best_child < 0 || !nodes[best_child].has_action) return Action{};
    has_selected = true;
    return nodes[best_child].action;
}

static std::vector<Action> simulate_tree_round(
    const RoundData& round,
    double pruning_thresh,
    double gamma,
    double lapse_rate,
    int max_steps,
    int max_depth,
    int max_expansions,
    bool random_tie_break,
    std::mt19937& rng,
    State* final_state
) {
    State current = round.initial_state;
    std::vector<Action> trajectory;

    for (int step = 0; step < max_steps; ++step) {
        if (conflict_edges(round, current).empty()) break;

        bool has_selected = false;
        Action selected = sample_tree_next_action(
            round,
            current,
            pruning_thresh,
            gamma,
            lapse_rate,
            max_depth,
            max_expansions,
            random_tie_break,
            rng,
            has_selected
        );
        if (!has_selected) break;
        trajectory.push_back(selected);
        current = apply_action(current, selected);
    }

    if (final_state) *final_state = current;
    return trajectory;
}

static std::vector<RoundData> read_rounds(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open rounds file: " + path);
    std::string tag;
    int n_rounds = 0;
    in >> tag >> n_rounds;
    if (tag != "ROUNDS") throw std::runtime_error("Expected ROUNDS header.");
    std::vector<RoundData> rounds;
    rounds.reserve(n_rounds);
    for (int i = 0; i < n_rounds; ++i) {
        RoundData round;
        int n_edges = 0;
        in >> tag >> round.round_id >> round.n_regions >> n_edges;
        if (tag != "ROUND") throw std::runtime_error("Expected ROUND row.");
        in >> tag;
        if (tag != "INIT") throw std::runtime_error("Expected INIT row.");
        round.initial_state.resize(round.n_regions);
        for (int j = 0; j < round.n_regions; ++j) in >> round.initial_state[j];
        round.adj.assign(round.n_regions, {});
        for (int e = 0; e < n_edges; ++e) {
            int u = 0, v = 0;
            in >> tag >> u >> v;
            if (tag != "EDGE") throw std::runtime_error("Expected EDGE row.");
            round.adj[u].push_back(v);
            round.adj[v].push_back(u);
        }
        for (auto& neighbors : round.adj) std::sort(neighbors.begin(), neighbors.end());
        rounds.push_back(std::move(round));
    }
    return rounds;
}

static void read_ibs_tasks(
    const std::string& path,
    std::unordered_map<int, RoundData>& rounds,
    std::vector<ObsTask>& tasks
) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open IBS task file: " + path);

    std::string tag;
    int n_rounds = 0;
    in >> tag >> n_rounds;
    if (tag != "ROUNDS") throw std::runtime_error("Expected ROUNDS header.");

    for (int i = 0; i < n_rounds; ++i) {
        RoundData round;
        int n_edges = 0;
        in >> tag >> round.round_id >> round.n_regions >> n_edges;
        if (tag != "ROUND") throw std::runtime_error("Expected ROUND row.");
        round.adj.assign(round.n_regions, {});
        for (int e = 0; e < n_edges; ++e) {
            int u = 0, v = 0;
            in >> tag >> u >> v;
            if (tag != "EDGE") throw std::runtime_error("Expected EDGE row.");
            round.adj[u].push_back(v);
            round.adj[v].push_back(u);
        }
        for (auto& neighbors : round.adj) std::sort(neighbors.begin(), neighbors.end());
        rounds[round.round_id] = std::move(round);
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

static double ibs_action_nll(
    const ObsTask& task,
    const std::unordered_map<int, RoundData>& rounds,
    double pruning_thresh,
    double gamma,
    double lapse_rate,
    int ibs_samples,
    int base_seed,
    int max_depth,
    int max_expansions,
    int max_tries,
    bool random_tie_break
) {
    const auto round_it = rounds.find(task.round_id);
    if (round_it == rounds.end()) {
        throw std::runtime_error("Missing round in IBS task: " + std::to_string(task.round_id));
    }
    const RoundData& round = round_it->second;

    double nll = 0.0;
    int times_left = ibs_samples;
    int tries = 0;
    while (times_left > 0 && tries < max_tries) {
        ++tries;
        int seed = base_seed + task.obs_index * 1000003 + tries;
        std::mt19937 rng(static_cast<unsigned int>(seed));
        bool has_action = false;
        Action sampled = sample_tree_next_action(
            round,
            task.state,
            pruning_thresh,
            gamma,
            lapse_rate,
            max_depth,
            max_expansions,
            random_tie_break,
            rng,
            has_action
        );
        if (has_action &&
            sampled.region == task.target_region &&
            sampled.new_color == task.target_new_color) {
            --times_left;
        } else {
            nll += 1.0 / (static_cast<double>(tries) * static_cast<double>(ibs_samples));
        }
    }
    if (times_left > 0) nll += static_cast<double>(times_left) * 3.5;
    return nll;
}

static int run_ibs_mode(int argc, char** argv) {
    if (argc != 13) {
        std::cerr
            << "Usage: tree_simulate --ibs TASKS_FILE pruning_thresh gamma lapse_rate "
            << "ibs_samples base_seed max_depth max_expansions max_tries n_workers random_tie_break\n";
        return 2;
    }

    const std::string task_path = argv[2];
    const double pruning_thresh = std::stod(argv[3]);
    const double gamma = std::stod(argv[4]);
    const double lapse_rate = std::stod(argv[5]);
    const int ibs_samples = std::stoi(argv[6]);
    const int base_seed = std::stoi(argv[7]);
    const int max_depth = std::stoi(argv[8]);
    const int max_expansions = std::stoi(argv[9]);
    const int max_tries = std::stoi(argv[10]);
    const int n_workers = std::max(1, std::stoi(argv[11]));
    const bool random_tie_break = std::stoi(argv[12]) != 0;

    std::unordered_map<int, RoundData> rounds;
    std::vector<ObsTask> tasks;
    read_ibs_tasks(task_path, rounds, tasks);

    std::vector<double> results(tasks.size(), 0.0);
    const int workers = std::min<int>(n_workers, std::max<int>(1, tasks.size()));
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        futures.push_back(std::async(std::launch::async, [&, worker]() {
            for (size_t i = worker; i < tasks.size(); i += workers) {
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
                    max_tries,
                    random_tie_break
                );
            }
        }));
    }
    for (auto& future : futures) future.get();

    for (size_t i = 0; i < tasks.size(); ++i) {
        std::cout << tasks[i].obs_index << " " << std::setprecision(12) << results[i] << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--ibs") {
        try {
            return run_ibs_mode(argc, argv);
        } catch (const std::exception& exc) {
            std::cerr << "tree_simulate --ibs error: " << exc.what() << "\n";
            return 1;
        }
    }

    if (argc != 10) {
        std::cerr
            << "Usage: tree_simulate ROUNDS_FILE pruning_thresh gamma lapse_rate "
            << "max_steps max_depth max_expansions seed random_tie_break\n";
        return 2;
    }

    try {
        const std::string rounds_path = argv[1];
        const double pruning_thresh = std::stod(argv[2]);
        const double gamma = std::stod(argv[3]);
        const double lapse_rate = std::stod(argv[4]);
        const int max_steps = std::stoi(argv[5]);
        const int max_depth = std::stoi(argv[6]);
        const int max_expansions = std::stoi(argv[7]);
        const int seed = std::stoi(argv[8]);
        const bool random_tie_break = std::stoi(argv[9]) != 0;

        auto rounds = read_rounds(rounds_path);
        std::mt19937 rng(static_cast<unsigned int>(seed));

        std::cout
            << "round,agent_step,state_before,region,old_color,new_color,"
            << "n_conflict_edges_before,n_conflict_edges_after,final_conflicts\n";
        for (const RoundData& round : rounds) {
            State state = round.initial_state;
            State final_state;
            auto actions = simulate_tree_round(
                round,
                pruning_thresh,
                gamma,
                lapse_rate,
                max_steps,
                max_depth,
                max_expansions,
                random_tie_break,
                rng,
                &final_state
            );

            for (size_t step = 0; step < actions.size(); ++step) {
                int before = static_cast<int>(conflict_edges(round, state).size());
                std::string before_text = state_text(state);
                state = apply_action(state, actions[step]);
                int after = static_cast<int>(conflict_edges(round, state).size());
                int final_conflicts = static_cast<int>(conflict_edges(round, final_state).size());
                std::cout
                    << round.round_id << ','
                    << step << ','
                    << '"' << before_text << '"' << ','
                    << actions[step].region << ','
                    << actions[step].old_color << ','
                    << actions[step].new_color << ','
                    << before << ','
                    << after << ','
                    << final_conflicts << '\n';
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << "tree_simulate error: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
