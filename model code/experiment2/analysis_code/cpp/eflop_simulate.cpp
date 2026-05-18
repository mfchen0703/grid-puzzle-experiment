#include <algorithm>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using State = std::vector<int>;

struct Action {
    int node = -1;
    int old_color = -1;
    int new_color = -1;
    std::string module;
};

struct RoundData {
    int round_id = 0;
    int n_regions = 0;
    std::vector<std::vector<int>> adj;
    State initial_state;
};

struct Trace {
    State final_state;
    std::vector<Action> actions;
};

struct ObsTask {
    int obs_index = 0;
    int round_id = 0;
    State state;
    int target_region = -1;
    int target_new_color = -1;
};

static std::string state_text(const State& state) {
    std::ostringstream out;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i) out << ' ';
        out << state[i];
    }
    return out.str();
}

static int count_conflicts(const RoundData& round, const State& state) {
    int out = 0;
    for (int u = 0; u < round.n_regions; ++u) {
        for (int v : round.adj[u]) {
            if (u < v && state[u] == state[v]) ++out;
        }
    }
    return out;
}

static std::vector<int> conflict_nodes(const RoundData& round, const State& state) {
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

static bool legal_color(const RoundData& round, const State& state, int node, int color) {
    for (int neighbor : round.adj[node]) {
        if (state[neighbor] == color) return false;
    }
    return true;
}

static std::vector<int> legal_colors_for_node(
    const RoundData& round,
    const State& state,
    int node
) {
    std::vector<int> out;
    for (int color = 0; color < 4; ++color) {
        if (legal_color(round, state, node, color)) out.push_back(color);
    }
    return out;
}

template <class T>
static T random_choice(std::mt19937& rng, const std::vector<T>& items) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(items.size()) - 1);
    return items[pick(rng)];
}

static Trace run_min_conflicts(
    const RoundData& round,
    const State& start,
    int max_steps,
    int stuck_window,
    std::mt19937& rng,
    bool& stuck,
    bool& solved
) {
    State current = start;
    std::vector<Action> actions;
    State best_state = current;
    int best_conflicts = count_conflicts(round, current);
    std::vector<Action> best_actions;
    int steps_since_improvement = 0;
    std::vector<int> history;
    stuck = false;
    solved = false;

    for (int step = 0; step < max_steps; ++step) {
        int before = count_conflicts(round, current);
        if (before == 0) {
            solved = true;
            return Trace{current, actions};
        }

        if (before < best_conflicts) {
            best_state = current;
            best_conflicts = before;
            best_actions = actions;
            steps_since_improvement = 0;
        } else {
            ++steps_since_improvement;
        }

        auto conflicts = conflict_nodes(round, current);
        if (conflicts.empty()) {
            solved = true;
            return Trace{current, actions};
        }

        std::vector<int> legal_conflict_nodes;
        for (int node : conflicts) {
            for (int color : legal_colors_for_node(round, current, node)) {
                if (color != current[node]) {
                    legal_conflict_nodes.push_back(node);
                    break;
                }
            }
        }

        if (legal_conflict_nodes.empty()) {
            stuck = true;
            return Trace{best_state, best_actions};
        }

        int node = random_choice(rng, legal_conflict_nodes);
        int old_color = current[node];
        std::vector<int> best_colors;
        int best_child_conflicts = -1;
        for (int color : legal_colors_for_node(round, current, node)) {
            if (color == old_color) continue;
            Action action{node, old_color, color, "min_conflicts"};
            State child = apply_action(current, action);
            int child_conflicts = count_conflicts(round, child);
            if (best_child_conflicts < 0 || child_conflicts < best_child_conflicts) {
                best_child_conflicts = child_conflicts;
                best_colors = {color};
            } else if (child_conflicts == best_child_conflicts) {
                best_colors.push_back(color);
            }
        }

        if (best_colors.empty()) {
            stuck = true;
            return Trace{best_state, best_actions};
        }

        int chosen_color = random_choice(rng, best_colors);
        Action action{node, old_color, chosen_color, "min_conflicts"};
        current = apply_action(current, action);
        actions.push_back(action);

        int after = count_conflicts(round, current);
        history.push_back(after);
        if (static_cast<int>(history.size()) > stuck_window) {
            history.erase(history.begin());
        }

        if (after < best_conflicts) {
            best_state = current;
            best_conflicts = after;
            best_actions = actions;
            steps_since_improvement = 0;
        }

        bool history_stuck = false;
        if (static_cast<int>(history.size()) >= stuck_window) {
            int min_history = *std::min_element(history.begin(), history.end());
            history_stuck = min_history >= best_conflicts;
        }
        if (steps_since_improvement >= stuck_window || history_stuck) {
            stuck = true;
            return Trace{best_state, best_actions};
        }
    }

    stuck = true;
    solved = count_conflicts(round, best_state) == 0;
    return Trace{best_state, best_actions};
}

static Trace run_eflop(
    const RoundData& round,
    const State& start,
    int max_depth,
    int max_visits_per_node,
    std::mt19937& rng
) {
    State current = start;
    std::vector<Action> actions;
    auto conflicts = conflict_nodes(round, current);
    if (conflicts.empty()) return Trace{current, actions};

    int v_start = random_choice(rng, conflicts);
    int old_color = current[v_start];
    std::vector<int> candidate_colors;
    for (int color = 0; color < 4; ++color) {
        if (color != old_color) candidate_colors.push_back(color);
    }
    if (candidate_colors.empty()) return Trace{current, actions};

    int seed_color = random_choice(rng, candidate_colors);
    Action seed_action{v_start, old_color, seed_color, "eflop"};
    current = apply_action(current, seed_action);
    actions.push_back(seed_action);

    std::deque<std::tuple<int, int, int>> queue;
    for (int neighbor : round.adj[v_start]) {
        if (current[neighbor] == current[v_start]) {
            queue.emplace_back(neighbor, v_start, 1);
        }
    }

    std::unordered_map<int, int> visited_count;
    while (!queue.empty()) {
        auto [node, source, depth] = queue.front();
        queue.pop_front();
        if (depth > max_depth) continue;
        if (visited_count[node] >= max_visits_per_node) continue;
        visited_count[node] += 1;

        int node_old_color = current[node];
        std::vector<int> best_colors;
        int best_score = -1;
        for (int color = 0; color < 4; ++color) {
            if (color == node_old_color) continue;
            Action action{node, node_old_color, color, "eflop"};
            State child = apply_action(current, action);
            int local_conflicts = 0;
            for (int neighbor : round.adj[node]) {
                if (neighbor != source && child[neighbor] == child[node]) {
                    ++local_conflicts;
                }
            }
            if (best_score < 0 || local_conflicts < best_score) {
                best_score = local_conflicts;
                best_colors = {color};
            } else if (local_conflicts == best_score) {
                best_colors.push_back(color);
            }
        }

        if (best_colors.empty()) continue;
        int chosen_color = random_choice(rng, best_colors);
        Action action{node, node_old_color, chosen_color, "eflop"};
        current = apply_action(current, action);
        actions.push_back(action);

        for (int neighbor : round.adj[node]) {
            if (neighbor == source) continue;
            if (current[neighbor] == current[node]) {
                queue.emplace_back(neighbor, node, depth + 1);
            }
        }
    }
    return Trace{current, actions};
}

static Trace simulate_eflop_repair(
    const RoundData& round,
    int max_outer_loops,
    int max_min_conflicts_steps,
    int max_eflop_retries,
    std::mt19937& rng
) {
    State current = round.initial_state;
    std::vector<Action> actions;

    for (int outer = 0; outer < max_outer_loops; ++outer) {
        if (count_conflicts(round, current) == 0) break;

        bool stuck = false;
        bool solved = false;
        Trace mc = run_min_conflicts(round, current, max_min_conflicts_steps, 20, rng, stuck, solved);
        actions.insert(actions.end(), mc.actions.begin(), mc.actions.end());
        current = mc.final_state;
        if (solved || count_conflicts(round, current) == 0) break;

        int current_conflicts = count_conflicts(round, current);
        bool accepted = false;
        for (int retry = 0; retry < max_eflop_retries; ++retry) {
            Trace ef = run_eflop(round, current, 5, 2, rng);
            bool stuck2 = false;
            bool solved2 = false;
            Trace mc2 = run_min_conflicts(round, ef.final_state, max_min_conflicts_steps, 20, rng, stuck2, solved2);
            int temp_conflicts = count_conflicts(round, mc2.final_state);
            if (temp_conflicts < current_conflicts || solved2) {
                actions.insert(actions.end(), ef.actions.begin(), ef.actions.end());
                actions.insert(actions.end(), mc2.actions.begin(), mc2.actions.end());
                current = mc2.final_state;
                accepted = true;
                break;
            }
        }
        if (!accepted) break;
    }
    return Trace{current, actions};
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

static Action sample_next_repair_action(
    const RoundData& round,
    const State& state,
    int max_min_conflicts_steps,
    int max_eflop_retries,
    std::mt19937& rng,
    bool& has_action
) {
    has_action = false;
    if (count_conflicts(round, state) == 0) return Action{};
    bool stuck = false;
    bool solved = false;
    Trace mc = run_min_conflicts(round, state, max_min_conflicts_steps, 20, rng, stuck, solved);
    if (!mc.actions.empty()) {
        has_action = true;
        return mc.actions.front();
    }
    if (solved || count_conflicts(round, mc.final_state) == 0) return Action{};

    int current_conflicts = count_conflicts(round, mc.final_state);
    for (int retry = 0; retry < max_eflop_retries; ++retry) {
        Trace ef = run_eflop(round, mc.final_state, 5, 2, rng);
        bool stuck2 = false;
        bool solved2 = false;
        Trace mc2 = run_min_conflicts(round, ef.final_state, max_min_conflicts_steps, 20, rng, stuck2, solved2);
        int temp_conflicts = count_conflicts(round, mc2.final_state);
        if (temp_conflicts < current_conflicts || solved2) {
            if (!ef.actions.empty()) {
                has_action = true;
                return ef.actions.front();
            }
            if (!mc2.actions.empty()) {
                has_action = true;
                return mc2.actions.front();
            }
            return Action{};
        }
    }
    return Action{};
}

static double ibs_action_nll(
    const ObsTask& task,
    const std::unordered_map<int, RoundData>& rounds,
    int ibs_samples,
    int base_seed,
    int max_min_conflicts_steps,
    int max_eflop_retries,
    int max_tries
) {
    const RoundData& round = rounds.at(task.round_id);
    double nll = 0.0;
    int times_left = ibs_samples;
    int tries = 0;
    while (times_left > 0 && tries < max_tries) {
        ++tries;
        int seed = base_seed + task.obs_index * 1000003 + tries;
        std::mt19937 rng(static_cast<unsigned int>(seed));
        bool has_action = false;
        Action sampled = sample_next_repair_action(
            round,
            task.state,
            max_min_conflicts_steps,
            max_eflop_retries,
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

static int run_ibs_mode(int argc, char** argv) {
    if (argc != 9) {
        std::cerr
            << "Usage: eflop_simulate --ibs TASKS_FILE ibs_samples base_seed "
            << "max_min_conflicts_steps max_eflop_retries max_tries n_workers\n";
        return 2;
    }
    const std::string task_path = argv[2];
    const int ibs_samples = std::stoi(argv[3]);
    const int base_seed = std::stoi(argv[4]);
    const int max_min_conflicts_steps = std::stoi(argv[5]);
    const int max_eflop_retries = std::stoi(argv[6]);
    const int max_tries = std::stoi(argv[7]);
    const int n_workers = std::max(1, std::stoi(argv[8]));

    std::unordered_map<int, RoundData> rounds;
    std::vector<ObsTask> tasks;
    read_ibs_tasks(task_path, rounds, tasks);

    std::vector<double> results(tasks.size(), 0.0);
    int workers = std::min<int>(n_workers, std::max<int>(1, tasks.size()));
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        futures.push_back(std::async(std::launch::async, [&, worker]() {
            for (size_t i = worker; i < tasks.size(); i += workers) {
                results[i] = ibs_action_nll(
                    tasks[i],
                    rounds,
                    ibs_samples,
                    base_seed,
                    max_min_conflicts_steps,
                    max_eflop_retries,
                    max_tries
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
            std::cerr << "eflop_simulate --ibs error: " << exc.what() << "\n";
            return 1;
        }
    }
    if (argc != 7) {
        std::cerr
            << "Usage: eflop_simulate ROUNDS_FILE max_outer_loops "
            << "max_min_conflicts_steps max_eflop_retries seed round_limit\n";
        return 2;
    }

    try {
        const std::string rounds_path = argv[1];
        const int max_outer_loops = std::stoi(argv[2]);
        const int max_min_conflicts_steps = std::stoi(argv[3]);
        const int max_eflop_retries = std::stoi(argv[4]);
        const int seed = std::stoi(argv[5]);
        const int round_limit = std::stoi(argv[6]);
        auto rounds = read_rounds(rounds_path);
        if (round_limit > 0 && round_limit < static_cast<int>(rounds.size())) {
            rounds.resize(round_limit);
        }
        std::mt19937 rng(static_cast<unsigned int>(seed));
        std::cout
            << "agent,round,agent_step,module,state_before,region,old_color,new_color,"
            << "n_conflict_edges_before,n_conflict_edges_after,final_conflicts,success,random_seed\n";
        for (const RoundData& round : rounds) {
            State state = round.initial_state;
            Trace trace = simulate_eflop_repair(
                round,
                max_outer_loops,
                max_min_conflicts_steps,
                max_eflop_retries,
                rng
            );
            int final_conflicts = count_conflicts(round, trace.final_state);
            for (size_t step = 0; step < trace.actions.size(); ++step) {
                int before = count_conflicts(round, state);
                std::string before_text = state_text(state);
                state = apply_action(state, trace.actions[step]);
                int after = count_conflicts(round, state);
                std::cout
                    << "eflop,"
                    << round.round_id << ','
                    << step << ','
                    << trace.actions[step].module << ','
                    << '"' << before_text << '"' << ','
                    << trace.actions[step].node << ','
                    << trace.actions[step].old_color << ','
                    << trace.actions[step].new_color << ','
                    << before << ','
                    << after << ','
                    << final_conflicts << ','
                    << (final_conflicts == 0 ? 1 : 0) << ','
                    << seed << '\n';
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << "eflop_simulate error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
