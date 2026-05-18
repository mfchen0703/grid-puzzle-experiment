#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct RoundData {
    int round_id = 0;
    int n_regions = 0;
    std::vector<int> initial_state;
    std::vector<std::pair<int, int>> edges;
};

struct ObservedAction {
    int obs_index = 0;
    int round_id = 0;
    int agent_step = 0;
    std::vector<int> state_before;
    int region = -1;
    int new_color = -1;
};

struct ModelScore {
    std::string model;
    double nll = 0.0;
};

struct Options {
    std::filesystem::path rounds_path;
    std::filesystem::path observed_actions_path;
    std::filesystem::path output_prefix;
    std::string simulate_agent;
    int round_limit = 0;
    int random_seed = 1;

    double tree_pruning = 2.0;
    double tree_gamma = 0.1;
    double tree_lapse = 0.05;
    double hsp2_pruning = 2.0;
    double hsp2_gamma = 0.1;
    double hsp2_lapse = 0.05;

    int max_agent_steps = 50;
    int max_depth = 8;
    int max_expansions = 1000;
    int max_outer_loops = 50;
    int max_min_conflicts_steps = 200;
    int max_eflop_retries = 5;

    int ibs_samples = 5;
    int base_seed = 10;
    int ibs_max_tries = 100;
    int n_workers = 8;
    int tree_random_tie_break = 1;
    std::string hsp2_likelihood_mode = "exact";
    bool keep_task_file = false;
};

static std::string shell_quote(const std::filesystem::path& path) {
    std::string s = path.string();
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string shell_quote_text(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string run_capture(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("Failed to run command: " + command);
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Command failed with status " + std::to_string(status) + ": " + command);
    }
    return output;
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

static std::string join_state(const std::vector<int>& state) {
    std::ostringstream out;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i) out << ' ';
        out << state[i];
    }
    return out.str();
}

static std::vector<int> parse_state(const std::string& text) {
    std::vector<int> state;
    std::istringstream in(text);
    int value = 0;
    while (in >> value) state.push_back(value);
    return state;
}

static std::string format_double(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

static std::vector<RoundData> read_rounds(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open rounds file: " + path.string());
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
        round.edges.reserve(n_edges);
        for (int e = 0; e < n_edges; ++e) {
            int u = 0, v = 0;
            in >> tag >> u >> v;
            if (tag != "EDGE") throw std::runtime_error("Expected EDGE row.");
            round.edges.emplace_back(u, v);
        }
        rounds.push_back(std::move(round));
    }
    return rounds;
}

static void write_rounds(
    const std::filesystem::path& path,
    const std::vector<RoundData>& rounds,
    int round_limit
) {
    int n = static_cast<int>(rounds.size());
    if (round_limit > 0) n = std::min(n, round_limit);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not write rounds file: " + path.string());
    out << "ROUNDS " << n << "\n";
    for (int i = 0; i < n; ++i) {
        const RoundData& round = rounds[i];
        out << "ROUND " << round.round_id << " " << round.n_regions << " " << round.edges.size() << "\n";
        out << "INIT";
        for (int color : round.initial_state) out << " " << color;
        out << "\n";
        for (auto [u, v] : round.edges) out << "EDGE " << u << " " << v << "\n";
    }
}

static std::vector<ObservedAction> read_observed_actions(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open observed actions: " + path.string());
    std::string header_line;
    if (!std::getline(in, header_line)) throw std::runtime_error("Observed action CSV is empty.");
    if (!header_line.empty() && header_line.back() == '\r') header_line.pop_back();
    auto headers = split_csv_line(header_line);
    std::map<std::string, int> col;
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) col[headers[i]] = i;
    for (const std::string& required : {"round", "state_before", "region", "new_color"}) {
        if (col.find(required) == col.end()) {
            throw std::runtime_error("Observed action CSV missing required column: " + required);
        }
    }

    std::vector<ObservedAction> actions;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto fields = split_csv_line(line);
        if (fields.size() < headers.size()) continue;
        ObservedAction action;
        action.obs_index = static_cast<int>(actions.size());
        action.round_id = std::stoi(fields[col["round"]]);
        if (col.find("agent_step") != col.end()) {
            action.agent_step = std::stoi(fields[col["agent_step"]]);
        } else {
            action.agent_step = action.obs_index;
        }
        action.state_before = parse_state(fields[col["state_before"]]);
        action.region = std::stoi(fields[col["region"]]);
        action.new_color = std::stoi(fields[col["new_color"]]);
        actions.push_back(std::move(action));
    }
    std::sort(actions.begin(), actions.end(), [](const ObservedAction& a, const ObservedAction& b) {
        return std::tie(a.round_id, a.agent_step, a.obs_index) <
            std::tie(b.round_id, b.agent_step, b.obs_index);
    });
    for (int i = 0; i < static_cast<int>(actions.size()); ++i) actions[i].obs_index = i;
    return actions;
}

static void write_task_file(
    const std::filesystem::path& path,
    const std::vector<RoundData>& rounds,
    const std::vector<ObservedAction>& actions
) {
    std::map<int, RoundData> round_by_id;
    for (const RoundData& round : rounds) round_by_id[round.round_id] = round;
    std::set<int> round_ids;
    for (const ObservedAction& action : actions) round_ids.insert(action.round_id);

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not write task file: " + path.string());
    out << "ROUNDS " << round_ids.size() << "\n";
    for (int round_id : round_ids) {
        auto it = round_by_id.find(round_id);
        if (it == round_by_id.end()) throw std::runtime_error("Round not found in compact rounds: " + std::to_string(round_id));
        const RoundData& round = it->second;
        out << "ROUND " << round.round_id << " " << round.n_regions << " " << round.edges.size() << "\n";
        for (auto [u, v] : round.edges) out << "EDGE " << u << " " << v << "\n";
    }
    out << "OBS " << actions.size() << "\n";
    for (const ObservedAction& action : actions) {
        out << "OBSROW "
            << action.obs_index << " "
            << action.round_id << " "
            << action.state_before.size() << " "
            << join_state(action.state_before) << " "
            << action.region << " "
            << action.new_color << "\n";
    }
}

static std::filesystem::path binary_dir_from_argv0(const char* argv0) {
    std::filesystem::path p = std::filesystem::absolute(argv0);
    return p.parent_path();
}

static std::filesystem::path ensure_limited_rounds_file(
    const Options& options,
    const std::vector<RoundData>& rounds
) {
    if (options.round_limit <= 0) return options.rounds_path;
    std::filesystem::path path = options.output_prefix;
    path += "_rounds_limited.txt";
    write_rounds(path, rounds, options.round_limit);
    return path;
}

static std::filesystem::path simulate_observed_actions(
    const Options& options,
    const std::filesystem::path& bin_dir,
    const std::filesystem::path& rounds_for_sim
) {
    if (options.simulate_agent.empty()) return options.observed_actions_path;
    std::filesystem::path observed_path = options.output_prefix;
    observed_path += "_observed_actions.csv";

    std::ostringstream cmd;
    if (options.simulate_agent == "tree") {
        cmd << shell_quote(bin_dir / "tree_simulate") << " "
            << shell_quote(rounds_for_sim) << " "
            << format_double(options.tree_pruning) << " "
            << format_double(options.tree_gamma) << " "
            << format_double(options.tree_lapse) << " "
            << options.max_agent_steps << " "
            << options.max_depth << " "
            << options.max_expansions << " "
            << options.random_seed << " "
            << options.tree_random_tie_break;
    } else if (options.simulate_agent == "hsp2") {
        cmd << shell_quote(bin_dir / "hsp2_ibs_fast") << " --simulate "
            << shell_quote(rounds_for_sim) << " "
            << format_double(options.hsp2_pruning) << " "
            << format_double(options.hsp2_gamma) << " "
            << format_double(options.hsp2_lapse) << " "
            << options.max_agent_steps << " "
            << options.max_depth << " "
            << options.max_expansions << " "
            << options.random_seed << " "
            << 0;
    } else if (options.simulate_agent == "eflop") {
        cmd << shell_quote(bin_dir / "eflop_simulate") << " "
            << shell_quote(rounds_for_sim) << " "
            << options.max_outer_loops << " "
            << options.max_min_conflicts_steps << " "
            << options.max_eflop_retries << " "
            << options.random_seed << " "
            << 0;
    } else {
        throw std::runtime_error("Unknown --simulate-agent: " + options.simulate_agent);
    }

    std::string csv = run_capture(cmd.str());
    std::ofstream out(observed_path);
    if (!out) throw std::runtime_error("Could not write observed action CSV: " + observed_path.string());
    out << csv;
    return observed_path;
}

static double sum_nll_output(const std::string& text) {
    std::istringstream in(text);
    int obs_index = 0;
    double value = 0.0;
    double total = 0.0;
    while (in >> obs_index >> value) total += value;
    return total;
}

static std::vector<ModelScore> score_models(
    const Options& options,
    const std::filesystem::path& bin_dir,
    const std::filesystem::path& task_path
) {
    std::vector<ModelScore> scores;

    {
        std::ostringstream cmd;
        cmd << shell_quote(bin_dir / "tree_simulate") << " --ibs "
            << shell_quote(task_path) << " "
            << format_double(options.tree_pruning) << " "
            << format_double(options.tree_gamma) << " "
            << format_double(options.tree_lapse) << " "
            << options.ibs_samples << " "
            << options.base_seed << " "
            << options.max_depth << " "
            << options.max_expansions << " "
            << options.ibs_max_tries << " "
            << options.n_workers << " "
            << options.tree_random_tie_break;
        scores.push_back({"tree", sum_nll_output(run_capture(cmd.str()))});
    }

    {
        std::ostringstream cmd;
        cmd << shell_quote(bin_dir / "hsp2_ibs_fast") << " "
            << shell_quote(task_path) << " "
            << format_double(options.hsp2_pruning) << " "
            << format_double(options.hsp2_gamma) << " "
            << format_double(options.hsp2_lapse) << " "
            << options.ibs_samples << " "
            << options.base_seed << " "
            << options.max_depth << " "
            << options.max_expansions << " "
            << options.ibs_max_tries << " "
            << options.n_workers << " "
            << shell_quote_text(options.hsp2_likelihood_mode);
        scores.push_back({"hsp2", sum_nll_output(run_capture(cmd.str()))});
    }

    {
        std::ostringstream cmd;
        cmd << shell_quote(bin_dir / "eflop_simulate") << " --ibs "
            << shell_quote(task_path) << " "
            << options.ibs_samples << " "
            << options.base_seed << " "
            << options.max_min_conflicts_steps << " "
            << options.max_eflop_retries << " "
            << options.ibs_max_tries << " "
            << options.n_workers;
        scores.push_back({"eflop", sum_nll_output(run_capture(cmd.str()))});
    }

    std::sort(scores.begin(), scores.end(), [](const ModelScore& a, const ModelScore& b) {
        return std::tie(a.nll, a.model) < std::tie(b.nll, b.model);
    });
    return scores;
}

static void write_scores_csv(
    const std::filesystem::path& path,
    const std::vector<ModelScore>& scores,
    int n_actions
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not write scores CSV: " + path.string());
    out << "model,nll,n_actions\n";
    out << std::setprecision(12);
    for (const ModelScore& score : scores) {
        out << score.model << "," << score.nll << "," << n_actions << "\n";
    }
}

static void write_summary_json(
    const std::filesystem::path& path,
    const Options& options,
    const std::filesystem::path& observed_path,
    const std::filesystem::path& scores_path,
    const std::vector<ModelScore>& scores,
    int n_actions
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not write summary JSON: " + path.string());
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"true_agent\": \"" << (options.simulate_agent.empty() ? "" : options.simulate_agent) << "\",\n";
    out << "  \"predicted_agent\": \"" << (scores.empty() ? "" : scores.front().model) << "\",\n";
    out << "  \"n_actions\": " << n_actions << ",\n";
    out << "  \"observed_path\": \"" << observed_path.string() << "\",\n";
    out << "  \"scores_path\": \"" << scores_path.string() << "\",\n";
    out << "  \"scores\": [\n";
    for (size_t i = 0; i < scores.size(); ++i) {
        out << "    {\"model\": \"" << scores[i].model << "\", \"nll\": " << scores[i].nll << "}";
        if (i + 1 < scores.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

static Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (key == "--rounds") options.rounds_path = need_value(key);
        else if (key == "--observed-actions") options.observed_actions_path = need_value(key);
        else if (key == "--output-prefix") options.output_prefix = need_value(key);
        else if (key == "--simulate-agent") options.simulate_agent = need_value(key);
        else if (key == "--round-limit") options.round_limit = std::stoi(need_value(key));
        else if (key == "--random-seed") options.random_seed = std::stoi(need_value(key));
        else if (key == "--tree-pruning") options.tree_pruning = std::stod(need_value(key));
        else if (key == "--tree-gamma") options.tree_gamma = std::stod(need_value(key));
        else if (key == "--tree-lapse") options.tree_lapse = std::stod(need_value(key));
        else if (key == "--hsp2-pruning") options.hsp2_pruning = std::stod(need_value(key));
        else if (key == "--hsp2-gamma") options.hsp2_gamma = std::stod(need_value(key));
        else if (key == "--hsp2-lapse") options.hsp2_lapse = std::stod(need_value(key));
        else if (key == "--max-agent-steps") options.max_agent_steps = std::stoi(need_value(key));
        else if (key == "--max-depth") options.max_depth = std::stoi(need_value(key));
        else if (key == "--max-expansions") options.max_expansions = std::stoi(need_value(key));
        else if (key == "--max-outer-loops") options.max_outer_loops = std::stoi(need_value(key));
        else if (key == "--max-min-conflicts-steps") options.max_min_conflicts_steps = std::stoi(need_value(key));
        else if (key == "--max-eflop-retries") options.max_eflop_retries = std::stoi(need_value(key));
        else if (key == "--ibs-samples") options.ibs_samples = std::stoi(need_value(key));
        else if (key == "--base-seed") options.base_seed = std::stoi(need_value(key));
        else if (key == "--ibs-max-tries") options.ibs_max_tries = std::stoi(need_value(key));
        else if (key == "--n-workers") options.n_workers = std::stoi(need_value(key));
        else if (key == "--tree-random-tie-break") options.tree_random_tie_break = std::stoi(need_value(key));
        else if (key == "--hsp2-likelihood-mode") options.hsp2_likelihood_mode = need_value(key);
        else if (key == "--keep-task-file") options.keep_task_file = true;
        else if (key == "--help") {
            std::cout
                << "Usage: cpp_model_recovery --rounds ROUNDS_FILE --output-prefix PREFIX "
                << "[--observed-actions CSV | --simulate-agent tree|hsp2|eflop] [options]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (options.rounds_path.empty()) throw std::runtime_error("--rounds is required");
    if (options.output_prefix.empty()) throw std::runtime_error("--output-prefix is required");
    if (options.simulate_agent.empty() && options.observed_actions_path.empty()) {
        throw std::runtime_error("Provide --observed-actions or --simulate-agent");
    }
    if (!options.simulate_agent.empty() && !options.observed_actions_path.empty()) {
        throw std::runtime_error("Use either --observed-actions or --simulate-agent, not both");
    }
    if (options.hsp2_likelihood_mode != "ibs" && options.hsp2_likelihood_mode != "exact") {
        throw std::runtime_error("--hsp2-likelihood-mode must be ibs or exact");
    }
    return options;
}

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        std::filesystem::path parent = options.output_prefix.parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);

        auto rounds = read_rounds(options.rounds_path);
        std::filesystem::path bin_dir = binary_dir_from_argv0(argv[0]);
        std::filesystem::path rounds_for_sim = ensure_limited_rounds_file(options, rounds);
        std::filesystem::path observed_path = simulate_observed_actions(options, bin_dir, rounds_for_sim);
        auto observed = read_observed_actions(observed_path);
        if (observed.empty()) throw std::runtime_error("No observed actions to recover.");

        std::filesystem::path task_path = options.output_prefix;
        task_path += "_tasks.txt";
        write_task_file(task_path, rounds, observed);

        auto scores = score_models(options, bin_dir, task_path);

        std::filesystem::path scores_path = options.output_prefix;
        scores_path += "_scores.csv";
        std::filesystem::path summary_path = options.output_prefix;
        summary_path += "_summary.json";
        write_scores_csv(scores_path, scores, static_cast<int>(observed.size()));
        write_summary_json(summary_path, options, observed_path, scores_path, scores, static_cast<int>(observed.size()));

        if (!options.keep_task_file) std::filesystem::remove(task_path);

        std::cout
            << "{\n"
            << "  \"true_agent\": \"" << (options.simulate_agent.empty() ? "" : options.simulate_agent) << "\",\n"
            << "  \"predicted_agent\": \"" << scores.front().model << "\",\n"
            << "  \"n_actions\": " << observed.size() << ",\n"
            << "  \"scores_path\": \"" << scores_path.string() << "\",\n"
            << "  \"summary_path\": \"" << summary_path.string() << "\"\n"
            << "}\n";
    } catch (const std::exception& exc) {
        std::cerr << "cpp_model_recovery error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
