#include "graph.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

enum class NetworkType { SmallWorld, Lattice2D };

struct SimulationConfig {
    NetworkType network = NetworkType::SmallWorld;
    int num_nodes = 600;
    int neighbors_per_node = 10;
    double rewiring_prob = 0.1;
    int lattice_radius = 1;
    int num_features = 5;
    int feature_dim = 3;
    int num_interactions = 100000;
    int num_runs = 20;
    std::string data_root = "data";
};

struct RunResult {
    int num_distinct_cultures = 0;
    int largest_culture_size = 0;
    double largest_culture_fraction = 0.0;
    double avg_similarity = 0.0;
    double entropy = 0.0;
    double fragmentation = 0.0;
    double edge_homophily = 0.0;
    double global_similarity = 0.0;
    double mean_degree = 0.0;
};

struct SweepResult {
    std::string network_type;
    std::string param_name;
    double param_value = 0.0;
    int num_distinct_cultures = 0;
    int largest_culture_size = 0;
    double largest_culture_fraction = 0.0;
    double avg_similarity = 0.0;
    double entropy = 0.0;
    double fragmentation = 0.0;
    double edge_homophily = 0.0;
    double global_similarity = 0.0;
    double mean_degree = 0.0;
};

std::string to_string(NetworkType network) {
    switch (network) {
        case NetworkType::SmallWorld: return "smallworld";
        case NetworkType::Lattice2D:  return "lattice";
    }
    return "unknown";
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --network [smallworld|lattice]   Select graph family (default: smallworld)\n"
              << "  --nodes <int>                    Number of nodes (default: 600)\n"
              << "  --neighbors <int>                Mean degree for small-world (default: 10, must be even)\n"
              << "  --rewiring <float>               Rewiring prob for small-world (default: 0.1)\n"
              << "  --radius <int>                   Manhattan radius for 2D lattice (default: 1 => 4 neighbors)\n"
              << "  --features <int>                 Number of cultural features F (default: 5)\n"
              << "  --feature-dim <int>              Feature dimension q (default: 3)\n"
              << "  --interactions <int>             Interaction steps (default: 100000)\n"
              << "  --runs <int>                     Monte Carlo runs per sweep point (default: 20)\n"
              << "  --data-root <path>               Base output folder (default: data)\n";
}

bool parse_args(int argc, char** argv, SimulationConfig& cfg) {
    auto require_value = [&](int& idx, const std::string& name) -> std::string {
        if (idx + 1 >= argc) {
            throw std::runtime_error("Missing value for " + name);
        }
        return std::string(argv[++idx]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--network") {
                std::string val = require_value(i, arg);
                if (val == "smallworld" || val == "small" || val == "sw") {
                    cfg.network = NetworkType::SmallWorld;
                } else if (val == "lattice" || val == "lattice2d" || val == "2d") {
                    cfg.network = NetworkType::Lattice2D;
                } else {
                    std::cerr << "Unknown network type: " << val << "\n";
                    return false;
                }
            } else if (arg == "--nodes") {
                cfg.num_nodes = std::stoi(require_value(i, arg));
            } else if (arg == "--neighbors") {
                cfg.neighbors_per_node = std::stoi(require_value(i, arg));
            } else if (arg == "--rewiring") {
                cfg.rewiring_prob = std::stod(require_value(i, arg));
            } else if (arg == "--radius") {
                cfg.lattice_radius = std::stoi(require_value(i, arg));
            } else if (arg == "--features") {
                cfg.num_features = std::stoi(require_value(i, arg));
            } else if (arg == "--feature-dim") {
                cfg.feature_dim = std::stoi(require_value(i, arg));
            } else if (arg == "--interactions") {
                cfg.num_interactions = std::stoi(require_value(i, arg));
            } else if (arg == "--runs") {
                cfg.num_runs = std::stoi(require_value(i, arg));
            } else if (arg == "--data-root") {
                cfg.data_root = require_value(i, arg);
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return false;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                print_usage(argv[0]);
                return false;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        print_usage(argv[0]);
        return false;
    }

    return true;
}

int nearest_perfect_square(int n) {
    if (n <= 1) {
        return 1;
    }
    int root = static_cast<int>(std::sqrt(static_cast<double>(n)));
    int lower = root * root;
    int upper = (root + 1) * (root + 1);
    if (lower == n) {
        return n;
    }
    return (n - lower <= upper - n) ? lower : upper;
}

void reset_directory(const std::string& path) {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec) {
        std::cerr << "Warning: could not clean " << path << " (" << ec.message() << ")\n";
    }
    fs::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("Could not create output directory " + path + ": " + ec.message());
    }
}

double compute_mean_degree(const Graph& g) {
    double total_deg = 0.0;
    const auto& adj = g.adjacency();
    for (const auto& row : adj) {
        total_deg += static_cast<double>(row.size());
    }
    return total_deg / static_cast<double>(g.N());
}

RunResult measure_final_state(const SimulationConfig& cfg,
                              int neighbors_per_node,
                              double rewiring_prob,
                              int lattice_radius,
                              int num_features,
                              int feature_dim) {
    std::unique_ptr<Graph> g;
    if (cfg.network == NetworkType::SmallWorld) {
        g = std::make_unique<StrogatzGraph>(cfg.num_nodes, neighbors_per_node, rewiring_prob,
                                            num_features, feature_dim);
    } else {
        g = std::make_unique<LatticeGraph>(cfg.num_nodes, lattice_radius, num_features, feature_dim);
    }

    for (int it = 0; it < cfg.num_interactions; ++it) {
        g->axelrod_interaction();
    }

    RunResult result;
    result.num_distinct_cultures    = g->count_distinct_cultures();
    result.largest_culture_size     = g->largest_culture_size();
    result.largest_culture_fraction = static_cast<double>(result.largest_culture_size) / cfg.num_nodes;
    result.avg_similarity           = g->average_similarity();
    result.entropy                  = g->entropy_cultures();
    result.fragmentation            = g->fragmentation_index();
    result.edge_homophily           = g->edge_homophily();
    result.global_similarity        = g->global_similarity();
    result.mean_degree              = compute_mean_degree(*g);

    return result;
}

template <typename T, typename Runner>
std::vector<SweepResult> sweep_parameter(const SimulationConfig& cfg,
                                         const std::string& network_name,
                                         const std::string& param_name,
                                         const std::vector<T>& values,
                                         Runner&& runner) {
    std::vector<SweepResult> results(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        results[i].network_type = network_name;
        results[i].param_name = param_name;
        results[i].param_value = static_cast<double>(values[i]);
    }

    for (int run = 0; run < cfg.num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << cfg.num_runs
                  << " for " << param_name << "\n";
        #pragma omp parallel for
        for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
            RunResult r = runner(values[idx]);
            #pragma omp critical
            {
                results[idx].num_distinct_cultures    += r.num_distinct_cultures;
                results[idx].largest_culture_size     += r.largest_culture_size;
                results[idx].largest_culture_fraction += r.largest_culture_fraction;
                results[idx].avg_similarity           += r.avg_similarity;
                results[idx].entropy                  += r.entropy;
                results[idx].fragmentation            += r.fragmentation;
                results[idx].edge_homophily           += r.edge_homophily;
                results[idx].global_similarity        += r.global_similarity;
                results[idx].mean_degree              += r.mean_degree;
                std::cout << "  " << param_name << "=" << values[idx]
                          << " -> cultures=" << r.num_distinct_cultures
                          << " largest_frac=" << r.largest_culture_fraction << "\n";
            }
        }
    }

    for (auto& r : results) {
        r.num_distinct_cultures    = static_cast<int>(r.num_distinct_cultures / static_cast<double>(cfg.num_runs) + 0.5);
        r.largest_culture_size     = static_cast<int>(r.largest_culture_size  / static_cast<double>(cfg.num_runs) + 0.5);
        r.largest_culture_fraction /= static_cast<double>(cfg.num_runs);
        r.avg_similarity           /= static_cast<double>(cfg.num_runs);
        r.entropy                  /= static_cast<double>(cfg.num_runs);
        r.fragmentation            /= static_cast<double>(cfg.num_runs);
        r.edge_homophily           /= static_cast<double>(cfg.num_runs);
        r.global_similarity        /= static_cast<double>(cfg.num_runs);
        r.mean_degree              /= static_cast<double>(cfg.num_runs);
    }

    return results;
}

void save_sweep_results(const std::string& filename,
                        const std::vector<SweepResult>& results) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }
    fprintf(f, "network_type,param_name,param_value,num_cultures,largest_culture_size,largest_culture_fraction,"
               "avg_similarity,entropy,fragmentation,edge_homophily,global_similarity,mean_degree\n");
    for (const auto& r : results) {
        fprintf(f, "%s,%s,%.10g,%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                r.network_type.c_str(), r.param_name.c_str(), r.param_value,
                r.num_distinct_cultures, r.largest_culture_size,
                r.largest_culture_fraction, r.avg_similarity, r.entropy,
                r.fragmentation, r.edge_homophily, r.global_similarity,
                r.mean_degree);
    }
    fclose(f);
    std::cout << "Saved results to " << filename << "\n";
}

int main(int argc, char** argv) {
    SimulationConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    if (cfg.network == NetworkType::SmallWorld && (cfg.neighbors_per_node % 2 != 0)) {
        std::cerr << "neighbors_per_node must be even for small-world networks.\n";
        return 1;
    }

    if (cfg.network == NetworkType::Lattice2D) {
        int adjusted_nodes = nearest_perfect_square(cfg.num_nodes);
        if (adjusted_nodes != cfg.num_nodes) {
            std::cout << "Adjusting num_nodes from " << cfg.num_nodes
                      << " to " << adjusted_nodes
                      << " to fit a square lattice.\n";
            cfg.num_nodes = adjusted_nodes;
        }
    }

    const std::string network_label = to_string(cfg.network);
    const std::string data_dir = cfg.data_root + "/" + network_label;

    try {
        reset_directory(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error preparing output directory: " << e.what() << "\n";
        return 1;
    }

    // Shared sweep grids
    std::vector<int> num_features_values = {3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30};
    std::vector<int> feature_dim_values  = {
        2, 3, 4, 5, 6, 7, 8, 10, 12, 15,
        18, 20, 25, 30, 40, 50, 60, 75, 90, 120
    };

    if (cfg.network == NetworkType::SmallWorld) {
        std::cout << "\n=== Sweeping neighbors per node (small-world) ===\n";
        std::vector<int> neighbors_values = {4, 6, 8, 10, 12, 14, 16, 18};
        auto neighbors_results = sweep_parameter(cfg, network_label, "neighbors_per_node",
            neighbors_values,
            [&](int k) {
                return measure_final_state(cfg, k, cfg.rewiring_prob,
                                           cfg.lattice_radius, cfg.num_features,
                                           cfg.feature_dim);
            });
        save_sweep_results(data_dir + "/sweep_neighbors_per_node.csv", neighbors_results);

        std::cout << "\n=== Sweeping rewiring probability (small-world) ===\n";
        std::vector<double> rewiring_values = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0};
        auto rewiring_results = sweep_parameter(cfg, network_label, "rewiring_prob",
            rewiring_values,
            [&](double p) {
                return measure_final_state(cfg, cfg.neighbors_per_node, p,
                                           cfg.lattice_radius, cfg.num_features,
                                           cfg.feature_dim);
            });
        save_sweep_results(data_dir + "/sweep_rewiring_prob.csv", rewiring_results);
    } else {
        std::cout << "\n=== Sweeping lattice radius (2D lattice) ===\n";
        std::vector<int> lattice_radius_values = {1, 2, 3, 4};
        auto radius_results = sweep_parameter(cfg, network_label, "lattice_radius",
            lattice_radius_values,
            [&](int radius) {
                return measure_final_state(cfg, cfg.neighbors_per_node, cfg.rewiring_prob,
                                           radius, cfg.num_features, cfg.feature_dim);
            });
        save_sweep_results(data_dir + "/sweep_lattice_radius.csv", radius_results);
    }

    std::cout << "\n=== Sweeping number of features ===\n";
    auto features_results = sweep_parameter(cfg, network_label, "num_features",
        num_features_values,
        [&](int f_num) {
            return measure_final_state(cfg, cfg.neighbors_per_node, cfg.rewiring_prob,
                                       cfg.lattice_radius, f_num, cfg.feature_dim);
        });
    save_sweep_results(data_dir + "/sweep_num_features.csv", features_results);

    std::cout << "\n=== Sweeping feature dimension ===\n";
    auto feature_dim_results = sweep_parameter(cfg, network_label, "feature_dim",
        feature_dim_values,
        [&](int f_dim) {
            return measure_final_state(cfg, cfg.neighbors_per_node, cfg.rewiring_prob,
                                       cfg.lattice_radius, cfg.num_features, f_dim);
        });
    save_sweep_results(data_dir + "/sweep_feature_dim.csv", feature_dim_results);

    return 0;
}
