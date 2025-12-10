#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct ConvergenceConfig {
    int feature_dim = 3;          // q = 3 ensures eventual consensus
    int default_nodes = 400;      // 20 x 20 lattice
    int default_radius = 1;
    int default_num_features = 5;
    int max_steps = 100000;         // maximum Axelrod sweeps
    double frag_threshold = 0.8; // convergence defined when fragmentation < threshold
    int runs_per_point = 30;       // average over this many seeds
    std::string out_dir = "data/lattice_convergence";
};

double steps_to_convergence(int num_nodes, int radius, int num_features, const ConvergenceConfig& cfg) {
    LatticeGraph g(num_nodes, radius, num_features, cfg.feature_dim);
    double frag = g.fragmentation_index();
    if (frag <= cfg.frag_threshold) {
        return 0.0;
    }

    for (int step = 1; step <= cfg.max_steps; ++step) {
        g.axelrod_step();
        frag = g.fragmentation_index();
        if (frag <= cfg.frag_threshold) {
            return static_cast<double>(step);
        }
    }
    return static_cast<double>(cfg.max_steps);
}

struct StatsRow {
    int sweep_value = 0;
    double mean_steps = 0.0;
    double sd_steps = 0.0;
    double mean_interactions = 0.0;
    double sd_interactions = 0.0;
};

StatsRow measure_point(int sweep_value,
                       int num_nodes,
                       int radius,
                       int num_features,
                       const ConvergenceConfig& cfg) {
    double sum = 0.0;
    double sumsq = 0.0;
    for (int run = 0; run < cfg.runs_per_point; ++run) {
        double steps = steps_to_convergence(num_nodes, radius, num_features, cfg);
        sum += steps;
        sumsq += steps * steps;
        std::cout << "  run " << (run + 1) << "/" << cfg.runs_per_point
                  << " steps=" << steps << "\n";
    }

    double mean = sum / static_cast<double>(cfg.runs_per_point);
    double var = std::max(0.0, sumsq / static_cast<double>(cfg.runs_per_point) - mean * mean);
    double sd = std::sqrt(var);

    StatsRow row;
    row.sweep_value = sweep_value;
    row.mean_steps = mean;
    row.sd_steps = sd;
    row.mean_interactions = mean * static_cast<double>(num_nodes);
    row.sd_interactions = sd * static_cast<double>(num_nodes);
    return row;
}

void write_header(std::ofstream& out,
                  const std::string& param_name,
                  const ConvergenceConfig& cfg) {
    out << "# convergence sweep for " << param_name << "\n";
    out << "# default_nodes " << cfg.default_nodes
        << " default_radius " << cfg.default_radius
        << " default_num_features " << cfg.default_num_features
        << " feature_dim " << cfg.feature_dim
        << " frag_threshold " << cfg.frag_threshold
        << " max_steps " << cfg.max_steps
        << " runs_per_point " << cfg.runs_per_point << "\n";
    out << "# columns: sweep_value mean_steps sd_steps mean_interactions sd_interactions\n";
}

void sweep_num_nodes(const std::vector<int>& values, const ConvergenceConfig& cfg) {
    std::string path = cfg.out_dir + "/convergence_num_nodes.txt";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Could not open " << path << "\n";
        return;
    }
    write_header(out, "num_nodes", cfg);

    for (int val : values) {
        std::cout << "Sweeping num_nodes=" << val << "\n";
        StatsRow row = measure_point(val, val, cfg.default_radius, cfg.default_num_features, cfg);
        out << row.sweep_value << " "
            << row.mean_steps << " " << row.sd_steps << " "
            << row.mean_interactions << " " << row.sd_interactions << "\n";
    }
    std::cout << "Saved node sweep to " << path << "\n";
}

void sweep_num_features(const std::vector<int>& values, const ConvergenceConfig& cfg) {
    std::string path = cfg.out_dir + "/convergence_num_features.txt";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Could not open " << path << "\n";
        return;
    }
    write_header(out, "num_features", cfg);

    for (int val : values) {
        std::cout << "Sweeping num_features=" << val << "\n";
        StatsRow row = measure_point(val, cfg.default_nodes, cfg.default_radius, val, cfg);
        out << row.sweep_value << " "
            << row.mean_steps << " " << row.sd_steps << " "
            << row.mean_interactions << " " << row.sd_interactions << "\n";
    }
    std::cout << "Saved feature sweep to " << path << "\n";
}

void sweep_radius(const std::vector<int>& values, const ConvergenceConfig& cfg) {
    std::string path = cfg.out_dir + "/convergence_radius.txt";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Could not open " << path << "\n";
        return;
    }
    write_header(out, "lattice_radius", cfg);

    for (int val : values) {
        std::cout << "Sweeping lattice_radius=" << val << "\n";
        StatsRow row = measure_point(val, cfg.default_nodes, val, cfg.default_num_features, cfg);
        out << row.sweep_value << " "
            << row.mean_steps << " " << row.sd_steps << " "
            << row.mean_interactions << " " << row.sd_interactions << "\n";
    }
    std::cout << "Saved radius sweep to " << path << "\n";
}

int main() {
    ConvergenceConfig cfg;
    fs::create_directories(cfg.out_dir);

    std::vector<int> node_values = {225, 400, 625, 900, 1225};       // 15^2 ... 35^2
    std::vector<int> feature_values = {2, 3, 4, 5, 6, 8, 10};
    std::vector<int> radius_values = {1, 2, 3, 4, 5, 6};

    std::cout << "Starting convergence sweeps (frag threshold "
              << cfg.frag_threshold << ")\n";
    sweep_num_nodes(node_values, cfg);
    sweep_num_features(feature_values, cfg);
    sweep_radius(radius_values, cfg);
    std::cout << "All sweeps complete. Output in " << cfg.out_dir << "\n";
    return 0;
}
