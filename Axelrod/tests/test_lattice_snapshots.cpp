#include "graph.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct SnapshotConfig {
    int num_nodes = 900;          // 30 x 30 lattice
    int lattice_radius = 1;       // Manhattan radius
    int num_features = 5;         // cultural features F
    std::vector<int> qs{5,10,15,20,21,22,23,24,25,30}; // feature dimension q values
    int max_steps = 10000;         // number of Axelrod sweeps to run
    int min_steps = 5000;         // do not stop before this many sweeps
    int save_interval = 20;       // save every N sweeps for smoother animation
    double stability_tolerance = 5e-4;
    int stability_window = 10;    // require this many consecutive close values to stop early
    std::string out_dir = "data/lattice_snapshots";
};

void write_file_header(std::ofstream& snapshot_out,
                       std::ofstream& frag_out,
                       const SnapshotConfig& cfg,
                       int q) {
    snapshot_out << "# 2D lattice snapshot stream\n";
    snapshot_out << "# q " << q
                 << " num_nodes " << cfg.num_nodes
                 << " lattice_radius " << cfg.lattice_radius
                 << " num_features " << cfg.num_features
                 << " max_steps " << cfg.max_steps
                 << " save_interval " << cfg.save_interval << "\n";
    snapshot_out << "# columns: step node x y f1 f2 ... fF\n";

    frag_out << "# fragmentation series for q=" << q << "\n";
    frag_out << "# columns: step fragmentation\n";
}

void save_positions_once(const LatticeGraph& g, const SnapshotConfig& cfg) {
    fs::create_directories(cfg.out_dir);
    std::string pos_path = cfg.out_dir + "/lattice_positions.txt";
    if (fs::exists(pos_path)) {
        return;
    }
    std::ofstream pos_out(pos_path);
    if (!pos_out.is_open()) {
        std::cerr << "Could not open " << pos_path << " for writing positions\n";
        return;
    }
    g.save_positions(pos_out);
    std::cout << "Saved lattice positions to " << pos_path << "\n";
}

void run_snapshot_experiment(int q, const SnapshotConfig& cfg) {
    fs::create_directories(cfg.out_dir);
    std::string snapshot_path = cfg.out_dir + "/lattice_q" + std::to_string(q) + "_snapshots.txt";
    std::string frag_path = cfg.out_dir + "/lattice_q" + std::to_string(q) + "_fragmentation.txt";

    std::ofstream snapshot_out(snapshot_path);
    std::ofstream frag_out(frag_path);
    if (!snapshot_out.is_open() || !frag_out.is_open()) {
        std::cerr << "Failed to open output files for q=" << q << "\n";
        return;
    }
    snapshot_out << std::fixed;
    frag_out << std::fixed << std::setprecision(6);

    LatticeGraph g(cfg.num_nodes, cfg.lattice_radius, cfg.num_features, q);
    save_positions_once(g, cfg);
    write_file_header(snapshot_out, frag_out, cfg, q);

    auto record_state = [&](int step) -> double {
        double frag_val = g.fragmentation_index();
        frag_out << step << " " << frag_val << "\n";
        g.save_snapshot(snapshot_out, step);
        return frag_val;
    };

    double last_frag = record_state(0);
    int stable_counter = 0;

    for (int step = 1; step <= cfg.max_steps; ++step) {
        g.axelrod_step();
        if (step % cfg.save_interval == 0 || step == cfg.max_steps) {
            double frag_val = record_state(step);
            if (std::abs(frag_val - last_frag) < cfg.stability_tolerance) {
                stable_counter += 1;
            } else {
                stable_counter = 0;
            }
            last_frag = frag_val;

            if (step >= cfg.min_steps && stable_counter >= cfg.stability_window) {
                std::cout << "q=" << q << " appeared stable after " << step
                          << " steps ("
                          << stable_counter << " consecutive near-constant snapshots)\n";
                break;
            }
        }
    }

    std::cout << "Saved snapshots to " << snapshot_path
              << " and fragmentation series to " << frag_path << "\n";
}

int main() {
    SnapshotConfig cfg;
    std::cout << "Running lattice snapshot experiments on q values: ";
    for (size_t i = 0; i < cfg.qs.size(); ++i) {
        std::cout << cfg.qs[i] << (i + 1 == cfg.qs.size() ? "\n" : ", ");
    }

    for (int q : cfg.qs) {
        run_snapshot_experiment(q, cfg);
    }
    std::cout << "Done.\n";
    return 0;
}
