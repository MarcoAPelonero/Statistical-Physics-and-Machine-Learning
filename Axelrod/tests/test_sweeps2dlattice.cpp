#include "graph.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <functional>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// In here we JUST measure the biggest cluster size though the largest culture size method normalized, the homophily, and the fragmentation index

struct SweepLatticeResult {
    int lattice_radius;
    int num_features;
    int feature_dim;
    double largest_culture_fraction;
    double homophily;
    double fragmentation;
};

SweepLatticeResult sweep_lattice_parameters(int num_nodes, int lattice_radius,
                              int num_features, int feature_dim, int num_interactions) {
    LatticeGraph g(num_nodes, lattice_radius, num_features, feature_dim);
    for (int it = 0; it < num_interactions; ++it) {
        g.axelrod_step();
    }

    int largest_culture_size = g.largest_culture_size();
    double largest_culture_fraction = static_cast<double>(largest_culture_size) / static_cast<double>(num_nodes);
    double homophily = g.edge_homophily();
    double fragmentation = g.fragmentation_index();

    SweepLatticeResult result;
    result.lattice_radius = lattice_radius;
    result.num_features = num_features;
    result.feature_dim = feature_dim;
    result.largest_culture_fraction = largest_culture_fraction;
    result.homophily = homophily;
    result.fragmentation = fragmentation;
    return result;
}

enum class SweepParam { LatticeRadius, NumFeatures, FeatureDim, NumNodes };

void sweep_and_save(const std::string &filename,
                    SweepParam sweep_param,
                    const std::vector<int> &values,
                    int def_num_nodes,
                    int def_radius,
                    int def_num_features,
                    int def_feature_dim,
                    int num_interactions,
                    int num_sweeps) {

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }

    outfile << "#sweep_param sweep_value avg_largest_fraction sd_largest_fraction avg_homophily sd_homophily avg_fragmentation sd_fragmentation\n";
    outfile << "#defaults " << def_num_nodes << " " << def_radius << " " << def_num_features << " " << def_feature_dim << "\n";

    for (int v : values) {
        // set parameters using defaults, then override the swept one
        int num_nodes = def_num_nodes;
        int radius = def_radius;
        int num_features = def_num_features;
        int feature_dim = def_feature_dim;

        switch (sweep_param) {
            case SweepParam::LatticeRadius: radius = v; break;
            case SweepParam::NumFeatures:  num_features = v; break;
            case SweepParam::FeatureDim:   feature_dim = v; break;
            case SweepParam::NumNodes:     num_nodes = v; break;
        }

        // Decide how many OpenMP kernels (threads) to use: at most 5, or the
        // maximum available if smaller. Set that before the parallel loop.
    #ifdef _OPENMP
        {
            int max_threads = omp_get_max_threads();
            int num_kernels = std::min(max_threads, 5);
            omp_set_num_threads(num_kernels);
            std::cout << "Using " << num_kernels << " OpenMP threads (max available "
                  << max_threads << ") for sweep value " << v << "\n";
        }
    #endif

        // Efficient parallel sampling: accumulate sums and sums-of-squares with OpenMP reductions
        double sum_largest = 0.0, sum_homophily = 0.0, sum_fragmentation = 0.0;
        double sumsq_largest = 0.0, sumsq_homophily = 0.0, sumsq_fragmentation = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum_largest,sum_homophily,sum_fragmentation,sumsq_largest,sumsq_homophily,sumsq_fragmentation)
#endif
        for (int it = 0; it < num_sweeps; ++it) {
            SweepLatticeResult s = sweep_lattice_parameters(num_nodes, radius, num_features, feature_dim, num_interactions);
            sum_largest += s.largest_culture_fraction;
            sum_homophily += s.homophily;
            sum_fragmentation += s.fragmentation;
            sumsq_largest += s.largest_culture_fraction * s.largest_culture_fraction;
            sumsq_homophily += s.homophily * s.homophily;
            sumsq_fragmentation += s.fragmentation * s.fragmentation;
            std::cout << "Completed sweep " << it + 1 << "/" << num_sweeps
                      << " for param value " << v << "\n";
        }

        // compute means
        double mean_largest = sum_largest / num_sweeps;
        double mean_homophily = sum_homophily / num_sweeps;
        double mean_fragmentation = sum_fragmentation / num_sweeps;

        // compute stddev via sums-of-squares (population stddev)
        double var_largest = sumsq_largest / num_sweeps - mean_largest * mean_largest;
        double var_homophily = sumsq_homophily / num_sweeps - mean_homophily * mean_homophily;
        double var_fragmentation = sumsq_fragmentation / num_sweeps - mean_fragmentation * mean_fragmentation;
        if (var_largest < 0 && var_largest > -1e-15) var_largest = 0; // guard tiny negative rounding
        if (var_homophily < 0 && var_homophily > -1e-15) var_homophily = 0;
        if (var_fragmentation < 0 && var_fragmentation > -1e-15) var_fragmentation = 0;

        double sd_largest = std::sqrt(std::max(0.0, var_largest));
        double sd_homophily = std::sqrt(std::max(0.0, var_homophily));
        double sd_fragmentation = std::sqrt(std::max(0.0, var_fragmentation));  
        // Write row without saving the constant parameters each time
        outfile << static_cast<int>(sweep_param) << " "   // numeric code for which param (or write a string instead)
                << v << " "
                << mean_largest << " " << sd_largest / std::sqrt(num_sweeps) << " "
                << mean_homophily << " " << sd_homophily / std::sqrt(num_sweeps) << " "
                << mean_fragmentation << " " << sd_fragmentation / std::sqrt(num_sweeps) << " " << "\n";
    }

    outfile.close();
}

constexpr int num_interactions = 10000;
constexpr int num_sweeps = 10;

int main() {
    // Design a set of dfeault variables and then a set of parameters to sweep across, 8 points per each
    int def_num_nodes = 400;
    int def_radius = 1;
    int def_num_features = 5;
    int def_feature_dim = 3;

    std::vector<int> lattice_radius_values = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> num_features_values = {3, 5, 7, 10, 12, 15, 18, 20};
    std::vector<int> feature_dim_values = {2, 5, 10, 15, 25, 40, 60, 80};
    std::vector<int> num_nodes_values = {225, 289, 324, 400, 484, 576, 676, 784};

    std::cout << "Starting radius sweep" << std::endl;
    sweep_and_save("data_sweeps/sweep_lattice_radius.txt",
                   SweepParam::LatticeRadius,
                   lattice_radius_values,
                   def_num_nodes, def_radius, def_num_features, def_feature_dim,
                   num_interactions, num_sweeps);

    std::cout << "Starting num features sweep" <<  std::endl;
    sweep_and_save("data_sweeps/sweep_num_features.txt",
                   SweepParam::NumFeatures,
                    num_features_values,
                    def_num_nodes, def_radius, def_num_features, def_feature_dim,
                    num_interactions * 30, num_sweeps);

    std::cout << "Starting feature dim sweep" <<  std::endl;
    sweep_and_save("data_sweeps/sweep_feature_dim.txt",
                   SweepParam::FeatureDim,
                    feature_dim_values,
                    def_num_nodes, def_radius, def_num_features, def_feature_dim,
                    num_interactions * 30, num_sweeps);

    std::cout << "Starting num nodes sweep" <<  std::endl;
    sweep_and_save("data_sweeps/sweep_num_nodes.txt",
                   SweepParam::NumNodes,
                   num_nodes_values,
                   def_num_nodes, def_radius, def_num_features, def_feature_dim,
                   num_interactions, num_sweeps);

    return 0;
}