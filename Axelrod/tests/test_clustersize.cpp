#include "graph.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>   // added

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem; // added

int threads = 1;

int main() {
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    // Make a graph with 500 nodes, 10 neighbors per node, rewiring prob 0.1, 5 features and 
    // an array of N different features between a max and a minimum feature dimension

    // Force remove the folder data/culture_data/ if it exists and create it again
    try {
        fs::remove_all("data/culture_data");
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Warning: couldn't remove directory 'data/culture_data': " << e.what() << "\n";
    }
    try {
        fs::create_directories("data/culture_data");
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error: couldn't create directory 'data/culture_data': " << e.what() << "\n";
        return 1;
    }

    int num_nodes = 900;
    int neighbors_per_node = 10;
    double rewiring_prob = 0.05;
    int num_features = 5;
    int min_feature_dim = 2;
    int max_feature_dim = 100;
    int feature_dimensionality = 20;
    std::vector<int> feature_dim; ;
    double step = (max_feature_dim - min_feature_dim) / static_cast<double>(feature_dimensionality - 1);
    for (int i = 0; i < feature_dimensionality; ++i) {
        feature_dim.push_back(static_cast<int>(min_feature_dim + i * step));
    }

    constexpr int num_interactions = 100000;

    // Parallelize over different feature dimensions with OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t idx = 0; idx < feature_dim.size(); ++idx) {
        int f_dim = feature_dim[idx];
        StrogatzGraph g(num_nodes, neighbors_per_node, rewiring_prob,
                            num_features, f_dim);
        g.convert_to_2d_lattice();
        std::ofstream outfile;
        outfile.open("data/culture_data/culture_distribution_fdim_" + std::to_string(f_dim) + ".txt");
        for (int it = 0 ; it < num_interactions; ++it) {
            g.axelrod_step();
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            std::cout << "Feature dimension: " << f_dim << "\n";
        }

        int biggest_cluster = g.largest_culture_size();
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            std::cout << "Biggest culture size in perc: " << static_cast<double>(biggest_cluster) / num_nodes << "\n";
        }
        // Get the biggest cluster size and print on outfile the final culture distribution
        g.measure_culture_histogram();
        g.save_culture_histogram(outfile);
        outfile.close();
    }

    return 0;
}