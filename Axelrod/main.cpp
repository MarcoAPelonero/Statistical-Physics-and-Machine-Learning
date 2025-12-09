#include "graph.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>

struct SweepResult {
    std::string param_name;
    int param_value;
    double param_value_double;
    int num_distinct_cultures;
    int largest_culture_size;
    double largest_culture_fraction;
    double avg_similarity;
    double entropy;
    double fragmentation;
    double edge_homophily;
    double global_similarity;
};

void sweep_different_parameters(int num_nodes, int neighbors_per_node,
                                double rewiring_prob, int num_features,
                                int feature_dim, int num_interactions, bool verbose=false) {

    StrogatzGraph g(num_nodes, neighbors_per_node, rewiring_prob,
                    num_features, feature_dim);

    for (int it = 0; it < num_interactions; ++it) {
        if (it % 100 == 0) {
            g.measure_culture_histogram();
            g.normalize_culture_distribution();
            std::vector<double> dist = g.get_culture_distribution();
            if (verbose) {
                std::cout << "Interaction " << it << ", culture distribution: ";
                for (double val : dist) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }
        }
        g.axelrod_interaction();
    }

    if (verbose) {
        const auto& features = g.get_node_features();
        for (size_t i = 0; i < features.size(); ++i) {
            std::cout << "Node " << i << ": ";
            for (size_t j = 0; j < features[i].size(); ++j) {
                std::cout << features[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
}

SweepResult measure_final_state(int num_nodes, int neighbors_per_node,
                                double rewiring_prob, int num_features,
                                int feature_dim, int num_interactions) {
    StrogatzGraph g(num_nodes, neighbors_per_node, rewiring_prob,
                    num_features, feature_dim);

    for (int it = 0; it < num_interactions; ++it) {
        g.axelrod_interaction();
    }

    SweepResult result;
    result.num_distinct_cultures = g.count_distinct_cultures();
    result.largest_culture_size = g.largest_culture_size();
    result.largest_culture_fraction = static_cast<double>(result.largest_culture_size) / num_nodes;
    result.avg_similarity = g.average_similarity();
    result.entropy = g.entropy_cultures();
    result.fragmentation = g.fragmentation_index();
    result.edge_homophily = g.edge_homophily();
    result.global_similarity = g.global_similarity();

    return result;
}

std::vector<SweepResult> average_sweep_results(const std::vector<std::vector<SweepResult>>& all_runs) {
    if (all_runs.empty() || all_runs[0].empty()) {
        return std::vector<SweepResult>();
    }
    
    size_t num_params = all_runs[0].size();
    size_t num_runs = all_runs.size();
    std::vector<SweepResult> averaged_results;
    
    for (size_t param_idx = 0; param_idx < num_params; ++param_idx) {
        SweepResult avg;
        avg.param_name = all_runs[0][param_idx].param_name;
        avg.param_value = all_runs[0][param_idx].param_value;
        avg.param_value_double = all_runs[0][param_idx].param_value_double;
        
        // Initialize accumulators
        double sum_num_cultures = 0.0;
        double sum_largest_size = 0.0;
        double sum_largest_frac = 0.0;
        double sum_avg_sim = 0.0;
        double sum_entropy = 0.0;
        double sum_frag = 0.0;
        double sum_homophily = 0.0;
        double sum_global_sim = 0.0;
        
        // Sum across all runs
        for (size_t run = 0; run < num_runs; ++run) {
            const SweepResult& r = all_runs[run][param_idx];
            sum_num_cultures += r.num_distinct_cultures;
            sum_largest_size += r.largest_culture_size;
            sum_largest_frac += r.largest_culture_fraction;
            sum_avg_sim += r.avg_similarity;
            sum_entropy += r.entropy;
            sum_frag += r.fragmentation;
            sum_homophily += r.edge_homophily;
            sum_global_sim += r.global_similarity;
        }
        
        // Average
        avg.num_distinct_cultures = static_cast<int>(sum_num_cultures / num_runs + 0.5);  // Round to nearest int
        avg.largest_culture_size = static_cast<int>(sum_largest_size / num_runs + 0.5);
        avg.largest_culture_fraction = sum_largest_frac / num_runs;
        avg.avg_similarity = sum_avg_sim / num_runs;
        avg.entropy = sum_entropy / num_runs;
        avg.fragmentation = sum_frag / num_runs;
        avg.edge_homophily = sum_homophily / num_runs;
        avg.global_similarity = sum_global_sim / num_runs;
        
        averaged_results.push_back(avg);
    }
    
    return averaged_results;
}

void save_sweep_results(const std::string& filename, const std::vector<SweepResult>& results) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }
    
    // Header
    outfile << "param_name,param_value,num_cultures,largest_culture_size,largest_culture_fraction,"
            << "avg_similarity,entropy,fragmentation,edge_homophily,global_similarity\n";
    
    for (const auto& r : results) {
        outfile << r.param_name << ","
                << (r.param_value != -1 ? std::to_string(r.param_value) : std::to_string(r.param_value_double))
                << "," << r.num_distinct_cultures
                << "," << r.largest_culture_size
                << "," << r.largest_culture_fraction
                << "," << r.avg_similarity
                << "," << r.entropy
                << "," << r.fragmentation
                << "," << r.edge_homophily
                << "," << r.global_similarity << "\n";
    }
    outfile.close();
    std::cout << "Saved results to " << filename << "\n";
}

int main () {
    // Define a set of default parameters for the graph
    int num_nodes = 600;
    int neighbors_per_node = 10;
    double rewiring_prob = 0.1;
    int num_features = 5;
    int feature_dim = 3;
    int num_interactions = 100000;
    int num_runs = 5;  // Number of runs to average across

    // Now for feature num, feature dim, rewiring prob and neighbors per node, define arrays of different values to sweep across, 8 per each 
    // of the parameters
    std::vector<int> neighbors_per_node_values = {4, 6, 8, 10, 12, 14, 16, 18};
    std::vector<double> rewiring_prob_values = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0};
    std::vector<int> num_features_values = {3, 4, 5, 6, 8, 10, 15, 20};
    std::vector<int> feature_dim_values = {2, 3, 5, 7, 10, 15, 20, 30};

    // Sweep across neighbors per node
    std::cout << "\n=== Sweeping neighbors per node ===\n";
    std::vector<std::vector<SweepResult>> neighbors_all_runs;
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << num_runs << "\n";
        std::vector<SweepResult> neighbors_results;
        #pragma omp parallel for
        for (size_t i = 0; i < neighbors_per_node_values.size(); ++i) {
            int k = neighbors_per_node_values[i];
            SweepResult res = measure_final_state(num_nodes, k, rewiring_prob,
                                                  num_features, feature_dim, num_interactions);
            res.param_name = "neighbors_per_node";
            res.param_value = k;
            res.param_value_double = -1;
            #pragma omp critical
            {
                neighbors_results.push_back(res);
                std::cout << "neighbors_per_node=" << k << " -> cultures=" << res.num_distinct_cultures 
                          << " largest_frac=" << res.largest_culture_fraction << "\n";
            }
        }
        neighbors_all_runs.push_back(neighbors_results);
    }
    std::vector<SweepResult> neighbors_averaged = average_sweep_results(neighbors_all_runs);
    save_sweep_results("data/sweep_neighbors_per_node.csv", neighbors_averaged);

    // Sweep across rewiring prob
    std::cout << "\n=== Sweeping rewiring probability ===\n";
    std::vector<std::vector<SweepResult>> rewiring_all_runs;
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << num_runs << "\n";
        std::vector<SweepResult> rewiring_results;
        #pragma omp parallel for
        for (size_t i = 0; i < rewiring_prob_values.size(); ++i) {
            double p = rewiring_prob_values[i];
            SweepResult res = measure_final_state(num_nodes, neighbors_per_node, p,
                                                  num_features, feature_dim, num_interactions);
            res.param_name = "rewiring_prob";
            res.param_value = -1;
            res.param_value_double = p;
            #pragma omp critical
            {
                rewiring_results.push_back(res);
                std::cout << "rewiring_prob=" << p << " -> cultures=" << res.num_distinct_cultures 
                          << " largest_frac=" << res.largest_culture_fraction << "\n";
            }
        }
        rewiring_all_runs.push_back(rewiring_results);
    }
    std::vector<SweepResult> rewiring_averaged = average_sweep_results(rewiring_all_runs);
    save_sweep_results("data/sweep_rewiring_prob.csv", rewiring_averaged);

    // Sweep across num features
    std::cout << "\n=== Sweeping number of features ===\n";
    std::vector<std::vector<SweepResult>> features_all_runs;
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << num_runs << "\n";
        std::vector<SweepResult> features_results;
        #pragma omp parallel for
        for (size_t i = 0; i < num_features_values.size(); ++i) {
            int f_num = num_features_values[i];
            SweepResult res = measure_final_state(num_nodes, neighbors_per_node, rewiring_prob,
                                                  f_num, feature_dim, num_interactions);
            res.param_name = "num_features";
            res.param_value = f_num;
            res.param_value_double = -1;
            #pragma omp critical
            {
                features_results.push_back(res);
                std::cout << "num_features=" << f_num << " -> cultures=" << res.num_distinct_cultures 
                          << " largest_frac=" << res.largest_culture_fraction << "\n";
            }
        }
        features_all_runs.push_back(features_results);
    }
    std::vector<SweepResult> features_averaged = average_sweep_results(features_all_runs);
    save_sweep_results("data/sweep_num_features.csv", features_averaged);

    // Sweep across feature dim
    std::cout << "\n=== Sweeping feature dimension ===\n";
    std::vector<std::vector<SweepResult>> feature_dim_all_runs;
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << num_runs << "\n";
        std::vector<SweepResult> feature_dim_results;
        #pragma omp parallel for
        for (size_t i = 0; i < feature_dim_values.size(); ++i) {
            int f_dim = feature_dim_values[i];
            SweepResult res = measure_final_state(num_nodes, neighbors_per_node, rewiring_prob,
                                                  num_features, f_dim, num_interactions);
            res.param_name = "feature_dim";
            res.param_value = f_dim;
            res.param_value_double = -1;
            #pragma omp critical
            {
                feature_dim_results.push_back(res);
                std::cout << "feature_dim=" << f_dim << " -> cultures=" << res.num_distinct_cultures 
                          << " largest_frac=" << res.largest_culture_fraction << "\n";
            }
        }
        feature_dim_all_runs.push_back(feature_dim_results);
    }
    std::vector<SweepResult> feature_dim_averaged = average_sweep_results(feature_dim_all_runs);
    save_sweep_results("data/sweep_feature_dim.csv", feature_dim_averaged);
    
    return 0;
}