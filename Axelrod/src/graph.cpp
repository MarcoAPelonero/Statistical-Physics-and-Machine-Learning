#include "graph.hpp"

#include <algorithm>
#include <random>
#include <unordered_set>

void StrogatzGraph::generate_graph() {
    if (neightbors_per_node % 2 != 0) {
        throw std::runtime_error("neighbors_per_node must be even.");
    }

    adj_list.assign(num_nodes, {});

    int K = neightbors_per_node;

    // 1) Build directed ring lattice: i -> i+1 ... i+K/2
    std::vector<std::vector<int>> directed(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 1; j <= K / 2; ++j) {
            int nbr = (i + j) % num_nodes;
            directed[i].push_back(nbr);
        }
    }

    // 2) Rewire each directed edge with probability rewiring_prob
    for (int i = 0; i < num_nodes; ++i) {
        auto& row = directed[i];
        for (int idx = 0; idx < static_cast<int>(row.size()); ++idx) {

            if (rand_real_0_1() < rewiring_prob) {
                int new_n;

                while (true) {
                    new_n = rand_int(0, num_nodes - 1);
                    if (new_n == i) continue; // avoid self-loop

                    bool exists = false;
                    for (int x : row) {
                        if (x == new_n) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) break;
                }

                row[idx] = new_n;
            }
        }
    }

    // 3) Make the graph undirected and clean duplicates
    for (int i = 0; i < num_nodes; ++i) {
        for (int n : directed[i]) {
            adj_list[i].push_back(n);
            adj_list[n].push_back(i);
        }
    }

    for (int i = 0; i < num_nodes; ++i) {
        auto& v = adj_list[i];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
}

void StrogatzGraph::convert_to_2d_lattice() {
    int L = static_cast<int>(std::sqrt(num_nodes));
    if (L * L != num_nodes) {
        throw std::runtime_error("convert_to_2d_lattice: num_nodes is not a perfect square.");
    }

    adj_list.assign(num_nodes, {});

    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            int node = x * L + y;

            // Right neighbor
            int right = x * L + ((y + 1) % L);
            adj_list[node].push_back(right);

            // Left neighbor
            int left = x * L + ((y - 1 + L) % L);
            adj_list[node].push_back(left);

            // Down neighbor
            int down = ((x + 1) % L) * L + y;
            adj_list[node].push_back(down);

            // Up neighbor
            int up = ((x - 1 + L) % L) * L + y;
            adj_list[node].push_back(up);
        }
    }
}

void StrogatzGraph::generate_node_features() {
    node_features.assign(num_nodes, std::vector<int>(num_features));

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_features; ++j) {
            node_features[i][j] = rand_int(0, feature_dim - 1);
        }
    }
}

const std::vector<std::vector<int>>& StrogatzGraph::get_node_features() const {
    return node_features;
}

couple StrogatzGraph::select_nodes_for_interaction() const {
    if (num_nodes == 0) {
        throw std::runtime_error("Graph has no nodes.");
    }

    // Pick focal node i
    int i = rand_int(0, num_nodes - 1);
    const auto& neighbors = adj_list[i];

    if (neighbors.empty()) {
        throw std::runtime_error("Node has no neighbors to interact with.");
    }

    // Pick random neighbor j of i
    int j = neighbors[rand_int(0, static_cast<int>(neighbors.size()) - 1)];

    return {i, j};
}

void StrogatzGraph::interaction(int node1, int node2) {
    // node1 and node2 must be valid indices
    if (node1 < 0 || node1 >= num_nodes ||
        node2 < 0 || node2 >= num_nodes) {
        throw std::runtime_error("interaction: node index out of range.");
    }

    auto& f1 = node_features[node1];
    const auto& f2 = node_features[node2];

    if (static_cast<int>(f1.size()) != num_features ||
        static_cast<int>(f2.size()) != num_features) {
        throw std::runtime_error("interaction: feature vector size mismatch.");
    }

    // Collect differing feature indices
    std::vector<int> diff;
    diff.reserve(num_features);
    for (int f = 0; f < num_features; ++f) {
        if (f1[f] != f2[f]) {
            diff.push_back(f);
        }
    }

    if (diff.empty()) {
        // Cultures are identical: nothing to copy
        return;
    }

    // Pick one differing feature uniformly at random
    int idx = rand_int(0, static_cast<int>(diff.size()) - 1);
    int fstar = diff[idx];

    // node1 copies node2 on feature fstar
    f1[fstar] = f2[fstar];
}

void StrogatzGraph::axelrod_interaction() {
    // 1. select a random pair (i,j) with j neighbor of i
    couple nodes = select_nodes_for_interaction();
    int i = nodes.node1;
    int j = nodes.node2;

    auto& fi = node_features[i];
    auto& fj = node_features[j];

    if (static_cast<int>(fi.size()) != num_features ||
        static_cast<int>(fj.size()) != num_features) {
        throw std::runtime_error("axelrod_interaction: feature vector size mismatch.");
    }

    // 2. compute similarity (overlap)
    int shared = 0;
    for (int f = 0; f < num_features; ++f) {
        if (fi[f] == fj[f]) {
            ++shared;
        }
    }

    if (shared == 0) {
        // no shared features -> no interaction possible
        return;
    }

    double sim = static_cast<double>(shared) / static_cast<double>(num_features);

    // 3. with probability sim, interact
    double r = rand_real_0_1();
    if (r >= sim) {
        return; // no interaction this time
    }

    // 4. if they interact, i copies one differing feature of j
    interaction(i, j);
}

void StrogatzGraph::axelrod_step() {
    for (int it = 0; it < num_nodes; ++it) {
        axelrod_interaction();
    }
}

void StrogatzGraph::measure_culture_histogram() {

    std::unordered_map<std::vector<int>, int, VecHash> freq;
    freq.reserve(num_nodes);

    for (const auto& culture : node_features) {
        freq[culture] += 1;
    }

    culture_histogram.clear();
    culture_histogram.reserve(freq.size());

    for (auto& kv : freq) {
        culture_histogram.push_back(kv.second);
    }
}

void StrogatzGraph::normalize_culture_distribution() {
    measure_culture_histogram();

    culture_distribution.clear();
    culture_distribution.reserve(culture_histogram.size());

    double total = 0.0;
    for (int count : culture_histogram) {
        total += static_cast<double>(count);
    }

    if (total == 0.0) {
        throw std::runtime_error("normalize_culture_distribution: total count is zero.");
    }

    for (int count : culture_histogram) {
        culture_distribution.push_back(static_cast<double>(count) / total);
    }
}

void StrogatzGraph::save_culture_histogram(std::ofstream& outfile) const {
    for (int count : culture_histogram) {
        outfile << count << " ";
    }
    outfile << "\n";
}

void StrogatzGraph::save_culture_distribution(std::ofstream& outfile) const {
    for (double p : culture_distribution) {
        outfile << p << " ";
    }
    outfile << "\n";
}

int StrogatzGraph::count_distinct_cultures() const {
    std::unordered_set<std::vector<int>, VecHash> uniq;
    uniq.reserve(num_nodes);
    for (auto& c : node_features) {
        uniq.insert(c);
    }
    return (int)uniq.size();
}

int StrogatzGraph::largest_culture_size() const {
    std::unordered_map<std::vector<int>, int, VecHash> freq;
    freq.reserve(num_nodes);

    for (auto& c : node_features) freq[c]++;

    int max_size = 0;
    for (auto& kv : freq) {
        if (kv.second > max_size) max_size = kv.second;
    }
    return max_size;
}

double StrogatzGraph::average_similarity() const {
    long long count_edges = 0;
    double total_sim = 0.0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j : adj_list[i]) {
            if (j <= i) continue; // avoid double counting
            count_edges++;

            int shared = 0;
            for (int f = 0; f < num_features; ++f)
                if (node_features[i][f] == node_features[j][f])
                    shared++;

            total_sim += (double)shared / num_features;
        }
    }

    if (count_edges == 0) return 0.0;
    return total_sim / count_edges;
}

double StrogatzGraph::entropy_cultures() const {
    std::unordered_map<std::vector<int>, int, VecHash> freq;
    freq.reserve(num_nodes);

    for (auto& c : node_features) freq[c]++;

    double total = (double)num_nodes;
    double H = 0.0;

    for (auto& kv : freq) {
        double p = kv.second / total;
        H -= p * std::log(p);
    }

    return H;
}

double StrogatzGraph::fragmentation_index() const {
    int L = largest_culture_size();
    return 1.0 - (double)L / (double)num_nodes;
}

double StrogatzGraph::edge_homophily() const {
    return average_similarity(); // same definition here
}

double StrogatzGraph::global_similarity() const {
    long long pairs = 0;
    double total_sim = 0.0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = i+1; j < num_nodes; ++j) {
            pairs++;

            int shared = 0;
            for (int f = 0; f < num_features; ++f)
                if (node_features[i][f] == node_features[j][f])
                    shared++;

            total_sim += (double)shared / num_features;
        }
    }

    if (pairs == 0) return 0.0;
    return total_sim / pairs;
}