#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>

Graph::Graph(int n, int f_num, int f_dim)
    : num_nodes(n), num_features(f_num), feature_dim(f_dim) {
    if (num_nodes <= 0) {
        throw std::runtime_error("Number of nodes must be positive.");
    }
    if (num_features <= 0 || feature_dim <= 0) {
        throw std::runtime_error("Feature counts must be positive.");
    }
}

void Graph::generate_node_features() {
    node_features.assign(num_nodes, std::vector<int>(num_features));

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_features; ++j) {
            node_features[i][j] = rand_int(0, feature_dim - 1);
        }
    }
}

couple Graph::select_nodes_for_interaction() const {
    if (num_nodes == 0) {
        throw std::runtime_error("Graph has no nodes.");
    }

    int i = rand_int(0, num_nodes - 1);
    const auto& neighbors = adj_list[i];

    if (neighbors.empty()) {
        throw std::runtime_error("Node has no neighbors to interact with.");
    }

    int j = neighbors[rand_int(0, static_cast<int>(neighbors.size()) - 1)];
    return {i, j};
}

void Graph::interaction(int node1, int node2) {
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

    std::vector<int> diff;
    diff.reserve(num_features);
    for (int f = 0; f < num_features; ++f) {
        if (f1[f] != f2[f]) {
            diff.push_back(f);
        }
    }

    if (diff.empty()) {
        return;
    }

    int idx = rand_int(0, static_cast<int>(diff.size()) - 1);
    int fstar = diff[idx];

    f1[fstar] = f2[fstar];
}

void Graph::axelrod_interaction() {
    couple nodes = select_nodes_for_interaction();
    int i = nodes.node1;
    int j = nodes.node2;

    auto& fi = node_features[i];
    auto& fj = node_features[j];

    if (static_cast<int>(fi.size()) != num_features ||
        static_cast<int>(fj.size()) != num_features) {
        throw std::runtime_error("axelrod_interaction: feature vector size mismatch.");
    }

    int shared = 0;
    for (int f = 0; f < num_features; ++f) {
        if (fi[f] == fj[f]) {
            ++shared;
        }
    }

    if (shared == 0) {
        return;
    }

    double sim = static_cast<double>(shared) / static_cast<double>(num_features);
    if (rand_real_0_1() >= sim) {
        return;
    }

    interaction(i, j);
}

void Graph::axelrod_step() {
    for (int it = 0; it < num_nodes; ++it) {
        axelrod_interaction();
    }
}

void Graph::measure_culture_histogram() {
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

void Graph::normalize_culture_distribution() {
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

void Graph::save_culture_histogram(std::ofstream& outfile) const {
    for (int count : culture_histogram) {
        outfile << count << " ";
    }
    outfile << "\n";
}

void Graph::save_culture_distribution(std::ofstream& outfile) const {
    for (double p : culture_distribution) {
        outfile << p << " ";
    }
    outfile << "\n";
}

int Graph::count_distinct_cultures() const {
    std::unordered_set<std::vector<int>, VecHash> uniq;
    uniq.reserve(num_nodes);
    for (auto& c : node_features) {
        uniq.insert(c);
    }
    return static_cast<int>(uniq.size());
}

int Graph::largest_culture_size() const {
    std::unordered_map<std::vector<int>, int, VecHash> freq;
    freq.reserve(num_nodes);

    for (auto& c : node_features) {
        freq[c]++;
    }

    int max_size = 0;
    for (auto& kv : freq) {
        if (kv.second > max_size) {
            max_size = kv.second;
        }
    }
    return max_size;
}

double Graph::average_similarity() const {
    long long count_edges = 0;
    double total_sim = 0.0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j : adj_list[i]) {
            if (j <= i) continue;
            count_edges++;

            int shared = 0;
            for (int f = 0; f < num_features; ++f) {
                if (node_features[i][f] == node_features[j][f]) {
                    shared++;
                }
            }

            total_sim += static_cast<double>(shared) / num_features;
        }
    }

    if (count_edges == 0) return 0.0;
    return total_sim / count_edges;
}

double Graph::entropy_cultures() const {
    std::unordered_map<std::vector<int>, int, VecHash> freq;
    freq.reserve(num_nodes);

    for (auto& c : node_features) {
        freq[c]++;
    }

    double total = static_cast<double>(num_nodes);
    double H = 0.0;

    for (auto& kv : freq) {
        double p = kv.second / total;
        H -= p * std::log(p);
    }

    return H;
}

double Graph::fragmentation_index() const {
    int L = largest_culture_size();
    return 1.0 - static_cast<double>(L) / static_cast<double>(num_nodes);
}

double Graph::edge_homophily() const {
    return average_similarity();
}

double Graph::global_similarity() const {
    long long pairs = 0;
    double total_sim = 0.0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = i + 1; j < num_nodes; ++j) {
            pairs++;

            int shared = 0;
            for (int f = 0; f < num_features; ++f) {
                if (node_features[i][f] == node_features[j][f]) {
                    shared++;
                }
            }

            total_sim += static_cast<double>(shared) / num_features;
        }
    }

    if (pairs == 0) return 0.0;
    return total_sim / pairs;
}

StrogatzGraph::StrogatzGraph(int n, int k, double p, int f_num, int f_dim)
    : Graph(n, f_num, f_dim), neighbors_per_node(k), rewiring_prob(p) {
    generate_graph();
    generate_node_features();
}

void StrogatzGraph::generate_graph() {
    if (neighbors_per_node % 2 != 0) {
        throw std::runtime_error("neighbors_per_node must be even.");
    }

    adj_list.assign(num_nodes, {});

    int K = neighbors_per_node;
    std::vector<std::vector<int>> directed(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 1; j <= K / 2; ++j) {
            int nbr = (i + j) % num_nodes;
            directed[i].push_back(nbr);
        }
    }

    for (int i = 0; i < num_nodes; ++i) {
        auto& row = directed[i];
        for (int idx = 0; idx < static_cast<int>(row.size()); ++idx) {
            if (rand_real_0_1() < rewiring_prob) {
                int new_n;

                while (true) {
                    new_n = rand_int(0, num_nodes - 1);
                    if (new_n == i) continue;

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

LatticeGraph::LatticeGraph(int n, int radius, int f_num, int f_dim)
    : Graph(n, f_num, f_dim), lattice_radius(radius) {
    if (lattice_radius <= 0) {
        throw std::runtime_error("lattice_radius must be positive.");
    }
    generate_graph();
    generate_node_features();
}

int LatticeGraph::lattice_side() const {
    int L = static_cast<int>(std::sqrt(num_nodes));
    if (L * L != num_nodes) {
        throw std::runtime_error("num_nodes must be a perfect square for a 2D lattice.");
    }
    return L;
}

void LatticeGraph::generate_graph() {
    int L = lattice_side();
    adj_list.assign(num_nodes, {});

    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            int node = x * L + y;

            for (int dx = -lattice_radius; dx <= lattice_radius; ++dx) {
                for (int dy = -lattice_radius; dy <= lattice_radius; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    if (std::abs(dx) + std::abs(dy) > lattice_radius) continue;

                    int nx = (x + dx + L) % L;
                    int ny = (y + dy + L) % L;
                    int neighbor = nx * L + ny;
                    adj_list[node].push_back(neighbor);
                }
            }

            auto& row = adj_list[node];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
    }
}
