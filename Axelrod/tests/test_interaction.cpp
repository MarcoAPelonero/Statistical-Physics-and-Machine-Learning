#include "graph.hpp"
#include <iostream>

int main() {
    StrogatzGraph g(100, 10, 0.1, 5, 3);

    int num_interactions = 10000000;
    for (int it = 0; it < num_interactions; ++it) {
        g.axelrod_interaction();
    }

    const auto& features = g.get_node_features();
    for (size_t i = 0; i < features.size(); ++i) {
        std::cout << "Node " << i << ": ";
        for (size_t j = 0; j < features[i].size(); ++j) {
            std::cout << features[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}