#include "importUtils.hpp"
#include <iostream>
#include <algorithm>

int main() {
    // Split: year <= 2015 → train, year > 2015 → test
    int splitYear = 2015;
    Dataset dataset("Dataset.jsonl", splitYear);

    std::cout << "Split year (inclusive for train): " << splitYear << "\n";
    std::cout << "Total articles : " << dataset.size()      << "\n";
    std::cout << "Train articles : " << dataset.trainSize()  << "\n";
    std::cout << "Test  articles : " << dataset.testSize()   << "\n\n";

    // Helper lambda to print an article entry
    auto printEntry = [](const std::vector<int>& entry) {
        std::cout << "  year = " << entry[0] << "  |  codes (" << entry.size() - 1 << "): ";
        for (size_t i = 1; i < entry.size(); ++i) {
            std::cout << entry[i];
            if (i + 1 < entry.size()) std::cout << ", ";
        }
        std::cout << "\n";
    };

    // Show first 3 train articles
    int showN = std::min(3, dataset.trainSize());
    std::cout << "=== First " << showN << " TRAIN entries ===\n";
    for (int i = 0; i < showN; ++i) {
        printEntry(dataset.train()[i]);
    }

    // Show first 3 test articles
    showN = std::min(3, dataset.testSize());
    std::cout << "\n=== First " << showN << " TEST entries ===\n";
    for (int i = 0; i < showN; ++i) {
        printEntry(dataset.test()[i]);
    }

    return 0;
}