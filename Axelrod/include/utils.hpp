#pragma once 

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <queue>
#include <random>
#include <unordered_map>

struct VecHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v) {
            h = h * 1315423911u + std::hash<int>()(x);
        }
        return h;
    }
};

// Random utilities are defined in the implementation file to avoid
// unused-function warnings when this header is included but the
// random helpers are not used in a translation unit.
int rand_int(int lo, int hi);
double rand_real_0_1();

double similarity(const std::vector<int>& features1, const std::vector<int>& features2);