#include "traininUtils.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>

Dataset generateDataset(int P, std::optional<unsigned> seed, int bits) {
    Dataset ds;
    ds = Dataset();

    if (!seed.has_value()) {
        // initialize seed from current clock time
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        seed = static_cast<unsigned>(now & 0xFFFFFFFFu);
    }
    // seed now has a value; construct rng from the unsigned seed
    std::mt19937 rng(seed.value());
    // bits is provided by caller; default is 10 at declaration site
    assert(bits > 0);
    std::uniform_int_distribution<int> bitDist(0, 1);

    auto makeScalar = [&](void) -> Scalar {
        Scalar s(bits);
        for (int idx = 0; idx < bits; ++idx) {
            s[idx] = bitDist(rng) ? +1 : -1;
        }
        return s;
    };

    auto compareScalars = [&](const Scalar& a, const Scalar& b) -> int {
        for (int idx = 0; idx < bits; ++idx) {
            const int bitA = (a[idx] + 1) / 2;
            const int bitB = (b[idx] + 1) / 2;
            if (bitA != bitB) {
                return (bitA > bitB) ? +1 : -1;
            }
        }
        return 0;
    };

    int generated = 0;
    while (generated < P) {
        Scalar sa = makeScalar();
        Scalar sb = makeScalar();
        int label = compareScalars(sa, sb);
        if (label == 0) continue; // resample until strictly ordered
        ds.add(sa, sb, label);
        ++generated;
    }
    return ds;
}

// Note: TrainPerceptron is implemented as a template in the header to allow
// training Perceptron<N> for arbitrary N. The template implementation is in
// include/traininUtils.hpp. This file keeps generateDataset and related helpers.
