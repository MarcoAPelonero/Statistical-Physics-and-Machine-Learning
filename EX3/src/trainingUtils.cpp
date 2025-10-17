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
    std::uniform_int_distribution<int> dist(0, (1 << bits) - 1);

    for (int i = 0; i < P; ++i) {
        int a = 0;
        int b = 0;
        // Resample until the pair encodes a strictly ordered comparison.
        do {
            a = dist(rng);
            b = dist(rng);
        } while (a == b);

    const Scalar sa = Scalar::fromInt(a, bits);
    const Scalar sb = Scalar::fromInt(b, bits);
        int label = 0;
        if (a > b) label = +1;
        else if (a < b) label = -1;
        ds.add(sa, sb, label);
    }
    return ds;
}

// Note: TrainPerceptron is implemented as a template in the header to allow
// training Perceptron<N> for arbitrary N. The template implementation is in
// include/traininUtils.hpp. This file keeps generateDataset and related helpers.
