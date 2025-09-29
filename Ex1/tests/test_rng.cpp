#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include "rng.hpp"   // from include/

using std::cout;
using std::endl;

struct Stats {
    double mean;
    double stddev;
};

Stats compute_stats(const std::vector<double>& x) {
    if (x.empty()) return {0.0, 0.0};
    double m = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    double v = 0.0;
    for (double xi : x) v += (xi - m) * (xi - m);
    v /= static_cast<double>(x.size());
    return {m, std::sqrt(v)};
}

void print_histogram(const std::string& title,
                     const std::vector<double>& data,
                     double min_v, double max_v,
                     int bins = 40,
                     int bar_width = 60)
{
    cout << "\n== " << title << " ==\n";
    if (data.empty() || max_v <= min_v || bins <= 0) {
        cout << "(no data)\n";
        return;
    }

    std::vector<std::size_t> counts(bins, 0);
    const double w = (max_v - min_v) / static_cast<double>(bins);

    for (double x : data) {
        if (x < min_v || x >= max_v) continue; // drop out-of-range
        int idx = static_cast<int>((x - min_v) / w);
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        counts[static_cast<std::size_t>(idx)]++;
    }

    std::size_t max_count = 0;
    for (auto c : counts) max_count = std::max(max_count, c);
    if (max_count == 0) max_count = 1;

    for (int i = 0; i < bins; ++i) {
        double a = min_v + i * w;
        double b = a + w;
        int hashes = static_cast<int>(std::round((counts[static_cast<std::size_t>(i)] / static_cast<double>(max_count)) * bar_width));
        cout << std::fixed << std::setprecision(2)
             << "[" << std::setw(6) << a << ", " << std::setw(6) << b << ") "
             << std::setw(6) << counts[static_cast<std::size_t>(i)] << " | ";
        for (int h = 0; h < hashes; ++h) cout << '#';
        cout << '\n';
    }

    Stats s = compute_stats(data);
    cout << "samples=" << data.size()
         << "  mean=" << std::setprecision(5) << s.mean
         << "  std=" << std::setprecision(5) << s.stddev << "\n";
}

static uint64_t parse_seed(int argc, char** argv, bool& has_seed) {
    has_seed = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        const std::string key = "--seed=";
        if (a.rfind(key, 0) == 0) {
            has_seed = true;
            return static_cast<uint64_t>(std::stoull(a.substr(key.size())));
        }
    }
    return 0ULL;
}

int main(int argc, char** argv) {
    constexpr std::size_t N = 20000;  

    bool has_seed = false;
    uint64_t seed = parse_seed(argc, argv, has_seed);

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen; 
    if (has_seed) { ugen.seed(seed); ggen.seed(seed + 1); }

    std::vector<double> u01 = ugen.next(N); 

    const double a = -2.0, b = 3.0;
    std::vector<double> u_a_b;
    u_a_b.reserve(N);
    for (double u : u01) u_a_b.push_back(a + (b - a) * u);

    print_histogram("Uniform [0,1)", u01, 0.0, 1.0, 40, 60);
    print_histogram("Uniform mapped to [-2, 3)", u_a_b, a, b, 40, 60);

    // --- Gaussian tests (mean=0) with two sigmas ---
    const double mean = 0.0;

    std::vector<double> g_s05 = ggen.next(N, mean, 0.5);
    std::vector<double> g_s20 = ggen.next(N, mean, 2.0);

    // Choose histogram windows ~ mean ± 4σ for visibility
    print_histogram("Gaussian N(0, 0.5^2)", g_s05, mean - 4.0 * 0.5, mean + 4.0 * 0.5, 40, 60);
    print_histogram("Gaussian N(0, 2.0^2)", g_s20, mean - 4.0 * 2.0, mean + 4.0 * 2.0, 40, 60);

    cout << "\nDone.\n";

    // Define 2 new generators with random seeds
    rng::UniformRandom ugen2;
    rng::GaussianRandom ggen2;
    // Generate single values and print them 
    cout << "\nSingle samples from new generators with random seeds:\n";
    cout << "Uniform: " << ugen2() << "\n";
    cout << "Gaussian: " << ggen2(mean, 1000.0) << "\n";
    return 0;
}
