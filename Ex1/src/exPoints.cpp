#include "exPoints.hpp"
#include <fstream>

void exPointOne() {}

void exPointTwo() {
    std::cout << "\n=== Exercise point n2 ===\n";

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    std::size_t N = 10;

    double min_v = 0.0;
    double max_v = 1.0;
    double std = 0.0;

    std::vector<double> uniform_numbers = ugen.next(N);
    std::vector<double> uniform_numbers_rescaled;
    uniform_numbers_rescaled.reserve(N);
    for (double u : uniform_numbers) uniform_numbers_rescaled.push_back(min_v + (max_v - min_v) * u);

    std::vector<double> dataPointsA = generateDataPointsA(uniform_numbers_rescaled, ggen, N, std);
    std::vector<double> dataPointsB = generateDataPointsB(uniform_numbers_rescaled, ggen, N, std);

    std::ofstream outfile("output.txt");
    for (std::size_t i = 0; i < N; ++i) {
        outfile << uniform_numbers[i] << " " << dataPointsA[i] << " " << dataPointsB[i] << "\n";
    }
    outfile.close();
}