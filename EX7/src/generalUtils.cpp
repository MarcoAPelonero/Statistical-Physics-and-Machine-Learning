#include "generalUtils.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace generalUtils {

int analyticalComparison(const Scalar& top, const Scalar& bottom) {
    const long int topVal = top.toInt();
    const long int botVal = bottom.toInt();
    if (topVal > botVal) return +1;
    if (topVal < botVal) return -1;
    return 0;
}

unsigned clockSeed() {
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    return static_cast<unsigned>(now & 0xFFFFFFFFu);
}

void appendRange(std::vector<double>& dest, double start, double end, double step, bool includeEnd) {
    if (step <= 0.0) return;
    if (includeEnd) {
        for (double value = start; value <= end; value += step) {
            dest.push_back(value);
        }
    } else {
        for (double value = start; value < end; value += step) {
            dest.push_back(value);
        }
    }
}

std::vector<double> makeRange(double start, double end, double step, bool includeEnd) {
    std::vector<double> values;
    if (step <= 0.0) return values;
    values.reserve(static_cast<size_t>(std::max(0.0, (end - start) / step)) + 1u);
    appendRange(values, start, end, step, includeEnd);
    return values;
}

} // namespace generalUtils

void generalUtils::report(const char* name, int trials, int correct, double elapsedMs) {
    const double accuracy = static_cast<double>(correct) / trials;
    const std::streamsize oldPrecision = std::cout.precision();
    const auto oldFlags = std::cout.flags();

    std::cout << name << " -> trials: " << trials
              << ", accuracy: " << std::fixed << std::setprecision(4) << (100.0 * accuracy) << "%"
              << ", error: " << (100.0 * (1.0 - accuracy)) << "%"
              << ", time: " << std::setprecision(3) << elapsedMs << " ms\n";

    std::cout.precision(oldPrecision);
    std::cout.flags(oldFlags);
}
