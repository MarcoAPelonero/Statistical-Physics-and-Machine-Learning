#include "generalUtils.hpp"

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
