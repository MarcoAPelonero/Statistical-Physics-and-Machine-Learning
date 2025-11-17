#ifndef GENERAL_UTILS_HPP
#define GENERAL_UTILS_HPP

#include "vector.hpp"

namespace generalUtils {

int analyticalComparison(const Scalar& top, const Scalar& bottom);

// Print a standardized evaluation report line.
// name: label to print, trials: number of trials, correct: number of correct outcomes,
// elapsedMs: total elapsed time in milliseconds.
void report(const char* name, int trials, int correct, double elapsedMs);

} // namespace generalUtils

#endif // GENERAL_UTILS_HPP
