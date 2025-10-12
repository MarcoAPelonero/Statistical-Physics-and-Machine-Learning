#include "generalUtils.hpp"

namespace generalUtils {

int analyticalComparison(const Scalar& top, const Scalar& bottom) {
    const long int topVal = top.toInt();
    const long int botVal = bottom.toInt();
    if (topVal > botVal) return +1;
    if (topVal < botVal) return -1;
    return 0;
}

} // namespace generalUtils
