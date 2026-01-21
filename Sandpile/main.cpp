#include "sandbox.hpp"
#include <iostream>
#include <fstream>

int main() {
    const int L = 10; // Example linear size
    Sandbox sandbox(L);
    std::ofstream ofs("sandbox_state.txt");
    sandbox.saveSandbox(ofs);
    ofs.close();
    return 0;
}