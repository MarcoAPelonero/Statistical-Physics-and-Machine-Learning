#include <iostream>
#include <string>
#include "exPoints.hpp"

int main(int argc, char** argv) {
    // Add a switch case if i want to select one excerice in specific or execute all of them based on the values read as argc

    // You should excpect 0 1 or 2 variables. Default case is execute all, if
    // the flag is swithced to single instead of all, then use the number to select which excercise to run

    switch (argc) {
        case 1:
            exPointOne();
            exPointTwo();
            break;
        case 3:
            if (std::string(argv[1]) == "single") {
                int exNum = std::stoi(argv[2]);
                switch (exNum) {
                    case 1:
                        exPointOne();
                        break;
                    case 2:
                        exPointTwo();
                        break;
                    default:
                        std::cerr << "Invalid exercise number. Please choose between 1 and 4.\n";
                        return 1;
                }
            } else {
                std::cerr << "Invalid argument. Use 'single <number>' to run a specific exercise.\n";
                return 1;
            }
        }
    return 0;
}