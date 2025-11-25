#include <iostream>
#include <string>
#include "exPoints.hpp"

int main(int argc, char** argv) {
    // Add a switch case if i want to select one excerice in specific or execute all of them based on the values read as argc

    // You should expect 0, 1, 2 or 3 variables. Default case is execute all exercises,
    // if the flag is switched to single, then use the number to select which exercise to run
    // if the flag is comparison, run only the comparison function
    // Examples: 
    // ./main.exe
    // ./main.exe default
    // ./main.exe comparison
    // ./main.exe single 3

    switch (argc) {
        case 1:
            exPointOne();
            exPointTwo();
            exPointThree();
            exPointFour();
            exPointFive();
            exPointSix();
            exPointSeven();
            break;
        case 2:
            if (std::string(argv[1]) == "default") {
                exPointOne();
                exPointTwo();
                exPointThree();
                exPointFour();
                exPointFive();
                exPointSix();
                exPointSeven();
            } else if (std::string(argv[1]) == "comparison") {
                comparison();
            } else {
                std::cerr << "Invalid argument. Use 'default', 'comparison', or 'single <number>'.\n";
                return 1;
            }
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
                    case 3:
                        exPointThree();
                        break;
                    case 4:
                        exPointFour();
                        break;
                    case 5:
                        exPointFive();
                        break;
                    case 6:
                        exPointSix();
                        break;
                    case 7:
                        exPointSeven();
                        break;
                    case 10:
                        extraPointOne();
                        break;
                    default:
                        std::cerr << "Invalid exercise number. Please choose between 1 and 13.\n";
                        return 1;
                }
            } else {
                std::cerr << "Invalid argument. Use 'single <number>' to run a specific exercise.\n";
                return 1;
            }
            break;
        default:
        std::cerr << "Too many arguments. Usage:\n";
        std::cerr << "  No arguments: run all exercises\n";
        std::cerr << "  default: run all exercises\n";
        std::cerr << "  comparison: run only comparison\n";
        std::cerr << "  single <number>: run specific exercise (1-13)\n";
        return 1;
    }
    return 0;
}
