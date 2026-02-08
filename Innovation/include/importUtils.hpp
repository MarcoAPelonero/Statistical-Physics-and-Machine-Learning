#pragma once 

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>

inline std::vector<std::string> readFile(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return lines;
    }

    lines.reserve(10000);

    std::string line;
    line.reserve(4096);

    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

inline int parseYear(const std::string& jsonLine) {
    const std::string pattern = "\"pub_year\": ";
    size_t pos = jsonLine.find(pattern);
    if (pos == std::string::npos) return -1;

    size_t numStart = pos + pattern.size();
    size_t numEnd   = numStart;
    while (numEnd < jsonLine.size() &&
           jsonLine[numEnd] >= '0' && jsonLine[numEnd] <= '9') {
        ++numEnd;
    }
    if (numEnd == numStart) return -1;
    return std::atoi(jsonLine.substr(numStart, numEnd - numStart).c_str());
}

inline std::vector<int> parseMeshCodes(const std::string& jsonLine) {
    std::vector<int> codes;

    const std::string uiPattern = "\"ui\": \"";
    size_t searchPos = 0;

    while (searchPos < jsonLine.size()) {
        size_t uiPos = jsonLine.find(uiPattern, searchPos);
        if (uiPos == std::string::npos) break;

        size_t codeStart = uiPos + uiPattern.length();
        if (codeStart >= jsonLine.size()) break;

        size_t codeEnd = codeStart;
        while (codeEnd < jsonLine.size() && jsonLine[codeEnd] != '\"') {
            ++codeEnd;
        }

        if (codeEnd > codeStart) {
            size_t digitStart = codeStart;
            while (digitStart < codeEnd &&
                   !(jsonLine[digitStart] >= '0' && jsonLine[digitStart] <= '9')) {
                ++digitStart;
            }
            if (digitStart < codeEnd) {
                codes.push_back(
                    std::atoi(jsonLine.substr(digitStart, codeEnd - digitStart).c_str()));
            }
        }

        searchPos = codeEnd + 1;
    }

    return codes;
}


class Dataset {
private:
    std::vector<std::vector<int>> trainSet;
    std::vector<std::vector<int>> testSet;

public:
    Dataset(const std::string& filename, int splitYear) {
        auto lines = readFile(filename);

        trainSet.reserve(lines.size());
        testSet.reserve(lines.size() / 4);

        for (const auto& line : lines) {
            int year = parseYear(line);
            if (year < 0) continue;                 // skip malformed lines

            std::vector<int> codes = parseMeshCodes(line);

            // Build the entry: year first, then all codes
            std::vector<int> entry;
            entry.reserve(1 + codes.size());
            entry.push_back(year);
            entry.insert(entry.end(), codes.begin(), codes.end());

            if (year <= splitYear) {
                trainSet.push_back(std::move(entry));
            } else {
                testSet.push_back(std::move(entry));
            }
        }

        trainSet.shrink_to_fit();
        testSet.shrink_to_fit();
    }

    const std::vector<std::vector<int>>& train() const { return trainSet; }
    const std::vector<std::vector<int>>& test()  const { return testSet;  }

    int trainSize() const { return (int)trainSet.size(); }
    int testSize()  const { return (int)testSet.size();  }
    int size()      const { return trainSize() + testSize(); }
};