#pragma once

// ===========================================================================
//  runPipeline.hpp  -  Sliding-window Word2Vec training (refactored from main)
// ===========================================================================

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "importUtils.hpp"
#include "trainPipeline.hpp"

inline int runTrainingPipeline() {

    const std::string dataFile   = "Dataset.jsonl";
    const std::string outputDir  = "output";
    const int         windowSize = 5;
    const int         maxPredYear = 2020;

#ifdef _WIN32
    system(("if not exist \"" + outputDir + "\" mkdir \"" + outputDir + "\"").c_str());
#else
    system(("mkdir -p " + outputDir).c_str());
#endif

    std::cout << "===========================================\n"
              << "  Loading dataset: " << dataFile << "\n"
              << "===========================================\n\n";

    auto lines = readFile(dataFile);
    if (lines.empty()) {
        std::cerr << "Error: no data loaded from " << dataFile << "\n";
        return 1;
    }

    std::vector<std::vector<int>> allEntries;
    std::map<int, std::vector<std::vector<int>>> byYear;
    allEntries.reserve(lines.size());

    for (const auto& line : lines) {
        int year = parseYear(line);
        if (year < 0) continue;
        auto codes = parseMeshCodes(line);
        if (codes.empty()) continue;

        std::vector<int> entry;
        entry.reserve(1 + codes.size());
        entry.push_back(year);
        entry.insert(entry.end(), codes.begin(), codes.end());

        byYear[year].push_back(entry);
        allEntries.push_back(std::move(entry));
    }
    lines.clear();
    lines.shrink_to_fit();

    if (allEntries.empty()) {
        std::cerr << "Error: no valid entries parsed from " << dataFile << "\n";
        return 1;
    }

    const int minYear    = byYear.begin()->first;
    const int maxYear    = byYear.rbegin()->first;
    const int numYears   = maxYear - minYear + 1;
    const int maxPossibleWindows = numYears - windowSize;
    const int numWindows = std::min(maxPossibleWindows,
                                    maxPredYear - minYear - windowSize + 1);

    std::cout << "Total entries : " << allEntries.size() << "\n"
              << "Year range    : " << minYear << " - " << maxYear
              << "  (" << numYears << " years)\n"
              << "Last pred year: " << maxPredYear << "\n"
              << "Windows       : " << numWindows << "\n\n";

    std::cout << "Entries per year:\n";
    for (const auto& [y, entries] : byYear)
        std::cout << "  " << y << " : " << entries.size() << "\n";
    std::cout << "\n";

    if (numWindows <= 0) {
        std::cerr << "Error: need at least " << (windowSize + 1)
                  << " distinct years\n";
        return 1;
    }

    const int    globalMinCount   = 5;
    const double globalTrimLowPct = 0.05;
    const double globalTrimHiPct  = 0.05;

    Vocabulary globalVocab;
    globalVocab.build(allEntries, globalMinCount, globalTrimLowPct, globalTrimHiPct);
    allEntries.clear();
    allEntries.shrink_to_fit();

    if (globalVocab.empty()) {
        std::cerr << "Error: global vocabulary is empty after filtering\n";
        return 1;
    }

    std::cout << "===========================================\n"
              << "  Global Vocabulary\n"
              << "===========================================\n";
    globalVocab.printStats();
    {
        std::string vocabFile = outputDir + "/global_vocab.txt";
        FILE* vf = std::fopen(vocabFile.c_str(), "w");
        if (vf) {
            std::fprintf(vf, "%d\n", globalVocab.size());
            for (int i = 0; i < globalVocab.size(); ++i)
                std::fprintf(vf, "%s %d\n",
                             formatMeshUI(globalVocab.idx2token[i]).c_str(),
                             globalVocab.frequencies[i]);
            std::fclose(vf);
        }
        std::cout << "  Saved to " << vocabFile << "\n\n";
    }

    TrainingConfig config;
    config.embeddingDim    = 64;
    config.negSamples      = 10;
    config.epochs          = 80;          // More epochs for better convergence
    config.batchSize       = 4096;
    config.learningRate    = 0.05f;       // Tuned LR for small dataset
    config.minLearningRate = 0.01f;      // Higher floor to prevent stalling
    config.minCount        = globalMinCount;
    config.trimLowPct      = 0.0f;
    config.trimHighPct     = 0.0f;
    config.numThreads      = 8;
    config.seed            = 42;
    config.verbose         = true;

    const int V = globalVocab.size();
    const int D = config.embeddingDim;

    std::vector<std::vector<float>> masterEmbeddings(numWindows);
    std::vector<int> predictionYears;
    predictionYears.reserve(numWindows);

    for (int w = 0; w < numWindows; ++w) {
        const int startYear = minYear + w;
        const int endYear   = startYear + windowSize - 1;
        const int predYear  = startYear + windowSize;
        predictionYears.push_back(predYear);

        std::cout
            << "\n===========================================\n"
            << "  Window " << (w + 1) << "/" << numWindows
            << "  |  Train [" << startYear << " - " << endYear
            << "]  ->  Predict " << predYear << "\n"
            << "===========================================\n";

        std::vector<std::vector<int>> windowEntries;
        for (int y = startYear; y <= endYear; ++y) {
            auto it = byYear.find(y);
            if (it != byYear.end())
                windowEntries.insert(windowEntries.end(),
                                     it->second.begin(), it->second.end());
        }
        std::cout << "  Training entries : " << windowEntries.size() << "\n";

        auto pairs = generateTrainingPairs(windowEntries, globalVocab);
        std::cout << "  Training pairs   : " << pairs.size() << "\n";

        if (pairs.empty()) {
            std::cout << "  [skip] no valid training pairs\n";
            masterEmbeddings[w].assign((size_t)V * D, 0.0f);
            continue;
        }

        w2v::Word2VecModel model;
        model.init(V, D, globalVocab, config.seed);
        TrainingPipeline::train(model, pairs, config);
        model.normalizeEmbeddings();
        // Post-process: remove dominant shared direction (Mu & Viswanath 2018)
        w2v::meanCenterAndNormalize(model.W_in.data(), V, D);

        masterEmbeddings[w].resize((size_t)V * D);
        std::memcpy(masterEmbeddings[w].data(), model.W_in.data(),
                    (size_t)V * D * sizeof(float));

        {
            std::string windowFile = outputDir + "/embeddings_"
                                   + std::to_string(predYear) + ".txt";
            FILE* wf = std::fopen(windowFile.c_str(), "w");
            if (wf) {
                std::fprintf(wf, "%d %d\n", V, D);
                for (int i = 0; i < V; ++i) {
                    std::fprintf(wf, "%s",
                                 formatMeshUI(globalVocab.idx2token[i]).c_str());
                    const float* emb = model.inputVec(i);
                    for (int d = 0; d < D; ++d)
                        std::fprintf(wf, " %g", emb[d]);
                    std::fprintf(wf, "\n");
                }
                std::fclose(wf);
            }
            std::cout << "  Saved: " << windowFile << "\n";
        }
    }

    std::string masterFile = outputDir + "/master_embeddings.txt";
    std::cout << "\n===========================================\n"
              << "  Writing master file: " << masterFile << "\n"
              << "===========================================\n";

    FILE* fp = std::fopen(masterFile.c_str(), "w");
    if (!fp) {
        std::cerr << "Error: cannot open " << masterFile << "\n";
        return 1;
    }
    std::fprintf(fp, "%d %d %d\n", V, D, numWindows);
    for (int w = 0; w < numWindows; ++w)
        std::fprintf(fp, "%d%c", predictionYears[w],
                     (w + 1 < numWindows) ? ' ' : '\n');
    for (int i = 0; i < V; ++i) {
        std::fprintf(fp, "%s", formatMeshUI(globalVocab.idx2token[i]).c_str());
        for (int w = 0; w < numWindows; ++w) {
            const float* emb = masterEmbeddings[w].data() + (size_t)i * D;
            for (int d = 0; d < D; ++d)
                std::fprintf(fp, " %g", emb[d]);
        }
        std::fprintf(fp, "\n");
    }
    std::fclose(fp);

    std::cout << "\n===========================================\n"
              << "  PIPELINE COMPLETE\n"
              << "===========================================\n"
              << "  Tokens             : " << V << "\n"
              << "  Embedding dim      : " << D << "\n"
              << "  Prediction years   : " << numWindows
              << "  (" << predictionYears.front()
              << " - " << predictionYears.back() << ")\n"
              << "  Master file        : " << masterFile << "\n"
              << "  Per-window files   : " << outputDir
              << "/embeddings_<YEAR>.txt\n"
              << "  Global vocabulary  : " << outputDir
              << "/global_vocab.txt\n"
              << "═══════════════════════════════════════════\n";

    return 0;
}
