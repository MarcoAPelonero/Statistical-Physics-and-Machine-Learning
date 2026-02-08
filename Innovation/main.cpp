// ═══════════════════════════════════════════════════════════════════════════
//  main.cpp  –  Complete sliding-window Word2Vec pipeline
//
//  Pipeline:
//    1. Import Dataset.jsonl
//    2. Build a GLOBAL vocabulary with top/bottom frequency trimming
//    3. For each 5-year sliding window starting from the earliest year:
//       a. Collect all entries in [startYear .. startYear+4]
//       b. Train a fresh Word2Vec model on those entries (global vocab)
//       c. Normalise embeddings, save per-window file
//       d. Store embeddings for the prediction year (startYear + 5)
//    4. Write a master embeddings file:
//         – every token in the global vocabulary
//         – one embedding vector per prediction year  (N_years − 5 total)
// ═══════════════════════════════════════════════════════════════════════════

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

// Format an internal int token back to a full MeSH UI  (e.g. 3267 → "D003267")
inline std::string formatMeshUI(int token) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "D%06d", token);
    return std::string(buf);
}

int main() {

    const std::string dataFile   = "Dataset.jsonl";
    const std::string outputDir  = "output";
    const int         windowSize = 5;
    const int         maxPredYear = 2020;   // stop here – later years are artifacts

    // Create output directory (silently ignore if it already exists)
#ifdef _WIN32
    system(("if not exist \"" + outputDir + "\" mkdir \"" + outputDir + "\"").c_str());
#else
    system(("mkdir -p " + outputDir).c_str());
#endif

    // ═══════════════════════════════════════════════════════════════════════
    //  1.  Load & parse the dataset
    // ═══════════════════════════════════════════════════════════════════════

    std::cout << "═══════════════════════════════════════════\n"
              << "  Loading dataset: " << dataFile << "\n"
              << "═══════════════════════════════════════════\n\n";

    auto lines = readFile(dataFile);
    if (lines.empty()) {
        std::cerr << "Error: no data loaded from " << dataFile << "\n";
        return 1;
    }

    // Parse each JSONL line into [year, code1, code2, ...]
    // and simultaneously group entries by year.
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

        byYear[year].push_back(entry);       // copy into year bucket
        allEntries.push_back(std::move(entry)); // move into flat list
    }

    lines.clear();   // free raw text – no longer needed
    lines.shrink_to_fit();

    if (allEntries.empty()) {
        std::cerr << "Error: no valid entries parsed from " << dataFile << "\n";
        return 1;
    }

    const int minYear    = byYear.begin()->first;
    const int maxYear    = byYear.rbegin()->first;
    const int numYears   = maxYear - minYear + 1;

    // Cap: last prediction year is maxPredYear
    // prediction year = minYear + w + windowSize, so w_max = maxPredYear - minYear - windowSize
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
                  << " distinct years for a sliding window of size "
                  << windowSize << "\n";
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  2.  Build global vocabulary (with top/bottom frequency trimming)
    // ═══════════════════════════════════════════════════════════════════════

    const int    globalMinCount   = 2;
    const double globalTrimLowPct = 0.05;   // drop bottom 5 % by frequency
    const double globalTrimHiPct  = 0.05;   // drop top    5 % by frequency

    Vocabulary globalVocab;
    globalVocab.build(allEntries, globalMinCount,
                      globalTrimLowPct, globalTrimHiPct);

    // allEntries no longer needed – free memory
    allEntries.clear();
    allEntries.shrink_to_fit();

    if (globalVocab.empty()) {
        std::cerr << "Error: global vocabulary is empty after filtering\n";
        return 1;
    }

    std::cout << "═══════════════════════════════════════════\n"
              << "  Global Vocabulary\n"
              << "═══════════════════════════════════════════\n";
    globalVocab.printStats();
    // Save vocab with full MeSH UIDs
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

    // ═══════════════════════════════════════════════════════════════════════
    //  3.  Training configuration  (shared across all windows)
    // ═══════════════════════════════════════════════════════════════════════

    TrainingConfig config;
    config.embeddingDim    = 100;
    config.negSamples      = 5;
    config.epochs          = 5;
    config.batchSize       = 4096;
    config.learningRate    = 0.025f;
    config.minLearningRate = 0.0001f;
    config.minCount        = globalMinCount;
    config.trimLowPct      = 0.0f;    // already trimmed globally
    config.trimHighPct     = 0.0f;
    config.numThreads      = 4;
    config.seed            = 42;
    config.verbose         = true;

    const int V = globalVocab.size();
    const int D = config.embeddingDim;

    // ═══════════════════════════════════════════════════════════════════════
    //  4.  Sliding-window training loop
    // ═══════════════════════════════════════════════════════════════════════
    //
    //  window w:
    //    training years  =  [minYear + w  ..  minYear + w + 4]
    //    prediction year =   minYear + w + 5
    //
    //  We store the (normalised) input embeddings for every token after
    //  each window, so each token ends up with numWindows embedding vectors.

    std::vector<std::vector<float>> masterEmbeddings(numWindows);
    std::vector<int> predictionYears;
    predictionYears.reserve(numWindows);

    for (int w = 0; w < numWindows; ++w) {

        const int startYear = minYear + w;
        const int endYear   = startYear + windowSize - 1;
        const int predYear  = startYear + windowSize;
        predictionYears.push_back(predYear);

        std::cout
            << "\n═══════════════════════════════════════════\n"
            << "  Window " << (w + 1) << "/" << numWindows
            << "  |  Train [" << startYear << " - " << endYear
            << "]  ->  Predict " << predYear << "\n"
            << "═══════════════════════════════════════════\n";

        // ── Collect entries for the 5-year training window ──────────────
        std::vector<std::vector<int>> windowEntries;
        for (int y = startYear; y <= endYear; ++y) {
            auto it = byYear.find(y);
            if (it != byYear.end())
                windowEntries.insert(windowEntries.end(),
                                     it->second.begin(),
                                     it->second.end());
        }

        std::cout << "  Training entries : " << windowEntries.size() << "\n";

        // ── Generate training pairs (using global vocabulary) ───────────
        auto pairs = generateTrainingPairs(windowEntries, globalVocab);
        std::cout << "  Training pairs   : " << pairs.size() << "\n";

        if (pairs.empty()) {
            std::cout << "  [skip] no valid training pairs for this window\n";
            masterEmbeddings[w].assign((size_t)V * D, 0.0f);
            continue;
        }

        // ── Initialise a fresh model (same seed → same starting point) ──
        w2v::Word2VecModel model;
        model.init(V, D, globalVocab, config.seed);

        // ── Train ───────────────────────────────────────────────────────
        TrainingPipeline::train(model, pairs, config);

        // ── L2-normalise embeddings ─────────────────────────────────────
        model.normalizeEmbeddings();

        // ── Copy embeddings into master storage ─────────────────────────
        masterEmbeddings[w].resize((size_t)V * D);
        std::memcpy(masterEmbeddings[w].data(),
                    model.W_in.data(),
                    (size_t)V * D * sizeof(float));

        // ── Save per-window embeddings file (with full MeSH UIDs) ─────
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

    // ═══════════════════════════════════════════════════════════════════════
    //  5.  Write master embeddings file
    // ═══════════════════════════════════════════════════════════════════════
    //
    //  Format:
    //    Line 1 :  <num_tokens> <embedding_dim> <num_prediction_years>
    //    Line 2 :  <pred_year_1> <pred_year_2> ... <pred_year_N>
    //    Line 3+:  <token_id> <e_y1_d1> ... <e_y1_dD> <e_y2_d1> ... <e_yN_dD>
    //
    //  Each token row therefore has  num_prediction_years × embedding_dim  floats.

    std::string masterFile = outputDir + "/master_embeddings.txt";

    std::cout << "\n═══════════════════════════════════════════\n"
              << "  Writing master file: " << masterFile << "\n"
              << "═══════════════════════════════════════════\n";

    FILE* fp = std::fopen(masterFile.c_str(), "w");
    if (!fp) {
        std::cerr << "Error: cannot open " << masterFile << " for writing\n";
        return 1;
    }

    // Header
    std::fprintf(fp, "%d %d %d\n", V, D, numWindows);

    // Prediction-year line
    for (int w = 0; w < numWindows; ++w)
        std::fprintf(fp, "%d%c", predictionYears[w],
                     (w + 1 < numWindows) ? ' ' : '\n');

    // One row per token: MeSH UID followed by all embeddings (year-major)
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

    // ═══════════════════════════════════════════════════════════════════════
    //  Done
    // ═══════════════════════════════════════════════════════════════════════

    std::cout << "\n═══════════════════════════════════════════\n"
              << "  PIPELINE COMPLETE\n"
              << "═══════════════════════════════════════════\n"
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
