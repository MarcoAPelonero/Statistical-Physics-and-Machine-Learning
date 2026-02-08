#pragma once

// ===========================================================================
//  innovation.hpp  -  Detect & score never-seen MeSH code pairs
//
//  Flow:
//    1. Load global vocab & per-year embeddings produced by the training pipeline
//    2. Parse Dataset.jsonl to get articles grouped by year
//    3. From the first 5 training years, collect all codes and co-occurring pairs
//    4. All possible code pairs minus co-occurred = "innovation candidates"
//    5. For each subsequent year, record which candidates first appear (discovery)
//    6. For every candidate, compute the cosine-similarity (dot product of
//       normalised embeddings) across all available years
//    7. Save  innovations_all.bin        - full binary dump
//            innovations_discovered.csv  - discovered subset (for plotting)
//            innovations_stats.txt       - summary
// ===========================================================================

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>

#include "importUtils.hpp"
#include "prepareData.hpp"
#include "word2vec.hpp"       // for w2v::dotProduct

// ── Helpers ─────────────────────────────────────────────────────────────────

inline int parseDCode(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && !std::isdigit((unsigned char)s[i])) ++i;
    return (i < s.size()) ? std::atoi(s.c_str() + i) : -1;
}

// Canonical pair key - both values are vocab INDICES (< 20 000)
inline int64_t encodePair(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    return (int64_t)a * 20000LL + b;
}

// Load the D-prefixed global_vocab.txt produced by the training pipeline
inline Vocabulary loadFormattedVocab(const std::string& filename) {
    Vocabulary vocab;
    std::ifstream in(filename);
    if (!in) { std::cerr << "Error: cannot open " << filename << "\n"; return vocab; }
    int n; in >> n;
    vocab.idx2token.resize(n);
    vocab.frequencies.resize(n);
    vocab.totalTokens = 0;
    for (int i = 0; i < n; ++i) {
        std::string ui; int freq;
        in >> ui >> freq;
        int token = parseDCode(ui);
        vocab.idx2token[i]   = token;
        vocab.frequencies[i] = freq;
        vocab.token2idx[token] = i;
        vocab.totalTokens   += freq;
    }
    return vocab;
}

// Scan output/ for embeddings_YYYY.txt and return sorted year list
inline std::vector<int> detectPredictionYears(const std::string& dir) {
    std::vector<int> yrs;
    for (int y = 1990; y <= 2030; ++y) {
        std::string path = dir + "/embeddings_" + std::to_string(y) + ".txt";
        FILE* fp = std::fopen(path.c_str(), "r");
        if (fp) { yrs.push_back(y); std::fclose(fp); }
    }
    return yrs;
}

// Load a per-year embedding file into a flat VxD float vector
// Uses fopen (MinGW ifstream has reliability issues in this binary).
inline std::vector<float> loadEmbeddingFile(const std::string& path,
                                            int V, int D) {
    std::vector<float> emb((size_t)V * D, 0.0f);
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) { std::cerr << "Error: cannot read " << path << "\n"; return emb; }
    int fV, fD;
    std::fscanf(fp, "%d %d", &fV, &fD);
    for (int i = 0; i < V; ++i) {
        char tok[32];
        std::fscanf(fp, "%s", tok);          // skip D-code string
        for (int d = 0; d < D; ++d)
            std::fscanf(fp, "%f", &emb[(size_t)i * D + d]);
    }
    std::fclose(fp);

    // Post-process: remove dominant shared direction & re-normalise
    w2v::meanCenterAndNormalize(emb.data(), V, D);

    return emb;
}

// Simple overwrite-line progress bar
inline void showProgress(const char* label, int64_t cur, int64_t tot, double sec) {
    if (tot <= 0) return;
    double frac = (double)cur / tot;
    int pct  = (int)(100.0 * frac);
    const int W = 30;
    int fill = (int)(W * frac);
    double eta = (cur > 0) ? sec * (tot - cur) / cur : 0.0;
    std::printf("\r  [%s] |", label);
    for (int i = 0; i < W; ++i) std::putchar(i < fill ? '#' : '.');
    std::printf("| %3d%%  %.1fM/%.1fM", pct, cur / 1e6, tot / 1e6);
    if (cur > 0) std::printf("  [%.0fs ETA %.0fs]", sec, eta);
    std::printf("   ");
    std::fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main entry
// ═══════════════════════════════════════════════════════════════════════════

inline int runInnovationPipeline() {
    using Clock = std::chrono::high_resolution_clock;

    const std::string dataFile  = "Dataset.jsonl";
    const std::string outputDir = "output";
    const int windowSize = 5;

    std::cout << "\n"
        "═══════════════════════════════════════════\n"
        "  Innovation Detection Pipeline\n"
        "═══════════════════════════════════════════\n\n";

    // ── 1. Load global vocabulary ──────────────────────────────────────
    std::cout << "  [1/7] Loading global vocabulary...\n";
    Vocabulary vocab = loadFormattedVocab(outputDir + "/global_vocab.txt");
    if (vocab.empty()) {
        std::cerr << "  Error: vocab empty - run 'train' first.\n"; return 1; }
    const int V = vocab.size();
    std::cout << "         " << V << " tokens\n";

    // ── 2. Detect prediction years ─────────────────────────────────────
    std::cout << "  [2/7] Scanning embedding files...\n";
    auto predYears = detectPredictionYears(outputDir);
    if (predYears.empty()) {
        std::cerr << "  Error: no embeddings - run 'train' first.\n"; return 1; }
    std::sort(predYears.begin(), predYears.end());
    const int numYears     = (int)predYears.size();
    const int firstPred    = predYears.front();
    const int lastPred     = predYears.back();
    const int trainStart   = firstPred - windowSize;
    const int trainEnd     = firstPred - 1;
    std::cout << "         " << numYears << " years: "
              << firstPred << " - " << lastPred << "\n"
              << "         Training window: " << trainStart
              << " - " << trainEnd << "\n";

    // ── 3. Load all embeddings ─────────────────────────────────────────
    std::cout << "  [3/7] Loading embeddings...\n";
    int D = 0;
    { FILE* probe = std::fopen((outputDir + "/embeddings_"
                          + std::to_string(firstPred) + ".txt").c_str(), "r");
      int pV; std::fscanf(probe, "%d %d", &pV, &D); std::fclose(probe); }
    std::cout << "         Dimension: " << D << "\n";

    std::vector<std::vector<float>> embeddings(numYears);
    for (int i = 0; i < numYears; ++i) {
        embeddings[i] = loadEmbeddingFile(
            outputDir + "/embeddings_" + std::to_string(predYears[i]) + ".txt",
            V, D);
        std::printf("\r         Loaded %d/%d (%d)   ", i + 1, numYears,
                    predYears[i]);
        std::fflush(stdout);
    }
    std::cout << "\n";

    // ── 4. Parse dataset ───────────────────────────────────────────────
    std::cout << "  [4/7] Parsing dataset...\n";
    auto lines = readFile(dataFile);
    if (lines.empty()) {
        std::cerr << "  Error: cannot read " << dataFile << "\n"; return 1; }

    // year -> list of articles (each article = vector of vocab indices)
    std::map<int, std::vector<std::vector<int>>> articlesByYear;
    int nArticles = 0;
    for (const auto& line : lines) {
        int year = parseYear(line);
        if (year < 0) continue;
        auto codes = parseMeshCodes(line);
        std::vector<int> filt;
        for (int c : codes) { int idx = vocab.getIndex(c); if (idx >= 0) filt.push_back(idx); }
        if (filt.size() >= 2) { articlesByYear[year].push_back(std::move(filt)); ++nArticles; }
    }
    lines.clear(); lines.shrink_to_fit();
    std::cout << "         " << nArticles << " articles with ≥ 2 valid codes\n";

    // ── 5. Build co-occurrence set & active codes (training window) ────
    std::cout << "  [5/7] Building co-occurrence set ("
              << trainStart << "-" << trainEnd << ")...\n";
    auto t5 = Clock::now();

    std::unordered_set<int64_t> coOccurred;
    std::unordered_set<int>     activeSet;
    int trainArts = 0;

    for (int y = trainStart; y <= trainEnd; ++y) {
        auto it = articlesByYear.find(y);
        if (it == articlesByYear.end()) continue;
        for (const auto& art : it->second) {
            for (int c : art) activeSet.insert(c);
            for (size_t i = 0; i < art.size(); ++i)
                for (size_t j = i + 1; j < art.size(); ++j)
                    coOccurred.insert(encodePair(art[i], art[j]));
            ++trainArts;
        }
    }

    std::vector<int> active(activeSet.begin(), activeSet.end());
    std::sort(active.begin(), active.end());
    const int N = (int)active.size();
    const int64_t totalPossible = (int64_t)N * (N - 1) / 2;
    const int64_t nCoOcc        = (int64_t)coOccurred.size();
    const int64_t estInnov      = totalPossible - nCoOcc;

    double t5s = std::chrono::duration<double>(Clock::now() - t5).count();
    std::cout << "         Training articles : " << trainArts         << "\n"
              << "         Active codes      : " << N                 << "\n"
              << "         Co-occurring pairs: " << nCoOcc            << "\n"
              << "         Total possible    : " << totalPossible     << "\n"
              << "         Est. innovations  : " << estInnov          << "\n"
              << "         Time: " << t5s << "s\n";

    // ── 6. Build discovery map (prediction years) ──────────────────────
    std::cout << "  [6/7] Tracking discoveries "
              << firstPred << "-" << lastPred << "...\n";
    auto t6 = Clock::now();

    std::unordered_map<int64_t, int> discoveryMap;
    for (int y : predYears) {
        auto it = articlesByYear.find(y);
        if (it == articlesByYear.end()) continue;
        int newInYear = 0;
        for (const auto& art : it->second) {
            for (size_t i = 0; i < art.size(); ++i)
                for (size_t j = i + 1; j < art.size(); ++j) {
                    int64_t key = encodePair(art[i], art[j]);
                    if (coOccurred.count(key)) continue;       // already known
                    if (discoveryMap.find(key) == discoveryMap.end()) {
                        discoveryMap[key] = y;
                        ++newInYear;
                    }
                }
        }
        std::cout << "         " << y << ": " << newInYear << " new\n";
    }
    double t6s = std::chrono::duration<double>(Clock::now() - t6).count();
    std::cout << "         Total discovered: " << discoveryMap.size()
              << "  (" << t6s << "s)\n";

    // ── 7. Compute dot products & save ─────────────────────────────────
    std::cout << "  [7/7] Computing dot products & writing output...\n";

    double estGB = (double)estInnov * (12 + 4 * numYears)
                   / (1024.0 * 1024.0 * 1024.0);
    std::printf("         Est. binary size: %.2f GB\n", estGB);

    std::string binFile   = outputDir + "/innovations_all.bin";
    std::string csvFile   = outputDir + "/innovations_discovered.csv";
    std::string statsFile = outputDir + "/innovations_stats.txt";

    FILE* binFp = std::fopen(binFile.c_str(), "wb");
    if (!binFp) { std::cerr << "Error: cannot create " << binFile << "\n"; return 1; }
    std::setvbuf(binFp, nullptr, _IOFBF, 4 * 1024 * 1024);

    FILE* csvFp = std::fopen(csvFile.c_str(), "w");
    if (!csvFp) { std::cerr << "Error: cannot create " << csvFile << "\n"; return 1; }

    // Binary header (placeholder count, fill later)
    int32_t hdrPairs = 0, hdrYears = numYears;
    std::fwrite(&hdrPairs, 4, 1, binFp);
    std::fwrite(&hdrYears, 4, 1, binFp);
    for (int y : predYears) { int32_t yy = y; std::fwrite(&yy, 4, 1, binFp); }

    // CSV header
    std::fprintf(csvFp, "codeA,codeB,discovery_year");
    for (int y : predYears) std::fprintf(csvFp, ",dot_%d", y);
    std::fprintf(csvFp, "\n");

    auto tComp = Clock::now();
    int64_t numInnov     = 0;
    int64_t numDisc      = 0;
    int64_t pairsChecked = 0;
    std::vector<float> dotBuf(numYears);

    for (int a = 0; a < N; ++a) {
        for (int b = a + 1; b < N; ++b) {
            int idxA = active[a];
            int idxB = active[b];
            int64_t key = encodePair(idxA, idxB);
            ++pairsChecked;

            if (pairsChecked % 500000 == 0) {
                double el = std::chrono::duration<double>(
                                Clock::now() - tComp).count();
                showProgress("Pairs", pairsChecked, totalPossible, el);
            }

            if (coOccurred.count(key)) continue;

            ++numInnov;

            // Dot products for every year
            for (int y = 0; y < numYears; ++y)
                dotBuf[y] = w2v::dotProduct(
                    embeddings[y].data() + (size_t)idxA * D,
                    embeddings[y].data() + (size_t)idxB * D, D);

            int32_t discYear = -1;
            { auto dit = discoveryMap.find(key);
              if (dit != discoveryMap.end()) { discYear = dit->second; ++numDisc; } }

            // Binary record
            int32_t tA = vocab.idx2token[idxA];
            int32_t tB = vocab.idx2token[idxB];
            std::fwrite(&tA,        4, 1,        binFp);
            std::fwrite(&tB,        4, 1,        binFp);
            std::fwrite(&discYear,  4, 1,        binFp);
            std::fwrite(dotBuf.data(), 4, numYears, binFp);

            // CSV row (discovered only)
            if (discYear > 0) {
                std::fprintf(csvFp, "D%06d,D%06d,%d", tA, tB, discYear);
                for (int y = 0; y < numYears; ++y)
                    std::fprintf(csvFp, ",%.6g", dotBuf[y]);
                std::fprintf(csvFp, "\n");
            }
        }
    }

    // Finish progress
    {   double el = std::chrono::duration<double>(Clock::now() - tComp).count();
        showProgress("Pairs", totalPossible, totalPossible, el);
        std::cout << "\n"; }

    // Patch binary header with actual count
    std::fseek(binFp, 0, SEEK_SET);
    hdrPairs = (int32_t)numInnov;
    std::fwrite(&hdrPairs, 4, 1, binFp);
    std::fclose(binFp);
    std::fclose(csvFp);

    double totalTime = std::chrono::duration<double>(Clock::now() - tComp).count();

    // ── Stats file ─────────────────────────────────────────────────────
    {
        FILE* sf = std::fopen(statsFile.c_str(), "w");
        if (sf) {
            std::fprintf(sf, "training_years: %d-%d\n", trainStart, trainEnd);
            std::fprintf(sf, "prediction_years: %d-%d\n", firstPred, lastPred);
            std::fprintf(sf, "active_codes: %d\n", N);
            std::fprintf(sf, "total_possible_pairs: %lld\n",
                         (long long)totalPossible);
            std::fprintf(sf, "co_occurred_pairs: %lld\n", (long long)nCoOcc);
            std::fprintf(sf, "innovation_candidates: %lld\n",
                         (long long)numInnov);
            std::fprintf(sf, "discovered_pairs: %lld\n", (long long)numDisc);
            std::fprintf(sf, "computation_seconds: %.1f\n", totalTime);
            std::fclose(sf);
        }
    }

    // ── Summary ────────────────────────────────────────────────────────
    std::cout
        << "\n═══════════════════════════════════════════\n"
        << "  INNOVATION PIPELINE COMPLETE\n"
        << "═══════════════════════════════════════════\n"
        << "  Active codes       : " << N          << "\n"
        << "  Innovation pairs   : " << numInnov   << "\n"
        << "  Discovered pairs   : " << numDisc    << "\n"
        << "  Computation time   : " << totalTime  << "s\n"
        << "  Binary output      : " << binFile    << "\n"
        << "  Discovered CSV     : " << csvFile    << "\n"
        << "  Stats              : " << statsFile  << "\n"
        << "═══════════════════════════════════════════\n";

    return 0;
}
