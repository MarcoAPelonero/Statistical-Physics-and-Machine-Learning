#pragma once

// ===========================================================================
//  nullModel.hpp  -  Cosine-similarity null model for innovation filtering
//
//  For each prediction year, estimate the null distribution of cosine
//  similarities among non-co-occurring code pairs by random sampling.
//  This provides a per-year baseline (μ, σ) against which each innovation
//  couple's embedding similarity can be compared via z-scores.
//
//  Steps:
//    1. Load embeddings for all prediction years
//    2. Build the baseline co-occurrence set from the initial training window
//    3. Sample N random non-co-occurring pairs from active codes
//    4. For each year, compute cosine similarity of the sampled pairs -> mu_Y, sigma_Y
//    5. For each innovation couple, compute z = (cos_sim - μ_Y) / σ_Y
//    6. Filter couples with z > threshold at discovery year
//
//  Output: output/null_model_filtered.csv
// ===========================================================================

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "importUtils.hpp"
#include "innovation.hpp"   // encodePair, parseDCode, loadFormattedVocab, etc.

// ─── Configuration ──────────────────────────────────────────────────────────

struct NullModelConfig {
    int      nullSamples    = 50000;   // random non-co-occurring pairs to sample
    double   zThreshold     = 3.0;     // z-score filter threshold
    int      numThreads     = 6;       // OpenMP threads
    unsigned seed           = 42;      // master RNG seed
    bool     verbose        = true;
};

// ─── Innovation couple loaded from CSV ──────────────────────────────────────

struct InnovationCouple {
    int      codeA, codeB;               // raw D-codes (e.g. 2, 3634)
    int      idxA, idxB;                 // vocab indices
    int      discoveryYear;
    int64_t  pairKey;                    // encodePair(idxA, idxB)
    std::vector<float> dotProducts;      // cosine sims: dot_2005, …, dot_2020
};

inline std::vector<InnovationCouple> loadInnovationsCSV(
    const std::string& path,
    const Vocabulary& vocab)
{
    std::vector<InnovationCouple> couples;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "  Error: cannot open " << path << "\n";
        return couples;
    }

    std::string line;
    std::getline(in, line);   // skip header

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        InnovationCouple ic;
        std::istringstream ss(line);
        std::string tok;

        // codeA
        std::getline(ss, tok, ',');
        ic.codeA = parseDCode(tok);

        // codeB
        std::getline(ss, tok, ',');
        ic.codeB = parseDCode(tok);

        // discovery_year
        std::getline(ss, tok, ',');
        ic.discoveryYear = std::stoi(tok);

        // dot products (one per prediction year)
        while (std::getline(ss, tok, ','))
            ic.dotProducts.push_back(std::stof(tok));

        // Resolve vocab indices
        ic.idxA = vocab.getIndex(ic.codeA);
        ic.idxB = vocab.getIndex(ic.codeB);
        if (ic.idxA < 0 || ic.idxB < 0) continue;

        ic.pairKey = encodePair(ic.idxA, ic.idxB);
        couples.push_back(std::move(ic));
    }
    return couples;
}

// ─── Per-year null distribution ─────────────────────────────────────────────

struct YearNullDist {
    float mu    = 0.0f;
    float sigma = 0.0f;
};

// ======================================================================================
//  Main entry
// ======================================================================================

inline int runNullModelPipeline() {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    const std::string dataFile   = "Dataset.jsonl";
    const std::string outputDir  = "output";
    const std::string csvFile    = outputDir + "/innovations_discovered.csv";
    const std::string outFile    = outputDir + "/null_model_filtered.csv";
    const std::string zScoreFile = outputDir + "/null_model_zscores.csv";
    const int windowSize = 5;

    NullModelConfig config;

    std::cout << "\n"
        "================================================================================\n"
        "  Null Model Pipeline  (cosine-sim null)\n"
        "================================================================================\n"
        "  Config:\n"
        "    Null samples          : " << config.nullSamples << "\n"
        "    Z threshold           : " << config.zThreshold << "\n\n";

    // ── 1. Load global vocabulary ──────────────────────────────────────
    std::cout << "  [1/7] Loading global vocabulary...\n";
    Vocabulary vocab = loadFormattedVocab(outputDir + "/global_vocab.txt");
    if (vocab.empty()) {
        std::cerr << "  Error: vocab empty - run 'train' first.\n"; return 1;
    }
    const int V = vocab.size();
    std::cout << "         " << V << " tokens\n";

    // ── 2. Detect prediction years ─────────────────────────────────────
    std::cout << "  [2/7] Detecting prediction years...\n";
    auto predYears = detectPredictionYears(outputDir);
    if (predYears.empty()) {
        std::cerr << "  Error: no embeddings found - run 'train' first.\n";
        return 1;
    }
    std::sort(predYears.begin(), predYears.end());
    const int numYears     = (int)predYears.size();
    const int firstPred    = predYears.front();
    const int trainStart   = firstPred - windowSize;
    const int trainEnd     = firstPred - 1;
    std::cout << "         " << numYears << " years: "
              << predYears.front() << " - " << predYears.back() << "\n"
              << "         Baseline window: " << trainStart
              << " - " << trainEnd << "\n";

    // ── 3. Load all embeddings ─────────────────────────────────────────
    std::cout << "  [3/7] Loading embeddings...\n";
    int D = 0;
    { FILE* probe = std::fopen((outputDir + "/embeddings_"
                          + std::to_string(firstPred) + ".txt").c_str(), "r");
      if (!probe) { std::cerr << "  Error: cannot open embedding probe file\n"; return 1; }
      int pV; std::fscanf(probe, "%d %d", &pV, &D); std::fclose(probe); }
    std::cout << "         Dimension: " << D << "\n";

    std::vector<std::vector<float>> embeddings(numYears);
    for (int i = 0; i < numYears; ++i) {
        embeddings[i] = loadEmbeddingFile(
            outputDir + "/embeddings_" + std::to_string(predYears[i]) + ".txt",
            V, D);
        if (config.verbose)
            std::printf("\r         Loaded %d/%d (%d)   ", i + 1, numYears,
                        predYears[i]);
    }
    std::cout << "\n";

    // ── 4. Load innovations CSV ────────────────────────────────────────
    std::cout << "  [4/7] Loading innovations_discovered.csv...\n";
    auto innovations = loadInnovationsCSV(csvFile, vocab);
    std::cout << "         " << innovations.size() << " innovation couples loaded\n";

    // Group by discovery year for summary
    std::map<int, int> innovCountByYear;
    for (const auto& ic : innovations)
        innovCountByYear[ic.discoveryYear]++;
    for (auto& [y, cnt] : innovCountByYear)
        std::cout << "         " << y << ": " << cnt << " couples\n";

    // ── 5. Parse dataset & build baseline co-occurrence set ────────────
    std::cout << "  [5/7] Building baseline co-occurrence set ("
              << trainStart << "-" << trainEnd << ")...\n";
    auto tParse = Clock::now();

    auto lines = readFile(dataFile);
    if (lines.empty()) {
        std::cerr << "  Error: cannot read " << dataFile << "\n"; return 1;
    }

    // Parse articles by year (vocab-index representation)
    std::map<int, std::vector<std::vector<int>>> articlesByYear;
    int nArticles = 0;
    for (const auto& line : lines) {
        int year = parseYear(line);
        if (year < 0) continue;
        auto codes = parseMeshCodes(line);
        std::vector<int> filt;
        for (int c : codes) {
            int idx = vocab.getIndex(c);
            if (idx >= 0) filt.push_back(idx);
        }
        if (filt.size() >= 2) {
            articlesByYear[year].push_back(std::move(filt));
            ++nArticles;
        }
    }
    lines.clear(); lines.shrink_to_fit();

    // Build co-occurrence set from baseline training window
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
    const int64_t nCoOcc = (int64_t)coOccurred.size();

    double tParseS = std::chrono::duration<double>(Clock::now() - tParse).count();
    std::cout << "         Articles parsed   : " << nArticles << "\n"
              << "         Training articles : " << trainArts << "\n"
              << "         Active codes      : " << N << "\n"
              << "         Co-occurring pairs: " << nCoOcc << "\n"
              << "         Time: " << tParseS << "s\n";

    if (N < 2) {
        std::cerr << "  Error: need at least 2 active codes.\n"; return 1;
    }

    // ── 6. Sample null pairs & compute per-year null distribution ──────
    std::cout << "  [6/7] Sampling " << config.nullSamples
              << " random non-co-occurring pairs...\n";
    auto tNull = Clock::now();

    // Sample random non-co-occurring pairs via rejection sampling
    struct SampledPair { int idxA, idxB; };
    std::vector<SampledPair> sampledPairs;
    sampledPairs.reserve(config.nullSamples);

    {
        std::mt19937 rng(config.seed);
        std::uniform_int_distribution<int> dist(0, N - 1);

        // Use a set to avoid duplicate pairs in the sample
        std::unordered_set<int64_t> sampled;
        int attempts = 0;
        const int maxAttempts = config.nullSamples * 20;

        while ((int)sampledPairs.size() < config.nullSamples
               && attempts < maxAttempts) {
            int a = dist(rng);
            int b = dist(rng);
            if (a == b) { ++attempts; continue; }
            int idxA = active[a];
            int idxB = active[b];
            int64_t key = encodePair(idxA, idxB);
            if (coOccurred.count(key)) { ++attempts; continue; }
            if (sampled.count(key))    { ++attempts; continue; }
            sampled.insert(key);
            sampledPairs.push_back({idxA, idxB});
            ++attempts;
        }
    }

    std::cout << "         Sampled " << sampledPairs.size() << " unique pairs\n";

    // Compute cosine-similarity null distribution per year
    std::vector<YearNullDist> nullDist(numYears);

    for (int yi = 0; yi < numYears; ++yi) {
        const float* emb = embeddings[yi].data();
        double sum = 0.0, sumSq = 0.0;
        const int count = (int)sampledPairs.size();

        for (const auto& sp : sampledPairs) {
            float dot = w2v::dotProduct(
                emb + (size_t)sp.idxA * D,
                emb + (size_t)sp.idxB * D, D);
            sum   += dot;
            sumSq += (double)dot * dot;
        }

        if (count > 0) {
            nullDist[yi].mu = (float)(sum / count);
            double var = sumSq / count
                       - (double)nullDist[yi].mu * nullDist[yi].mu;
            nullDist[yi].sigma = (var > 0.0) ? (float)std::sqrt(var) : 0.0f;
        }

        if (config.verbose)
            std::printf("         %d: mu = %.4f, sigma = %.4f\n",
                        predYears[yi], nullDist[yi].mu, nullDist[yi].sigma);
    }

    double tNullS = std::chrono::duration<double>(Clock::now() - tNull).count();
    std::cout << "         Null model time: " << tNullS << "s\n";

    // ── 7. Compute z-scores & write output ─────────────────────────────
    std::cout << "  [7/7] Computing z-scores & writing output...\n";

    // Z-score log (per-couple, per-year)
    FILE* zFp = std::fopen(zScoreFile.c_str(), "w");
    if (zFp) {
        std::fprintf(zFp, "codeA,codeB,discovery_year");
        for (int y : predYears)
            std::fprintf(zFp, ",z_%d", y);
        std::fprintf(zFp, ",decision\n");
    }

    struct OutputRow {
        int   coupleIdx;
        float discoveryZ;
        int   lastSurprisingYear;
    };
    std::vector<OutputRow> survivors;

    int accepted = 0, rejected = 0, skipped = 0;

    for (int i = 0; i < (int)innovations.size(); ++i) {
        const auto& ic = innovations[i];

        // Find discovery year index
        int discYi = -1;
        for (int yi = 0; yi < numYears; ++yi)
            if (predYears[yi] == ic.discoveryYear) { discYi = yi; break; }
        if (discYi < 0 || discYi >= (int)ic.dotProducts.size()) {
            ++skipped;
            continue;
        }

        // Compute z at discovery year
        float dot_disc = ic.dotProducts[discYi];
        float z_disc = 0.0f;
        if (nullDist[discYi].sigma > 1e-12f)
            z_disc = (dot_disc - nullDist[discYi].mu) / nullDist[discYi].sigma;

        // Log z-scores for all years
        if (zFp) {
            std::fprintf(zFp, "D%06d,D%06d,%d", ic.codeA, ic.codeB,
                         ic.discoveryYear);
            for (int yi = 0; yi < numYears; ++yi) {
                float z = 0.0f;
                if (yi < (int)ic.dotProducts.size() &&
                    nullDist[yi].sigma > 1e-12f)
                    z = (ic.dotProducts[yi] - nullDist[yi].mu)
                      / nullDist[yi].sigma;
                std::fprintf(zFp, ",%.4f", z);
            }
            std::fprintf(zFp, ",%s\n",
                         (z_disc > config.zThreshold) ? "ACCEPT" : "REJECT");
        }

        // Filter: z at discovery must exceed threshold
        if (z_disc <= config.zThreshold) {
            ++rejected;
            continue;
        }
        ++accepted;

        // Find last year where z > threshold (starting from discovery)
        int lastSurp = ic.discoveryYear;
        for (int yi = discYi + 1; yi < numYears; ++yi) {
            if (yi >= (int)ic.dotProducts.size()) break;
            float z = 0.0f;
            if (nullDist[yi].sigma > 1e-12f)
                z = (ic.dotProducts[yi] - nullDist[yi].mu)
                  / nullDist[yi].sigma;
            if (z > config.zThreshold)
                lastSurp = predYears[yi];
            else
                break;  // stop tracking once it falls below threshold
        }

        survivors.push_back({i, z_disc, lastSurp});
    }

    if (zFp) std::fclose(zFp);

    // Sort by discovery year, then codes
    std::sort(survivors.begin(), survivors.end(),
              [&](const OutputRow& a, const OutputRow& b) {
                  const auto& ia = innovations[a.coupleIdx];
                  const auto& ib = innovations[b.coupleIdx];
                  if (ia.discoveryYear != ib.discoveryYear)
                      return ia.discoveryYear < ib.discoveryYear;
                  if (ia.codeA != ib.codeA) return ia.codeA < ib.codeA;
                  return ia.codeB < ib.codeB;
              });

    // Write filtered output CSV
    FILE* outFp = std::fopen(outFile.c_str(), "w");
    if (!outFp) {
        std::cerr << "  Error: cannot create " << outFile << "\n";
        return 1;
    }

    // Header
    std::fprintf(outFp, "codeA,codeB,discovery_year,z_score,last_surprising_year");
    for (int y : predYears) std::fprintf(outFp, ",dot_%d", y);
    for (int y : predYears) std::fprintf(outFp, ",mu_%d", y);
    for (int y : predYears) std::fprintf(outFp, ",sigma_%d", y);
    std::fprintf(outFp, "\n");

    for (const auto& row : survivors) {
        const auto& ic = innovations[row.coupleIdx];
        std::fprintf(outFp, "D%06d,D%06d,%d,%.4f,%d",
                     ic.codeA, ic.codeB, ic.discoveryYear,
                     row.discoveryZ, row.lastSurprisingYear);

        // Dot products (raw cosine similarities from innovation pipeline)
        for (int yi = 0; yi < numYears; ++yi) {
            float v = (yi < (int)ic.dotProducts.size())
                    ? ic.dotProducts[yi] : 0.0f;
            std::fprintf(outFp, ",%.6g", v);
        }

        // Per-year null model mu (GLOBAL - same for all couples)
        for (int yi = 0; yi < numYears; ++yi)
            std::fprintf(outFp, ",%.6g", nullDist[yi].mu);

        // Per-year null model sigma (GLOBAL - same for all couples)
        for (int yi = 0; yi < numYears; ++yi)
            std::fprintf(outFp, ",%.6g", nullDist[yi].sigma);

        std::fprintf(outFp, "\n");
    }
    std::fclose(outFp);

    double totalTime = std::chrono::duration<double>(Clock::now() - t0).count();

    std::cout << "\n"
        "================================================================================\n"
        "  NULL MODEL PIPELINE COMPLETE\n"
        "================================================================================\n"
        "  Total innovations in CSV  : " << innovations.size()   << "\n"
        "  Skipped (year mismatch)   : " << skipped              << "\n"
        "  Accepted (z > threshold)  : " << accepted             << "\n"
        "  Rejected (z <= threshold) : " << rejected             << "\n"
        "  Survivors in output       : " << survivors.size()     << "\n"
        "  Output                    : " << outFile              << "\n"
        "  Z-score log               : " << zScoreFile           << "\n"
        "  Total time                : " << totalTime            << "s\n"
        "================================================================================\n";

    return 0;
}
