#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  trainPipeline.hpp  –  Batched, OpenMP-parallel training pipeline
//
//  Generic: input entries have entry[0] = metadata (skipped), entry[1..n] =
//  tokens.  Change the data, swap the parser, this pipeline still works.
//
//  Workflow:
//    1.  Build vocabulary (with min-count filtering)
//    2.  Generate full-context training pairs
//    3.  For each epoch, shuffle then process pairs in configurable batches
//        using Hogwild!-parallel SGNS with linear LR decay
//
//  Use `TrainingPipeline::run()` for the one-call path, or call the
//  individual public helpers for a custom flow.
// ═══════════════════════════════════════════════════════════════════════════

#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "prepareData.hpp"
#include "word2vec.hpp"

// ── Hyperparameters ─────────────────────────────────────────────────────────

struct TrainingConfig {
    int      embeddingDim    = 100;      // dimensionality
    int      negSamples      = 5;        // negatives per positive pair
    int      epochs          = 5;        // full passes over the data
    int      batchSize       = 4096;     // pairs per mini-batch
    float    learningRate    = 0.025f;   // initial α
    float    minLearningRate = 0.0001f;  // floor for α decay
    int      minCount        = 1;        // discard tokens rarer than this
    float    trimLowPct      = 0.05f;    // drop bottom frequency percentile
    float    trimHighPct     = 0.05f;    // drop top frequency percentile
    int      numThreads      = 4;        // OpenMP threads
    unsigned seed            = 42;       // master RNG seed
    bool     verbose         = true;     // progress output
};

// ── Pipeline result ─────────────────────────────────────────────────────────

struct TrainingResult {
    w2v::Word2VecModel model;
    Vocabulary         vocab;
};

// ── Pipeline ────────────────────────────────────────────────────────────────

class TrainingPipeline {
public:

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║  One-call convenience: vocab → pairs → train → return               ║
    // ╚═══════════════════════════════════════════════════════════════════════╝

    static TrainingResult run(
        const std::vector<std::vector<int>>& entries,
        const TrainingConfig& config = TrainingConfig())
    {
        if (config.verbose)
            std::cout << "\n══════ Training Pipeline ══════\n\n";

        // 1. Vocabulary
        Vocabulary vocab;
        vocab.build(entries, config.minCount, config.trimLowPct, config.trimHighPct);
        if (vocab.empty()) {
            std::cerr << "Error: vocabulary empty after filtering (minCount="
                      << config.minCount << ")\n";
            return { w2v::Word2VecModel(), vocab };
        }
        if (config.verbose) {
            std::cout << "[vocab]\n";
            vocab.printStats();
        }

        // 2. Training pairs
        auto pairs = generateTrainingPairs(entries, vocab);
        if (config.verbose)
            std::cout << "\n[pairs] " << pairs.size()
                      << " training pairs generated\n";

        // 3. Model
        w2v::Word2VecModel model;
        model.init(vocab.size(), config.embeddingDim, vocab, config.seed);
        if (config.verbose)
            std::cout << "[model] " << vocab.size() << " x "
                      << config.embeddingDim << " embeddings initialised\n\n";

        // 4. Train
        train(model, pairs, config);

        return { std::move(model), std::move(vocab) };
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║  Public helper: train an already-initialised model on given pairs    ║
    // ╚═══════════════════════════════════════════════════════════════════════╝

    static void train(
        w2v::Word2VecModel& model,
        std::vector<TrainingPair>& pairs,
        const TrainingConfig& config)
    {
        if (pairs.empty()) {
            if (config.verbose) std::cout << "[skip] no training pairs\n";
            return;
        }

        const size_t totalPairs = pairs.size();
        const size_t totalWork  = totalPairs * config.epochs;
        size_t       workDone   = 0;

        int numThreads = config.numThreads;
#ifdef _OPENMP
        omp_set_num_threads(numThreads);
#else
        numThreads = 1;
#endif

        const int batchSize = std::max(config.batchSize, 1);
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < config.epochs; ++epoch) {

            // Deterministic shuffle per epoch
            {
                std::mt19937 shuffleRng(config.seed + (unsigned)epoch);
                std::shuffle(pairs.begin(), pairs.end(), shuffleRng);
            }

            double epochLoss   = 0.0;
            size_t numBatches  = (totalPairs + batchSize - 1) / batchSize;

            for (size_t b = 0; b < numBatches; ++b) {

                size_t bStart = b * batchSize;
                size_t bEnd   = std::min(bStart + (size_t)batchSize, totalPairs);
                int    bLen   = (int)(bEnd - bStart);

                // Linear learning-rate decay over the whole run
                float lr = config.learningRate
                         * (1.0f - (float)workDone / (float)totalWork);
                lr = std::max(lr, config.minLearningRate);

                double batchLoss = 0.0;

                // ── Parallel batch (Hogwild!) ──────────────────────────────
#pragma omp parallel reduction(+:batchLoss)
                {
                    int tid = 0;
#ifdef _OPENMP
                    tid = omp_get_thread_num();
#endif
                    std::mt19937 rng(config.seed
                                    + (unsigned)epoch * 100000u
                                    + (unsigned)b     * 1000u
                                    + (unsigned)tid);
                    std::vector<float> gradBuf(model.embDim, 0.0f);

#pragma omp for schedule(static)
                    for (int i = 0; i < bLen; ++i) {
                        size_t idx = bStart + (size_t)i;
                        batchLoss += model.trainPairWithBuffer(
                            pairs[idx].center, pairs[idx].context,
                            config.negSamples, lr, rng, gradBuf);
                    }
                }

                epochLoss += batchLoss;
                workDone  += (size_t)bLen;
            }

            // ── Epoch summary ──────────────────────────────────────────────
            if (config.verbose) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed =
                    std::chrono::duration<double>(now - t0).count();
                std::cout << "  [epoch " << (epoch + 1) << "/" << config.epochs
                          << "]  avg_loss=" << (epochLoss / totalPairs)
                          << "  lr=" << (config.learningRate
                                        * (1.0f - (float)workDone / (float)totalWork))
                          << "  " << elapsed << "s\n";
            }
        }

        // ── Summary ────────────────────────────────────────────────────────
        if (config.verbose) {
            auto now   = std::chrono::high_resolution_clock::now();
            double tot = std::chrono::duration<double>(now - t0).count();
            double pps = (double)totalWork / tot;
            std::cout << "\n[done] " << tot << "s total, "
                      << (pps / 1e6) << " M pairs/s\n";
        }
    }
};
