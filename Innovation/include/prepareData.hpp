#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  prepareData.hpp  –  Vocabulary construction & training-pair generation
//
//  Generic: expects entries where entry[0] = metadata (e.g. year, skipped)
//           and entry[1..n] = context tokens (ints).
// ═══════════════════════════════════════════════════════════════════════════

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

// ── Vocabulary ──────────────────────────────────────────────────────────────

struct Vocabulary {
    std::unordered_map<int, int> token2idx;   // original token → contiguous index
    std::vector<int>             idx2token;   // contiguous index → original token
    std::vector<int>             frequencies; // contiguous index → count
    int totalTokens = 0;

    // Build from dataset entries. Tokens rarer than minCount are discarded.
    // Optionally trim bottom/top frequency percentiles.
    void build(const std::vector<std::vector<int>>& entries,
               int minCount = 1,
               double trimLowPct = 0.0,
               double trimHighPct = 0.0) {
        std::unordered_map<int, int> rawCounts;
        for (const auto& entry : entries) {
            for (size_t i = 1; i < entry.size(); ++i)   // skip entry[0] (metadata)
                rawCounts[entry[i]]++;
        }

        token2idx.clear();
        idx2token.clear();
        frequencies.clear();
        totalTokens = 0;

        std::vector<std::pair<int, int>> filtered;
        filtered.reserve(rawCounts.size());
        for (const auto& kv : rawCounts) {
            if (kv.second >= minCount)
                filtered.push_back(kv);
        }

        // Deterministic ordering (sorted by original token value)
        std::sort(filtered.begin(), filtered.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::unordered_set<int> dropTokens;
        if (!filtered.empty() && (trimLowPct > 0.0 || trimHighPct > 0.0)) {
            std::vector<std::pair<int, int>> byFreq = filtered;
            std::sort(byFreq.begin(), byFreq.end(),
                      [](const auto& a, const auto& b) {
                          if (a.second != b.second) return a.second < b.second;
                          return a.first < b.first;
                      });

            const int n = (int)byFreq.size();
            int trimLow  = (trimLowPct  > 0.0) ? (int)(n * trimLowPct)  : 0;
            int trimHigh = (trimHighPct > 0.0) ? (int)(n * trimHighPct) : 0;

            if (trimLow + trimHigh >= n && n > 0) {
                trimLow  = std::min(trimLow,  n - 1);
                trimHigh = std::min(trimHigh, n - 1 - trimLow);
            }

            for (int i = 0; i < trimLow; ++i)
                dropTokens.insert(byFreq[i].first);
            for (int i = 0; i < trimHigh; ++i)
                dropTokens.insert(byFreq[n - 1 - i].first);
        }

        for (const auto& [token, count] : filtered) {
            if (!dropTokens.empty() &&
                dropTokens.find(token) != dropTokens.end())
                continue;
            int idx = (int)idx2token.size();
            token2idx[token] = idx;
            idx2token.push_back(token);
            frequencies.push_back(count);
            totalTokens += count;
        }
    }

    int  size()     const { return (int)idx2token.size(); }
    bool empty()    const { return idx2token.empty(); }

    bool contains(int token) const {
        return token2idx.find(token) != token2idx.end();
    }

    int getIndex(int token) const {
        auto it = token2idx.find(token);
        return (it != token2idx.end()) ? it->second : -1;
    }

    // ── I/O ─────────────────────────────────────────────────────────────────

    bool save(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) return false;
        out << size() << "\n";
        for (int i = 0; i < size(); ++i)
            out << idx2token[i] << " " << frequencies[i] << "\n";
        return out.good();
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) return false;
        int n;  in >> n;
        token2idx.clear();
        idx2token.resize(n);
        frequencies.resize(n);
        totalTokens = 0;
        for (int i = 0; i < n; ++i) {
            int tok, freq;
            in >> tok >> freq;
            idx2token[i]   = tok;
            frequencies[i] = freq;
            token2idx[tok] = i;
            totalTokens   += freq;
        }
        return in.good();
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────

    void printStats() const {
        if (empty()) { std::cout << "  (empty vocabulary)\n"; return; }
        int maxFreq = *std::max_element(frequencies.begin(), frequencies.end());
        int minFreq = *std::min_element(frequencies.begin(), frequencies.end());
        double avgFreq = (double)totalTokens / size();
        std::cout << "  Vocab size    : " << size()       << "\n"
                  << "  Total tokens  : " << totalTokens  << "\n"
                  << "  Min frequency : " << minFreq      << "\n"
                  << "  Max frequency : " << maxFreq      << "\n"
                  << "  Avg frequency : " << avgFreq      << "\n";
    }
};

// ── Training pair ───────────────────────────────────────────────────────────

struct TrainingPair {
    int center;   // vocab index of centre word
    int context;  // vocab index of context word
};

// ── Full-context pair generation ────────────────────────────────────────────
//  For every entry, every *ordered* pair (i, j) with i ≠ j among the valid
//  tokens becomes one training pair.  Correct for unordered-set data such as
//  MeSH codes (avoids imposing a false ordering).

inline std::vector<TrainingPair> generateTrainingPairs(
    const std::vector<std::vector<int>>& entries,
    const Vocabulary& vocab)
{
    // Pre-count for a single reservation
    size_t totalPairs = 0;
    for (const auto& entry : entries) {
        int valid = 0;
        for (size_t i = 1; i < entry.size(); ++i)
            if (vocab.contains(entry[i])) ++valid;
        totalPairs += (size_t)valid * (valid - 1);
    }

    std::vector<TrainingPair> pairs;
    pairs.reserve(totalPairs);

    for (const auto& entry : entries) {
        // Collect vocab indices for this entry's tokens
        std::vector<int> indices;
        indices.reserve(entry.size() - 1);
        for (size_t i = 1; i < entry.size(); ++i) {
            int idx = vocab.getIndex(entry[i]);
            if (idx >= 0) indices.push_back(idx);
        }
        // Full context — all ordered pairs
        for (size_t i = 0; i < indices.size(); ++i)
            for (size_t j = 0; j < indices.size(); ++j)
                if (i != j) pairs.push_back({indices[i], indices[j]});
    }
    return pairs;
}

// ── Utility: shuffle pairs in-place (deterministic) ─────────────────────────

inline void shufflePairs(std::vector<TrainingPair>& pairs, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::shuffle(pairs.begin(), pairs.end(), rng);
}
