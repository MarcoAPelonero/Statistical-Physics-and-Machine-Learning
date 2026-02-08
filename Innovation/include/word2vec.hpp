#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  word2vec.hpp  –  Skip-Gram with Negative Sampling (SGNS)
//
//  • SIMD-accelerated dot-product & axpy  (AVX → SSE → scalar fallback)
//  • Negative-sampling via pre-built unigram^0.75 table
//  • Hogwild!-safe trainPairWithBuffer (thread-local grad buffer, no locks)
//  • Utilities: save/load weights, text export, cosine similarity,
//    nearest neighbours, analogy, L2 normalisation.
// ═══════════════════════════════════════════════════════════════════════════

#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>
#include <cstdio>

// ── SIMD headers (compile-time detection) ───────────────────────────────────
#if defined(__AVX__) || defined(__AVX2__)
  #include <immintrin.h>
  #define W2V_USE_AVX
#elif defined(__SSE__)
  #include <xmmintrin.h>
  #define W2V_USE_SSE
#endif

#include "prepareData.hpp"

namespace w2v {

// ═════════════════════════════════════════════════════════════════════════════
//  SIMD-accelerated primitives
// ═════════════════════════════════════════════════════════════════════════════

inline float dotProduct(const float* a, const float* b, int n) {
    float sum = 0.0f;
    int i = 0;

#ifdef W2V_USE_AVX
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }
    // 256 → 128
    __m128 hi  = _mm256_extractf128_ps(vsum, 1);
    __m128 lo  = _mm256_castps256_ps128(vsum);
    __m128 r4  = _mm_add_ps(lo, hi);
    // horizontal sum (SSE1-compatible)
    __m128 shuf = _mm_shuffle_ps(r4, r4, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(r4, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum  = _mm_cvtss_f32(sums);

#elif defined(W2V_USE_SSE)
    __m128 vsum = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
    }
    __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(vsum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum  = _mm_cvtss_f32(sums);
#endif

    for (; i < n; ++i) sum += a[i] * b[i];        // scalar tail
    return sum;
}

// y[i] += alpha * x[i]
inline void axpy(float alpha, const float* x, float* y, int n) {
    int i = 0;

#ifdef W2V_USE_AVX
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        vy = _mm256_add_ps(vy, _mm256_mul_ps(va, vx));
        _mm256_storeu_ps(y + i, vy);
    }
#elif defined(W2V_USE_SSE)
    __m128 va = _mm_set1_ps(alpha);
    for (; i + 3 < n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vy = _mm_loadu_ps(y + i);
        vy = _mm_add_ps(vy, _mm_mul_ps(va, vx));
        _mm_storeu_ps(y + i, vy);
    }
#endif

    for (; i < n; ++i) y[i] += alpha * x[i];       // scalar tail
}

// ═════════════════════════════════════════════════════════════════════════════
//  Fast sigmoid (clamped for numerical safety)
// ═════════════════════════════════════════════════════════════════════════════

inline float sigmoid(float x) {
    if (x >  6.0f) return 1.0f;
    if (x < -6.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Negative-sampling table  (unigram^0.75 distribution)
// ═════════════════════════════════════════════════════════════════════════════

class NegativeSampler {
    std::vector<int> table;
    static constexpr int TABLE_SIZE = 10'000'000;

public:
    NegativeSampler() = default;

    void build(const Vocabulary& vocab) {
        if (vocab.empty()) return;
        table.resize(TABLE_SIZE);

        double totalPow = 0.0;
        std::vector<double> powered(vocab.size());
        for (int i = 0; i < vocab.size(); ++i) {
            powered[i] = std::pow((double)vocab.frequencies[i], 0.75);
            totalPow  += powered[i];
        }

        int    idx        = 0;
        double cumulative = powered[0] / totalPow;
        for (int i = 0; i < TABLE_SIZE; ++i) {
            table[i] = idx;
            if ((double)(i + 1) / TABLE_SIZE > cumulative &&
                idx + 1 < vocab.size()) {
                ++idx;
                cumulative += powered[idx] / totalPow;
            }
        }
    }

    inline int sample(std::mt19937& rng) const {
        if (table.empty()) return 0;
        return table[rng() % TABLE_SIZE];
    }
};

// ═════════════════════════════════════════════════════════════════════════════
//  Word2Vec Model
// ═════════════════════════════════════════════════════════════════════════════

class Word2VecModel {
public:
    int vocabSize = 0;
    int embDim    = 0;

    std::vector<float> W_in;       // input  embeddings  [vocabSize × embDim]
    std::vector<float> W_out;      // output embeddings  [vocabSize × embDim]
    NegativeSampler    sampler;

    Word2VecModel() = default;

    // ── Initialisation ──────────────────────────────────────────────────────

    void init(int vocabSize_, int embDim_,
              const Vocabulary& vocab, unsigned seed = 42)
    {
        vocabSize = vocabSize_;
        embDim    = embDim_;

        W_in .resize((size_t)vocabSize * embDim);
        W_out.resize((size_t)vocabSize * embDim);

        // Xavier-style uniform init: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        // Simplified to U(-1/sqrt(d), 1/sqrt(d)) for embeddings
        std::mt19937 rng(seed);
        float scale = 1.0f / std::sqrt((float)embDim);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto& w : W_in)  w = dist(rng);
        for (auto& w : W_out) w = dist(rng);

        sampler.build(vocab);
    }

    // ── Raw pointer access (row-major layout) ───────────────────────────────

    float*       inputVec (int idx)       { return W_in .data() + (size_t)idx * embDim; }
    const float* inputVec (int idx) const { return W_in .data() + (size_t)idx * embDim; }
    float*       outputVec(int idx)       { return W_out.data() + (size_t)idx * embDim; }
    const float* outputVec(int idx) const { return W_out.data() + (size_t)idx * embDim; }

    // ═════════════════════════════════════════════════════════════════════════
    //  Training  (SGNS on a single centre–context pair)
    // ═════════════════════════════════════════════════════════════════════════

    // Efficient version: caller supplies a pre-allocated gradient buffer
    // of size embDim (avoids allocation in the hot loop).
    // Thread-safe under Hogwild! (benign races on shared weight rows).

    float trainPairWithBuffer(int center, int context,
                              int negSamples, float lr,
                              std::mt19937& rng,
                              std::vector<float>& grad)
    {
        float* vIn = inputVec(center);
        float  loss = 0.0f;

        // Zero the gradient accumulator
        std::memset(grad.data(), 0, embDim * sizeof(float));

        // ── positive sample ──
        {
            float* vOut = outputVec(context);
            float  dot  = dotProduct(vIn, vOut, embDim);
            float  sig  = sigmoid(dot);
            float  g    = (1.0f - sig) * lr;
            axpy(g, vOut, grad.data(), embDim);   // grad += g * vOut
            axpy(g, vIn,  vOut,        embDim);   // vOut += g * vIn
            loss -= std::log(sig + 1e-10f);
        }

        // ── negative samples ──
        for (int n = 0; n < negSamples; ++n) {
            int neg = sampler.sample(rng);
            if (neg == context) continue;

            float* vOut = outputVec(neg);
            float  dot  = dotProduct(vIn, vOut, embDim);
            float  sig  = sigmoid(dot);
            float  g    = -sig * lr;
            axpy(g, vOut, grad.data(), embDim);
            axpy(g, vIn,  vOut,        embDim);
            loss -= std::log(1.0f - sig + 1e-10f);
        }

        // Apply accumulated gradient to input embedding
        axpy(1.0f, grad.data(), vIn, embDim);

        return loss;
    }

    // Convenience wrapper (allocates grad buffer — fine for one-off calls)
    float trainPair(int center, int context,
                    int negSamples, float lr, std::mt19937& rng)
    {
        std::vector<float> grad(embDim, 0.0f);
        return trainPairWithBuffer(center, context, negSamples, lr, rng, grad);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Query utilities
    // ═════════════════════════════════════════════════════════════════════════

    // Copy of input embedding
    std::vector<float> getEmbedding(int idx) const {
        std::vector<float> emb(embDim);
        std::memcpy(emb.data(), inputVec(idx), embDim * sizeof(float));
        return emb;
    }

    // Cosine similarity
    float cosineSimilarity(int idx1, int idx2) const {
        const float* a = inputVec(idx1);
        const float* b = inputVec(idx2);
        float dot   = dotProduct(a, b, embDim);
        float normA = dotProduct(a, a, embDim);
        float normB = dotProduct(b, b, embDim);
        if (normA < 1e-12f || normB < 1e-12f) return 0.0f;
        return dot / (std::sqrt(normA) * std::sqrt(normB));
    }

    // k nearest neighbours of a vocab index (by cosine similarity)
    std::vector<std::pair<int, float>> nearestNeighbors(int idx, int k) const {
        std::vector<std::pair<float, int>> sims;
        sims.reserve(vocabSize);
        for (int i = 0; i < vocabSize; ++i) {
            if (i == idx) continue;
            sims.push_back({cosineSimilarity(idx, i), i});
        }
        int n = std::min(k, (int)sims.size());
        std::partial_sort(sims.begin(), sims.begin() + n, sims.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first;
                          });
        std::vector<std::pair<int, float>> result(n);
        for (int i = 0; i < n; ++i)
            result[i] = {sims[i].second, sims[i].first};
        return result;
    }

    // Analogy: "a is to b as c is to ?"  →  vec(a) − vec(b) + vec(c)
    std::vector<std::pair<int, float>> analogy(int a, int b, int c,
                                               int k = 5) const
    {
        std::vector<float> target(embDim);
        const float* va = inputVec(a);
        const float* vb = inputVec(b);
        const float* vc = inputVec(c);
        for (int d = 0; d < embDim; ++d)
            target[d] = va[d] - vb[d] + vc[d];

        float tNorm = std::sqrt(dotProduct(target.data(), target.data(), embDim));

        std::vector<std::pair<float, int>> sims;
        sims.reserve(vocabSize);
        for (int i = 0; i < vocabSize; ++i) {
            if (i == a || i == b || i == c) continue;
            const float* vi = inputVec(i);
            float dot  = dotProduct(target.data(), vi, embDim);
            float norm = std::sqrt(dotProduct(vi, vi, embDim));
            float sim  = (tNorm > 1e-12f && norm > 1e-12f)
                         ? dot / (tNorm * norm) : 0.0f;
            sims.push_back({sim, i});
        }

        int n = std::min(k, (int)sims.size());
        std::partial_sort(sims.begin(), sims.begin() + n, sims.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first;
                          });
        std::vector<std::pair<int, float>> result(n);
        for (int i = 0; i < n; ++i)
            result[i] = {sims[i].second, sims[i].first};
        return result;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Persistence
    // ═════════════════════════════════════════════════════════════════════════

    // Binary format: [vocabSize (int)][embDim (int)][W_in (floats)][W_out (floats)]
    bool saveWeights(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Error: cannot write " << filename << "\n";
            return false;
        }
        out.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
        out.write(reinterpret_cast<const char*>(&embDim),    sizeof(embDim));
        out.write(reinterpret_cast<const char*>(W_in .data()),
                  W_in .size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(W_out.data()),
                  W_out.size() * sizeof(float));
        return out.good();
    }

    bool loadWeights(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "Error: cannot read " << filename << "\n";
            return false;
        }
        in.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
        in.read(reinterpret_cast<char*>(&embDim),    sizeof(embDim));
        W_in .resize((size_t)vocabSize * embDim);
        W_out.resize((size_t)vocabSize * embDim);
        in.read(reinterpret_cast<char*>(W_in .data()),
                W_in .size() * sizeof(float));
        in.read(reinterpret_cast<char*>(W_out.data()),
                W_out.size() * sizeof(float));
        return in.good();
    }

    // word2vec text format: "vocabSize embDim\ntoken e1 e2 ... ed\n..."
    // Uses fprintf — std::ofstream crashes on large float writes in MinGW.
    bool saveEmbeddingsText(const std::string& filename,
                            const Vocabulary& vocab) const
    {
        FILE* fp = std::fopen(filename.c_str(), "w");
        if (!fp) {
            std::cerr << "Error: cannot write " << filename << "\n";
            return false;
        }
        std::fprintf(fp, "%d %d\n", vocabSize, embDim);
        for (int i = 0; i < vocabSize; ++i) {
            std::fprintf(fp, "%d", vocab.idx2token[i]);
            const float* emb = inputVec(i);
            for (int d = 0; d < embDim; ++d)
                std::fprintf(fp, " %g", emb[d]);
            std::fprintf(fp, "\n");
        }
        std::fclose(fp);
        return true;
    }

    // L2-normalise every input embedding in-place
    void normalizeEmbeddings() {
        for (int i = 0; i < vocabSize; ++i) {
            float* v    = inputVec(i);
            float  norm = std::sqrt(dotProduct(v, v, embDim));
            if (norm > 1e-12f) {
                float inv = 1.0f / norm;
                for (int d = 0; d < embDim; ++d) v[d] *= inv;
            }
        }
    }
};

// ═════════════════════════════════════════════════════════════════════════════
//  Post-processing: remove the dominant shared direction
//
//  SGNS on small/skewed corpora leaves a strong "mean direction" in the
//  embedding space.  Every vector is partially aligned with it, which
//  inflates pairwise cosine similarity.  Subtracting the mean and
//  re-normalising removes this artifact.
//  (cf. Mu & Viswanath 2018, "All-but-the-Top")
// ═════════════════════════════════════════════════════════════════════════════

inline void meanCenterAndNormalize(float* emb, int V, int D) {
    // 1. Compute mean embedding
    std::vector<float> mean(D, 0.0f);
    for (int i = 0; i < V; ++i) {
        const float* row = emb + (size_t)i * D;
        for (int d = 0; d < D; ++d) mean[d] += row[d];
    }
    float invV = 1.0f / V;
    for (int d = 0; d < D; ++d) mean[d] *= invV;

    // 2. Subtract mean from every vector
    for (int i = 0; i < V; ++i) {
        float* row = emb + (size_t)i * D;
        for (int d = 0; d < D; ++d) row[d] -= mean[d];
    }

    // 3. Re-normalise to unit length
    for (int i = 0; i < V; ++i) {
        float* row = emb + (size_t)i * D;
        float norm = 0.0f;
        for (int d = 0; d < D; ++d) norm += row[d] * row[d];
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
            float inv = 1.0f / norm;
            for (int d = 0; d < D; ++d) row[d] *= inv;
        }
    }
}

} // namespace w2v
