#include "importUtils.hpp"
#include "trainPipeline.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    std::cout << "===========================================\n";
    std::cout << "  testTrain -- end-to-end pipeline test\n";
    std::cout << "===========================================\n\n";

    // ── 1. Load dataset ────────────────────────────────────────────────────
    int splitYear = 2015;
    Dataset dataset("Dataset.jsonl", splitYear);
    std::cout << "Loaded dataset  (split at " << splitYear << ")\n"
              << "  Train : " << dataset.trainSize() << "\n"
              << "  Test  : " << dataset.testSize()  << "\n\n";

    assert(dataset.trainSize() > 0 && "train set must not be empty");

    // ── 2. Configure ───────────────────────────────────────────────────────
    TrainingConfig config;
    config.embeddingDim    = 64;
    config.negSamples      = 5;
    config.epochs          = 3;
    config.batchSize       = 4096;
    config.learningRate    = 0.025f;
    config.minLearningRate = 0.0001f;
    config.minCount        = 2;
    config.numThreads      = 4;
    config.seed            = 42;
    config.verbose         = true;

    auto result = TrainingPipeline::run(dataset.train(), config);

    assert(result.vocab.size() > 0       && "vocab must not be empty");
    assert(result.vocab.size() == result.model.vocabSize);
    assert(result.model.embDim == config.embeddingDim);

    std::cout << "\n-- Post-training checks --\n\n";

    {
        int topIdx = 0;
        for (int i = 1; i < result.vocab.size(); ++i)
            if (result.vocab.frequencies[i] > result.vocab.frequencies[topIdx])
                topIdx = i;

        std::cout << "Most frequent token: " << result.vocab.idx2token[topIdx]
                  << " (freq=" << result.vocab.frequencies[topIdx] << ")\n";
        std::cout << "  5 nearest neighbours:\n";

        auto nn = result.model.nearestNeighbors(topIdx, 5);
        for (auto& [idx, sim] : nn)
            std::cout << "    " << result.vocab.idx2token[idx]
                      << "  sim=" << sim << "\n";
    }

    {
        float self = result.model.cosineSimilarity(0, 0);
        std::cout << "\ncos(0,0) = " << self << "  (should be ~1.0)\n";
        assert(std::abs(self - 1.0f) < 1e-4f);
    }

    {
        const char* modelFile = "test_model.bin";
        const char* vocabFile = "test_vocab.txt";

        assert(result.model.saveWeights(modelFile));
        assert(result.vocab.save(vocabFile));

        w2v::Word2VecModel loaded;
        assert(loaded.loadWeights(modelFile));
        assert(loaded.vocabSize == result.model.vocabSize);
        assert(loaded.embDim    == result.model.embDim);

        for (int i = 0; i < std::min(10, loaded.vocabSize); ++i) {
            auto e1 = result.model.getEmbedding(i);
            auto e2 = loaded.getEmbedding(i);
            for (int d = 0; d < loaded.embDim; ++d)
                assert(std::abs(e1[d] - e2[d]) < 1e-6f);
        }
        std::cout << "Binary save/load round-trip OK\n";

        Vocabulary loadedVocab;
        assert(loadedVocab.load(vocabFile));
        assert(loadedVocab.size() == result.vocab.size());
        for (int i = 0; i < loadedVocab.size(); ++i) {
            assert(loadedVocab.idx2token[i]   == result.vocab.idx2token[i]);
            assert(loadedVocab.frequencies[i] == result.vocab.frequencies[i]);
        }
        std::cout << "Vocab save/load round-trip OK\n";

        std::remove(modelFile);
        std::remove(vocabFile);
    }

    {
        const char* txtFile = "test_embeddings.txt";
        assert(result.model.saveEmbeddingsText(txtFile, result.vocab));
        std::cout << "Text embedding export OK\n";
        std::remove(txtFile);
    }

    if (result.vocab.size() >= 4) {
        auto res = result.model.analogy(0, 1, 2, 3);
        assert((int)res.size() <= 3);
        std::cout << "Analogy function OK\n";
    }

    {
        result.model.normalizeEmbeddings();
        float self = result.model.cosineSimilarity(0, 0);
        assert(std::abs(self - 1.0f) < 1e-4f);
        std::cout << "Normalise + cosine OK\n";
    }

    std::cout << "\n===========================================\n";
    std::cout << "  ALL TESTS PASSED\n";
    std::cout << "===========================================\n";
    return 0;
}
