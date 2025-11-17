#ifndef LEARNINGRULES_HPP
#define LEARNINGRULES_HPP

#include "vector.hpp"

namespace LearningRules {
    
    struct HebbianRule {
        template <typename WeightVec, typename InputVec>
        void operator()(WeightVec& weights, const InputVec& input, int /*out*/, int label, double scale) const {
            assert(input.getSize() == weights.getSize());
            for (int i = 0; i < weights.getSize(); ++i) {
                weights[i] += scale * static_cast<double>(label) * static_cast<double>(input[i]);
            }
        }
    };

    struct PerceptronRule {
        template <typename WeightVec, typename InputVec>
        void operator()(WeightVec& weights, const InputVec& input, int out, int label, double scale) const {
            assert(input.getSize() == weights.getSize());
            if (out == label) return;
            for (int i = 0; i < weights.getSize(); ++i) {
                weights[i] += scale * static_cast<double>(label) * static_cast<double>(input[i]);
            }
        }
    };

    struct NoisyPerceptronRule {
        std::mt19937 rng;
        std::normal_distribution<double> noise;

        NoisyPerceptronRule(unsigned seed = 1234, double sigma = 0.01)
            : rng(seed), noise(0.0, sigma) {}

        template <typename WeightVec, typename InputVec>
        void operator()(WeightVec& weights, const InputVec& input, int out, int label, double scale) {
            assert(input.getSize() == weights.getSize());
            if (out == label) return;

            for (int i = 0; i < weights.getSize(); ++i) {
                double n = noise(rng);
                weights[i] += scale * double(label) * double(input[i]) + n;
            }
        }
    };

} // namespace LearningRules

#endif // LEARNINGRULES_HPP