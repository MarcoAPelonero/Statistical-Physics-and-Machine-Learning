#ifndef LEARNINGRULES_HPP
#define LEARNINGRULES_HPP

#include "vector.hpp"
#include <random>
#include <cmath>

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
        mutable std::mt19937 rng;
        mutable std::normal_distribution<double> noise;

        NoisyPerceptronRule(unsigned seed = 1234, double sigma = std::sqrt(50.0))
            : rng(seed), noise(0.0, sigma) {}

        template <typename WeightVec, typename InputVec>
        void operator()(WeightVec& weights, const InputVec& input, int out, int label, double scale) const {
            assert(input.getSize() == weights.getSize());
            if (out == label) return;

            double eta = noise(rng);
            double factor = 1.0 + eta;

            for (int i = 0; i < weights.getSize(); ++i) {
                weights[i] += scale * static_cast<double>(label) * static_cast<double>(input[i]) * factor;
            }
        }
    };

} // namespace LearningRules

#endif // LEARNINGRULES_HPP