#ifndef LEARNINGRULES_HPP
#define LEARNINGRULES_HPP

#include "vector.hpp"
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

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

namespace InstantLearningRules {
    // Shared helper: solve A * x = b using Gaussian elimination with partial pivoting
    // Template accepts any WeightVec-like type which is indexable and sized.
    template <typename WeightVec>
    void solveLinearSystem(std::vector<std::vector<double>>& A,
                           std::vector<double>& b,
                           WeightVec& x) {
        const int n = static_cast<int>(b.size());

        // Ensure output vector has correct size
        if (x.getSize() != n) {
            // If WeightVec supports resize-like semantics, try to set it; otherwise assume it's correctly sized.
            // We avoid calling unknown API; rely on it being pre-sized by the caller.
        }

        // Forward elimination with partial pivoting
        for (int k = 0; k < n - 1; ++k) {
            int maxRow = k;
            double maxVal = std::abs(A[k][k]);
            for (int i = k + 1; i < n; ++i) {
                if (std::abs(A[i][k]) > maxVal) {
                    maxVal = std::abs(A[i][k]);
                    maxRow = i;
                }
            }

            if (maxRow != k) {
                std::swap(A[k], A[maxRow]);
                std::swap(b[k], b[maxRow]);
            }

            for (int i = k + 1; i < n; ++i) {
                if (std::abs(A[k][k]) < 1e-10) continue;
                double factor = A[i][k] / A[k][k];
                for (int j = k; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }

        // Back substitution
        for (int i = n - 1; i >= 0; --i) {
            double sum = b[i];
            for (int j = i + 1; j < n; ++j) {
                sum -= A[i][j] * x[j];
            }
            if (std::abs(A[i][i]) < 1e-10) {
                x[i] = 0.0;
            } else {
                x[i] = sum / A[i][i];
            }
        }
    }

    template <typename PerceptronType, typename DatasetType>
    double AdalineCostFunction(const PerceptronType& perceptron, const DatasetType& dataset) {
        const int P = dataset.getSize();  // number of patterns
        double cost = 0.0;

        for (int mu = 0; mu < P; ++mu) {
            const auto& elem = dataset[mu];
            Vector<int> S = Vector<int>::concat(elem.top.toVector(), elem.bottom.toVector());
            // Evaluate perceptron response on S
            const double out = perceptron.evalProd(S);
            double diff = static_cast<double>(elem.label) - out ;
            cost += diff * diff;
        }

        return cost * 0.5;
    }

    template <typename PerceptronType, typename DatasetType>
    void AdalineCorrection(PerceptronType& perceptron,
                        const DatasetType& dataset,
                        double tol = 1e-8,
                        int maxIter = 100000,
                        double gamma = 0.05)
    {
        auto weights = perceptron.getWeights();     
        const int P = dataset.getSize();
        const int N = weights.getSize();

        //-----------------------------------------------------
        // Precompute all input vectors S^mu once (major speedup)
        //-----------------------------------------------------
        std::vector<std::vector<double>> S(P, std::vector<double>(N));

        for (int mu = 0; mu < P; ++mu) {
            const auto& elem = dataset[mu];
            Vector<int> v = Vector<int>::concat(elem.top.toVector(), elem.bottom.toVector());
            for (int j = 0; j < N; ++j)
                S[mu][j] = static_cast<double>(v[j]);
        }

        //-----------------------------------------------------
        // Gradient descent loop
        //-----------------------------------------------------
        std::vector<double> grad(N);
        double prevEnergy = std::numeric_limits<double>::infinity();

        for (int iter = 0; iter < maxIter; ++iter) {

            // reset gradient
            std::fill(grad.begin(), grad.end(), 0.0);

            // compute energy and full gradient
            double energy = 0.0;

            for (int mu = 0; mu < P; ++mu) {
                // Compute out = J · S^mu
                double out = 0.0;
                for (int j = 0; j < N; ++j)
                    out += weights[j] * S[mu][j];

                double diff = out - static_cast<double>(dataset[mu].label);

                energy += diff * diff;

                // accumulate gradient: diff * S^mu
                for (int j = 0; j < N; ++j)
                    grad[j] += diff * S[mu][j];
            }

            energy *= 0.5;

            //---------------------------------------------
            // stopping criterion: EXACT same math
            //---------------------------------------------
            if (std::abs(energy - prevEnergy) < tol)
                break;

            prevEnergy = energy;

            //---------------------------------------------
            // gradient descent step: EXACT same math
            //---------------------------------------------
            double eff_gamma = gamma / static_cast<double>(P);

            for (int j = 0; j < N; ++j)
                weights[j] -= eff_gamma * grad[j];
        }
    }


    struct PseudoInverseRule {
        // Train the perceptron using pseudoinverse method
        // Takes the entire dataset and computes weights in one shot
        template <typename PerceptronType, typename DatasetType>
        void operator()(PerceptronType& perceptron, const DatasetType& dataset) const {
            auto weights = perceptron.getWeights();
            const int P = dataset.getSize();  // number of patterns
            const int N = weights.getSize();  // input dimension
            
            if (P == 0) return;
            
            // Build matrix X (P x N) where each row is an input pattern
            // and vector y (P x 1) of labels
            std::vector<std::vector<double>> X(P, std::vector<double>(N));
            std::vector<double> y(P);
            
            for (int mu = 0; mu < P; ++mu) {
                const auto& elem = dataset[mu];
                y[mu] = static_cast<double>(elem.label);
                
                // Convert top and bottom Scalars to vectors and concatenate
                Vector<int> S = Vector<int>::concat(elem.top.toVector(), elem.bottom.toVector());
                assert(S.getSize() == N);
                for (int i = 0; i < N; ++i) {
                    X[mu][i] = static_cast<double>(S[i]);
                }
            }
            
            // Compute X^T * X (Gram matrix, N x N)
            std::vector<std::vector<double>> XTX(N, std::vector<double>(N, 0.0));
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int mu = 0; mu < P; ++mu) {
                        XTX[i][j] += X[mu][i] * X[mu][j];
                    }
                }
            }
            
            // Compute X^T * y (N x 1)
            std::vector<double> XTy(N, 0.0);
            for (int i = 0; i < N; ++i) {
                for (int mu = 0; mu < P; ++mu) {
                    XTy[i] += X[mu][i] * y[mu];
                }
            }
            
            // Solve (X^T X) w = X^T y using simple Gaussian elimination
            // This computes w = (X^T X)^{-1} X^T y = X^+ y
            solveLinearSystem(XTX, XTy, weights);
        }
    };
    
    struct RidgeRegressionRule {
        double lambda; // regularization parameter

        RidgeRegressionRule(double lambda_ = 1.0) : lambda(lambda_) {}

        template <typename PerceptronType, typename DatasetType>
        void operator()(PerceptronType& perceptron, const DatasetType& dataset) const {
            auto weights = perceptron.getWeights();
            const int P = dataset.getSize();  // number of patterns
            const int N = weights.getSize();  // input dimension
            
            if (P == 0) return;
            
            // Build matrix X (P x N) where each row is an input pattern
            // and vector y (P x 1) of labels
            std::vector<std::vector<double>> X(P, std::vector<double>(N));
            std::vector<double> y(P);
            
            for (int mu = 0; mu < P; ++mu) {
                const auto& elem = dataset[mu];
                y[mu] = static_cast<double>(elem.label);
                
                // Convert top and bottom Scalars to vectors and concatenate
                Vector<int> S = Vector<int>::concat(elem.top.toVector(), elem.bottom.toVector());
                assert(S.getSize() == N);
                for (int i = 0; i < N; ++i) {
                    X[mu][i] = static_cast<double>(S[i]);
                }
            }
            
            // Compute X^T * X + lambda * I (Regularized Gram matrix, N x N)
            std::vector<std::vector<double>> XTX(N, std::vector<double>(N, 0.0));
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int mu = 0; mu < P; ++mu) {
                        XTX[i][j] += X[mu][i] * X[mu][j];
                    }
                    if (i == j) {
                        XTX[i][j] += lambda; // Add regularization term
                    }
                }
            }
            
            // Compute X^T * y (N x 1)
            std::vector<double> XTy(N, 0.0);
            for (int i = 0; i < N; ++i) {
                for (int mu = 0; mu < P; ++mu) {
                    XTy[i] += X[mu][i] * y[mu];
                }
            }
            // Solve (X^T X + lambda I) w = X^T y using simple Gaussian elimination
            // This computes w = (X^T X + lambda I)^{-1} X
            solveLinearSystem(XTX, XTy, weights);
        }
    };

    struct AdalineRule{
        double tol;
        AdalineRule(double threshold = 1e-8) : tol(threshold) {}
        template <typename PerceptronType, typename DatasetType>
        void operator()(PerceptronType& perceptron, const DatasetType& dataset) const {
            perceptron.resetWeights(0.0);
            auto weights = perceptron.getWeights();
            // Sum all the vectors in the dataset 
            for (auto mu = 0; mu < dataset.getSize(); ++mu) {
                const auto& elem = dataset[mu];
                Vector<int> S = Vector<int>::concat(elem.top.toVector(), elem.bottom.toVector());
                for (int i = 0; i < weights.getSize(); ++i) {
                    weights[i] += static_cast<double>(S[i]);
                }
            }
            // Normalize to unitary lenght
            double norm = 0.0;
            for (int i = 0; i < weights.getSize(); ++i) {
                norm += weights[i] * weights[i];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-10) {  // Avoid division by zero
                for (int i = 0; i < weights.getSize(); ++i) {
                    weights[i] /= norm;
                }
            }
            // Apply Adaline correction to refine weights
            AdalineCorrection(perceptron, dataset, tol);
        }
    };

    struct BayesRule {
        int numPerceptrons;
        double ridgeLambda;

        BayesRule(int numPerceptrons_ = 10, double ridgeLambda_ = 0.001) : numPerceptrons(numPerceptrons_), ridgeLambda(ridgeLambda_) {}


        template <typename PerceptronType, typename DatasetType>
        void operator()(PerceptronType& perceptron, const DatasetType& dataset) const {
            // Define a population of perceptrons
            std::vector<PerceptronType> perceptrons;
            for (int p = 0; p < numPerceptrons; ++p) {
                PerceptronType pc(perceptron); // Copy constructor
                pc.resetWeights(0.0);
                perceptrons.push_back(pc);
            }
            // Train each perceptron using Ridge Regression
            RidgeRegressionRule ridgeRule(ridgeLambda);
            for (auto& pc : perceptrons) {
                ridgeRule(pc, dataset);
            }
            // Average weights over all perceptrons
            auto weights = perceptron.getWeights();
            int N = weights.getSize();
            std::vector<double> avgWeights(N, 0.0);
            
            for (const auto& pc : perceptrons) {
                const auto& pcWeightsArray = pc.weights();
                for (int i = 0; i < N; ++i) {
                    avgWeights[i] += pcWeightsArray[i];
                }
            }
            
            // Set the averaged weights back
            const auto& templateWeights = perceptrons[0].weights();
            auto finalWeights = templateWeights; // Copy to get proper type
            for (int i = 0; i < N; ++i) {
                finalWeights[i] = avgWeights[i] / static_cast<double>(numPerceptrons);
            }
            perceptron.setWeights(finalWeights);
        }
    };
} // namespace InstantLearningRules

#endif // LEARNINGRULES_HPP