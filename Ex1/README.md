# Statistical Physics and Machine Learning - Exercise 1

## Project Overview

This project explores fundamental concepts in machine learning through the lens of polynomial fitting and optimization algorithms. We implement and compare ```powershell
# Compile the project first
mingw32-make compile

# Run the GD vs SGD comparison
bin\main.exe single 5
```erent gradient-based optimization methods, specifically **Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)**, for polynomial regression tasks.

The project demonstrates key machine learning concepts including:
- **Polynomial fitting** with different orders (1st, 3rd, and 10th degree)
- **Gradient-based optimization** algorithms
- **Regularization techniques** (L2 regularization)
- **Batch processing** strategies
- **Overfitting and underfitting** analysis

This work is particularly relevant in statistical physics where polynomial approximations are commonly used to model complex systems and where optimization techniques are crucial for parameter estimation in physical models.

## Quick Start Guide

### Prerequisites
- C++17 compatible compiler (g++ recommended)
- Python 3.x with matplotlib and numpy
- Make utility (mingw32-make on Windows)

### Compilation

To compile the project:

```powershell
mingw32-make compile
```

This creates the executable `bin\main.exe` along with all necessary object files in the `build\` directory.

### Running Exercises

The program supports several execution modes:

#### Run All Exercises (Default)
```powershell
bin\main.exe
# or
bin\main.exe default
```

#### Run Specific Exercise
```powershell
bin\main.exe single <exercise_number>
```

Available exercises:
- `1`: Basic polynomial fitting
- `2`: Regularization analysis
- `3`: Batch size comparison
- `4`: Learning rate optimization
- `5`: GD vs SGD comparison

#### Run Comparison Only
```powershell
bin\main.exe comparison
```

### Visualization

After running exercises, generate plots using Python:

```powershell
# Set backend for headless operation (optional)
$env:MPLBACKEND = "Agg"

# Generate all plots
python -c "import excercisePlotter as p; p.exFivePlotter()"
```

This creates PNG files alongside the corresponding data files for visualization.

### Output Files

The program generates several output files:
- `fitted_output.txt` / `fitted_output_with_test.txt`: Fitting results
- `gd_vs_sgd_fullbatch.txt/.png`: Full-batch comparison
- `gd_vs_sgd_minibatch2.txt/.png`: Mini-batch comparison
- `output.txt`: General output and logs

## Mathematical Foundation

### Polynomial Regression Model

We define a polynomial function $\hat{y}(\vec{x};\vec{\theta})$, where:
- $\vec{x}$ represents the input variables
- $\vec{\theta} = \{\theta_0, \theta_1, ..., \theta_d\}$ are the **parameters** to be learned
- $d$ is the polynomial degree

The polynomial model takes the form:
$$
\hat{y}(x; \theta) = \sum_{k=0}^d \theta_k x^k = \theta_0 + \theta_1 x + \theta_2 x^2 + ... + \theta_d x^d
$$

### Loss Function with Regularization

Given a dataset of $N$ points $(x_i, y_i)$, we define the **L2 regularized Mean Squared Error** loss function:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left(\hat{y}(x_i; \theta) - y_i\right)^2 + \frac{\lambda}{2}\|\theta\|_2^2
$$

Where:
- The first term is the **data fitting error** (MSE)
- The second term is the **L2 regularization** penalty
- $\lambda \geq 0$ is the **regularization strength**
- $\|\theta\|_2^2 = \sum_{k=0}^d \theta_k^2$ prevents overfitting

## Gradient Descent (GD)

### Mathematical Derivation

For polynomial regression, the gradient of the loss function with respect to parameter $\theta_k$ is:

$$
\frac{\partial\mathcal{L}(\theta)}{\partial \theta_k} = \frac{2}{M} \sum_{i \in \mathcal{B}} \left(\hat{y}(x_i; \theta) - y_i\right) x_i^k + \lambda\theta_k
$$

Where:
- $M$ is the **batch size** 
- $\mathcal{B}$ is the current batch of data points
- The gradient combines both fitting error and regularization terms

### Update Rule

The iterative parameter update follows:
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$

Where $\eta > 0$ is the **learning rate**, controlling the step size of each update.

### Characteristics of Standard GD

- **Uses entire dataset** in each iteration ($M = N$)
- **Deterministic** updates (same gradient every epoch)
- **Smooth convergence** but potentially slow
- **Guaranteed decrease** in loss function (with appropriate $\eta$)

## Stochastic Gradient Descent (SGD)

### Core Concept

**Stochastic Gradient Descent** modifies the standard GD approach by using only a subset of the data (or single data points) to compute gradient estimates at each iteration. Instead of computing the exact gradient using all $N$ data points, SGD uses a **mini-batch** of size $M < N$ or even individual samples ($M = 1$).

### Mathematical Framework

The SGD gradient estimation becomes:
$$
\nabla_\theta \mathcal{L}_{SGD} = \frac{2}{M} \sum_{i \in \mathcal{B}_t} \left(\hat{y}(x_i; \theta) - y_i\right) x_i^k + \lambda\theta_k
$$

Where $\mathcal{B}_t$ is a **randomly sampled mini-batch** at iteration $t$, with $|\mathcal{B}_t| = M$.

### Variants of SGD

1. **True SGD**: $M = 1$ (single sample per iteration)
2. **Mini-batch SGD**: $1 < M < N$ (small batches)
3. **Full-batch GD**: $M = N$ (entire dataset)

### Advantages of SGD

#### 🚀 **Computational Efficiency**
- **Faster iterations**: Each update requires processing only $M$ samples instead of $N$
- **Memory efficient**: Reduced memory footprint for large datasets
- **Early progress**: Can make significant progress before seeing the entire dataset

#### 📈 **Optimization Benefits**
- **Escape local minima**: Stochastic noise helps escape shallow local optima
- **Better generalization**: The noise acts as implicit regularization
- **Online learning**: Can adapt to new data in real-time applications

#### ⚡ **Scalability**
- **Large datasets**: Essential for datasets where full-batch computation is impractical
- **Parallel processing**: Mini-batches can be processed in parallel across multiple cores/GPUs

### Disadvantages of SGD

#### 🎯 **Convergence Issues**
- **Noisy updates**: High variance in gradient estimates leads to oscillatory behavior
- **No guaranteed decrease**: Loss function may increase in individual iterations
- **Slower final convergence**: May require more iterations to reach high precision

#### ⚙️ **Hyperparameter Sensitivity**
- **Learning rate tuning**: More sensitive to learning rate selection
- **Batch size selection**: Requires careful tuning of batch size $M$
- **Convergence criteria**: Harder to determine when optimization has converged

#### 📊 **Reproducibility**
- **Stochastic behavior**: Results depend on random batch sampling
- **Requires multiple runs**: Need averaging over multiple runs for reliable results

### SGD vs GD: Key Trade-offs

| Aspect | Full-batch GD | Mini-batch/Stochastic GD |
|--------|---------------|---------------------------|
| **Computation per iteration** | $O(N)$ | $O(M)$ where $M \ll N$ |
| **Memory usage** | High ($\propto N$) | Low ($\propto M$) |
| **Convergence smoothness** | Smooth, deterministic | Noisy, stochastic |
| **Escape local minima** | Difficult | Easier due to noise |
| **Hyperparameter sensitivity** | Lower | Higher |
| **Parallelization** | Limited | Excellent |
| **Large-scale applicability** | Poor | Excellent |

### Batch Size Effects

The choice of batch size $M$ creates a spectrum of behaviors:

- **$M = 1$** (True SGD): Maximum stochasticity, fastest iterations, highest variance
- **$M = $ small** (Mini-batch): Balance between speed and stability
- **$M = N$** (Full-batch): Deterministic, smooth, but computationally expensive

### Implementation Considerations

In practice, effective SGD implementations often include:
- **Learning rate scheduling**: Decreasing $\eta$ over time
- **Momentum**: Adding momentum terms to reduce oscillations
- **Adaptive methods**: Adam, RMSprop, etc., that adapt learning rates per parameter

## Practical Comparison: GD vs SGD (Exercise 5)

This section demonstrates the theoretical concepts through practical implementation and comparison of GD and SGD algorithms.

### Experimental Setup

**Exercise 5** (`exPointFive`) implements a comprehensive comparison by:

1. **Generating fresh datasets** for unbiased comparison
2. **Fitting three polynomial orders** (1st, 3rd, and 10th degree) to demonstrate different complexity scenarios
3. **Testing both algorithms** under identical conditions
4. **Varying batch sizes** to show the spectrum from full-batch GD to true SGD

### Running the Comparison

#### Generate Comparison Data

```powershell
mingw32-make compile
.in\main.exe single 5
```

This execution generates comprehensive comparison data:

#### Output Files

- **`gd_vs_sgd_fullbatch.txt/.png`**: 
  - Compares GD vs SGD with **full batch size** ($M = N$)
  - Shows that when batch size equals dataset size, SGD behaves identically to GD
  - Demonstrates the **deterministic limit** of SGD

- **`gd_vs_sgd_minibatch2.txt/.png`**: 
  - Compares GD vs SGD with **mini-batch size 2** ($M = 2$)
  - Highlights the **stochastic nature** of SGD with small batches
  - Shows convergence differences and oscillatory behavior

### Visualization and Analysis

#### Generate Plots

```powershell
# Optional: Set backend for headless operation
$env:MPLBACKEND = "Agg"

# Generate comprehensive comparison plots
python -c "import excercisePlotter as p; p.exFivePlotter()"
```

#### What the Plots Show

The generated visualizations include:

1. **8×2 Comparison Matrices**: 
   - **Left panels**: Polynomial fits overlaid on data points
   - **Right panels**: Parameter value comparisons between GD and SGD
   - **Separate views**: For datasets A and B to show robustness

2. **Convergence Analysis**:
   - **Learning curves**: Loss vs iteration for both algorithms
   - **Parameter evolution**: How $\theta_k$ values change over time
   - **Variance analysis**: Stability differences between methods

### Key Observations

#### When Batch Size = Dataset Size ($M = N$)
- SGD becomes **identical to GD**
- **Deterministic convergence** with smooth trajectories
- **Same final parameters** (within numerical precision)

#### When Batch Size << Dataset Size ($M \ll N$)
- SGD shows **stochastic behavior** with oscillations
- **Faster initial progress** but noisier convergence
- **Different final parameters** due to stochastic effects
- **May escape poor local minima** that trap GD

#### Polynomial Order Effects
- **1st order** (linear): Both methods converge easily
- **3rd order**: Moderate complexity, shows clear differences
- **10th order**: High complexity, demonstrates overfitting risks and optimization challenges

### Performance Metrics

The comparison evaluates:
- **Convergence speed**: Iterations to reach target accuracy
- **Final accuracy**: Quality of fit on training data
- **Generalization**: Performance on test data (when available)
- **Stability**: Variance across multiple runs
- **Computational efficiency**: Time per iteration

### Insights for Statistical Physics

This comparison is particularly relevant for statistical physics applications where:
- **Large datasets** from simulations benefit from SGD's computational efficiency
- **Complex energy landscapes** may have local minima that SGD can escape
- **Real-time data processing** (e.g., from experiments) requires online learning capabilities
- **Parameter estimation** in physical models needs balance between accuracy and computational cost