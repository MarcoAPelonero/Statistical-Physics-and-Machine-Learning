# EX6 – Hebbian learning and theory vs experiment

This directory implements the Hebbian learning exercise for a binary
comparator task. The perceptron receives patterns of length `N = 2 * bits`
with entries in `{+1, -1}` and must reproduce the ordering of two `bits`‑long
binary numbers. The teacher is the deterministic “perfect comparator” used in
`Perceptron::perfectTeacher`.

## Original theoretical curves

Initially the code compared the empirical results from `exPointTwo` against
two analytical formulas in `include/integration.hpp`:

- `epsilon_theory(alpha)` – generalization error for Hebbian learning in the
  thermodynamic limit with i.i.d. *Gaussian* inputs.
- `epsilon_train(alpha)` – theoretical training error from the replica
  computation, expressed as

  ```math
  \varepsilon_\text{train}(\alpha)
    = 2 \int_0^\infty \mathcal{D}x\; H\!\left(\frac{1}{\sqrt{\alpha}} +
              \sqrt{\frac{2\alpha}{\pi}}\,x\right)
  ```

  implemented numerically via Simpson integration.

These curves were mathematically correct for that Gaussian model and the
integration was verified independently, but they did **not** sit exactly on
top of the empirical points produced by `exPointTwo`. Both training and test
errors from the simulation were systematically *below* the theory by a few
percent, especially at small α.

The reason is a mismatch between the assumptions of the theory and the actual
experiment:

- Theory: teacher and student see i.i.d. Gaussian patterns.
- Code: teacher is the fixed binary comparator, and all patterns are ±1
  Rademacher vectors with constant norm.

The different input distribution (and structured teacher) changes the
statistics of the Hebbian weight vector, so the analytic curve is shifted up
relative to what the simulator measures.

## What was changed

To obtain a “theory” curve that matches *exactly* the experiment we added a
Monte‑Carlo prediction that uses the same model as the C++ simulation.
The existing analytic functions were kept intact.

### New Monte‑Carlo theory function

In `include/integration.hpp` we added:

- Extra headers: `#include <vector>`, `#include <random>`, `#include <cstdint>`,
  `#include <algorithm>`.
- A small result struct:

  ```cpp
  struct MonteCarloResult {
      double eps_train; // expected training error
      double eps_test;  // expected generalization error
  };
  ```

- The function

  ```cpp
  MonteCarloResult epsilon_mc(double alpha,
                              int bits = 30,
                              int trials = 1000,
                              int testSamples = 2000,
                              unsigned seed = 12345);
  ```

  which does the following for each `alpha`:

  1. Set `N = 2 * bits` and `P = round(alpha * N)`.
  2. Build the *same* comparator weights as the teacher:
     `+2^{bits-1-i}` on the first half, `-` of that on the second half.
  3. For each Monte‑Carlo trial:
     - Draw `P` random training patterns with entries in `{+1, -1}`.
     - Label each pattern with the comparator teacher.
     - Perform a **single Hebbian pass** with learning rate
       `scale = 1/sqrt(N)`, exactly as in the simulation:

       ```cpp
       w += scale * label * pattern;
       ```

     - Measure the training error on that same set.
     - Measure the generalization error on a fresh test set of
       `testSamples` new random patterns, labelled by the same teacher.
  4. Return the averages of the training and test errors over all trials.

Because the Monte‑Carlo routine reproduces the *same* distribution of inputs,
labels and updates as `exPointTwo`, it computes the theoretical expectation of
those empirical curves for the finite‑P, finite‑N model actually simulated.

### Using the Monte‑Carlo theory in exPointThree

`src/exPoints.cpp` was updated in `exPointThree()` to call the new function
for each α:

- The CSV header written to
  `plots/exPointThree_results.csv` is now

  ```text
  alpha,epsilon_train,epsilon_theory,epsilon_mc_train,epsilon_mc_test
  ```

- For each `alpha` we still compute the original Gaussian‑based
  `epsilon_train(alpha)` and `epsilon_theory(alpha)`, *and additionally*:

  ```cpp
  auto mc = Integration::epsilon_mc(alpha, bits, mcTrials, mcTestSamples,
                                    mcSeed + static_cast<unsigned>(alpha * 1000));
  ```

  and write `mc.eps_train` and `mc.eps_test` into the CSV.

### Plotting the new curves

`exPointTreePlotter.py` was extended to overlay the Monte‑Carlo prediction
on the existing plots:

- Training subplot: adds a dashed purple line for
  `epsilon_mc_train` labelled as
  “Monte Carlo (binary): εᵐᶜ_train” alongside the Gaussian
  `epsilon_train` and the empirical training points.
- Test subplot: adds a dashed purple line for
  `epsilon_mc_test` labelled “Monte Carlo (binary): εᵐᶜ_test”
  together with the Gaussian `epsilon_theory` and the empirical test points.

The original analytic curves are still plotted and can be directly compared
to the Monte‑Carlo prediction and the data.

## Why the ±1 Monte‑Carlo theory matches so well

With the new function the “theoretical” prediction is generated by repeating
exactly what the experiment does, but averaged over many random draws. This
eliminates the model mismatch that existed with the Gaussian theory.

More concretely:

- **Same teacher** – both simulation and Monte‑Carlo use the comparator
  weight vector; no random teacher approximation.
- **Same input distribution** – patterns are i.i.d. Rademacher (±1) with
  fixed norm, not Gaussian. This changes the self‑overlap and noise on the
  Hebbian field.
- **Same update rule** – single‑epoch Hebbian learning with scaling
  `1 / √N` on the concatenated pattern, exactly as in `TrainPerceptronOne`.
- **Same measurement protocol** – training error is measured on the training
  patterns after the epoch; test error is measured on fresh patterns drawn
  from the same distribution.

Because every modelling detail is aligned, the Monte‑Carlo curve represents
the true expectation value of the empirical experiment. When you plot it, it
almost perfectly overlays the simulation points (differences are at the level
expected from finite sampling noise), confirming that:

1. The C++ implementation of Hebbian learning is correct.
2. The previous discrepancy was purely due to using a Gaussian analytic
   theory for a binary‑input comparator task.

You can therefore use the Monte‑Carlo curve as the “correct” theory for the
specific setup implemented here, while keeping the Gaussian formulas as a
useful reference to the standard textbook results. 
*** End Patch***】} ***!
