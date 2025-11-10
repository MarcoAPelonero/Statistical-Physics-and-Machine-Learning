import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import bisect

def H(u):
    return 0.5 * erfc(u / np.sqrt(2))

def I_of_R(R, num_samples=100000, rng=None, return_std_error=True):
    """
    Monte Carlo estimate of I(R) = E_w[ exp(-0.5 (1+R) v^2) / H(-sqrt(R) v) ]

    Returns
    -------
    mean, std_error
      If `return_std_error` is False, returns only mean.
    """
    if rng is None:
        # use numpy's default generator for reproducibility when caller doesn't provide one
        rng = np.random.default_rng()

    # ensure R is safe for sqrt (clip tiny/negative numerical values)
    R_safe = float(np.clip(R, 0.0, 1.0))

    v = rng.normal(size=num_samples)
    denom = H(-np.sqrt(R_safe) * v)
    # protect against tiny denominators
    denom = denom + 1e-24
    weights = np.exp(-0.5 * (1.0 + R_safe) * v**2) / denom

    mean = float(np.mean(weights))
    if not return_std_error:
        return mean

    if num_samples > 1:
        sample_std = float(np.std(weights, ddof=1))
        std_error = sample_std / np.sqrt(num_samples)
    else:
        std_error = 0.0

    return mean, std_error

def quenched_equation(R, alpha, num_samples):
    lhs = R / np.sqrt(1 - R)
    # use only the Monte Carlo mean inside the equation; we keep std_error for reporting
    I_mean, _ = I_of_R(R, num_samples)
    rhs = (alpha / np.pi) * I_mean
    return lhs - rhs

def solve_R_of_alpha(alpha_values, num_samples=100000):
    R_values = []
    R_std_errors = []
    I_means = []
    for alpha in alpha_values:
        try:
            R_root = bisect(lambda R: quenched_equation(R, alpha, num_samples),
                            1e-6, 0.9999, maxiter=50)
            # estimate the mean and std error of I(R_root) for reporting
            I_mean, I_std_err = I_of_R(R_root, num_samples)
            R_values.append(R_root)
            R_std_errors.append(I_std_err)
            I_means.append(I_mean)
        except Exception:
            R_values.append(np.nan)
            R_std_errors.append(np.nan)
            I_means.append(np.nan)
    return np.array(R_values), np.array(R_std_errors), np.array(I_means)

def epsilon_from_R(R):
    return (1/np.pi) * np.arccos(R)

def exPointOne():
    alpha_values = np.linspace(0.5, 5.0, 15)
    num_samples = 100000
    R_values, R_std_errors, I_means = solve_R_of_alpha(alpha_values, num_samples)
    eps_values = epsilon_from_R(R_values)

    # helper: invert lhs(R) = rhs_value to get R (lhs is monotonic on (0,1))
    def invert_lhs_to_R(rhs_value):
        def lhs_minus_rhs(R):
            return (R / np.sqrt(1.0 - R)) - rhs_value
        try:
            R_sol = bisect(lhs_minus_rhs, 1e-8, 1.0 - 1e-8, maxiter=50)
            return R_sol
        except Exception:
            return np.nan

    # propagate I std error into eps error by perturbing rhs = (alpha/pi)*I
    eps_errs = []
    for alpha, R, Imean, Ierr in zip(alpha_values, R_values, I_means, R_std_errors):
        if np.isnan(R) or np.isnan(Imean) or np.isnan(Ierr):
            eps_errs.append(np.nan)
            continue
        rhs = (alpha / np.pi) * Imean
        rhs_plus = (alpha / np.pi) * (Imean + Ierr)
        rhs_minus = (alpha / np.pi) * max(0.0, (Imean - Ierr))

        R_plus = invert_lhs_to_R(rhs_plus)
        R_minus = invert_lhs_to_R(rhs_minus)

        eps_plus = epsilon_from_R(R_plus) if not np.isnan(R_plus) else np.nan
        eps_minus = epsilon_from_R(R_minus) if not np.isnan(R_minus) else np.nan

        if np.isnan(eps_plus) or np.isnan(eps_minus):
            eps_errs.append(np.nan)
        else:
            # symmetric error estimate
            eps_errs.append(0.5 * abs(eps_plus - eps_minus))

    eps_errs = np.array(eps_errs)

    plt.errorbar(alpha_values, eps_values, yerr=eps_errs, fmt='o-', capsize=3, label="Quenched ε(α)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\epsilon(\alpha)$")
    plt.title("Exercise 05 – Quenched generalization error")
    plt.grid(True)
    plt.legend()
    plt.savefig("ex05_quenched_epsilon_vs_alpha.png")

    for a, R, eps, Ierr, epe in zip(alpha_values, R_values, eps_values, R_std_errors, eps_errs):
        print(f"α = {a:.2f}  →  R = {R:.4f},  ε = {eps:.4f},  I_std_error = {Ierr:.4e},  ε_std_error = {epe:.4e}")
    plt.clf()

def f1(R, alpha, num_samples=100_000, rng=None):
    # f1(R, α) = (α/π) * sqrt(1 - R) * I(R, α)
    R_clip = np.clip(R, 1e-9, 1 - 1e-9)
    Ival, _ = I_of_R(R_clip, num_samples=num_samples, rng=rng)
    return (alpha / np.pi) * np.sqrt(1.0 - R_clip) * Ival

def f2(R, alpha, num_samples=100_000, rng=None):
    # f2(R, α) = 1 - (π^2 R^2) / (α^2 I(R, α)^2)
    R_clip = np.clip(R, 1e-9, 1 - 1e-9)
    Ival, _ = I_of_R(R_clip, num_samples=num_samples, rng=rng)
    return 1.0 - (np.pi**2 * R_clip**2) / (alpha**2 * (Ival**2 + 1e-24))

# ---------------------------------------------------------
# exPointThree: implement the STRATEGY (do not run f1/f2)
# ---------------------------------------------------------
def fixed_point_iterate(alpha, R0, a, num_iter, f_callable,
                        num_samples=100_000, seed=None, clip=True,
                        tol=None, return_on_convergence=True):
    """
    Generic relaxed fixed-point iteration:
        R_{i+1} = (1 - a) R_i + a * f(R_i, α)

    Parameters
    ----------
    alpha : float
    R0    : float in (0, 1)
    a     : float in (0, 1]  (relaxation / learning-rate)
    num_iter : int
    f_callable : function(R, alpha, num_samples, rng) -> float
    num_samples : int, MC samples for I(R, α) inside f
    seed : None or int, for reproducibility
    clip : keep R in (0,1) to ensure well-defined updates

    Returns
    -------
    Rs : np.ndarray of shape (k+1,) with R_0, ..., R_k (k <= num_iter)
    """
    rng = np.random.default_rng(seed)
    R = float(R0)
    Rs = [np.clip(R, 1e-9, 1 - 1e-9) if clip else R]

    for _ in range(num_iter):
        f_val = f_callable(R, alpha, num_samples=num_samples, rng=rng)
        R_next = (1.0 - a) * R + a * f_val
        if clip:
            R_next = float(np.clip(R_next, 1e-9, 1.0 - 1e-9))
        # Early stopping if requested
        if tol is not None and return_on_convergence and abs(R_next - R) < tol:
            R = R_next
            Rs.append(R)
            break
        R = R_next
        Rs.append(R)

    return np.array(Rs)

def exPointTwo():
    """
    Implements ONLY the iterative strategy required by the exercise.
    - Provides `fixed_point_iterate(...)` to be used with either f1 or f2.
    - Does NOT run any specific case and does NOT plot.
    """
    # Strategy implemented above; nothing else to do here by design.
    pass

# ---------------------------------------------------------
# exPointFour: USE the strategy to solve for f1 and f2
# ---------------------------------------------------------
def exPointThree():
    # Required parameters from the sheet
    cases = [
        {"alpha": 5.0, "R0": 0.5, "a": 0.5},
        {"alpha": 0.1, "R0": 0.5, "a": 0.5},
    ]
    num_iter = 40
    num_samples = 100_000
    seed = 42  # reproducible MC

    # --- Plot for f1 ---
    plt.figure()
    for c in cases:
        Rs = fixed_point_iterate(
            alpha=c["alpha"], R0=c["R0"], a=c["a"],
            num_iter=num_iter, f_callable=f1,
            num_samples=num_samples, seed=seed, clip=True
        )
        plt.plot(Rs, marker='o', label=fr"$\alpha={c['alpha']}$, $R_0={c['R0']}$, $a={c['a']}$")
    plt.xlabel("Iteration $i$")
    plt.ylabel(r"$R_i$")
    plt.title(r"Fixed-point convergence with $f_1(R,\alpha)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("ex03_f1_convergence.png")
    # --- Plot for f2 ---
    plt.figure()
    for c in cases:
        Rs = fixed_point_iterate(
            alpha=c["alpha"], R0=c["R0"], a=c["a"],
            num_iter=num_iter, f_callable=f2,
            num_samples=num_samples, seed=seed, clip=True
        )
        plt.plot(Rs, marker='o', label=fr"$\alpha={c['alpha']}$, $R_0={c['R0']}$, $a={c['a']}$")
    plt.xlabel("Iteration $i$")
    plt.ylabel(r"$R_i$")
    plt.title(r"Fixed-point convergence with $f_2(R,\alpha)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("ex03_f2_convergence.png")
    plt.clf()

def exPointThreeExtra():
    # Same as exPointThree but you make 4x2 plots in a single figure
    # For 4 different learning rates a = [0.1, 0.3, 0.5, 0.7] at fixed alpha 0, on each plot you plot 
    # the convergence for alpha = [0.5, 1.5, 3.0, 5.0], for both f1 and f2
    cases = [
        {"alpha": 0.1, "R0": 0.5},
        {"alpha": 1.0, "R0": 0.5},
        {"alpha": 2.0, "R0": 0.5},
        {"alpha": 5.0, "R0": 0.5},
    ]
    a_values = [0.1, 0.3, 0.5, 0.7]
    num_iter = 40
    num_samples = 100_000
    seed = 42  # reproducible MC
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    for i, a in enumerate(a_values):
        ax_f1 = axes[i, 0]
        ax_f2 = axes[i, 1]
        for c in cases:
            Rs_f1 = fixed_point_iterate(
                alpha=c["alpha"], R0=c["R0"], a=a,
                num_iter=num_iter, f_callable=f1,
                num_samples=num_samples, seed=seed, clip=True
            )
            ax_f1.plot(Rs_f1, marker='o', label=fr"$\alpha={c['alpha']}$")
            Rs_f2 = fixed_point_iterate(
                alpha=c["alpha"], R0=c["R0"], a=a,
                num_iter=num_iter, f_callable=f2,
                num_samples=num_samples, seed=seed, clip=True
            )
            ax_f2.plot(Rs_f2, marker='o', label=fr"$\alpha={c['alpha']}$")
        ax_f1.set_xlabel("Iteration $i$")
        ax_f1.set_ylabel(r"$R_i$")
        ax_f1.set_title(fr"Convergence with $f_1(R,\alpha)$, $a={a}$")
        ax_f1.grid(True)
        ax_f1.legend()
        ax_f2.set_xlabel("Iteration $i$")
        ax_f2.set_ylabel(r"$R_i$")
        ax_f2.set_title(fr"Convergence with $f_2(R,\alpha)$, $a={a}$")
        ax_f2.grid(True)
        ax_f2.legend()
    plt.tight_layout()
    plt.savefig("ex03_f1_f2_convergence_grid.png")
    plt.clf()

# ---------------------------------------------------------
# exPointFour: curves vs alpha and adaptive choice
# ---------------------------------------------------------
def choose_update_function(alpha: float, a: float):
    """
    Heuristic chooser guided by the provided stability map:

    - Around the "sweet zone" near (alpha≈3.4, a≈0.17) both work; prefer f1.
    - For very small learning rates (a <= 0.20) and moderate alpha (alpha <= 6),
      f1 tends to be more stable (green region brighter in the figure), otherwise switch to f2.
    - For medium learning rates (0.20 < a <= 0.40), f2 is generally more robust; use f1
      only for tiny alpha (alpha < 1.0).
    - For larger learning rates (a > 0.40), use f2.

    Returns the callable f1 or f2.
    """
    if a <= 0.20:
        return f1 if alpha <= 6.0 else f2
    if a <= 0.40:
        return f1 if alpha < 1.0 else f2
    return f2


def exPointFour():
    """
    Produce R*(alpha) curves for both f1 and f2 on alpha in [0.1, 10],
    at a chosen learning rate a=0.17 (close to the stability sweet spot),
    and overlay an adaptive selection that picks which update to use
    based on (alpha, a).
    
    Returns the alpha values and corresponding epsilon values for use in exPointFive.
    """
    # Grid and params
    a = 0.17
    R0 = 0.5
    num_iter = 60
    num_samples = 10_000  # lighter for a full sweep
    seed = 123
    alphas = np.linspace(0.1, 10.0, 60)

    # Containers
    R_final_f1 = []
    R_final_f2 = []
    R_final_adapt = []
    chosen_idx = []  # 1 for f1, 2 for f2

    for alpha in alphas:
        # f1
        Rs1 = fixed_point_iterate(alpha, R0, a, num_iter, f1,
                                   num_samples=num_samples, seed=seed,
                                   clip=True, tol=1e-5)
        R_final_f1.append(float(Rs1[-1]))

        # f2
        Rs2 = fixed_point_iterate(alpha, R0, a, num_iter, f2,
                                   num_samples=num_samples, seed=seed,
                                   clip=True, tol=1e-5)
        R_final_f2.append(float(Rs2[-1]))

        # adaptive choice
        chooser = choose_update_function(alpha, a)
        RsA = fixed_point_iterate(alpha, R0, a, num_iter, chooser,
                                   num_samples=num_samples, seed=seed,
                                   clip=True, tol=1e-5)
        R_final_adapt.append(float(RsA[-1]))
        chosen_idx.append(1 if chooser is f1 else 2)

    R_final_f1 = np.array(R_final_f1)
    R_final_f2 = np.array(R_final_f2)
    R_final_adapt = np.array(R_final_adapt)
    chosen_idx = np.array(chosen_idx)

    # Convert to epsilon
    eps_final_f1 = epsilon_from_R(R_final_f1)
    eps_final_f2 = epsilon_from_R(R_final_f2)
    eps_final_adapt = epsilon_from_R(R_final_adapt)

    # Plot R values
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, R_final_f1, label=r"$R^*(\alpha)$ via $f_1$", color="tab:green")
    plt.plot(alphas, R_final_f2, label=r"$R^*(\alpha)$ via $f_2$", color="tab:purple")

    # Mark adaptive picks
    mask_f1 = chosen_idx == 1
    mask_f2 = ~mask_f1
    plt.scatter(alphas[mask_f1], R_final_adapt[mask_f1], s=18, color="tab:green", marker="o", alpha=0.9,
                label="adaptive pick → f1")
    plt.scatter(alphas[mask_f2], R_final_adapt[mask_f2], s=18, color="tab:purple", marker="x", alpha=0.9,
                label="adaptive pick → f2")

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$R^*(\alpha)$")
    plt.title(r"Fixed-point solution vs $\alpha$ (a = 0.17) with adaptive selection")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ex04_Rstar_vs_alpha_f1_f2_adaptive.png")
    plt.clf()

    # Plot epsilon values
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, eps_final_f1, label=r"$\varepsilon^*(\alpha)$ via $f_1$", color="tab:green")
    plt.plot(alphas, eps_final_f2, label=r"$\varepsilon^*(\alpha)$ via $f_2$", color="tab:purple")
    plt.scatter(alphas[mask_f1], eps_final_adapt[mask_f1], s=18, color="tab:green", marker="o", alpha=0.9,
                label="adaptive pick → f1")
    plt.scatter(alphas[mask_f2], eps_final_adapt[mask_f2], s=18, color="tab:purple", marker="x", alpha=0.9,
                label="adaptive pick → f2")

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\varepsilon^*(\alpha)$")
    plt.title(r"Fixed-point solution vs $\alpha$ (a = 0.17) with adaptive selection")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ex04_epsilon_vs_alpha_f1_f2_adaptive.png")
    plt.clf()

    return alphas, eps_final_adapt

def exPointFive():
    """
    Comparison plot of three epsilon(alpha) curves:
    1. Adaptive mixed-stability method from exPointFour
    2. Parametric quenched solution from exPointOne
    3. Experimental data from EX4/dataset_annealed.csv
    """
    import pandas as pd
    import os
    
    # Get adaptive method results
    print("Computing adaptive method results...")
    alpha_adaptive, eps_adaptive = exPointFour()
    
    # Get parametric solution from exPointOne (recompute with finer grid)
    print("Computing parametric quenched solution...")
    alpha_parametric = np.linspace(0.5, 10.0, 30)
    num_samples = 100000
    R_parametric, _, _ = solve_R_of_alpha(alpha_parametric, num_samples)
    eps_parametric = epsilon_from_R(R_parametric)
    
    # Load experimental data from EX4
    ex4_path = os.path.join(os.path.dirname(__file__), '..', 'EX4', 'dataset_annealed.csv')
    try:
        df = pd.read_csv(ex4_path)
        # Keep only rows with 20 bits as requested
        df = df[df['bits'] == 10].copy()
        if df.empty:
            raise ValueError("No rows with bits == 20 found in dataset.csv")
        # Convert accuracy to epsilon: epsilon = (100 - accuracy) / 100
        # alpha = dataset_size / bits
        alpha_exp = df['dataset_size'] / (df['bits'] * 2 )
        eps_exp = (100.0 - df['mean_accuracy']) / 100.0
        eps_std_exp = df['std_accuracy'] / 100.0  # convert std to same scale
        
        # Filter to reasonable alpha range for comparison
        mask = (alpha_exp <= 10.0) & (alpha_exp >= 0.1)
        alpha_exp = alpha_exp[mask].values
        eps_exp = eps_exp[mask].values
        eps_std_exp = eps_std_exp[mask].values
        has_exp_data = True
    except Exception as e:
        print(f"Warning: Could not load experimental data: {e}")
        has_exp_data = False
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    # Plot adaptive method
    plt.plot(alpha_adaptive, eps_adaptive, 'b-', linewidth=2, 
             label='Adaptive mixed-stability (EX5)', alpha=0.8)
    
    # Plot parametric solution
    plt.plot(alpha_parametric, eps_parametric, 'g--', linewidth=2,
             label='Parametric quenched (EX5 Point 1)', alpha=0.8)
    
    # Plot experimental data with error bars
    if has_exp_data:
        plt.errorbar(alpha_exp, eps_exp, yerr=eps_std_exp, 
                    fmt='ro', markersize=6, capsize=4, capthick=1.5,
                    label='Experimental data (bits=20)', alpha=0.7)
    
    plt.xlabel(r'$\alpha$ (training set size / input dimension)', fontsize=12)
    plt.ylabel(r'$\varepsilon(\alpha)$ (generalization error)', fontsize=12)
    plt.title('Comparison of generalization error: Mixed-stability vs Parametric vs Experimental', 
              fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 10.5)
    plt.ylim(0, max(0.5, np.max(eps_adaptive) * 1.1))
    plt.tight_layout()
    plt.savefig("ex05_epsilon_comparison_all_methods.png", dpi=150)
    plt.clf()
    
    print("\n=== exPointFive Summary ===")
    print(f"Adaptive method: α ∈ [{alpha_adaptive[0]:.2f}, {alpha_adaptive[-1]:.2f}]")
    print(f"Parametric method: α ∈ [{alpha_parametric[0]:.2f}, {alpha_parametric[-1]:.2f}]")
    if has_exp_data:
        print(f"Experimental data: α ∈ [{alpha_exp[0]:.2f}, {alpha_exp[-1]:.2f}], {len(alpha_exp)} points")
    print("Plot saved: ex05_epsilon_comparison_all_methods.png")

if __name__ == "__main__":
    exPointFive()