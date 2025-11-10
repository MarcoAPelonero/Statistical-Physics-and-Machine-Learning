import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from main import f1, f2, fixed_point_iterate
from scipy.ndimage import gaussian_filter

def stability_metric(Rs, alpha, f_callable, num_samples=10_000, rng=None):
    """
    True fixed-point residual metric:
        S = |R_last - f(R_last, α)|

    Smaller S → closer to a self-consistent fixed point → more stable.
    """
    if rng is None:
        rng = np.random.default_rng()
    R_last = Rs[-1]
    f_val = f_callable(R_last, alpha, num_samples=num_samples, rng=rng)
    return abs(R_last - f_val)


def stability_surface(f_callable, a_values, alpha_values,
                      num_iter=40, num_samples=50_000,
                      n_avg=5, seed=42, smooth_sigma=1.0):
    """
    Compute an averaged fixed-point residual stability surface.

    Parameters
    ----------
    f_callable : callable
        Function f1 or f2.
    a_values : array
        Learning rate grid.
    alpha_values : array
        Alpha grid.
    num_iter : int
        Fixed-point iterations.
    num_samples : int
        Monte Carlo samples per call.
    n_avg : int
        Number of independent Monte Carlo runs to average (reduces noise ~ 1/sqrt(n_avg)).
    seed : int
        Base random seed; each (a, alpha, run) uses a distinct offset.
    smooth_sigma : float
        Gaussian smoothing (in grid cells) applied to the averaged surface.

    Returns
    -------
    S_smooth : np.ndarray
        Smoothed 2D surface of residual |R* - f(R*,α)|.
    """
    n_a, n_alpha = len(a_values), len(alpha_values)
    S_accum = np.zeros((n_a, n_alpha))

    for k in range(n_avg):
        print(f"  Averaging pass {k+1}/{n_avg}...")
        S = np.zeros_like(S_accum)
        for i, a in enumerate(a_values):
            for j, alpha in enumerate(alpha_values):
                local_seed = seed + k * 10000 + i * 100 + j
                Rs = fixed_point_iterate(
                    alpha=alpha, R0=0.5, a=a,
                    num_iter=num_iter, f_callable=f_callable,
                    num_samples=num_samples, seed=local_seed, clip=True
                )
                # Compute fixed-point residual at the final R
                rng = np.random.default_rng(local_seed)
                R_last = Rs[-1]
                f_val = f_callable(R_last, alpha, num_samples=num_samples, rng=rng)
                S[i, j] = abs(R_last - f_val)
        S_accum += S

    # Average over Monte Carlo runs
    S_mean = S_accum / n_avg

    # Optional light smoothing to remove high-frequency noise
    if smooth_sigma > 0.0:
        S_mean = gaussian_filter(S_mean, sigma=smooth_sigma)

    return S_mean

# ======================================================
# --- Plotting utilities ---
# ======================================================

def plot_stability_surface(a_values, alpha_values, S, title, cmap="Greens_r"):
    """
    Plot a single stability surface (log10 residual) for one function.
    Low residual = high stability → bright color (reversed colormap).
    """
    A, ALPHA = np.meshgrid(alpha_values, a_values)
    logS = np.log10(S + 1e-12)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(ALPHA, A, logS,
                           cmap=cmap, edgecolor='none', alpha=0.9)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a$")
    ax.set_zlabel(r"$\log_{10}|R^* - f(R^*,\alpha)|$")
    ax.set_title(title)

    cbar = fig.colorbar(surf, shrink=0.6, aspect=10)
    cbar.set_label("log₁₀ fixed-point residual (low = stable)")
    plt.tight_layout()
    return fig, ax


def plot_3d_overlap(a_values, alpha_values, S1, S2,
                    title="Normalized 3D overlap: f₁ (green) vs f₂ (purple)"):
    """
    Overlay both normalized stability surfaces in one 3D plot.
    Higher = more stable (smaller residual).
    """
    A, ALPHA = np.meshgrid(alpha_values, a_values)

    # Convert to log space and invert → higher = more stable
    Z1 = np.log10(S1 + 1e-12)
    Z2 = np.log10(S2 + 1e-12)
    Z1n = 1 - (Z1 - np.nanmin(Z1)) / (np.nanmax(Z1) - np.nanmin(Z1) + 1e-12)
    Z2n = 1 - (Z2 - np.nanmin(Z2)) / (np.nanmax(Z2) - np.nanmin(Z2) + 1e-12)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    s1 = ax.plot_surface(ALPHA, A, Z1n, cmap='Greens', alpha=0.65, edgecolor='none')
    s2 = ax.plot_surface(ALPHA, A, Z2n, cmap='Purples', alpha=0.50, edgecolor='none')

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a$")
    ax.set_zlabel("normalized stability (high = good)")
    ax.set_title(title)

    cbar1 = fig.colorbar(s1, ax=ax, shrink=0.6, aspect=12, pad=0.02)
    cbar1.set_label('f₁ normalized stability (high = good)')
    cbar2 = fig.colorbar(s2, ax=ax, shrink=0.6, aspect=12, pad=0.10)
    cbar2.set_label('f₂ normalized stability (high = good)')

    plt.tight_layout()
    plt.savefig("ex05_stability_3d_overlap_normalized.png", dpi=300)
    plt.show()
    return fig, ax


def highlight_sweetzone(alpha_values, a_values, S1, S2,
                        stability_threshold=0.7):
    """
    Plot 2D normalized stability maps with joint 'sweet zone' highlight.
    Sweet zone = both f₁ and f₂ have small residuals (convergent).
    """
    logS1 = np.log10(S1 + 1e-12)
    logS2 = np.log10(S2 + 1e-12)

    # Normalize to [0,1], invert → higher = more stable
    norm1 = 1 - (logS1 - logS1.min()) / (logS1.max() - logS1.min() + 1e-12)
    norm2 = 1 - (logS2 - logS2.min()) / (logS2.max() - logS2.min() + 1e-12)

    # Combined sweet zone: both above threshold
    sweet_mask = (norm1 > stability_threshold) & (norm2 > stability_threshold)

    # Compute centroid of sweet zone
    if np.any(sweet_mask):
        idxs = np.argwhere(sweet_mask)
        mean_i, mean_j = idxs.mean(axis=0)
        sweet_a = a_values[int(round(mean_i))]
        sweet_alpha = alpha_values[int(round(mean_j))]
    else:
        sweet_a, sweet_alpha = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(9, 6))
    c1 = ax.contourf(alpha_values, a_values, norm1, levels=30, cmap="Greens", alpha=0.6)
    c2 = ax.contourf(alpha_values, a_values, norm2, levels=30, cmap="Purples", alpha=0.4)

    # Sweet zone hatch overlay
    ax.contourf(alpha_values, a_values, sweet_mask.astype(float),
                levels=[0.5, 1.5], colors='none', hatches=['////'], alpha=0)

    if not np.isnan(sweet_a):
        ax.scatter(sweet_alpha, sweet_a, s=120, c='red', edgecolors='black',
                   label=fr"Sweet zone center: α≈{sweet_alpha:.2f}, a≈{sweet_a:.2f}")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a$")
    ax.set_title("Normalized fixed-point stability overlap ('sweet zone')", fontsize=14)
    ax.legend(loc='upper right')

    fig.colorbar(c1, ax=ax, shrink=0.75, label="f₁ normalized stability (high = good)")
    fig.colorbar(c2, ax=ax, shrink=0.75, label="f₂ normalized stability (high = good)")

    plt.tight_layout()
    plt.savefig("ex05_stability_sweetzone.png", dpi=300)
    plt.show()

    if not np.isnan(sweet_a):
        print(f"🟢 Sweet zone center near α ≈ {sweet_alpha:.3f}, a ≈ {sweet_a:.3f}")
    else:
        print("⚠️ No region found where both stabilities exceed threshold.")


# ======================================================
# --- Complete pipeline ---
# ======================================================

def exPointStability3D():
    a_values = np.linspace(0.05, 0.9, 15)
    alpha_values = np.linspace(0.1, 20.0, 80)
    num_iter = 40
    num_samples = 5000
    seed = 42

    print("Computing fixed-point stability surfaces...")
    S1 = stability_surface(f1, a_values, alpha_values,
                           num_iter=num_iter, num_samples=num_samples, seed=seed)
    S2 = stability_surface(f2, a_values, alpha_values,
                           num_iter=num_iter, num_samples=num_samples, seed=seed)

    print("Plotting results...")
    highlight_sweetzone(alpha_values, a_values, S1, S2, stability_threshold=0.7)
    plot_stability_surface(a_values, alpha_values, S1,
                           title=r"Fixed-point residual surface for $f_1(R,\alpha)$", cmap="Greens_r")
    plot_stability_surface(a_values, alpha_values, S2,
                           title=r"Fixed-point residual surface for $f_2(R,\alpha)$", cmap="Purples_r")
    plot_3d_overlap(a_values, alpha_values, S1, S2)


# ======================================================
# --- Main entry ---
# ======================================================

if __name__ == "__main__":
    exPointStability3D()
