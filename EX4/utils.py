# utils.py  (clean ranges + robust iteration)
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_context("talk")
    sns.set_style("whitegrid")
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# -----------------------------
# Data loading
# -----------------------------
def _default_dataset_path() -> Path:
    here = Path(__file__).resolve().parent
    p1 = here / "dataset.csv"
    if p1.exists():
        return p1
    return Path("/mnt/data/dataset.csv")

def load_ex3_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """
    Expected CSV columns (case-insensitive):
      bits, dataset_size, mean_accuracy, std_accuracy
    Adds: alpha=P/N, eps=1-acc, eps_std
    """
    csv_path = Path(path) if path is not None else _default_dataset_path()
    df = pd.read_csv(csv_path)

    # Normalize names (case-insensitive)
    low = {c.lower(): c for c in df.columns}
    df = df.rename(columns={
        low.get("bits", "bits"): "bits",
        low.get("dataset_size", "dataset_size"): "dataset_size",
        low.get("mean_accuracy", "mean_accuracy"): "mean_accuracy",
        low.get("std_accuracy", "std_accuracy"): "std_accuracy",
    })

    # Each pattern concatenates two Scalars (top|bottom), so perceptron input size is 2*bits.
    df["alpha"]   = df["dataset_size"] / (2.0 * df["bits"])
    df["eps"]     = 1.0 - df["mean_accuracy"] / 100.0
    df["eps_std"] = df["std_accuracy"] / 100.0
    return df[["bits", "dataset_size", "alpha", "eps", "eps_std", "mean_accuracy", "std_accuracy"]]


# -----------------------------
# Annealed parametric curve
# -----------------------------
def annealed_parametric(num_points: int = 2000,
                        alpha_cap: float | None = 10.0,
                        eps_lo: float = 1e-6,
                        eps_hi: float = 0.5 - 1e-6):
    """
    Parametric:
        alpha(eps) = (1 - eps) * pi * cot(pi * eps),   eps∈(0, 1/2)
    Returns (eps, alpha) **clipped** so that alpha <= alpha_cap (if provided).
    """
    eps = np.linspace(eps_lo, eps_hi, num_points)
    alpha = (1.0 - eps) * np.pi * (np.cos(np.pi * eps) / np.sin(np.pi * eps))
    if alpha_cap is not None:
        m = alpha <= alpha_cap
        eps, alpha = eps[m], alpha[m]
    return eps, alpha


# -----------------------------
# Fixed-point maps and solver
# -----------------------------
def f1(alpha: float, eps: float) -> float:
    # 1 - (alpha/pi) * tan(pi*eps)
    return 1.0 - (alpha / math.pi) * math.tan(math.pi * eps)

def f2(alpha: float, eps: float) -> float:
    # (1/pi) * arctan( pi*(1-eps)/alpha )
    return (1.0 / math.pi) * math.atan(math.pi * (1.0 - eps) / alpha)

def fixed_point_iterate(alpha: float,
                        eps0: float,
                        a: float,
                        fkind: str = "f1",
                        max_iter: int = 2000,
                        tol: float = 1e-12,
                        clip_eps: tuple[float, float] = (1e-10, 0.5 - 1e-10)):
    """
    Damped iteration: eps_{i+1} = (1 - a)*eps_i + a*f(eps_i, alpha)
    - Hard clips to (0, 1/2) to avoid tan singularities.
    - Early-stops and flags divergence if we hit the upper boundary.
    Returns (history np.ndarray, converged: bool, diverged: bool).
    """
    if fkind not in ("f1", "f2"):
        raise ValueError("fkind must be 'f1' or 'f2'")
    f = f1 if fkind == "f1" else f2

    lo, hi = clip_eps
    eps = float(np.clip(eps0, lo, hi))
    hist = [eps]

    for _ in range(max_iter):
        val = f(alpha, eps)
        if not np.isfinite(val):
            # blow-up in map => divergence
            return np.array(hist, float), False, True

        val = float(np.clip(val, lo, hi))
        new_eps = (1.0 - a) * eps + a * val

        # if we get too close to the boundary, treat as instability
        if new_eps > hi * 0.999999:
            hist.append(hi)
            return np.array(hist, float), False, True

        hist.append(new_eps)
        if abs(new_eps - eps) < tol:
            return np.array(hist, float), True, False
        eps = new_eps

    return np.array(hist, float), False, False  # max_iter reached


def solve_eps_for_alpha(alpha: float,
                        fkind: str = "f2",
                        a: float = 0.5,
                        eps0: float = 0.25,
                        tol: float = 1e-12,
                        max_iter: int = 10000) -> float:
    hist, _, _ = fixed_point_iterate(alpha, eps0=eps0, a=a, fkind=fkind,
                                     max_iter=max_iter, tol=tol)
    return float(hist[-1])


def sweep_eps_curve(alpha_max: float = 10.0,
                    num: int = 300,
                    fkind: str = "f2",
                    a: float = 0.5,
                    eps0: float = 0.25,
                    tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    ε(α) for α in [1e-4, alpha_max]. Warm-start for stability.
    """
    alphas = np.linspace(1e-4, alpha_max, num)
    eps_vals = np.empty_like(alphas)
    cur = eps0
    for i, a_val in enumerate(alphas):
        cur = solve_eps_for_alpha(a_val, fkind=fkind, a=a, eps0=cur, tol=tol)
        eps_vals[i] = cur
    return alphas, eps_vals


# -----------------------------
# Plotting helpers
# -----------------------------
def _ensure_fig_dir(fig_dir: str | Path = "figs") -> Path:
    p = Path(fig_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _auto_alpha_cap(df: pd.DataFrame | None, fallback: float = 10.0) -> float:
    if df is None or df.empty:
        return fallback
    return float(max(fallback, 1.1 * df["alpha"].max()))

def _auto_alpha_cap_multi(dfs: list[pd.DataFrame | None], fallback: float = 10.0) -> float:
    max_alpha = None
    for df in dfs:
        if df is None or df.empty:
            continue
        cur = float(df["alpha"].max())
        max_alpha = cur if max_alpha is None else max(max_alpha, cur)
    if max_alpha is None:
        return fallback
    return float(max(fallback, 1.1 * max_alpha))

def plot_parametric_vs_data(df: pd.DataFrame,
                            save_to: str | Path = "figs/parametric_vs_data.png",
                            title: str = "Annealed ε(α): parametric vs Exercise 03",
                            alpha_cap: float | None = None):
    """
    Scatter Ex3 data (α, ε) with error bars; overlay clipped parametric curve.
    """
    fig_dir = _ensure_fig_dir(Path(save_to).parent)
    if alpha_cap is None:
        alpha_cap = _auto_alpha_cap(df, 10.0)
    eps_p, alpha_p = annealed_parametric(alpha_cap=alpha_cap)

    plt.figure(figsize=(9, 6))
    if _HAS_SNS:
        import seaborn as sns
        sns.scatterplot(data=df, x="alpha", y="eps", hue="bits", s=50, legend=True)
    else:
        for b in sorted(df["bits"].unique()):
            sub = df[df["bits"] == b]
            plt.scatter(sub["alpha"], sub["eps"], label=f"N={b}", alpha=0.85)
    plt.errorbar(df["alpha"], df["eps"], yerr=df["eps_std"],
                 fmt="none", ecolor="gray", alpha=0.6, capsize=2)

    plt.plot(alpha_p, eps_p, lw=2, label="Parametric (annealed)")

    plt.xlim(0.0, alpha_cap)
    plt.ylim(0.0, 0.5)
    plt.xlabel(r"$\alpha = P/N$")
    plt.ylabel(r"$\varepsilon$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=180)
    plt.close()

def plot_parametric_with_extra(df_base: pd.DataFrame,
                               df_extra: pd.DataFrame,
                               save_to: str | Path = "figs/parametric_with_extra.png",
                               title: str = "Annealed ε(α): parametric vs data (with extras)",
                               base_label: str = "Exercise 03 (acc→ε)",
                               extra_label: str = "Random labels (extra point)")-> None:
    """
    Overlay parametric curve, Exercise 03 dataset, and an additional dataset
    (e.g. extra random-label sweep).
    """
    cap = _auto_alpha_cap_multi([df_base, df_extra], 10.0)
    _ensure_fig_dir(Path(save_to).parent)
    eps_p, alpha_p = annealed_parametric(alpha_cap=cap)

    plt.figure(figsize=(9, 6))

    # Base dataset (multiple bit-widths)
    if _HAS_SNS:
        import seaborn as sns
        sns.scatterplot(data=df_base, x="alpha", y="eps", hue="bits", s=50,
                        legend=True, palette="deep")
    else:
        for b in sorted(df_base["bits"].unique()):
            sub = df_base[df_base["bits"] == b]
            plt.scatter(sub["alpha"], sub["eps"], label=f"{base_label} N={b}", alpha=0.85)

    plt.errorbar(df_base["alpha"], df_base["eps"], yerr=df_base["eps_std"],
                 fmt="none", ecolor="gray", alpha=0.5, capsize=2)

    # Extra dataset (single bit-width; plot with squares)
    if df_extra is not None and not df_extra.empty:
        plt.errorbar(df_extra["alpha"], df_extra["eps"], yerr=df_extra["eps_std"],
                     fmt="s", ms=6, mfc="none", mec="black", ecolor="black",
                     capsize=2, label=extra_label)

    plt.plot(alpha_p, eps_p, lw=2, label="Parametric (annealed)")

    plt.xlim(0.0, cap)
    plt.ylim(0.0, 0.5)
    plt.xlabel(r"$\alpha = P/N$")
    plt.ylabel(r"$\varepsilon$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=180)
    plt.close()

def plot_iterations(alpha: float,
                    eps0: float,
                    a_list: list[float],
                    fkind: str,
                    save_to: str | Path,
                    max_points: int = 300):
    """
    Plot ε_i vs i. Show only the stable prefix; flag divergence.
    """
    _ensure_fig_dir(Path(save_to).parent)
    plt.figure(figsize=(9, 6))

    def _stable_prefix_index(hist: np.ndarray,
                             conv: bool,
                             div: bool,
                             tol: float = 1e-8,
                             window: int = 8,
                             max_points_local: int = max_points) -> int:
        """
        Return an index n such that hist[:n] is the portion we consider "stable" to plot.
        Rules:
        - If converged: return full length.
        - If diverged: return up to the divergence point (full length — usually small).
        - If no convergence: try to detect when the sequence has entered a small-amplitude
          oscillation or reached a small moving-std; return the earliest index where the
          moving std over `window` falls below `tol` (i.e., effectively stable).
        - If none detected, return min(len(hist), max_points_local) to avoid over-plotting.
        """
        L = len(hist)
        if L <= 1:
            return L
        if conv or div:
            return min(L, max_points_local)

        # moving std test
        w = max(2, min(window, L // 2))
        ms = None
        for i in range(w, L + 1):
            chunk = hist[i - w:i]
            mstd = float(np.std(chunk))
            if ms is None:
                ms = []
            ms.append(mstd)
            # small moving std => effectively stable
            if mstd < tol:
                return min(i, max_points_local)

        # periodic / small set detection: look for few unique values (within tol)
        # check last up to 2*w points
        tail = hist[max(0, L - 2 * w):]
        # cluster values by rounding to tolerance
        if len(tail) > 0:
            rounded = np.round(tail / max(tol, 1e-12))
            unique_count = len(np.unique(rounded))
            if unique_count <= 4:
                # find earliest index where the tail's behavior starts
                for j in range(w, L + 1):
                    sub = hist[j - w:j]
                    if len(sub) > 0 and len(np.unique(np.round(sub / max(tol, 1e-12)))) <= 4:
                        return min(j, max_points_local)

        # fallback limit to avoid plotting huge noisy series
        return min(L, max_points_local)

    for a in a_list:
        hist, conv, div = fixed_point_iterate(alpha, eps0=eps0, a=a, fkind=fkind)
        n = _stable_prefix_index(hist, conv, div, tol=1e-9, window=12, max_points_local=max_points)
        label = f"a={a} ({'conv' if conv else 'div' if div else 'no conv'})"
        plt.plot(np.arange(n), hist[:n], lw=2, label=label)

    plt.xlabel("iteration i")
    plt.ylabel(r"$\varepsilon_i$")
    plt.ylim(0.0, 0.5)
    plt.title(f"Fixed-point: {fkind} at α={alpha}, ε0={eps0}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=180)
    plt.close()

def plot_curves_overlay(df: pd.DataFrame,
                        alpha_solved: np.ndarray,
                        eps_solved: np.ndarray,
                        save_to: str | Path = "figs/overlay_all.png",
                        title: str = "ε(α): parametric + fixed-point + data",
                        alpha_cap: float | None = None):
    """
    Overlay parametric curve (clipped), solver ε(α), and the data with error bars.
    """
    if alpha_cap is None:
        alpha_cap = _auto_alpha_cap(df, 10.0)
    _ensure_fig_dir(Path(save_to).parent)
    eps_p, alpha_p = annealed_parametric(alpha_cap=alpha_cap)

    plt.figure(figsize=(9, 6))
    # Data
    plt.errorbar(df["alpha"], df["eps"], yerr=df["eps_std"],
                 fmt="o", ms=4, alpha=0.85, label="Exercise 03 (acc→ε)",
                 ecolor="gray", capsize=2)

    # Parametric & numerical curve (restricted to cap)
    m = alpha_solved <= alpha_cap
    plt.plot(alpha_p, eps_p, lw=2, label="Parametric (annealed)")
    plt.plot(alpha_solved[m], eps_solved[m], lw=2, ls="--", label="Fixed-point ε(α)")

    plt.xlim(0.0, alpha_cap)
    plt.ylim(0.0, 0.5)
    plt.xlabel(r"$\alpha = P/N$")
    plt.ylabel(r"$\varepsilon$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=180)
    plt.close()
