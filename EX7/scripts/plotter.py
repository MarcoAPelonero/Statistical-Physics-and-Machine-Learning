import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
import mpmath as mp
import argparse



DATA_DIR = Path('data')


@dataclass
class AggregatedCurves:
    alpha: np.ndarray
    train_mean: np.ndarray
    test_mean: np.ndarray
    train_se: np.ndarray
    test_se: np.ndarray


def _apply_theme():
    sns.set_theme(style='whitegrid', palette='tab10')


def _load_curves(filename: str) -> AggregatedCurves:
    data = np.loadtxt(DATA_DIR / filename, delimiter=',', skiprows=1)
    return _aggregate_curves(data)


def _aggregate_curves(data: np.ndarray) -> AggregatedCurves:
    alpha = data[:, 0]
    train_error = data[:, 1]
    test_error = data[:, 2]

    unique_alpha = np.unique(alpha)
    mean_train, mean_test = [], []
    se_train, se_test = [], []

    for value in unique_alpha:
        mask = alpha == value
        n = np.sum(mask)
        train_vals = train_error[mask]
        test_vals = test_error[mask]
        mean_train.append(np.mean(train_vals))
        mean_test.append(np.mean(test_vals))
        se_train.append(np.std(train_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        se_test.append(np.std(test_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    return AggregatedCurves(
        alpha=unique_alpha,
        train_mean=np.array(mean_train),
        test_mean=np.array(mean_test),
        train_se=np.array(se_train),
        test_se=np.array(se_test),
    )


def _plot_with_band(ax, x, mean, se, label, *, marker, linestyle='-', color=None, fill_alpha=0.25):
    (line,) = ax.plot(
        x,
        mean,
        marker=marker,
        linewidth=2.2,
        label=label,
        linestyle=linestyle,
        color=color,
    )
    band_color = color if color is not None else line.get_color()
    ax.fill_between(x, mean - se, mean + se, alpha=fill_alpha, color=band_color)


def _finalize_axes(ax, title, legend_kwargs=None):
    ax.set_xlabel('Alpha (P/N)')
    ax.set_ylabel('Error Rate')
    ax.set_title(title)
    legend_args = {'frameon': True}
    if legend_kwargs:
        legend_args.update(legend_kwargs)
    ax.legend(**legend_args)
    ax.grid(linestyle='--', alpha=0.6)
    ax.figure.tight_layout()


def _plot_standard_curves(curves: AggregatedCurves, title: str, plot_train_curves: bool = True):
    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    if plot_train_curves:
        _plot_with_band(ax, curves.alpha, curves.train_mean, curves.train_se, 'Training Error Rate', marker='o')
    _plot_with_band(ax, curves.alpha, curves.test_mean, curves.test_se, 'Test Error Rate', marker='s')
    _finalize_axes(ax, title)
    plt.show()

sqrt2 = mp.sqrt(2)

_GH_QUAD_ORDER = 80

def _prepare_gauss_hermite_nodes(order: int):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return tuple(mp.mpf(node) for node in nodes), tuple(mp.mpf(weight) for weight in weights)

_GH_NODES, _GH_WEIGHTS = _prepare_gauss_hermite_nodes(_GH_QUAD_ORDER)
_GH_SCALE = mp.sqrt(2)
_GH_NORMALIZER = 1 / mp.sqrt(2 / mp.pi)
_I_CACHE = {}


def H(x):
    # Gaussian tail H(x) = ∫_x^∞ Dt
    return 0.5 * mp.erfc(x / sqrt2)


def I_of_R(R):
    """
    Approximate ∫ Dt exp(-R^2 t^2 / 2) / H(-R t) with Gauss–Hermite.
    Works with mpf *and* mpc (needed during root-finding).
    """
    # allow complex R during iterations
    if isinstance(R, (mp.mpf, mp.mpc)):
        R_val = R
    else:
        R_val = mp.mpf(R)

    key = mp.nstr(R_val, 20)
    cached = _I_CACHE.get(key)
    if cached is not None:
        return cached

    total = mp.mpf('0')
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS):
        t = _GH_SCALE * node
        denom = H(-R_val * t)
        # avoid division by an absurdly small tail
        if abs(denom) <= mp.mpf('1e-30'):
            denom = mp.mpf('1e-30')
        total += weight * mp.e**(-R_val**2 * t**2 / 2) / denom

    result = _GH_NORMALIZER * total
    _I_CACHE[key] = result
    return result


def F_of_R(R, alpha):
    """
    Fixed-point equation for R_B at given α:
        R^2 / sqrt(1 - R^2) = (α/π) I(R)
    """
    alpha = mp.mpf(alpha)
    return R**2 / mp.sqrt(1 - R**2) - (alpha / mp.pi) * I_of_R(R)


def solve_RB(alpha, *, n_grid=200, tol=1e-8, max_iter=80):
    """
    Robust solver for F_of_R(R, alpha) = 0 on 0 < R < 1.
    Uses coarse bracketing + bisection (no fragile Newton steps).
    """
    alpha = mp.mpf(alpha)

    # search interval (stay away from exactly 0 and 1)
    a = mp.mpf('1e-5')
    b = mp.mpf('0.999999')

    # coarse grid to find a sign change
    Rs = [a + (b - a) * i / n_grid for i in range(n_grid + 1)]
    Fs = [F_of_R(R, alpha) for R in Rs]

    low = None
    high = None
    for R1, R2, F1, F2 in zip(Rs[:-1], Rs[1:], Fs[:-1], Fs[1:]):
        if F1 == 0:
            return mp.re(R1)
        if F1 * F2 < 0:
            low, high = R1, R2
            f_low, f_high = F1, F2
            break

    # as a safety net, if grid missed the exact sign change,
    # just use the last two points (near 1)
    if low is None:
        low, high = Rs[-2], Rs[-1]
        f_low, f_high = Fs[-2], Fs[-1]

    # bisection refinement
    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = F_of_R(mid, alpha)
        if abs(f_mid) < tol:
            return mp.re(mid)
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    # final value after max_iter
    return mp.re((low + high) / 2)


def bayes_generalisation_error(alpha):
    """
    ε_B(α) = (1/π) arccos R_B(α)
    """
    R = solve_RB(alpha)
    if isinstance(R, mp.mpc):  # drop tiny imaginary noise
        R = mp.re(R)
    return (1 / mp.pi) * mp.acos(R)



def exPointTwoPlotter():
    curves = _load_curves('exPointTwo_results.csv')
    # use raw alpha values from file (no artificial shift)
    
    _plot_standard_curves(curves, 'Training and Test Error Rates vs Alpha')


def extraPointOnePlotter():
    curves = _load_curves('extraPointOne_results.csv')
    _plot_standard_curves(curves, 'Training and Test Error Rates vs Alpha (Ridge Regression)')


def exPointFivePlotter():
    curves = _load_curves('exPointFive_results.csv')
    _plot_standard_curves(curves, 'Training and Test Error Rates vs Alpha (Adaline)')


def exPointSixPlotter():
    pseudo_inverse = _load_curves('exPointTwo_results.csv')
    pseudo_inverse_ridge = _load_curves('extraPointOne_results.csv')
    adaline = _load_curves('exPointFive_results.csv')
    _apply_theme()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    _plot_with_band(
        ax,
        pseudo_inverse_ridge.alpha,
        pseudo_inverse_ridge.test_mean,
        pseudo_inverse_ridge.test_se,
        'Test Error (Pseudoinverse + Ridge)',
        marker='o',
        color='C2',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        pseudo_inverse.alpha,
        pseudo_inverse.test_mean,
        pseudo_inverse.test_se,
        'Test Error (Pseudoinverse)',
        marker='s',
        color='C0',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        adaline.alpha,
        adaline.test_mean,
        adaline.test_se,
        'Test Error (Adaline)',
        marker='^',
        color='C1',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        pseudo_inverse_ridge.alpha,
        pseudo_inverse_ridge.train_mean,
        pseudo_inverse_ridge.train_se,
        'Training Error (Pseudoinverse + Ridge)',
        marker='o',
        linestyle='--',
        color='C2',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        pseudo_inverse.alpha,
        pseudo_inverse.train_mean,
        pseudo_inverse.train_se,
        'Training Error (Pseudoinverse)',
        marker='o',
        linestyle='--',
        color='C0',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        adaline.alpha,
        adaline.train_mean,
        adaline.train_se,
        'Training Error (Adaline)',
        marker='D',
        linestyle='--',
        color='C1',
        fill_alpha=0.15,
    )
    _finalize_axes(ax, 'Comparison: Pseudoinverse vs Adaline', legend_kwargs={'fontsize': 10})
    plt.show()


def exPointSevenPlotter():
    curves = _load_curves('exPointSeven_results.csv')
    _plot_standard_curves(curves, 'Training and Test Error Rates vs Alpha', plot_train_curves=False)

def exPointEightPlotter():
    """
    Load PointSeven results and superimpose the analytic Bayes generalisation curve.
    """
    curves = _load_curves('exPointSeven_results.csv')
    _apply_theme()

    alpha_vals = curves.alpha
    bayes_eps = np.array([float(bayes_generalisation_error(a)) for a in alpha_vals])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    _plot_with_band(
        ax,
        curves.alpha,
        curves.test_mean,
        curves.test_se,
        'Empirical Test Error',
        marker='o',
        color='C0',
        fill_alpha=0.20,
    )

    ax.plot(
        alpha_vals,
        bayes_eps,
        label='Bayes Theoretical Error',
        linestyle='-',
        linewidth=2.5,
        marker='s',
        markersize=6,
        color='C3',
    )

    _finalize_axes(ax, 'Empirical vs Bayes Theoretical Generalisation Error')
    plt.show()

def exPointNinePlotter():
    # Plot the curves from  the theoretical model, seven, five, two, and extra1
    curves_seven = _load_curves('exPointSeven_results.csv')
    curves_five = _load_curves('exPointFive_results.csv')
    curves_two = _load_curves('exPointTwo_results.csv')
    curves_extra1 = _load_curves('extraPointOne_results.csv')

    alpha_vals = curves_seven.alpha
    bayes_eps = np.array([float(bayes_generalisation_error(a)) for a in alpha_vals])

    _apply_theme()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    _plot_with_band(
        ax,
        curves_seven.alpha,
        curves_seven.test_mean,
        curves_seven.test_se,
        'Test Error (Bayes)',
        marker='o',
        color='C0',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        curves_five.alpha,
        curves_five.test_mean,
        curves_five.test_se,
        'Test Error (Adaline)',
        marker='s',
        color='C1',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        curves_two.alpha,
        curves_two.test_mean,
        curves_two.test_se,
        'Test Error (Pseudoinverse)',
        marker='^',
        color='C2',
        fill_alpha=0.15,
    )
    _plot_with_band(
        ax,
        curves_extra1.alpha,
        curves_extra1.test_mean,
        curves_extra1.test_se,
        'Test Error (Ridge Regression)',
        marker='D',
        color='C3',
        fill_alpha=0.15,
    )
    ax.plot(
        alpha_vals,
        bayes_eps,
        label='Bayes Theoretical Error',
        linestyle='-',
        linewidth=2.5,
        marker='o',
        markersize=6,
        color='C4',
    )
    _finalize_axes(ax, 'Comparison of Test Errors Across Experiments', legend_kwargs={'fontsize': 10})
    plt.show()


def _build_dispatch_table():
    return {
        '2': exPointTwoPlotter,
        'two': exPointTwoPlotter,
        '5': exPointFivePlotter,
        'five': exPointFivePlotter,
        '6': exPointSixPlotter,
        'six': exPointSixPlotter,
        '7': exPointSevenPlotter,
        'seven': exPointSevenPlotter,
        '8': exPointEightPlotter,
        'eight': exPointEightPlotter,
        '9': exPointNinePlotter,
        'nine': exPointNinePlotter,
        'extra1': extraPointOnePlotter,
        'extra': extraPointOnePlotter,
        '': exPointEightPlotter,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('which', nargs='?', default='', help='Which example to plot (number or name). e.g. 2 -> exPointTwo')
    args = parser.parse_args()

    dispatch = _build_dispatch_table()
    key = str(args.which).lower()
    if key in dispatch:
        dispatch[key]()
    else:
        valid = ', '.join(sorted(k for k in dispatch.keys() if k))
        print(f"Unknown plotter '{args.which}'. Valid options: {valid}")
