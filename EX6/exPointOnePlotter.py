import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

def group_stats(path, N, n_trials=None):
    df = pd.read_csv(path)
    g = df.groupby('train_size').agg({'test_error_rate': ['mean', 'std']}).reset_index()
    g.columns = ['train_size', 'test_err_mean', 'test_err_std']
    g['alpha'] = g['train_size'] / float(N)
    if n_trials is None:
        try:
            n_trials = int(df['trial'].nunique())
        except Exception:
            n_trials = 1
    g['test_err_se'] = g['test_err_std'] / np.sqrt(n_trials)
    return g.sort_values('alpha')

def fit_powerlaw(alpha, y, yerr=None):
    # Fit y = A * alpha^{-b}  ->  log(y) = log(A) - b log(alpha)
    mask = (alpha > 0) & (y > 0)
    x = np.log(alpha[mask])
    z = np.log(y[mask])
    if len(x) < 2:
        raise RuntimeError('Not enough points to fit')

    if yerr is not None:
        # approximate variance in log space: var(log y) ≈ (yerr / y)^2
        v = (yerr[mask] / y[mask]) ** 2
        # weights for polyfit = 1/sigma, and polyfit expects w such that it minimizes sum(w*(y - p(x))^2)
        w = 1.0 / np.sqrt(v)
        p = np.polyfit(x, z, 1, w=w)
    else:
        p = np.polyfit(x, z, 1)

    slope, intercept = p[0], p[1]
    b = -slope
    A = math.exp(intercept)

    # compute R^2 in log-space
    z_pred = np.polyval(p, x)
    ss_res = np.sum((z - z_pred) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    # return parameters and mask to allow plotting fitted curve over original alpha range
    return {'A': A, 'b': b, 'r2': r2, 'mask': mask}

def main():
    plots = Path('plots')
    p6 = plots / 'exPointSix_results.csv'
    p2 = plots / 'exPointTwo_results.csv'

    if not p6.exists():
        raise FileNotFoundError(f"Missing {p6}")
    if not p2.exists():
        raise FileNotFoundError(f"Missing {p2}")

    # Simulation sizes (must match C++): exPointSix bits=10 -> N=20; exPointTwo bits=30 -> N=60
    N6 = 2 * 10
    N2 = 2 * 30
    n6 = 100
    n2 = 1000

    g6 = group_stats(p6, N6, n6)
    g2 = group_stats(p2, N2, n2)

    alpha6 = g6['alpha'].to_numpy()
    y6 = g6['test_err_mean'].to_numpy()
    se6 = g6['test_err_se'].to_numpy()

    alpha2 = g2['alpha'].to_numpy()
    y2 = g2['test_err_mean'].to_numpy()
    se2 = g2['test_err_se'].to_numpy()

    # Fit power laws (use all positive points)
    fit6 = fit_powerlaw(alpha6, y6, se6)
    fit2 = fit_powerlaw(alpha2, y2, se2)

    # Print results
    print('Fit results (y = A * alpha^{-b})')
    print('ExPointSix (Perceptron): A = {:.5g}, b = {:.5g}, R^2 = {:.5g}'.format(fit6['A'], fit6['b'], fit6['r2']))
    print('ExPointTwo (Hebbian):   A = {:.5g}, b = {:.5g}, R^2 = {:.5g}'.format(fit2['A'], fit2['b'], fit2['r2']))

    # Prepare smooth curves for plotting
    alpha_grid6 = np.linspace(alpha6.min(), alpha6.max(), 200)
    y6_fit = fit6['A'] * alpha_grid6 ** (-fit6['b'])

    alpha_grid2 = np.linspace(alpha2.min(), alpha2.max(), 200)
    y2_fit = fit2['A'] * alpha_grid2 ** (-fit2['b'])

    # Plot on linear axes and log-log axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

    # linear plot
    ax1.errorbar(alpha6, y6, yerr=se6, marker='o', linestyle='-', label='ExPointSix data', color='C0')
    ax1.plot(alpha_grid6, y6_fit, '--', color='C0', label=f'Fit6: A={fit6["A"]:.3g}, b={fit6["b"]:.3g}')

    ax1.errorbar(alpha2, y2, yerr=se2, marker='s', linestyle='--', label='ExPointTwo data', color='C1')
    ax1.plot(alpha_grid2, y2_fit, '-.', color='C1', label=f'Fit2: A={fit2["A"]:.3g}, b={fit2["b"]:.3g}')

    ax1.set_xlabel('Load (α = P/N)')
    ax1.set_ylabel('Generalization Error Rate')
    ax1.set_title('Power-law fits (linear scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # log-log plot
    ax2.errorbar(alpha6, y6, yerr=se6, marker='o', linestyle='None', label='ExPointSix data', color='C0')
    ax2.plot(alpha_grid6, y6_fit, '--', color='C0', label=f'Fit6: b={fit6["b"]:.3g}')
    ax2.errorbar(alpha2, y2, yerr=se2, marker='s', linestyle='None', label='ExPointTwo data', color='C1')
    ax2.plot(alpha_grid2, y2_fit, '-.', color='C1', label=f'Fit2: b={fit2["b"]:.3g}')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Load (α = P/N) [log]')
    ax2.set_ylabel('Generalization Error Rate [log]')
    ax2.set_title('Power-law fits (log-log)')
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    out = plots / 'exPoint_powerlaw_fits.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Plot saved to {out}')
    plt.show()

if __name__ == '__main__':
    main()
