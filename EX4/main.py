# main.py  (void functions only; sane α ranges; robust iteration plots)
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import utils

# ---------- Exercise points ----------
def exPointOne(csv_path: str | None = None) -> None:
    df = utils.load_ex3_dataset(csv_path)
    cap = max(10.0, 1.1 * float(df["alpha"].max()))
    utils.plot_parametric_vs_data(
        df,
        save_to=Path("figs") / "parametric_vs_data.png",
        title="Annealed ε(α): parametric vs Exercise 03",
        alpha_cap=cap
    )
    print("Saved: figs/parametric_vs_data.png")

def exPointTwo(alpha: float = 5.0, eps0: float = 0.25) -> None:
    utils.plot_iterations(alpha=alpha, eps0=eps0, a_list=[0.5, 0.9, 0.1],
                          fkind="f1", save_to=Path("figs") / "iterations_f1.png",
                          max_points=300)
    print("Saved: figs/iterations_f1.png")

def exPointThree(alpha: float = 5.0, eps0: float = 0.25) -> None:
    utils.plot_iterations(alpha=alpha, eps0=eps0, a_list=[0.5, 0.9, 0.1],
                          fkind="f2", save_to=Path("figs") / "iterations_f2.png",
                          max_points=300)
    print("Saved: figs/iterations_f2.png")

def exPointFour(alpha_max: float = 10.0, method: str = "f2", a: float = 0.5) -> None:
    alphas, eps_vals = utils.sweep_eps_curve(alpha_max=alpha_max, num=300,
                                             fkind=method, a=a, eps0=0.25)
    from matplotlib import pyplot as plt
    utils._ensure_fig_dir("figs")
    plt.figure(figsize=(9, 6))
    plt.plot(alphas, eps_vals, lw=2, label=f"Fixed-point (method={method}, a={a})")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\varepsilon$")
    plt.title("High-precision ε(α) sweep")
    plt.ylim(0.0, 0.5)
    plt.xlim(0.0, alpha_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/eps_curve_only.png", dpi=180)
    plt.close()
    print("Saved: figs/eps_curve_only.png")

    # Small numeric sample
    idx = np.linspace(0, len(alphas) - 1, 8, dtype=int)
    print("Sample ε(α):")
    for i in idx:
        print(f"  alpha={alphas[i]:6.3f}  eps={eps_vals[i]:.6f}")

def exPointFive(csv_path: str | None = None, alpha_max: float = 10.0,
                method: str = "f2", a: float = 0.5) -> None:
    df = utils.load_ex3_dataset(csv_path)
    cap = max(alpha_max, 1.1 * float(df["alpha"].max()))
    alpha_solved, eps_solved = utils.sweep_eps_curve(alpha_max=cap, num=300,
                                                     fkind=method, a=a, eps0=0.25)
    utils.plot_curves_overlay(
        df,
        alpha_solved,
        eps_solved,
        save_to=Path("figs") / "overlay_all.png",
        title="ε(α): parametric + fixed-point + Exercise 03 data",
        alpha_cap=cap
    )
    print("Saved: figs/overlay_all.png")

def extraPointOne(csv_path: str | None = None,
                  extra_csv_path: str | None = None) -> None:
    df_base = utils.load_ex3_dataset(csv_path)
    extra_path = Path(extra_csv_path) if extra_csv_path is not None else Path("dataset_annealed.csv")
    if not extra_path.exists():
        raise FileNotFoundError(f"Extra dataset not found at '{extra_path}'")
    df_extra = utils.load_ex3_dataset(extra_path)
    utils.plot_parametric_with_extra(
        df_base,
        df_extra,
        save_to=Path("figs") / "extra_point_one.png",
        title="Annealed ε(α) with random-label sweep",
        extra_label="Random labels (extra point)",
    )
    print("Saved: figs/extra_point_one.png")


# ---------- CLI ----------
def _parse_args():
    p = argparse.ArgumentParser(description="Exercise 04 runner")
    p.add_argument("-e", "--exercise", type=int, choices=[1, 2, 3, 4, 5],
                   help="Which exercise point to run (1..5). Omit to run --all.")
    p.add_argument("--csv", type=str, default=None, help="Path to dataset.csv (default: auto-detect).")
    p.add_argument("--alpha", type=float, default=5.0, help="Alpha for points 2-3.")
    p.add_argument("--eps0", type=float, default=0.25, help="Initial epsilon for points 2-3.")
    p.add_argument("--method", type=str, default="f2", choices=["f1", "f2"], help="Fixed-point method for points 4-5.")
    p.add_argument("--a", type=float, default=0.5, help="Damping a for points 2-5 where relevant.")
    p.add_argument("--alpha_max", type=float, default=10.0, help="Max α for sweep (points 4-5).")
    p.add_argument("--all", action="store_true", help="Run all points in sequence.")
    p.add_argument("--extra", type=int, choices=[1], help="Run extra point visualisations (currently only 1).")
    p.add_argument("--extra-csv", type=str, default=None,
                   help="Path to extra-point dataset (default: dataset_annealed.csv).")
    return p.parse_args()

def main():
    args = _parse_args()
    if args.extra is not None:
        if args.extra == 1:
            extraPointOne(args.csv, args.extra_csv)
        return

    if args.all or (args.exercise is None):
        exPointOne(args.csv)
        exPointTwo(alpha=args.alpha, eps0=args.eps0)
        exPointThree(alpha=args.alpha, eps0=args.eps0)
        exPointFour(alpha_max=args.alpha_max, method=args.method, a=args.a)
        exPointFive(csv_path=args.csv, alpha_max=args.alpha_max, method=args.method, a=args.a)
        return

    if args.exercise == 1:
        exPointOne(args.csv)
    elif args.exercise == 2:
        exPointTwo(alpha=args.alpha, eps0=args.eps0)
    elif args.exercise == 3:
        exPointThree(alpha=args.alpha, eps0=args.eps0)
    elif args.exercise == 4:
        exPointFour(alpha_max=args.alpha_max, method=args.method, a=args.a)
    elif args.exercise == 5:
        exPointFive(csv_path=args.csv, alpha_max=args.alpha_max, method=args.method, a=args.a)

    # No direct exercise provided but extra requested already handled above.

if __name__ == "__main__":
    main()
