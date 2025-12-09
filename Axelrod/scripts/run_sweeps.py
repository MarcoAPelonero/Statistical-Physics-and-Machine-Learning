#!/usr/bin/env python3
"""
Convenience launcher to run all sweeps with a single command and
produce the CSVs expected by the plotting script.

By default it runs both:
  - Small-world sweeps (neighbors + rewiring + features + q) -> data/smallworld
  - 2D lattice sweeps (radius + features + q)               -> data/lattice

Then you can plot with:
  python scripts/plot_axelrod_results.py --network smallworld
  python scripts/plot_axelrod_results.py --network lattice
"""
import argparse
import os
import shutil
import subprocess
import sys
from typing import List


def default_binary() -> str:
    exe = "main.exe" if os.name == "nt" else "main"
    return os.path.join("bin", exe)


def run(cmd: List[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def build_cmd(binary: str,
              network: str,
              nodes: int,
              neighbors: int,
              rewiring: float,
              radius: int,
              features: int,
              feature_dim: int,
              interactions: int,
              runs: int,
              data_root: str) -> List[str]:
    cmd = [
        binary,
        "--network", network,
        "--nodes", str(nodes),
        "--features", str(features),
        "--feature-dim", str(feature_dim),
        "--interactions", str(interactions),
        "--runs", str(runs),
        "--data-root", data_root,
    ]
    if network == "smallworld":
        cmd.extend(["--neighbors", str(neighbors),
                    "--rewiring", str(rewiring)])
    else:
        cmd.extend(["--radius", str(radius)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run Axelrod sweeps for plotting.")
    parser.add_argument("--binary", default=default_binary(),
                        help="Path to compiled main binary (default: bin/main[.exe])")
    parser.add_argument("--network", choices=["smallworld", "lattice", "all"],
                        default="all", help="Which network(s) to run.")
    parser.add_argument("--nodes", type=int, default=600,
                        help="Number of nodes (rounded to nearest square for lattice).")
    parser.add_argument("--neighbors", type=int, default=10,
                        help="Mean degree for small-world (even).")
    parser.add_argument("--rewiring", type=float, default=0.1,
                        help="Rewiring probability for small-world.")
    parser.add_argument("--radius", type=int, default=1,
                        help="Manhattan radius for 2D lattice (1 => 4 neighbors).")
    parser.add_argument("--features", type=int, default=5,
                        help="Number of cultural features F.")
    parser.add_argument("--feature-dim", type=int, default=3,
                        help="Feature dimension q.")
    parser.add_argument("--interactions", type=int, default=100000,
                        help="Interaction steps per run.")
    parser.add_argument("--runs", type=int, default=20,
                        help="Monte Carlo runs per sweep point.")
    parser.add_argument("--data-root", default="data",
                        help="Base output directory.")

    args = parser.parse_args()

    binary = args.binary
    if not os.path.isfile(binary):
        print(f"Error: binary not found at {binary}. Build first with `make compile`.", file=sys.stderr)
        sys.exit(1)

    # Ensure MinGW runtime DLLs are on PATH (Windows)
    compiler_path = shutil.which("g++")
    if compiler_path:
        mingw_bin = os.path.dirname(compiler_path)
        path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        # Move mingw bin to the front to avoid picking incompatible DLLs first.
        path_parts = [p for p in path_parts if os.path.normcase(p) != os.path.normcase(mingw_bin)]
        path_parts.insert(0, mingw_bin)
        os.environ["PATH"] = os.pathsep.join(path_parts)
        print(f"Ensured MinGW bin is first on PATH: {mingw_bin}")

    targets = ["smallworld", "lattice"] if args.network == "all" else [args.network]

    for net in targets:
        cmd = build_cmd(
            binary=binary,
            network=net,
            nodes=args.nodes,
            neighbors=args.neighbors,
            rewiring=args.rewiring,
            radius=args.radius,
            features=args.features,
            feature_dim=args.feature_dim,
            interactions=args.interactions,
            runs=args.runs,
            data_root=args.data_root,
        )
        run(cmd)


if __name__ == "__main__":
    main()
