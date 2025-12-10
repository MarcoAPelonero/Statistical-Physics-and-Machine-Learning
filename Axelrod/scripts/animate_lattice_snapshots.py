import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


def _features_to_colors_gradient(features: np.ndarray, max_value: int) -> np.ndarray:
    """
    Map each feature vector to an RGB color using a gradient-based scheme.
    
    Uses the feature values to create smooth color transitions:
    - Feature 0 controls Hue (color wheel position)
    - Feature 1 controls Saturation  
    - Feature 2 controls Value/Brightness
    - Additional features blend into hue offset
    
    This ensures that similar cultures have similar colors.
    """
    import colorsys
    
    num_nodes = features.shape[0]
    num_features = features.shape[1]
    colors = np.zeros((num_nodes, 3), dtype=np.float64)
    
    # Normalize features to [0, 1]
    norm_features = features.astype(np.float64) / max(max_value, 1)
    
    for i in range(num_nodes):
        f = norm_features[i]
        
        if num_features >= 3:
            # Use first 3 features for HSV
            h = f[0]  # Hue from feature 0
            # Add contribution from other features to create more distinct colors
            if num_features > 3:
                h = (h + 0.1 * np.sum(f[3:])) % 1.0
            s = 0.4 + 0.6 * f[1]  # Saturation: 0.4-1.0 range
            v = 0.5 + 0.5 * f[2]  # Value: 0.5-1.0 range
        elif num_features == 2:
            h = f[0]
            s = 0.4 + 0.6 * f[1]
            v = 0.85
        else:  # num_features == 1
            h = f[0]
            s = 0.7
            v = 0.85
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[i] = [r, g, b]
    
    return colors


def load_snapshots(path: str) -> Dict:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    if df.empty:
        raise ValueError(f"No snapshot data in {path}")

    num_features = df.shape[1] - 4
    columns = ["step", "node", "x", "y"] + [f"f{i}" for i in range(num_features)]
    df.columns = columns

    steps = sorted(df["step"].unique())
    first_step = steps[0]
    base = df[df["step"] == first_step].sort_values("node")

    positions = base[["x", "y"]].to_numpy()
    feature_cols = [f"f{i}" for i in range(num_features)]
    max_value = int(df[feature_cols].to_numpy().max())

    color_frames: List[np.ndarray] = []
    feature_history: List[np.ndarray] = []
    for step in steps:
        sdf = df[df["step"] == step].sort_values("node")
        feats = sdf[feature_cols].to_numpy(dtype=np.int64)
        feature_history.append(feats)

        colors = _features_to_colors_gradient(feats, max_value)
        side = int(round(np.sqrt(len(colors))))
        grid = colors.reshape(side, side, 3)
        color_frames.append(grid)

    side = color_frames[0].shape[0]
    return {
        "path": path,
        "steps": steps,
        "positions": positions,
        "features": feature_history,
        "color_frames": color_frames,
        "num_features": num_features,
        "side": side,
    }


def load_fragmentation(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    frag_df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["step", "fragmentation"],
        engine="python",
    )
    return frag_df


def discover_runs(input_dir: str) -> List[Dict]:
    runs: List[Dict] = []
    pattern = os.path.join(input_dir, "lattice_q*_snapshots.txt")
    for snap_path in glob.glob(pattern):
        match = re.search(r"q(\d+)", os.path.basename(snap_path))
        q_val = int(match.group(1)) if match else None
        frag_path = os.path.join(input_dir, f"lattice_q{q_val}_fragmentation.txt")

        run = load_snapshots(snap_path)
        run["q"] = q_val
        try:
            frag_df = load_fragmentation(frag_path)
        except FileNotFoundError:
            frag_df = pd.DataFrame(columns=["step", "fragmentation"])
        run["fragmentation"] = frag_df
        runs.append(run)
    
    # Sort by q value numerically
    runs.sort(key=lambda r: r["q"] if r["q"] is not None else 0)
    return runs


def build_animation(runs: List[Dict], interval_ms: int):
    if not runs:
        raise ValueError("No runs to animate.")

    ncols = len(runs)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 7))
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    images = []
    frag_lines = []
    frag_markers = []
    max_frames = max(len(run["steps"]) for run in runs)

    for idx, run in enumerate(runs):
        ax_top = axes[0, idx]
        img = ax_top.imshow(run["color_frames"][0], origin="upper", interpolation="nearest")
        ax_top.set_title(f"q={run['q']} step={run['steps'][0]}")
        ax_top.set_xlim(-0.5, run["side"] - 0.5)
        ax_top.set_ylim(-0.5, run["side"] - 0.5)
        ax_top.set_aspect("equal")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        images.append(img)

        ax_bot = axes[1, idx]
        frag_df = run["fragmentation"]
        line, = ax_bot.plot([], [], color="tab:blue", lw=1.5)
        marker, = ax_bot.plot([], [], "o", color="tab:orange", markersize=4)
        max_step = frag_df["step"].max() if not frag_df.empty else run["steps"][-1]
        ax_bot.set_xlim(0, max_step if max_step > 0 else 1)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.set_xlabel("Step")
        ax_bot.set_ylabel("Fragmentation")
        frag_lines.append(line)
        frag_markers.append(marker)

    def update(frame: int):
        artists = []
        for idx, run in enumerate(runs):
            step_idx = min(frame, len(run["steps"]) - 1)
            images[idx].set_data(run["color_frames"][step_idx])
            axes[0, idx].set_title(f"q={run['q']} step={run['steps'][step_idx]}")

            frag_df = run["fragmentation"]
            if not frag_df.empty:
                mask = frag_df["step"] <= run["steps"][step_idx]
                frag_lines[idx].set_data(frag_df["step"][mask], frag_df["fragmentation"][mask])
                frag_markers[idx].set_data(frag_df["step"][mask], frag_df["fragmentation"][mask])
            artists.extend([images[idx], frag_lines[idx], frag_markers[idx]])
        return artists
    return {
        "fig": fig,
        "update": update,
        "max_frames": max_frames,
        "frag_lines": frag_lines,
        "frag_markers": frag_markers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Animate lattice snapshots and fragmentation curves for multiple q values."
    )
    parser.add_argument(
        "--input-dir",
        default="data/lattice_snapshots",
        help="Directory containing lattice_q*_snapshots.txt and fragmentation files.",
    )
    parser.add_argument(
        "--output",
        default="figures/lattice_snapshots.mp4",
        help="Path to save the animation (.mp4 or .gif). Default: figures/lattice_snapshots.mp4",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=80,
        help="Animation interval in milliseconds (default: 80, lower = faster).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the animation window.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.input_dir)
    built = build_animation(runs, args.interval)
    fig = built["fig"]
    update = built["update"]
    max_frames = built["max_frames"]

    # Build the animation object (uses the existing update function)
    anim = FuncAnimation(fig, update, frames=max_frames, interval=args.interval, blit=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = out_path.suffix.lower()

        # Save first and last frames as PNGs
        base = out_path.with_suffix("")
        first_png = base.with_name(base.name + "_first.png")
        last_png = base.with_name(base.name + "_last.png")
        update(0)
        fig.savefig(first_png, dpi=200)
        update(max_frames - 1)
        fig.savefig(last_png, dpi=200)

        # Save animation
        if fmt == ".gif":
            anim.save(out_path, writer="pillow", dpi=150)
        else:
            anim.save(out_path, writer="ffmpeg", dpi=150)
        print(f"Saved animation to {out_path}")
        print(f"Saved first/last frames to {first_png} and {last_png}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
