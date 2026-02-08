#!/usr/bin/env python3
"""
avalancheAnimator.py

Create a 3D animation (surface) from an avalanche frames file produced by the Sandbox.

Examples (PowerShell):
  python .\avalancheAnimator.py --frames avalanche_frames.txt --skip 20 --interval 20 --save avalanche.mp4
  python .\avalancheAnimator.py --run-cmd "build\\testout.exe 64 5000" --frames avalanche_frames.txt --save out.mp4

Notes:
- If --save is given, we write only an MP4 (no GUI window).
- Optional smoothing via --smooth-sigma; optional upsampling via --upsample.
- NEW: Optional --interp to generate intermediate frames for flowy animation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
import shlex
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from scipy.ndimage import gaussian_filter, zoom
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------- IO -----------------------------

def parse_frames_from_file(path: Path) -> List[np.ndarray]:
    text = path.read_text()
    parts = [p.strip() for p in text.split('---')]
    frames: List[np.ndarray] = []
    for p in parts:
        if not p:
            continue
        rows = [line.strip() for line in p.splitlines() if line.strip()]
        if not rows:
            continue
        grid = [list(map(int, row.split())) for row in rows]
        frames.append(np.asarray(grid, dtype=float))
    return frames


def run_generator(cmd: str) -> int:
    print(f"Running generator command: {cmd}")
    try:
        res = subprocess.run(cmd, shell=True)
        return res.returncode
    except Exception as e:
        print(f"Failed to run generator: {e}")
        return 1


# -------------------------- Smoothing -------------------------

def _fallback_gaussian_like_blur(Z: np.ndarray, passes: int = 1) -> np.ndarray:
    k = np.array([1.0, 2.0, 1.0], dtype=Z.dtype)
    k /= k.sum()

    def blur1d_along_axis(A: np.ndarray, axis: int) -> np.ndarray:
        A_pad = np.pad(A, pad_width=[(1, 1) if ax == axis else (0, 0) for ax in range(A.ndim)],
                       mode='edge')
        out = (k[0] * np.take(A_pad, indices=range(0, A_pad.shape[axis] - 2), axis=axis) +
               k[1] * np.take(A_pad, indices=range(1, A_pad.shape[axis] - 1), axis=axis) +
               k[2] * np.take(A_pad, indices=range(2, A_pad.shape[axis]), axis=axis))
        return out

    out = Z
    for _ in range(max(1, passes)):
        out = blur1d_along_axis(out, axis=1)
        out = blur1d_along_axis(out, axis=0)
    return out


def smooth_and_resample(Z: np.ndarray,
                        sigma: float = 0.0,
                        upsample: int = 1,
                        fallback_passes: int = 1) -> np.ndarray:
    if upsample < 1:
        upsample = 1

    if _HAVE_SCIPY:
        ZZ = Z
        if upsample > 1:
            ZZ = zoom(ZZ, zoom=upsample, order=1)
        if sigma > 0.0:
            ZZ = gaussian_filter(ZZ, sigma=sigma * max(1, upsample))
        return ZZ
    else:
        ZZ = np.kron(Z, np.ones((upsample, upsample), dtype=Z.dtype)) if upsample > 1 else Z.copy()
        if sigma > 0.0:
            passes = max(fallback_passes, int(round(sigma)))
            ZZ = _fallback_gaussian_like_blur(ZZ, passes=passes)
        return ZZ


# --------------------- Temporal Interpolation ---------------------

def interpolate_frames(frames: List[np.ndarray], interp_factor: int = 1) -> List[np.ndarray]:
    """Linearly interpolate between each pair of frames for smoother motion."""
    if interp_factor <= 1:
        return frames
    out = []
    for i in range(len(frames) - 1):
        A, B = frames[i], frames[i + 1]
        for j in range(interp_factor):
            t = j / interp_factor
            out.append((1 - t) * A + t * B)
    out.append(frames[-1])
    return out


# ------------------------- Animation --------------------------

def make_animation(frames: List[np.ndarray],
                   skip: int = 1,
                   interval: int = 100,
                   cmap: str = 'viridis',
                   save: Optional[Path] = None,
                   elev: float = 45,
                   azim: float = -60,
                   smooth_sigma: float = 0.0,
                   upsample: int = 1,
                   dpi: int = 100,
                   z_pad: float = 1.0,
                   antialiased: bool = True,
                   ffmpeg_codec: Optional[str] = None,
                   ffmpeg_extra_args: Optional[List[str]] = None) -> None:
    if len(frames) == 0:
        raise ValueError("No frames to animate")

    vis_frames_raw = frames[::max(1, skip)]

    vis_frames: List[np.ndarray] = []
    for Z in vis_frames_raw:
        vis_frames.append(smooth_and_resample(Z, sigma=smooth_sigma, upsample=upsample))

    rows, cols = vis_frames[0].shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    allmin = float(min(np.min(f) for f in vis_frames))
    allmax = float(max(np.max(f) for f in vis_frames))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.set_zlim(allmin, allmax + z_pad)
    ax.view_init(elev=elev, azim=azim)

    surf = ax.plot_surface(X, Y, vis_frames[0], cmap=cmap,
                           vmin=allmin, vmax=allmax, linewidth=0,
                           antialiased=antialiased)

    prev_Z = None
    alpha = 0.6  # temporal blending for perceptual smoothness

    def update(i):
        nonlocal prev_Z
        ax.cla()
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_zlim(allmin, allmax + z_pad)
        ax.view_init(elev=elev, azim=azim)
        Z = vis_frames[i]
        if prev_Z is not None:
            Z = alpha * prev_Z + (1 - alpha) * Z
        prev_Z = Z
        ax.plot_surface(X, Y, Z, cmap=cmap, vmin=allmin, vmax=allmax,
                        linewidth=0, antialiased=antialiased)
        return ax,

    ani = FuncAnimation(fig, update, frames=len(vis_frames), interval=interval, blit=False)

    if save:
        fps = max(1, 1000 // max(1, interval))
        try:
            writer = FFMpegWriter(fps=fps, codec=ffmpeg_codec, extra_args=ffmpeg_extra_args)
        except TypeError:
            writer = FFMpegWriter(fps=fps)
        try:
            bar = tqdm(total=len(vis_frames), desc="Saving frames", unit="f") if tqdm else None
            with writer.saving(fig, str(save), dpi=dpi):
                for i in range(len(vis_frames)):
                    update(i)
                    writer.grab_frame()
                    if bar:
                        bar.update(1)
            if bar:
                bar.close()
            print(f"Saved MP4: {save}")
        except Exception as e:
            print("Failed to save animation:", e)
        finally:
            plt.close(fig)
    else:
        plt.show()


# ---------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser(description='Animate avalanche frames as a 3D surface.')
    p.add_argument('--frames', '-f', type=str, default='avalanche_frames.txt')
    p.add_argument('--skip', '-k', type=int, default=1)
    p.add_argument('--interval', '-i', type=int, default=100)
    p.add_argument('--run-cmd', '-r', type=str, default=None)
    p.add_argument('--save', '-s', type=str, default=None)
    p.add_argument('--cmap', type=str, default='viridis')
    p.add_argument('--elev', type=float, default=45.0)
    p.add_argument('--azim', type=float, default=-60.0)
    p.add_argument('--smooth-sigma', type=float, default=0.0)
    p.add_argument('--upsample', type=int, default=1)
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--zpad', type=float, default=1.0)
    p.add_argument('--ffmpeg-codec', type=str, default=None)
    p.add_argument('--ffmpeg-extra-args', type=str, default=None)
    p.add_argument('--interp', '-t', type=int, default=1,
                   help='Temporal interpolation factor for flowy animation (e.g. 4)')

    args = p.parse_args()
    frames_path = Path(args.frames)

    if args.run_cmd:
        rc = run_generator(args.run_cmd)
        if rc != 0:
            print(f"Generator returned non-zero exit code {rc}; aborting.")
            sys.exit(rc)

    if not frames_path.exists():
        print(f"Frames file not found: {frames_path}")
        sys.exit(1)

    frames = parse_frames_from_file(frames_path)
    if len(frames) == 0:
        print("No frames parsed from file. Exiting.")
        sys.exit(1)

    # Apply temporal interpolation
    if args.interp > 1:
        print(f"Interpolating frames with factor {args.interp}...")
        frames = interpolate_frames(frames, interp_factor=args.interp)

    savepath = Path(args.save) if args.save else None
    ffmpeg_extra_args = shlex.split(args.ffmpeg_extra_args) if args.ffmpeg_extra_args else None

    make_animation(
        frames=frames,
        skip=args.skip,
        interval=args.interval,
        cmap=args.cmap,
        save=savepath,
        elev=args.elev,
        azim=args.azim,
        smooth_sigma=args.smooth_sigma,
        upsample=args.upsample,
        dpi=args.dpi,
        z_pad=args.zpad,
        ffmpeg_codec=args.ffmpeg_codec,
        ffmpeg_extra_args=ffmpeg_extra_args,
    )


if __name__ == '__main__':
    try:
        matplotlib.animation.writers['ffmpeg']
    except Exception:
        print("Warning: ffmpeg not found by Matplotlib. Install FFmpeg and ensure it's on PATH.")
    main()
