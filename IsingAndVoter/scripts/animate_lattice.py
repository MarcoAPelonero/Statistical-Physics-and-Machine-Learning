import sys
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Usage:
    python scripts/animate_lattice.py lattice_ising.txt --output final_frame.png
    python scripts/animate_lattice.py lattice_voter.txt

The input file should contain blocks separated by lines starting with '# Step'.
Each block contains lines of the form:
    u v weight state_u
We only use the `state_u` to reconstruct the lattice state per time step.

This script plots/saves only the LAST frame from the file.

If explicit lines like 'NODE <id> <state>' are present, those will be used as fallback.
"""


def parse_steps(path):
    steps = []
    current_nodes = {}
    max_node = -1
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# Step'):
                if current_nodes:
                    steps.append(current_nodes)
                current_nodes = {}
                continue
            parts = line.split()
            if parts[0].upper() == 'NODE' and len(parts) >= 3:
                try:
                    nid = int(parts[1]); st = int(parts[2])
                    current_nodes[nid] = st
                    max_node = max(max_node, nid)
                except Exception:
                    pass
                continue
            # Edge line: u v weight [state_u]
            try:
                u = int(parts[0]); v = int(parts[1])
                if len(parts) >= 4:
                    st = int(parts[3])
                    current_nodes[u] = st
                    max_node = max(max_node, u, v)
                else:
                    max_node = max(max_node, u, v)
            except Exception:
                # ignore malformed lines
                pass
    # Append last block if any
    if current_nodes:
        steps.append(current_nodes)
    return steps, max_node


def infer_grid_size(max_node):
    # graph constructed from LxL lattice; nodes 0..L*L-1
    n = max_node + 1
    L = int(round(np.sqrt(n)))
    if L * L != n:
        # Not a perfect square; still use rounded sqrt for layout
        # but inform user via stderr
        print(f"[warn] Node count {n} is not a perfect square; using L={L}")
    return L


def nodes_to_grid(nodes_state, L):
    # nodes are indexed row-major: node = i*L + j
    grid = np.zeros((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            nid = i * L + j
            grid[i, j] = nodes_state.get(nid, 0)
    return grid


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/animate_lattice.py <lattice_file> [--output final.png]")
        sys.exit(1)

    lattice_file = sys.argv[1]
    out_png = None
    for k in range(2, len(sys.argv)):
        if sys.argv[k] == '--output' and k + 1 < len(sys.argv):
            out_png = sys.argv[k + 1]

    steps, max_node = parse_steps(lattice_file)
    if not steps:
        print("No steps found in input.")
        sys.exit(1)

    L = infer_grid_size(max_node)

    # Only process the LAST frame
    final_grid = nodes_to_grid(steps[-1], L)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Create custom purple-pink colormap: -1 to purple, +1 to pink
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#9B30FF', '#FFB6C1']  # purple to pink
    cmap = LinearSegmentedColormap.from_list('purple_pink', colors)
    
    im = ax.imshow(final_grid, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Lattice Final State - Step {len(steps)-1}')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    # Save if output file specified
    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"Saved final frame to {out_png}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
