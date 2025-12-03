import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Usage:
    python scripts/animate_voter_diffusion.py lattice_voter_diffusion.txt

This script creates an animation of the voter model diffusion, saves:
- The animation as 'voter_diffusion_animation.gif'
- The first frame as 'voter_diffusion_first_frame.png'
- The last frame as 'voter_diffusion_last_frame.png'

The input file should contain blocks separated by lines starting with '# Step'.
Each block contains lines of the form:
    u v weight state_u
"""


def parse_steps(path):
    """Parse the lattice file and extract all time steps."""
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
                    nid = int(parts[1])
                    st = int(parts[2])
                    current_nodes[nid] = st
                    max_node = max(max_node, nid)
                except Exception:
                    pass
                continue
            # Edge line: u v weight state_u
            try:
                u = int(parts[0])
                v = int(parts[1])
                if len(parts) >= 4:
                    st = int(parts[3])
                    current_nodes[u] = st
                    max_node = max(max_node, u, v)
                else:
                    max_node = max(max_node, u, v)
            except Exception:
                pass
    # Append last block if any
    if current_nodes:
        steps.append(current_nodes)
    return steps, max_node


def infer_grid_size(max_node):
    """Infer the grid size from the maximum node ID."""
    n = max_node + 1
    L = int(round(np.sqrt(n)))
    if L * L != n:
        print(f"[warn] Node count {n} is not a perfect square; using L={L}")
    return L


def nodes_to_grid(nodes_state, L):
    """Convert node dictionary to 2D grid."""
    grid = np.zeros((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            nid = i * L + j
            grid[i, j] = nodes_state.get(nid, 0)
    return grid


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/animate_voter_diffusion.py <lattice_file>")
        sys.exit(1)

    lattice_file = sys.argv[1]
    
    print("Parsing lattice file...")
    steps, max_node = parse_steps(lattice_file)
    if not steps:
        print("No steps found in input.")
        sys.exit(1)

    print(f"Found {len(steps)} time steps")
    L = infer_grid_size(max_node)
    print(f"Grid size: {L}x{L}")

    # Convert all steps to grids
    print("Converting to grids...")
    grids = [nodes_to_grid(step, L) for step in steps]
    
    # Create custom purple-pink colormap: -1 to purple, +1 to pink
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#9B30FF', '#FFB6C1']  # purple to pink
    cmap = LinearSegmentedColormap.from_list('purple_pink', colors)
    
    # Save first frame
    print("Saving first frame...")
    fig_first, ax_first = plt.subplots(figsize=(6, 6))
    im_first = ax_first.imshow(grids[0], cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax_first.set_title(f'Voter Model - Step 0')
    ax_first.set_xticks([])
    ax_first.set_yticks([])
    fig_first.tight_layout()
    fig_first.savefig('voter_diffusion_first_frame.png', dpi=200)
    plt.close(fig_first)
    print("Saved voter_diffusion_first_frame.png")
    
    # Save last frame
    print("Saving last frame...")
    fig_last, ax_last = plt.subplots(figsize=(6, 6))
    im_last = ax_last.imshow(grids[-1], cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax_last.set_title(f'Voter Model - Step {len(steps)-1}')
    ax_last.set_xticks([])
    ax_last.set_yticks([])
    fig_last.tight_layout()
    fig_last.savefig('voter_diffusion_last_frame.png', dpi=200)
    plt.close(fig_last)
    print("Saved voter_diffusion_last_frame.png")
    
    # Create animation
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grids[0], cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    title = ax.set_title('Voter Model - Step 0')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    
    # Sample frames for animation (every 10th frame to keep file size reasonable)
    frame_skip = max(1, len(grids) // 1000)  # Aim for ~1000 frames max
    animation_frames = list(range(0, len(grids), frame_skip))
    print(f"Animation will have {len(animation_frames)} frames (sampling every {frame_skip} steps)")
    
    def animate(frame_idx):
        step_idx = animation_frames[frame_idx]
        im.set_data(grids[step_idx])
        title.set_text(f'Voter Model - Step {step_idx}')
        return [im, title]
    
    print("Rendering animation (this may take a while)...")
    anim = animation.FuncAnimation(fig, animate, frames=len(animation_frames),
                                   interval=200, blit=True, repeat=True)
    
    # Save animation
    writer = animation.PillowWriter(fps=20)
    anim.save('voter_diffusion_animation.gif', writer=writer)
    plt.close(fig)
    print("Saved voter_diffusion_animation.gif")
    
    print("\nAnimation complete!")
    print("Generated files:")
    print("  - voter_diffusion_animation.gif")
    print("  - voter_diffusion_first_frame.png")
    print("  - voter_diffusion_last_frame.png")


if __name__ == '__main__':
    main()
