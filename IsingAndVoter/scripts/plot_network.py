import networkx as nx
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python plot_graph.py edges.txt")
    exit(1)

edgefile = sys.argv[1]

# Build graph and read optional node state column.
G = nx.Graph()
node_state = {}
with open(edgefile, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # New explicit node state line: "NODE <id> <state>"
        if parts[0].upper() == 'NODE' and len(parts) >= 3:
            try:
                nid = int(parts[1])
                nst = int(parts[2])
                node_state[nid] = nst
            except Exception:
                pass
            continue

        if line.startswith('#'):
            continue

        # Expect lines like: u v weight [node_state]
        try:
            u = int(parts[0])
            v = int(parts[1])
        except Exception:
            continue
        weight = 1.0
        if len(parts) >= 3:
            try:
                weight = float(parts[2])
            except Exception:
                pass
        G.add_edge(u, v, weight=weight)
        # Backward compatibility: if a node state is present (4th column), record it for the source node
        if len(parts) >= 4:
            try:
                node_state[u] = int(parts[3])
            except Exception:
                pass

# Draw (spring layout by default)
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)

# Determine node colors from `node_state` if available. Use green for +1, red for -1.
colors = []
for n in G.nodes():
    st = node_state.get(n, None)
    if st == 1:
        colors.append('green')
    elif st == -1:
        colors.append('red')
    else:
        colors.append('skyblue')

nx.draw(
    G, pos,
    node_color=colors,
    node_size=300,
    edge_color="gray",
    linewidths=1,
    width=0.8,
    with_labels=True,
    font_size=8
)

# Optional legend
import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color='green', label='+1'), mpatches.Patch(color='red', label='-1'), mpatches.Patch(color='skyblue', label='unknown')]
plt.legend(handles=legend_handles, loc='upper right')

plt.title("Graph Visualization")
plt.show()