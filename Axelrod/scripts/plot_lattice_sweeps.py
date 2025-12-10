import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import StringIO

def _is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def parse_sweep_string(text):
    """
    Parse sweep data from a string and return a pandas DataFrame.

    The function handles:
    - A header line that may start with '#' (e.g. "#sweep_param ...")
    - A header line without '#'
    - Files with no header (in which case columns are auto-numbered)
    - Whitespace-separated numeric data lines

    Returns an empty DataFrame if there's no data.
    """
    lines = text.splitlines()
    header_line = None
    data_lines = []
    defaults = None

    # Collect leading comment lines (they may contain header and a defaults line)
    comments = []
    first_nonempty_idx = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        first_nonempty_idx = i
        stripped = line.lstrip()
        if stripped.startswith('#'):
            comments.append(stripped.lstrip('#').strip())
            continue
        break

    # If we collected comment lines, treat the first as header (if present)
    if comments:
        header_line = comments[0]
        # Look for a defaults line among the comment lines (e.g. 'defaults 900 1 5 3')
        for c in comments[1:]:
            parts = c.split()
            if parts and parts[0].lower() == 'defaults':
                vals = parts[1:]
                # Map defaults to named parameters in this order:
                # num_nodes, lattice_radius, num_features, feature_dim
                keys = ['num_nodes', 'lattice_radius', 'num_features', 'feature_dim']
                defaults = {}
                for k, v in zip(keys, vals):
                    try:
                        defaults[k] = int(v)
                    except Exception:
                        try:
                            defaults[k] = float(v)
                        except Exception:
                            defaults[k] = v
                break

    # Determine the start of data lines
    if first_nonempty_idx is None:
        return pd.DataFrame()

    # If there were leading comments, data starts after the last comment
    if comments:
        # find index after the last leading comment
        data_start = first_nonempty_idx
        # advance past any consecutive comment lines
        i = data_start
        while i < len(lines) and lines[i].lstrip().startswith('#'):
            i += 1
        data_lines = lines[i:]
    else:
        # No comment header; inspect the first meaningful line
        line = lines[first_nonempty_idx]
        parts = line.strip().split()
        if any(not _is_number(tok) for tok in parts):
            header_line = line.strip()
            data_lines = lines[first_nonempty_idx+1:]
        else:
            header_line = None
            data_lines = lines[first_nonempty_idx:]

    if not data_lines:
        df = pd.DataFrame()
        if defaults:
            df.attrs = {'defaults': defaults}
        return df

    # Read the data portion with pandas (whitespace-separated)
    data_text = "\n".join([l for l in data_lines if l.strip()])
    df = pd.read_csv(StringIO(data_text), delim_whitespace=True, header=None, comment=None)

    # Apply header if provided and column counts match
    if header_line:
        cols = header_line.split()
        if len(cols) == df.shape[1]:
            df.columns = cols
        else:
            # If header tokens don't match column count, keep numeric columns
            # but attempt to assign as many as possible
            if len(cols) < df.shape[1]:
                df.columns = cols + [f"col_{i}" for i in range(len(cols), df.shape[1])]
            # otherwise leave default integer column names

    # Convert numeric columns to numeric dtypes where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Attach defaults (if any) to the DataFrame attrs for later use
    if defaults:
        df.attrs = getattr(df, 'attrs', {})
        df.attrs['defaults'] = defaults

    return df

def plot_observables(ax, df, param, title, observables, obs_colors=None, fill_alpha=0.25):
        # Determine x-axis column: prefer the named param if present, otherwise fall back to common names
        if param in df.columns:
            xcol = param
        elif 'sweep_value' in df.columns:
            xcol = 'sweep_value'
        elif 'sweep_param' in df.columns:
            xcol = 'sweep_param'
        else:
            # use the first column as a last resort
            xcol = df.columns[0] if len(df.columns) > 0 else None

        # Map logical observable names to the actual columns found in files
        obs_column_map = {
            'largest_cluster_size': 'avg_largest_fraction',
            'homophily': 'avg_homophily',
            'global_similarity': 'avg_global_similarity'
        }
        obs_std_map = {
            'largest_cluster_size': 'sd_largest_fraction',
            'homophily': 'sd_homophily',
            'global_similarity': 'sd_global_similarity'
        }

        for obs in observables:
            col = obs if obs in df.columns else obs_column_map.get(obs)
            yerr_col = obs_std_map.get(obs)
            y = df.get(col)
            yerr = df.get(yerr_col)
            if xcol is not None and y is not None:
                # choose color for this observable
                color = None
                if obs_colors and obs in obs_colors:
                    color = obs_colors[obs]
                else:
                    # fallback to matplotlib cycle
                    color = next(ax._get_lines.prop_cycler)['color']

                # plot main line with markers
                ax.plot(df[xcol], y, marker='o', linestyle='-', color=color, label=obs)

                # draw filled error region if yerr is present
                if yerr is not None:
                    # ensure numpy arrays for arithmetic
                    import numpy as _np
                    xvals = _np.array(df[xcol])
                    yvals = _np.array(y)
                    yerrvals = _np.array(yerr)
                    lower = yvals - yerrvals
                    upper = yvals + yerrvals
                    ax.fill_between(xvals, lower, upper, color=color, alpha=fill_alpha)

        # Set labels and title (use readable versions)
        xlabel = (param.replace('_', ' ').title()) if param else (xcol or '')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True)

def read_sweep_dataframe(filepath):
    """
    Read sweep data from `filepath` and return a pandas DataFrame.
    Delegates to `parse_sweep_string` after loading the file contents.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        raise
    return parse_sweep_string(text)

if __name__ == "__main__":
    data_dir = "data_sweeps_backup"
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    # Example: read one of the sweep files into a DataFrame
    lattice_file = os.path.join(data_dir, "sweep_lattice_radius.txt")
    try:
        lattice_df = read_sweep_dataframe(lattice_file)
        print("Loaded lattice sweep DataFrame:")
        print(lattice_df.head())
    except Exception as e:
        print(f"Could not read {lattice_file}: {e}")

    feature_dim_data = read_sweep_dataframe(os.path.join(data_dir, "sweep_feature_dim.txt"))
    num_nodes_data = read_sweep_dataframe(os.path.join(data_dir, "sweep_num_nodes.txt"))
    lattice_radius_data = read_sweep_dataframe(os.path.join(data_dir, "sweep_lattice_radius.txt"))
    num_features_data = read_sweep_dataframe(os.path.join(data_dir, "sweep_num_features.txt"))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Example plotting code (customize as needed)
    # Plot 3 errorbars for each subplot and for each of the observables: largest_cluster_size, homophily, global_similarity
    # Each subplot has 3 curves (errorbars) for each observable. The graphs are for the invidial sweeps
    observables = ['largest_cluster_size', 'homophily', 'global_similarity']  
    sweep_data = [feature_dim_data, num_nodes_data, lattice_radius_data, num_features_data]
    sweep_params = ['feature_dim', 'num_nodes', 'lattice_radius', 'num_features']

    titles = ['Sweep: Feature Dimension', 'Sweep: Number of Nodes', 'Sweep: Lattice Radius', 'Sweep: Number of Features']
    # Prepare colors for observables from matplotlib cycle so legend matches plots
    default_colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)
    if default_colors is None:
        default_colors = ['tab:blue', 'tab:orange', 'tab:green']
    obs_colors = {obs: default_colors[i % len(default_colors)] for i, obs in enumerate(observables)}

    for ax, df, param, title in zip(axs.flatten(), sweep_data, sweep_params, titles):
        plot_observables(ax, df, param, title, observables, obs_colors=obs_colors, fill_alpha=0.25)

    # Look for defaults attached to any of the loaded sweep DataFrames
    defaults_found = None
    for df in sweep_data:
        if hasattr(df, 'attrs') and df.attrs.get('defaults'):
            defaults_found = df.attrs.get('defaults')
            break

    if defaults_found:
        defaults_text = ', '.join(f"{k}={v}" for k, v in defaults_found.items())
        suptitle_text = f"Sweep Comparisons — Defaults: {defaults_text}"
    else:
        suptitle_text = 'Sweep Comparisons'

    fig.suptitle(suptitle_text, fontsize=18, fontweight='bold')

    # Create a single legend outside the subplots using proxy artists so it's stable
    # Create proxy artists for legend using matching colors (line + marker)
    import matplotlib.lines as mlines
    proxy_lines = [mlines.Line2D([0], [0], color=obs_colors[obs], marker='o', linestyle='-', markersize=6, label=obs) for obs in observables]
    fig.legend(handles=proxy_lines, labels=observables, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

    # Reserve room on the right for the legend, then save the figure
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out_path = os.path.join(out_dir, 'sweep_comparison.png')
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved figure to {out_path}")
