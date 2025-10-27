# Exercise 2
The theme of the exercise is to build and implement a Single Layer Perceptron on an extremely simple task, select highier or bigger numbers from a dataset of randomly generated couples.

## EX2

This folder contains the implementation for the perceptron tasks. For Exercise 7, the plotting script `plotter.py` has been updated to visualize:

- Errors vs. epochs (log scale) for training with different P (e.g., P=500 and P=2000).
- Final index profile required by the exercise: for each weight index i, it plots
	sign(J*_i J_i) · log2(|J_i|), comparing the trained perceptron(s) against the perfect perceptron J*.
- A heatmap showing the evolution of the transformed weights during training.
- An optional animation (GIF) of the index profile evolving over epochs.

## How to run

Using the provided training logs (`training_P500.log` and `training_P2000.log`):

```
python plotter.py --save --animate
```

Options:

- `--log500 PATH`     Path to the log for P=500 (default: `training_P500.log`).
- `--log2000 PATH`    Path to the log for P=2000 (default: `training_P2000.log`).
- `--save`            Save figures into `bin/` instead of just showing them.
- `--animate`         Save a GIF animation of the index profile evolution into `bin/`.

Outputs are saved under `bin/`:

- `errors_vs_epochs.png`
- `final_index_profile.png`
- `evolution_heatmaps.png`
- `evolution_P500.gif` (if `--animate`)
- `evolution_P2000.gif` (if `--animate`)
 - `errors_vs_epochs_P500.png` and `errors_vs_epochs_P2000.png`
 - `final_index_profile_P500.png` and `final_index_profile_P2000.png`

The perfect perceptron weights are defined as:

- J*_i = 2^(10 - i) for 1 ≤ i ≤ 10
- J*_i = -J*_{i-10} for 11 ≤ i ≤ 20

The index profile transforms weights as: sign(J*_i J_i) · log2(|J_i|).