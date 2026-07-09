# plotting/plot_regret.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Modified to accept a dynamic file prefix ('regret' or 'reward')
def load_data(result_dir, filename_prefix):
    result_dir = Path(result_dir)
    data_path = result_dir / f"{filename_prefix}s.npy"

    if data_path.exists():
        return np.load(data_path)

    mean_path = result_dir / f"mean_{filename_prefix}.npy"
    if mean_path.exists():
        return np.load(mean_path)[None, :]

    raise FileNotFoundError(f"No {filename_prefix}s.npy found in {result_dir}")


# Modified to search dynamically for the targeted metric file
def discover_agents(result_dir, filename_prefix):
    """
    Scans recursively the input folder and its sub folders
    to find the data files
    """
    result_dir = Path(result_dir)
    target_file = f"{filename_prefix}s.npy"
    
    dirs_with_data = [p.parent for p in result_dir.rglob(target_file)]
    
    if dirs_with_data:
        return sorted(list(set(dirs_with_data))) # set() avoids potential duplicates
        
    raise FileNotFoundError(f"No {target_file} files found in {result_dir}")


def plot_series(ax, regrets, label, color=None, linestyle=None, alpha=0.2):
    regrets = np.asarray(regrets)
    if regrets.ndim == 1:
        regrets = regrets[None, :]

    mean = regrets.mean(axis=0)
    std = regrets.std(axis=0, ddof=1 if len(regrets) > 1 else 0)
    ci95 = 1.96 * std / np.sqrt(len(regrets))
    x = np.arange(len(mean))

    line, = ax.plot(
        x, 
        mean, 
        label=label, 
        linewidth=2,
        color=color,
        linestyle=linestyle,
    )
    ax.fill_between(
        x,
        mean - ci95,
        mean + ci95,
        color=line.get_color(),
        alpha=alpha,
        linewidth=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative regret or time-averaged reward from saved experiment results."
    )
    
    # Added required mutually exclusive group for metric target choice
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--reward",
        action="store_true",
        help="Plot time-averaged reward",
    )
    group.add_argument(
        "--regret",
        action="store_true",
        help="Plot cumulative regret",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Result directory(ies). Multi-agent dirs contain one subfolder per agent.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (one per --input, or one per agent subfolder).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output figure (png/pdf).",
    )
    parser.add_argument(
        "--title",
        default=None, # Changed default to None to adapt dynamically below
        help="Plot title",
    )
    parser.add_argument(
        "--min", default=None, type=float
    )
    parser.add_argument(
        "--max", default=None, type=float
    )
    parser.add_argument(
        "--marker",
        action="store_true",
        help="If true, groups same sub-folders by color and agents by distinct linestyles.",
    )
    args = parser.parse_args()

    # Configure plot labels and dynamic prefixes based on the argument flag
    if args.regret:
        filename_prefix = "regret"
        y_label = "Cumulative Regret"
        default_title = "Cumulative Regret"
    else:
        filename_prefix = "reward"
        y_label = "Time-averaged Reward"
        default_title = "Time-averaged Reward"

    plot_title = args.title if args.title is not None else default_title

    min, max = args.min, args.max
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = args.labels or []
    label_idx = 0
    plotted = 0

    colors_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    # Agent 1: Solid line ("-")
    # Agent 2: Long dashed line ((0, (10, 5))) instead of "--"
    # Agent 3: Dotted line with tight spacing ((0, (1, 3))) 
    # Agent 4: Short dashed line with equal spacing ((0, (4, 4)))
    linestyles_palette = ["-", (0, (10, 5)), (0, (1, 3)), (0, (4, 4))]
    subfolder_colors = {}
    agent_linestyles = {}

    for input_path in args.input:
        input_path = Path(input_path)
        # Pass the dynamic file prefix down to find matching logs
        agent_dirs = discover_agents(input_path, filename_prefix)

        for agent_dir in agent_dirs:
            color = None
            linestyle = None
            agent_name = agent_dir.name.lower()
            parent_name = agent_dir.parent.name

            if args.marker:
                if parent_name not in subfolder_colors:
                    subfolder_colors[parent_name] = colors_palette[len(subfolder_colors) % len(colors_palette)]
                color = subfolder_colors[parent_name]

                if agent_name not in agent_linestyles:
                    agent_linestyles[agent_name] = linestyles_palette[len(agent_linestyles) % len(linestyles_palette)]
                linestyle = agent_linestyles[agent_name]

                if label_idx < len(labels):
                    label = labels[label_idx]
                else:
                    label = f"{parent_name}/{agent_dir.name}"
            else:
                if label_idx < len(labels):
                    label = labels[label_idx]
                elif len(agent_dirs) > 1:
                    label = f"{agent_dir.parent.name}/{agent_dir.name}"
                else:
                    label = input_path.name

            # Load the requested metric type dataset
            regrets = load_data(agent_dir, filename_prefix)
            plot_series(ax, regrets, label, color=color, linestyle=linestyle)
            label_idx += 1
            plotted += 1

    if plotted == 0:
        raise SystemExit(f"No data plotted. Check --input paths for {filename_prefix}s.npy.")

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label) 
    ax.set_title(plot_title)
    ax.set_ylim(min, max)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
