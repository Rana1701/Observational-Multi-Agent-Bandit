# plotting/plot_regret.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_regrets(result_dir):
    result_dir = Path(result_dir)
    regrets_path = result_dir / "regrets.npy"

    if regrets_path.exists():
        return np.load(regrets_path)

    mean_path = result_dir / "mean_regret.npy"
    if mean_path.exists():
        return np.load(mean_path)[None, :]

    raise FileNotFoundError(f"No regrets.npy found in {result_dir}")


def discover_agents(result_dir):
    result_dir = Path(result_dir)
    subdirs = [
        p for p in result_dir.iterdir()
        if p.is_dir() and (p / "regrets.npy").exists()
    ]
    if subdirs:
        return sorted(subdirs)

    if (result_dir / "regrets.npy").exists():
        return [result_dir]

    raise FileNotFoundError(f"No result files found in {result_dir}")


def plot_series(ax, regrets, label, color=None, marker=None, alpha=0.2):
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
        marker=marker,
        markevery=max(len(x) // 20, 1),
        markersize=5,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Cumulative Regret")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 8))

    labels = args.labels or []
    label_idx = 0
    plotted = 0

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    agent_markers = {
        "ucb": "o",
        "greedy": "s",
        "tucb": "^",
        "alphaoptimal": "D",
    }

    for input_idx, input_path in enumerate(args.input):
        input_path = Path(input_path)

        color = colors[input_idx % len(colors)]
        agent_dirs = discover_agents(input_path)

        for agent_dir in agent_dirs:

            agent_name = agent_dir.name.lower()
            marker = agent_markers.get(agent_name, "x")

            if label_idx < len(labels):
                label = labels[label_idx]
            elif len(agent_dirs) > 1:
                label = f"{input_path.name}/{agent_dir.name}"
            else:
                label = input_path.name

            regrets = load_regrets(agent_dir)

            plot_series(
                ax,
                regrets,
                label,
                color=color,
                marker=marker,
            )

            label_idx += 1
            plotted += 1

    if plotted == 0:
        raise SystemExit("No data plotted. Check --input paths.")

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()