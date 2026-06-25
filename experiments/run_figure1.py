import numpy as np
import argparse
import os
import sys
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.common import load_config

from agents.ts import TS
from agents.ucb import UCB
from agents.greedy import Greedy


def run_single_rep(args_tuple):

    rep_idx, horizon, probs, seed = args_tuple

    run_seed = seed + rep_idx

    np.random.seed(run_seed)
    random.seed(run_seed)

    results = {}

    algorithms = {
        "UCB": UCB,
        "TS": TS,
        "Greedy": Greedy,
    }

    for name, AgentClass in algorithms.items():

        bandit = BernoulliBandit(probs=probs)
        agent = AgentClass(bandit)

        cumulative_reward = 0
        avg_rewards = []

        for t in range(horizon):

            agent.getNextAction()

            cumulative_reward += agent.reward

            avg_rewards.append(
                cumulative_reward / (t + 1)
            )

        results[name] = avg_rewards

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Reproduce Figure 1 (right): Time-averaged reward"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    exp = config["experiment"]
    env = config["environment"]

    runs = exp.get("runs", 20)
    horizon = exp.get("horizon", 500)
    seed = exp.get("seed", 42)
    n_jobs = exp.get("n_jobs", 4)

    probs = env["probs"]

    tasks = [
        (i, horizon, probs, seed)
        for i in range(runs)
    ]

    print(f"Running {runs} replicates...")

    with Pool(processes=n_jobs) as pool:
        all_runs = list(pool.imap(run_single_rep, tasks))

    agents = ["UCB", "TS", "Greedy"]

    mean_curves = {}
    stderr_curves = {}

    for agent in agents:

        data = np.array(
            [run[agent] for run in all_runs]
        )

        mean_curves[agent] = data.mean(axis=0)

        stderr_curves[agent] = (
            data.std(axis=0, ddof=1)
            / np.sqrt(runs)
        )

    plt.figure(figsize=(8, 5))

    colors = {
        "UCB": "green",
        "TS": "orange",
        "Greedy": "red",
    }

    x = np.arange(1, horizon + 1)

    for agent in agents:

        mean = mean_curves[agent]
        se = stderr_curves[agent]

        plt.plot(
            x,
            mean,
            label=agent,
            color=colors[agent],
            linewidth=2,
        )

        plt.fill_between(
            x,
            mean - 2 * se,
            mean + 2 * se,
            color=colors[agent],
            alpha=0.2,
        )

    plt.xlabel("Round t")
    plt.ylabel("Time-averaged reward")
    plt.title("5-Armed Bernoulli Bandit (Δ = 0.2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 0.6)
    output_dir = exp.get(
        "output_dir",
        "results/figure1"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "time_averaged_reward.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()