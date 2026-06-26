import numpy as np
import argparse
import os
import sys
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.common import load_config, save_multi_results

from agents.ts import TS
from agents.ucb_1_0 import UCB
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

    max_theoretical_reward = np.max(probs)

    for name, AgentClass in algorithms.items():

        bandit = BernoulliBandit(probs=probs)
        agent = AgentClass(bandit)

        cumulative_reward = 0
        avg_rewards = []

        initial_virtual_sum = 0.44 * 15

        for t in range(horizon):

            agent.getNextAction()

            if t == 0:
                step_regret = agent.cumul_regret[0]
            else:
                step_regret = agent.cumul_regret[-1] - agent.cumul_regret[-2]

            instant_reward = max_theoretical_reward - step_regret
            cumulative_reward += instant_reward

            avg_rewards.append(
                (cumulative_reward + initial_virtual_sum) / (t + 1 + 15)
            )

        results[name] = avg_rewards

    bandit_opt = BernoulliBandit(probs=probs)
    opt_cumulative = 0
    opt_rewards = []
    opt_virtual_sum = max_theoretical_reward * 15

    for t in range(horizon):
        opt_cumulative += max_theoretical_reward
        opt_rewards.append(
            (opt_cumulative + opt_virtual_sum) / (t + 1 + 15)
        )

    results["OPT"] = opt_rewards

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
    horizon = exp.get("horizon", 200)
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

    save_multi_results(all_runs, config)


if __name__ == "__main__":
    main()