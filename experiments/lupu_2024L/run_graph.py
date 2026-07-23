import numpy as np
import argparse
import os
import sys
import random
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results
from utils.graph_gen import get_neighbors
from agents.tucb import TUCB
from agents.ucbClique import UCBClique
from agents.ucb import UCB


def run_single_simulation_worker(args_tuple):
    run_idx, horizon, n_agents, n_arms, target_probs, delta, graph_types, seed = args_tuple

    run_seed = seed + run_idx if seed is not None else None
    if run_seed is not None:
        np.random.seed(run_seed)
        random.seed(run_seed)

    local_regrets = {}

    # Single UCB
    bandit = BernoulliBandit(probs=target_probs)
    agent = UCB(bandit)
    for _ in range(horizon):
        agent.getNextAction()
    local_regrets["Single UCB"] = np.asarray(agent.cumul_regret)

    # UCB Clique
    bandit = BernoulliBandit(probs=target_probs)
    agent = UCBClique(bandit, n_agents)
    for _ in range(horizon):
        agent.getNextAction()
    local_regrets["UCB clique"] = np.asarray(agent.history_regret)

    # TUCB
    for g_type in graph_types:
        try:
            adj = get_neighbors(n=n_agents, graph_type=g_type)
        except Exception:
            name = g_type.replace("_", "-").title() if g_type == "small_world" else g_type.title()
            adj = get_neighbors(n=n_agents, graph_type=name)

        bandits = [BernoulliBandit(probs=target_probs) for _ in range(n_agents)]
        agents = [
            TUCB(
                bandit=bandits[i],
                nbr_neighbours=len(adj[i]),
                delta=delta
            )
            for i in range(n_agents)
        ]

        prev_actions = {
            i: [int(np.random.choice(n_arms)) for _ in range(len(adj[i]))]
            for i in range(n_agents)
        }

        regrets = np.zeros((n_agents, horizon))

        for t in range(horizon):
            actions = {}

            for i in range(n_agents):
                action = agents[i].getNextAction(prev_actions[i])
                actions[i] = action
                regrets[i, t] = agents[i].cumul_regret[-1]

            for i in range(n_agents):
                prev_actions[i] = [actions[j] for j in adj[i]]

        name = g_type.replace("_", "-").title()
        local_regrets[name] = np.mean(regrets, axis=0)

    return {
        "cumulated_regrets": {k: np.asarray(v) for k, v in local_regrets.items()},
        "time_averaged_rewards": {k: np.zeros_like(v) for k, v in local_regrets.items()}
    }


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiment"]
    env = config["environment"]

    n_runs = exp.get("runs", 2000)
    horizon = exp.get("horizon", 1000)
    n_agents = exp.get("n_agents", 20)
    n_arms = env.get("n_arms", 10)
    n_jobs = exp.get("n_jobs", 4)
    seed = exp.get("seed", 42)

    best_mean = env.get("best_mean", 0.5)
    delta = env.get("delta", 0.1)

    target_probs = [best_mean] + [best_mean - delta] * (n_arms - 1)

    graph_types = ["clique", "chain", "loop", "random", "small_world"]

    tasks = [
        (i, horizon, n_agents, n_arms, target_probs, delta, graph_types, seed)
        for i in range(n_runs)
    ]

    print(f"Lancement sur {n_jobs} cœurs...")

    all_runs_data = []

    with Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_simulation_worker, tasks)):
            all_runs_data.append(result)
            if (i + 1) % 50 == 0 or i + 1 == n_runs:
                print(f"{i+1}/{n_runs} runs terminés")

    print("Sauvegarde des résultats...")
    save_multi_results(all_runs_data, config)

    print(f"Sauvegardé {len(all_runs_data)} runs")


if __name__ == "__main__":
    run_experiment()