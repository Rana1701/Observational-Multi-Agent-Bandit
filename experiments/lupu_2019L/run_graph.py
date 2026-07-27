import numpy as np
import argparse
import os
import sys
import random
from multiprocessing import Pool
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results
from agents.tucb import TUCB
from agents.ucb import UCB
from agents.ucbClique import UCBClique


def generate_graph(n, graph_type):
    if graph_type == "clique": G = nx.complete_graph(n)
    elif graph_type == "loop": G = nx.cycle_graph(n)
    elif graph_type == "chain": G = nx.path_graph(n)
    elif graph_type == "random": G = nx.erdos_renyi_graph(n, p=0.5)
    elif graph_type == "small-world": G = nx.barabasi_albert_graph(n, m=1)
    else: raise ValueError(graph_type)

    return {i: list(G.neighbors(i)) for i in range(n)}


def run_single_simulation_worker(args):
    run_idx, horizon, n_agents, n_arms, seed = args
    np.random.seed(seed + run_idx)
    random.seed(seed + run_idx)
    results = {}

    # =====================================================
    # Random 10 arms problem
    # =====================================================
    probs = np.random.uniform(low=0.1, high=0.9, size=n_arms)

    # =====================================================
    # Single UCB
    # =====================================================
    bandit = BernoulliBandit(probs=probs)
    agent = UCB(bandit)
    for _ in range(horizon):
        agent.getNextAction()
    results["Single UCB"] = np.array(agent.cumul_regret)

    # =====================================================
    # UCB Clique
    # =====================================================
    bandit = BernoulliBandit(probs=probs)
    agent = UCBClique(bandit, n_agents)
    for _ in range(horizon):
        agent.getNextAction()
    results["UCB clique"] = np.array(agent.history_regret)

    # =====================================================
    # Target UCB graphs
    # =====================================================
    graph_types = ["clique", "chain", "loop", "random", "small-world"]

    for graph_type in graph_types:
        neighbors = generate_graph(n_agents, graph_type)
        bandits = [BernoulliBandit(probs=probs) for _ in range(n_agents)]
        agents = [TUCB(bandit=bandits[i], nbr_neighbours=max(1, len(neighbors[i]))) for i in range(n_agents)]
        prev_actions = {i: [np.random.randint(n_arms) for _ in range(max(1, len(neighbors[i])))] for i in range(n_agents)}
        regrets = np.zeros((n_agents, horizon))

        for t in range(horizon):
            actions = {}
            for i in range(n_agents):
                actions[i] = agents[i].getNextAction(prev_actions[i])
                regrets[i, t] = agents[i].cumul_regret[-1]

            for i in range(n_agents):
                neigh = neighbors[i]
                prev_actions[i] = [actions[i]] if len(neigh) == 0 else [actions[j] for j in neigh]

        name = graph_type.replace("-", " ").title()
        results[name] = np.mean(regrets, axis=0)

    return {
        "cumulated_regrets": results,
        "time_averaged_rewards": {k: np.zeros_like(v) for k, v in results.items()}
    }


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    exp, env = config["experiment"], config["environment"]

    runs, horizon, n_agents = exp["runs"], exp["horizon"], exp["n_agents"]
    n_arms, seed, n_jobs = env["n_arms"], exp["seed"], exp["n_jobs"]

    tasks = [(i, horizon, n_agents, n_arms, seed) for i in range(runs)]
    all_results = []

    print(f"Lancement {runs} runs sur {n_jobs} workers")

    with Pool(processes=n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_simulation_worker, tasks)):
            all_results.append(res)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{runs}")

    save_multi_results(all_results, config)
    print("Terminé")


if __name__ == "__main__":
    run_experiment()
