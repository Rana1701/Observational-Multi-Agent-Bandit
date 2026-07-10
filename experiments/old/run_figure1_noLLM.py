import numpy as np
import argparse
import os
import sys
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results

from utils.experiment_utils import (
    load_config,
    parallel_runs,
    run_seed,
    build_bandit,
    create_agent,
    AGENTS,
    build_llm_prompt,
    uses_llm,
    get_llm_model_name,
    save_multi_results,
)

def run_single_rep(task, shared_model=None):
    run_idx, cfg = task
    probs = cfg["environment"].get("probs", None)
    exp = cfg["experiment"]
    seed = run_seed(exp.get("seed"), run_idx)
    np.random.seed(seed)
    agent_cfgs = cfg["agents"]
    order = exp.get("order") or [a["name"] for a in agent_cfgs]
    interaction = exp.get("interaction", "sequential")
    horizon = exp["horizon"]

    results = {}
    agents = {}
    cfg_by_name = {}

    max_theoretical_reward = np.max(probs)

    bandit = BernoulliBandit(probs=probs)
    for a in agent_cfgs:
        name = a["name"]
        cfg_by_name[name] = a

        agents[name] = create_agent(
            AGENTS[a["class"]],
            bandit,
            a.get("params"),
            shared_model=shared_model,
        )

        agent = agents[name]

        cumulative_reward = 0
        avg_rewards = []

        #initial_virtual_sum =0

        for t in range(horizon):
            agent.getNextAction()
            reward = agent.reward 
            cumulative_reward += reward

            avg_rewards.append(
                (cumulative_reward ) / (t + 1 +1 )
            )

        results[name] = avg_rewards

    bandit_opt = BernoulliBandit(probs=probs)
    opt_cumulative = 0
    opt_rewards = []

    for t in range(horizon):
        opt_cumulative += max_theoretical_reward
        opt_rewards.append(
            (opt_cumulative ) / (t + 1 )
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
    runs = exp.get("runs", 20)
    n_jobs = exp.get("n_jobs", 4)

    tasks = [(i, config) for i in range(runs)]
    print(f"Running {runs} replicates...")

    with Pool(processes=n_jobs) as pool:
        all_runs = pool.map(run_single_rep, tasks)

    save_multi_results(all_runs, config)


if __name__ == "__main__":
    main()