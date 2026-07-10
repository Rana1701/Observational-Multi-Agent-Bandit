import argparse
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from vllm import LLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import ( 
    load_config,
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
    random.seed(seed)
    np.random.seed(seed)    
    agent_cfgs = cfg["agents"]
    order = exp.get("order") or [a["name"] for a in agent_cfgs]
    horizon = exp["horizon"]

    out = {
        "time_averaged_rewards": {},
        "cumulated_regrets": {}
    }

    agents = {}
    cfg_by_name = {}

    if probs is not None:
        max_theoretical_reward = np.max(probs)
    else:
        max_theoretical_reward = cfg["environment"].get("best_mean", 0.9)

    global_history = {name: [] for name in order}
    
    bandit = build_bandit(cfg["environment"], seed)
    n_arms = bandit.n_arms
    other_action_counts = [0] * n_arms

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

        cumulative_reward = 0.0
        cumulative_regret = 0.0
        
        rewards_ts = []
        regrets_ts = []

        for t in range(horizon):
            if a.get("class") == "LLM":
                cfg_by_name[name]["_other_action_counts"] = other_action_counts
                prompt = build_llm_prompt(cfg_by_name[name], agent)
                action = agent.getNextAction(prompt)
                
            elif a.get("observes"):
                observed = [
                    global_history[o][t]
                    for o in a.get("observes", [])
                    if o in global_history and len(global_history[o]) > t
                ]
                action = agent.getNextAction(observed or None)
                
            else:
                action = agent.getNextAction()

            global_history[name].append(action)

            reward = agent.reward 
            cumulative_reward += reward
            rewards_ts.append(cumulative_reward / (t + 1))

            if hasattr(bandit, "regret"):
                cumulative_regret += bandit.regret(action)
            else:
                expected_reward = bandit.probs[action]
                cumulative_regret += max_theoretical_reward - expected_reward
            regrets_ts.append(cumulative_regret)

        if name in cfg.get("track_other_actions_for", order):
            for act in global_history[name]:
                if 0 <= act < len(other_action_counts):
                    other_action_counts[act] += 1

        out["time_averaged_rewards"][name] = np.array(rewards_ts)
        out["cumulated_regrets"][name] = np.array(regrets_ts)

    # ===== BASELINE OPT =====
    opt_cum = 0.0
    opt_curve = []
    for t in range(horizon):
        opt_cum += max_theoretical_reward
        opt_curve.append(opt_cum / (t + 1))

    out["time_averaged_rewards"]["OPT"] = np.array(opt_curve)
    out["cumulated_regrets"]["OPT"] = np.zeros(horizon) 

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent experiment saving both rewards and regrets."
    )
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    exp = config["experiment"]
    runs = exp.get("runs", 20)
    n_jobs = exp.get("n_jobs", 4)
    tasks = [(i, config) for i in range(runs)]
    all_runs = []

    if uses_llm(config):
        if n_jobs > 1:
            print("LLM experiments run sequentially (model cannot be shared across workers).")
            n_jobs = 1

        model_name = get_llm_model_name(config)
        shared_model = LLM(model=model_name, max_model_len=4096)

        for i in range(runs):
            res = run_single_rep((i, config), shared_model)
            all_runs.append(res)
    else:
        print(f"Running {runs} replicates...")
        with Pool(processes=n_jobs) as pool:
            all_runs = pool.map(run_single_rep, tasks)

    # all_runs contient maintenant une liste de dictionnaires à deux clés: 
    # [{"time_averaged_rewards": {...}, "cumulated_regrets": {...}}, ...]
    save_multi_results(all_runs, config)


if __name__ == "__main__":
    main()
