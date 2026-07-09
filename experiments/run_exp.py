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
    np.random.seed(seed)

    horizon = exp["horizon"]
    agent_cfgs = cfg["agents"]

    order = exp.get("order") or [a["name"] for a in agent_cfgs]
    interaction = exp.get("interaction", "sequential")
    probs = cfg["environment"].get("probs", None)
    bandit = build_bandit(cfg["environment"], seed)
    max_theoretical_reward = np.max(probs) or cfg["environment"].get("best_mean", 0.9)

    agents = {}
    cfg_by_name = {}

    # ===== INIT AGENTS =====
    for a in agent_cfgs:
        name = a["name"]
        cfg_by_name[name] = a

        agents[name] = create_agent(
            AGENTS[a["class"]],
            bandit,
            a.get("params"),
            shared_model=shared_model,
        )

    # ===== OUTPUT STORAGE =====
    # Rewards
    cumulative_reward = {name: 0.0 for name in order}
    rewards_ts = {name: [] for name in order}
    
    # Cumulated Regret
    cumulative_regret = {name: 0.0 for name in order}
    regrets_ts = {name: [] for name in order}
    
    last_actions = {name: 0 for name in order}
    other_action_counts = [0] * bandit.n_arms
    
    # OPT baseline
    opt_cum = 0.0
    opt_curve = []

    # ===== MAIN LOOP =====
    for t in range(horizon):
        current_actions = {}

        # --- MODE SEQUENTIAL ---
        if interaction == "sequential":
            for i, name in enumerate(order):
                agent = agents[name]
                cfg_a = cfg_by_name[name]

                # LLM case
                if cfg_a.get("class") == "LLM":
                    cfg_a["_other_action_counts"] = other_action_counts
                    prompt = build_llm_prompt(cfg_a, agent)
                    print("the prompt is: ", prompt)
                    action = agent.getNextAction(prompt)
                # Agent normal
                else:
                    observed = [
                        current_actions[o]
                        for o in cfg_a.get("observes", [])
                        if o in current_actions
                    ]
                    action = agent.getNextAction(observed or None)

                reward = agent.reward
                current_actions[name] = action

                # Time-averaged reward
                cumulative_reward[name] += reward
                rewards_ts[name].append(cumulative_reward[name] / (t + 1))

                # Cumulated regret
                expected_reward = bandit.probs[action]
                regret_instantane = max_theoretical_reward - expected_reward
                cumulative_regret[name] += regret_instantane
                regrets_ts[name].append(cumulative_regret[name])

        # --- MODE SIMULTANEOUS ---
        else:
            actions = {}
            for name in order:
                agent = agents[name]
                cfg_a = cfg_by_name[name]

                if cfg_a.get("class") == "LLM":
                    cfg_a["_other_action_counts"] = other_action_counts
                    prompt = build_llm_prompt(cfg_a, agent)
                    print(prompt)
                    action = agent.getNextAction(prompt)
                else:
                    action = agent.getNextAction(None)

                actions[name] = action

            # Environnement update 
            for name in order:
                reward = bandit.pull(actions[name])

                # Rewards
                cumulative_reward[name] += reward
                rewards_ts[name].append(cumulative_reward[name] / (t + 1))

                # Regret
                expected_reward = bandit.probs[actions[name]]
                regret_instantane = max_theoretical_reward - reward
                cumulative_regret[name] += regret_instantane
                regrets_ts[name].append(cumulative_regret[name])

            current_actions = actions

        # OPT line 
        opt_cum += max_theoretical_reward
        opt_curve.append(opt_cum / (t + 1))
    
        # Global stats update
        for obs_name in cfg.get("track_other_actions_for", order):
            action = current_actions[obs_name]
            if 0 <= action < len(other_action_counts):
                other_action_counts[action] += 1

        last_actions = current_actions

    # ===== RESULTS =====
    out = {
        "time_averaged_rewards": {},
        "cumulated_regrets": {}
    }

    for name in order:
        out["time_averaged_rewards"][name] = np.array(rewards_ts[name])
        out["cumulated_regrets"][name] = np.array(regrets_ts[name])

    out["time_averaged_rewards"]["OPT"] = np.array(opt_curve)
    # OPT regret is always zero
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
