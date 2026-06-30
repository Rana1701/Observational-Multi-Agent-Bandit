import numpy as np
import argparse
import os
import sys
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from vllm import LLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.common import load_config, save_multi_results

from utils.common import (
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

    horizon = exp["horizon"]
    agent_cfgs = cfg["agents"]

    order = exp.get("order") or [a["name"] for a in agent_cfgs]
    interaction = exp.get("interaction", "sequential")

    bandit = BernoulliBandit(probs=probs)
    max_theoretical_reward = np.max(probs)

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
    cumulative = {name: 0.0 for name in order}
    rewards_ts = {name: [] for name in order}

    # OPT baseline
    opt_cum = 0.0
    opt_curve = []

    # ===== MAIN LOOP =====
    for t in range(horizon):

        current_actions = {}

        # SEQUENTIAL 
        if interaction == "sequential":

            for i, name in enumerate(order):
                agent = agents[name]
                cfg_a = cfg_by_name[name]

                # ---- LLM case ----
                if cfg_a.get("class") == "LLM":
                    prompt = build_llm_prompt(cfg_a, agent)
                    action = agent.getNextAction(prompt)

                # ---- normal agent ----
                else:
                    observed = [
                        current_actions[o]
                        for o in cfg_a.get("observes", [])
                        if o in current_actions
                    ]
                    action = agent.getNextAction(observed or None)

                reward = bandit.pull(action)

                current_actions[name] = action

                cumulative[name] += reward
                rewards_ts[name].append(cumulative[name] / (t + 1))

        # SIMULTANEOUS
        else:

            actions = {}

            for name in order:
                agent = agents[name]
                cfg_a = cfg_by_name[name]

                if cfg_a.get("class") == "LLM":
                    prompt = build_llm_prompt(cfg_a, agent)
                    action = agent.getNextAction(prompt)
                else:
                    action = agent.getNextAction(None)

                actions[name] = action

            # env step AFTER all actions
            for name in order:
                reward = bandit.pull(actions[name])

                cumulative[name] += reward
                rewards_ts[name].append(cumulative[name] / (t + 1))

            current_actions = actions

        # OPT 
        opt_cum += max_theoretical_reward
        opt_curve.append(opt_cum / (t + 1))

    out = {}

    for name in order:
        out[name] = np.array(rewards_ts[name])

    out["OPT"] = np.array(opt_curve)

    return out

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
    else :
        print(f"Running {runs} replicates...")

        with Pool(processes=n_jobs) as pool:
            all_runs = pool.map(run_single_rep, tasks)

    save_multi_results(all_runs, config)


if __name__ == "__main__":
    main()