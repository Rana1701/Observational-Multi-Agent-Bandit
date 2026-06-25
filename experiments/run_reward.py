import argparse
import os
import sys
import numpy as np
from vllm import LLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def run_single_multi_time_avg(cfg, run_idx, shared_model=None):

    exp = cfg["experiment"]
    seed = run_seed(exp.get("seed"), run_idx)
    bandit = build_bandit(cfg["environment"], seed)

    agent_cfgs = cfg["agents"]
    order = exp.get("order") or [a["name"] for a in agent_cfgs]
    interaction = exp.get("interaction", "sequential")
    horizon = exp["horizon"]

    agents = {}
    cfg_by_name = {}

    for a in agent_cfgs:
        name = a["name"]
        cfg_by_name[name] = a

        agents[name] = create_agent(
            AGENTS[a["class"]],
            bandit,
            a.get("params"),
            shared_model=shared_model,
        )

    rewards = {name: [] for name in order}
    last_actions = {name: 0 for name in order}
    other_action_counts = [0] * bandit.n_arms

    for _ in range(horizon):

        current_actions = {}

        # ===== SIMULTANEOUS =====
        if interaction == "simultaneous":

            for name in order:

                cfg_a = cfg_by_name[name]
                agent = agents[name]

                observed = [
                    last_actions[o]
                    for o in cfg_a.get("observes", [])
                ]

                if cfg_a.get("class") == "LLM":
                    cfg_a["_other_action_counts"] = other_action_counts
                    prompt = build_llm_prompt(cfg_a, agent)
                    action = agent.getNextAction(prompt)
                else:
                    action = agent.getNextAction(observed or None)

                reward = bandit.pull(action)

                current_actions[name] = action
                rewards[name].append(reward)

        # ===== SEQUENTIAL =====
        else:

            for i, name in enumerate(order):

                cfg_a = cfg_by_name[name]
                agent = agents[name]

                observed = [
                    current_actions[o]
                    for o in cfg_a.get("observes", [])
                ]

                if cfg_a.get("class") == "TUCB":

                    k = cfg_a.get("params", {}).get("nbr_neighbours", 1)

                    prev = [
                        current_actions[n]
                        for n in order[:i]
                    ][:k]

                    action = agent.getNextAction(prev or None)

                elif cfg_a.get("class") == "LLM":

                    cfg_a["_other_action_counts"] = other_action_counts
                    prompt = build_llm_prompt(cfg_a, agent)
                    action = agent.getNextAction(prompt)

                else:
                    action = agent.getNextAction(observed or None)

                reward = bandit.pull(action)

                current_actions[name] = action
                rewards[name].append(reward)

        # update stats
        for obs_name in cfg.get("track_other_actions_for", order):
            a = current_actions[obs_name]
            if 0 <= a < len(other_action_counts):
                other_action_counts[a] += 1

        last_actions = current_actions

    out = {}

    for name, r in rewards.items():
        r = np.asarray(r, dtype=float)
        out[name] = np.cumsum(r) / (np.arange(len(r)) + 1)

    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    n_jobs = args.jobs or cfg["experiment"].get("n_jobs", 1)

    shared_model = None

    if uses_llm(cfg):

        if n_jobs > 1:
            print("LLM detected → forcing n_jobs=1")
            n_jobs = 1

        model_name = get_llm_model_name(cfg)
        shared_model = LLM(model=model_name, max_model_len=4096)

        def run_fn(cfg, run_idx):
            return run_single_multi_time_avg(cfg, run_idx, shared_model)

        run_function = run_fn

    else:
        run_function = run_single_multi_time_avg

    all_results = parallel_runs(run_function, cfg, n_jobs)

    save_multi_results(all_results, cfg)


if __name__ == "__main__":
    main()