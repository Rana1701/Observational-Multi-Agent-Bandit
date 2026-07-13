import argparse
import os
import sys
import random
import numpy as np
from multiprocessing import Pool
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def batch_generate(model, prompts):
    """Generate multiple LLM responses in parallel."""
    if not prompts:
        return []

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        top_p=0.9,
        stop=["}"],
    )

    outputs = model.generate(
        prompts,
        sampling_params
    )

    responses = []
    for output in outputs:
        text = output.outputs[0].text.strip()

        if not text.endswith("}"):
            text += "}"

        responses.append(text)

    return responses


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

    cumulative_reward = {name: 0.0 for name in order}
    cumulative_regret = {name: 0.0 for name in order}

    rewards_ts = {name: [] for name in order}
    regrets_ts = {name: [] for name in order}

    # Main time loop
    for t in range(horizon):
        current_actions = {}

        for a in agent_cfgs:
            name = a["name"]
            agent = agents[name]

            if a.get("class") == "LLM":
                cfg_by_name[name]["_other_action_counts"] = other_action_counts.copy()

                prompt = build_llm_prompt(
                    cfg_by_name[name],
                    agent
                )

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

            current_actions[name] = action
            global_history[name].append(action)

            reward = agent.reward

            cumulative_reward[name] += reward
            rewards_ts[name].append(
                cumulative_reward[name] / (t + 1)
            )

            if hasattr(bandit, "regret"):
                cumulative_regret[name] += bandit.regret(action)
            else:
                expected_reward = bandit.probs[action]
                cumulative_regret[name] += (
                    max_theoretical_reward - expected_reward
                )

            regrets_ts[name].append(
                cumulative_regret[name]
            )

        # Update observed actions after all agents have played
        for name, action in current_actions.items():
            if name in exp.get("track_other_actions_for", []):
                if 0 <= action < len(other_action_counts):
                    other_action_counts[action] += 1

    for name in order:
        out["time_averaged_rewards"][name] = np.array(
            rewards_ts[name]
        )

        out["cumulated_regrets"][name] = np.array(
            regrets_ts[name]
        )

    # Optimal baseline
    if exp.get("add_opt", False):
        opt_cum = 0.0
        opt_curve = []

        for t in range(horizon):
            opt_cum += max_theoretical_reward
            opt_curve.append(
                opt_cum / (t + 1)
            )

        out["time_averaged_rewards"]["OPT"] = np.array(opt_curve)
        out["cumulated_regrets"]["OPT"] = np.zeros(horizon)

    return out

def init_batched_runs(config, shared_model):
    """Initialize experiment replicas."""
    exp = config["experiment"]
    states = []

    for run_idx in range(exp.get("runs", 20)):
        seed = run_seed(exp.get("seed"), run_idx)
        random.seed(seed)
        np.random.seed(seed)

        bandit = build_bandit(config["environment"], seed)
        order = exp.get("order") or [a["name"] for a in config["agents"]]

        agents = {}
        cfg_by_name = {}

        for a in config["agents"]:
            name = a["name"]
            cfg_by_name[name] = a
            agents[name] = create_agent(
                AGENTS[a["class"]],
                bandit,
                a.get("params"),
                shared_model=shared_model,
            )

        states.append({
            "bandit": bandit,
            "agents": agents,
            "cfg": cfg_by_name,
            "order": order,
            "history": {n: [] for n in order},
            "other_counts": [0] * bandit.n_arms,
            "actions": {},
            "reward": {n: 0.0 for n in order},
            "regret": {n: 0.0 for n in order},
            "rewards_ts": {n: [] for n in order},
            "regrets_ts": {n: [] for n in order},
            "best": np.max(bandit.probs) if hasattr(bandit, "probs") else config["environment"].get("best_mean", 0.9)
        })

    return states


def run_batched_llm_experiment(config, model):
    """Run replicas with batched LLM inference."""
    states = init_batched_runs(config, model)
    horizon = config["experiment"]["horizon"]
    track = config["experiment"].get("track_other_actions_for", [])

    for t in range(horizon):
        prompts, refs = [], []

        # Collect all LLM prompts
        for s in states:
            for name in s["order"]:
                cfg = s["cfg"][name]
                agent = s["agents"][name]

                if cfg.get("class") == "LLM":
                    cfg["_other_action_counts"] = s["other_counts"].copy()
                    prompts.append(build_llm_prompt(cfg, agent))
                    refs.append((s, name, agent))

        # One batched generation
        responses = batch_generate(model, prompts)

        for (s, name, agent), response in zip(refs, responses):
            s["actions"][name] = agent.getNextAction(response)

        # Non LLM agents
        for s in states:
            for name in s["order"]:
                if s["cfg"][name].get("class") != "LLM":
                    s["actions"][name] = s["agents"][name].getNextAction()

        # Update statistics
        for s in states:
            for name, action in s["actions"].items():
                agent = s["agents"][name]

                if s["cfg"][name].get("class") == "LLM":
                    reward = agent.getReward(action)
                    agent.history[str(action)]["pulls"] += 1
                    agent.history[str(action)]["reward"] += reward
                    agent.t += 1

                reward = agent.reward
                s["history"][name].append(action)

                s["reward"][name] += reward
                s["rewards_ts"][name].append(
                    s["reward"][name] / (t + 1)
                )

                if hasattr(s["bandit"], "regret"):
                    s["regret"][name] += s["bandit"].regret(action)
                else:
                    s["regret"][name] += s["best"] - s["bandit"].probs[action]

                s["regrets_ts"][name].append(
                    s["regret"][name]
                )

            # Update tracked actions
            for name in track:
                action = s["actions"].get(name)
                if action is not None:
                    s["other_counts"][action] += 1

            s["actions"] = {}

    results = []

    for s in states:
        out = {
            "time_averaged_rewards": {n: np.array(s["rewards_ts"][n])  for n in s["order"]},
            "cumulated_regrets": {n: np.array(s["regrets_ts"][n]) for n in s["order"]}}

        if config["experiment"].get("add_opt", False):
            out["time_averaged_rewards"]["OPT"] = np.ones(horizon) * s["best"]
            out["cumulated_regrets"]["OPT"] = np.zeros(horizon)

        results.append(out)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent experiment."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiment"]
    runs = exp.get("runs", 20)

    if uses_llm(config):
        model = LLM(model=get_llm_model_name(config),max_model_len=4096)
        results = run_batched_llm_experiment(config,model)

    else:
        tasks = [(i, config) for i in range(runs)]
        with Pool(exp.get("n_jobs", 4)) as pool:
            results = pool.map(run_single_rep,tasks)

    save_multi_results(results,config)
    print(f"Saved {len(results)} runs")


if __name__ == "__main__":
    main()