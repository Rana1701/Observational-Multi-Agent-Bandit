import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from agents.ucbClique import UCBClique

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.alphaOptimal import alphaOptimal
from agents.ts import TS
from environnement.bernoulli_bandit import BernoulliBandit
from agents.ucb import UCB
from agents.tucb import TUCB
from agents.greedy import Greedy
from agents.greedy_follower import GreedyFollower
from agents.e_greedy import EpsilonGreedy
from agents.llm import LLMAgent
from agents.ucb_1_0 import UCB1
from utils.prompt_builder import (
    build_prompt_history,
    build_prompt_noHistory,
    build_prompt_ucb_history,
    build_prompt_exploit,
    build_prompt_ucb_noHistory,
    build_prompt_explore,
    build_prompt_krishnamurthy,
    build_prompt,
)

AGENTS = {
    "UCB": UCB,
    "UCB1": UCB1,
    "UCBClique": UCBClique,
    "TUCB": TUCB,
    "Greedy": Greedy,
    "GreedyFollower": GreedyFollower,
    "EpsilonGreedy": EpsilonGreedy,
    "LLM": LLMAgent,
    "TS" : TS,
    "alphaOptimal": alphaOptimal,
}

PROMPT_BUILDERS = {
    "default": build_prompt,
    "history": build_prompt_history ,
    "no_history": build_prompt_noHistory ,
    "ucb": build_prompt_ucb_noHistory,
    "ucb_history": build_prompt_ucb_history,
    "exploit": build_prompt_exploit,
    "explore": build_prompt_explore,
    "krishnamurthy": build_prompt_krishnamurthy,
}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_seed(base_seed, run_idx):
    return None if base_seed is None else int(base_seed) + run_idx


def build_bandit(environment_cfg, seed):
    if seed is not None:
        np.random.seed(seed)
    return BernoulliBandit(**environment_cfg)


def create_agent(agent_class, bandit, params=None, shared_model=None):
    params = dict(params or {})

    if agent_class is LLMAgent:
        if shared_model is not None:
            params["model"] = shared_model
        elif "model" in params and isinstance(params["model"], str):
            params["name_parameter"] = params.pop("model")

    return agent_class(bandit, **params)


def uses_llm(cfg):
    if cfg.get("agent") == "LLM":
        return True
    return any(a.get("class") == "LLM" for a in cfg.get("agents", []))

def build_llm_prompt(agent_cfg, agent):
    prompt_name = agent_cfg.get("prompt", "default")
    builder = PROMPT_BUILDERS.get(prompt_name)
    #print(f" prompt name {prompt_name}, prompt builder {builder}")
    if builder is None:
        return None

    if prompt_name == "krishnamurthy_solo":
        other_actions = None
        return builder(agent.bandit, agent.t, agent.history, other_actions,
                         agent_cfg.get("prompt_tuple", [])[0], agent_cfg.get("prompt_tuple", [])[1],
                         agent_cfg.get("prompt_tuple", [])[2], agent_cfg.get("prompt_tuple", [])[3],
                         agent_cfg.get("prompt_tuple", [])[4])

    other_actions = agent_cfg.get("_other_action_counts", None)
    return builder(agent.bandit, agent.t, agent.history, other_actions, horizon=agent_cfg.get("horizon", 500))

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"

def get_llm_model_name(cfg):
    if "agent" in cfg:
        return cfg.get("agent_params", {}).get(
            "model",
            DEFAULT_LLM,
        )

    for agent_cfg in cfg.get("agents", []):
        if agent_cfg.get("class") == "LLM":
            return agent_cfg.get("params", {}).get(
                "model",
                DEFAULT_LLM,
            )

    return DEFAULT_LLM

""""
def run_single_multi(cfg, run_idx, shared_model=None):
    exp = cfg["experiment"]
    seed = run_seed(exp.get("seed"), run_idx)
    bandit = build_bandit(cfg["environment"], seed)

    agent_cfgs = cfg["agents"]
    order = exp.get("order") or [a["name"] for a in agent_cfgs]
    interaction = exp.get("interaction", "sequential")

    agents = {}
    for agent_cfg in agent_cfgs:
        name = agent_cfg["name"]
        agents[name] = create_agent(
            AGENTS[agent_cfg["class"]],
            bandit,
            agent_cfg.get("params"),
            shared_model=shared_model,
        )

    cfg_by_name = {a["name"]: a for a in agent_cfgs}

    regrets = {name: [] for name in order}
    last_actions = {name: 0 for name in order}
    other_action_counts = [0] * bandit.n_arms

    horizon = exp["horizon"]

    for _ in range(horizon):
        current_actions = {}

        if interaction == "simultaneous":
            for name in order:
                agent_cfg = cfg_by_name[name]
                agent = agents[name]

                observed = [
                    last_actions[obs_name]
                    for obs_name in agent_cfg.get("observes", [])
                ]

                if agent_cfg.get("class") == "LLM":
                    agent_cfg["_other_action_counts"] = other_action_counts
                    prompt = build_llm_prompt(agent_cfg, agent)
                    current_actions[name] = agent.getNextAction(prompt)
                else:
                    current_actions[name] = agent.getNextAction(observed or None)

        else:
            for i, name in enumerate(order):
                agent_cfg = cfg_by_name[name]
                agent = agents[name]

                observed = [
                    current_actions[obs_name]
                    for obs_name in agent_cfg.get("observes", [])
                ]

                #  TUCB SPECIAL HANDLING
                if agent_cfg.get("class") == "TUCB":
                    k = agent_cfg.get("params", {}).get("nbr_neighbours", 1)

                    prev = [
                        current_actions[n]
                        for n in order[:i]
                    ][:k]

                    current_actions[name] = agent.getNextAction(prev or None)

                elif agent_cfg.get("class") == "LLM":
                    agent_cfg["_other_action_counts"] = other_action_counts
                    prompt = build_llm_prompt(agent_cfg, agent)
                    current_actions[name] = agent.getNextAction(prompt)
                else:
                    current_actions[name] = agent.getNextAction(observed or None)

        # update global stats
        for obs_name in cfg.get("track_other_actions_for", order):
            action = current_actions[obs_name]
            if 0 <= action < len(other_action_counts):
                other_action_counts[action] += 1

        last_actions = current_actions

    # collect regrets
    for name in order:
        regrets[name] = np.asarray(agents[name].cumul_regret, dtype=float)

    return regrets
"""

def parallel_runs(run_fn, cfg, n_jobs):
    runs = cfg["experiment"]["runs"]
    n_jobs = min(n_jobs, runs)

    if n_jobs <= 1:
        return [run_fn(cfg, i) for i in range(runs)]

    results = [None] * runs
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_fn, cfg, i): i for i in range(runs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def save_multi_results(all_runs, cfg):
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    first_run = next(iter(all_runs))
    agent_names = list(first_run["cumulated_regrets"].keys())

    metrics_map = {
        "cumulated_regrets": "regret",
        "time_averaged_rewards": "reward"
    }

    for name in agent_names:
        agent_dir = output_dir / name
        agent_dir.mkdir(parents=True, exist_ok=True)

        for dict_key, file_prefix in metrics_map.items():
            agent_runs = np.asarray([run[dict_key][name] for run in all_runs])

            np.save(agent_dir / f"{file_prefix}s.npy", agent_runs)
            np.save(agent_dir / f"mean_{file_prefix}.npy", agent_runs.mean(axis=0))
            np.save(agent_dir / f"std_{file_prefix}.npy", agent_runs.std(axis=0))

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Results saved to {output_dir} for agents: {', '.join(agent_names)}")

