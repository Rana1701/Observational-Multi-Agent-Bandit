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
    """Generate multiple LLM responses in one vLLM call."""
    if not prompts:
        return []

    params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_p=0.9,
        stop=["</Answer>"]
    )

    outputs = model.generate(
        prompts,
        params
    )

    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("</Answer>"):
            text += "</Answer>"
        responses.append(text)

    return responses


def init_run(cfg, run_idx, shared_model=None):
    """Initialize one experiment run."""
    exp = cfg["experiment"]
    seed = run_seed(exp.get("seed"), run_idx)

    random.seed(seed)
    np.random.seed(seed)

    bandit = build_bandit(
        cfg["environment"],
        seed
    )

    order = exp.get("order") or [
        a["name"] for a in cfg["agents"]
    ]

    agents = {}
    cfg_by_name = {}

    for a in cfg["agents"]:
        name = a["name"]
        cfg_by_name[name] = a

        agents[name] = create_agent(
            AGENTS[a["class"]],
            bandit,
            a.get("params"),
            shared_model=shared_model
        )

    best_reward = (
        np.max(bandit.probs)
        if hasattr(bandit, "probs")
        else cfg["environment"].get("best_mean", 0.9)
    )

    return {
        "bandit": bandit,
        "agents": agents,
        "cfg": cfg_by_name,
        "order": order,
        "history": {n: [] for n in order},
        "other_counts": [0] * bandit.n_arms,
        "reward": {n: 0.0 for n in order},
        "regret": {n: 0.0 for n in order},
        "rewards_ts": {n: [] for n in order},
        "regrets_ts": {n: [] for n in order},
        "best": best_reward
    }


def run_single_rep(task, shared_model=None):
    """Run one experiment replica (used for non LLM experiments)."""
    run_idx, cfg = task
    state = init_run(
        cfg,
        run_idx,
        shared_model
    )

    horizon = cfg["experiment"]["horizon"]

    for t in range(horizon):
        actions = {}

        for name in state["order"]:
            agent = state["agents"][name]
            agent_cfg = state["cfg"][name]

            if agent_cfg.get("class") == "LLM":
                agent_cfg["_other_action_counts"] = (
                    state["other_counts"].copy()
                )

                action = agent.getNextAction(
                    build_llm_prompt(
                        agent_cfg,
                        agent
                    )
                )

            elif agent_cfg.get("observes"):
                obs = [
                    state["history"][o][t]
                    for o in agent_cfg["observes"]
                    if len(state["history"][o]) > t
                ]

                action = agent.getNextAction(
                    obs or None
                )

            else:
                action = agent.getNextAction()

            actions[name] = action
            state["history"][name].append(action)

            reward = agent.reward

            state["reward"][name] += reward
            state["rewards_ts"][name].append(
                state["reward"][name] / (t + 1)
            )

            state["regret"][name] += (
                state["bandit"].regret(action)
                if hasattr(state["bandit"], "regret")
                else state["best"] - state["bandit"].probs[action]
            )

            state["regrets_ts"][name].append(
                state["regret"][name]
            )

        for name in cfg["experiment"].get("track_other_actions_for", []):
            if name in actions:
                state["other_counts"][actions[name]] += 1

    return format_result(state, cfg)

def init_batched_runs(cfg, model):
    """Initialize all runs sharing the same LLM."""
    states = []

    for run_idx in range(cfg["experiment"].get("runs", 20)):
        states.append(
            init_run(
                cfg,
                run_idx,
                model
            )
        )

    return states


def run_batched_llm_experiment(cfg, model):
    """Run multiple replicas with batched LLM inference."""
    states = init_batched_runs(
        cfg,
        model
    )

    horizon = cfg["experiment"]["horizon"]
    track = cfg["experiment"].get(
        "track_other_actions_for",
        []
    )

    for t in range(horizon):
        prompts = []
        refs = []

        # Collect all LLM requests from all runs
        for state in states:
            for name in state["order"]:
                agent_cfg = state["cfg"][name]

                if agent_cfg.get("class") == "LLM":
                    agent_cfg["_other_action_counts"] = (
                        state["other_counts"].copy()
                    )

                    prompts.append(
                        build_llm_prompt(
                            agent_cfg,
                            state["agents"][name]
                        )
                    )

                    refs.append(
                        (
                            state,
                            name
                        )
                    )

        # One batched generation for all runs
        responses = batch_generate(
            model,
            prompts
        )

        actions_by_state = {}

        # Apply LLM responses
        for (state, name), response in zip(
            refs,
            responses
        ):
            agent = state["agents"][name]
            action = (agent.getNextActionFromResponse(response))
            actions_by_state.setdefault(
                id(state),
                {}
            )[name] = action

        # Compute non LLM actions
        for state in states:
            actions = actions_by_state.setdefault(
                id(state),
                {}
            )

            for name in state["order"]:
                agent_cfg = state["cfg"][name]
                if agent_cfg.get("class") != "LLM":
                    agent = state["agents"][name]
                    if agent_cfg.get("observes"):
                        obs = [
                            state["history"][o][t]
                            for o in agent_cfg["observes"]
                            if len(state["history"][o]) > t
                        ]

                        action = agent.getNextAction(
                            obs or None
                        )
                    else:
                        action = agent.getNextAction()

                    actions[name] = action

        # Update all runs
        for state in states:
            actions = actions_by_state.get(id(state),{})
            for name, action in actions.items():
                agent = state["agents"][name]
                # LLM response already updates reward internally only partially
                # so update reward here for batched execution
                if state["cfg"][name].get("class") == "LLM":
                    reward = agent.getReward(action)
                    agent.history[str(action)]["pulls"] += 1
                    agent.history[str(action)]["reward"] += reward
                    agent.t += 1
                    if t<3 :
                        print(f"number of parsing errors : {agent.error}")
                    if t == horizon - 1 :
                        print(f"number of parsing errors : {agent.error}")
                else:
                    reward = agent.reward
                state["history"][name].append(action)
                state["reward"][name] += reward
                state["rewards_ts"][name].append(state["reward"][name] / (t + 1))

                state["regret"][name] += (state["bandit"].regret(action)
                    if hasattr(state["bandit"], "regret")
                    else state["best"] - state["bandit"].probs[action]
                )

                state["regrets_ts"][name].append(
                    state["regret"][name]
                )

            for name in track:
                if name in actions:
                    state["other_counts"][
                        actions[name]
                    ] += 1

    return [
        format_result(
            state,
            cfg
        )
        for state in states
    ]


def format_result(state, cfg):
    """Convert state to experiment output format."""
    out = {
        "time_averaged_rewards": {},
        "cumulated_regrets": {}
    }

    for name in state["order"]:
        out["time_averaged_rewards"][name] = np.array(
            state["rewards_ts"][name]
        )

        out["cumulated_regrets"][name] = np.array(
            state["regrets_ts"][name]
        )

    if cfg["experiment"].get("add_opt", False):
        horizon = cfg["experiment"]["horizon"]

        out["time_averaged_rewards"]["OPT"] = (
            np.ones(horizon) * state["best"]
        )

        out["cumulated_regrets"]["OPT"] = np.zeros(
            horizon
        )

    return out

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent experiment."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    exp = cfg["experiment"]
    runs = exp.get(
        "runs",
        20
    )

    results = []

    if uses_llm(cfg):
        print("Loading LLM once...")

        model = LLM(
            model=get_llm_model_name(cfg),
            max_model_len=4096,
            max_num_seqs=runs
        )

        results = run_batched_llm_experiment(
            cfg,
            model
        )

    else:
        print(
            f"Running {runs} replicas..."
        )

        tasks = [
            (i, cfg)
            for i in range(runs)
        ]

        with Pool(
            processes=exp.get("n_jobs", 4)
        ) as pool:
            results = pool.map(
                run_single_rep,
                tasks
            )

    save_multi_results(
        results,
        cfg
    )

    print(
        f"Saved {len(results)} runs"
    )


if __name__ == "__main__":
    main()