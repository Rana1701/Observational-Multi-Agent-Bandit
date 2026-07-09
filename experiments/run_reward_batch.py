import numpy as np
import argparse
import os
import sys
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.experiment_utils import (
    load_config,
    save_multi_results,
    create_agent,
    AGENTS,
    build_llm_prompt,
    uses_llm,
    get_llm_model_name,
    run_seed,
)
from environnement.bernoulli_bandit import BernoulliBandit

def batch_generate(model, prompts):
    if not prompts:
        return []

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        top_p=0.9,
        stop=["}"],
    )

    outputs = model.generate(
        prompts,
        sampling_params
    )

    responses = []

    for out in outputs:
        text = out.outputs[0].text.strip()

        if not text.endswith("}"):
            text += "}"
        responses.append(text)
    return responses

def init_single_runs(config, model):

    exp = config["experiment"]
    runs = exp.get("runs", 20)
    probs = config["environment"].get("probs")
    states = []

    for run_id in range(runs):
        seed = run_seed(
            exp.get("seed"),
            run_id)
        np.random.seed(seed)

        bandit = BernoulliBandit(probs=probs)

        agents = {}
        cfg_by_name = {}

        for a in config["agents"]:
            name = a["name"]
            cfg_by_name[name] = a
            agents[name] = create_agent(
                AGENTS[a["class"]],
                bandit,
                a.get("params"),
                shared_model=model)

        states.append(
            {
                "bandit": bandit,
                "agents": agents,
                "cfg_by_name": cfg_by_name,
                "order": exp.get(
                    "order",
                    [a["name"] for a in config["agents"]]
                ),
                "cumulative":
                    {name:0.0 for name in agents},
                "rewards_ts":
                    {name:[] for name in agents},
                "current_actions": {},
                "other_action_counts":
                    [0]*bandit.n_arms,
                "opt_cum":0.0,
                "opt_curve":[],
            }
        )

    return states

def run_batched_experiment(config, model):
    exp = config["experiment"]
    horizon = exp["horizon"]
    interaction = exp.get(
        "interaction",
        "sequential"
    )
    states = init_single_runs(
        config,
        model
    )
    for t in range(horizon):
        prompts = []
        llm_refs = []
        for state in states:
            state["current_actions"] = {}
            for name in state["order"]:
                agent = state["agents"][name]
                cfg = state["cfg_by_name"][name]
                # AGENT LLM
                if cfg.get("class") == "LLM":
                    cfg["_other_action_counts"] = (
                        state["other_action_counts"]
                    )
                    prompt = build_llm_prompt(
                        cfg,
                        agent
                    )
                    prompts.append(prompt)
                    llm_refs.append(
                        (state,name,agent))
                # AGENT CLASSIQUE
                else:
                    if interaction == "sequential":
                        observed = [
                            state["current_actions"][o]
                            for o in cfg.get("observes", [])
                            if o in state["current_actions"]
                        ]

                        action = agent.getNextAction(
                            observed or None
                        )

                    else:

                        action = agent.getNextAction(
                            None
                        )
                    state["current_actions"][name] = action

        # ETAPE 2 : UN SEUL APPEL VLLM
        responses = batch_generate(
            model,
            prompts
        )
    
        # ETAPE 3 : APPLICATION DES REPONSES LLM
        for (state, name, agent), response in zip(
            llm_refs,
            responses
        ):
            action = agent.getNextActionFromResponse(response)
            state["current_actions"][name] = action

        # ETAPE 4 : CALCUL RECOMPENSES
        for state in states:
            for name in state["order"]:
                agent = state["agents"][name]
                reward = agent.reward
                state["cumulative"][name] += reward
                state["rewards_ts"][name].append(
                    state["cumulative"][name]
                    /
                    (t+1)
                )

            state["opt_cum"] += np.max(
                state["bandit"].probs
            )

            state["opt_curve"].append(
                state["opt_cum"]
                /
                (t+1)
            )
            # ACTIONS OBSERVED
            for obs_name in exp.get(
                "track_other_actions_for",
                state["order"]
            ):

                action = state["current_actions"].get(
                    obs_name
                )
                if action is not None:
                    if (
                        0 <= action <
                        len(state["other_action_counts"])
                    ):
                        state["other_action_counts"][action] += 1

    # FORMAT RESULTATS
    results = []

    for state in states:
        out = {}
        for name, values in state["rewards_ts"].items():
            out[name] = np.array(values)
        out["OPT"] = np.array(state["opt_curve"])
        results.append(out)

    return results

def main():

    parser = argparse.ArgumentParser(
        description="Run LLM bandit experiments with vLLM batching"
    )
    parser.add_argument(
        "--config",
        required=True
    )
    args = parser.parse_args()
    config = load_config(
        args.config
    )
    if not uses_llm(config):
        raise RuntimeError(
            "This script is only for LLM experiments"
        )
    model_name = get_llm_model_name(
        config
    )
    print(
        f"Loading LLM model: {model_name}"
    )
    model = LLM(
        model=model_name,
        max_model_len=4096
    )
    print(
        "Running batched experiment..."
    )
    results = run_batched_experiment(
        config,
        model
    )
    save_multi_results(
        results,
        config
    )
    print(
        "Experiment finished."
    )

if __name__ == "__main__":

    main()