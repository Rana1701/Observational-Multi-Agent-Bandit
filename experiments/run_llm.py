import numpy as np
import argparse
import os
import sys
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit

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


def batch_generate(model, prompts):
    """
    Génère plusieurs réponses LLM en parallèle.
    prompts = [prompt_run1, prompt_run2, ...]
    """

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

    for out in outputs:
        text = out.outputs[0].text.strip()

        if not text.endswith("}"):
            text += "}"

        responses.append(text)

    return responses



def init_runs(config, shared_model):

    exp = config["experiment"]
    runs = exp.get("runs", 20)
    states = []
    probs = config["environment"].get("probs")

    for run_id in range(runs):

        seed = run_seed(
            exp.get("seed"),
            run_id
        )

        np.random.seed(seed)

        bandit = BernoulliBandit(
            probs=probs
        )

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


        state = {
            "bandit": bandit,
            "agents": agents,
            "cfg_by_name": cfg_by_name,
            "order": exp.get(
                "order",
                [a["name"] for a in config["agents"]]
            ),
            "cumulative": {
                name:0
                for name in agents
            },
            "rewards_ts": {
                name:[]
                for name in agents
            },
            "other_action_counts":[0]*bandit.n_arms,
            "current_actions":{},
            "opt_curve":[],
            "opt_cum":0,
        }
        states.append(state)


    return states


def run_batched_experiment(config, model):

    exp = config["experiment"]
    horizon = exp["horizon"]
    states = init_runs(
        config,
        model
    )

    for t in range(horizon):
        prompts = []
        llm_refs = []

        # Construction des prompts des 20 runs

        for state in states:
            for name in state["order"]:
                agent = state["agents"][name]
                cfg = state["cfg_by_name"][name]
                if cfg.get("class") == "LLM":
                    
                    if t > 98:
                        print ("Parsing errors", agent.error)
                    
                    cfg["_other_action_counts"] = (
                        state["other_action_counts"]
                    )

                    prompt = build_llm_prompt(
                        cfg,
                        agent
                    )

                    prompts.append(prompt)

                    llm_refs.append(
                        (state,name,agent)
                    )

        #print("prompts : ")
        #print(prompts)
        # UN SEUL APPEL VLLM
        responses = batch_generate(
            model,
            prompts
        )
        print("responses : ")
        print(responses)

        # Application des réponses
        for (state,name,agent),response in zip(
            llm_refs,
            responses
        ):

            action = agent.getNextActionFromResponse(response)
            reward = agent.getReward(action)

            agent.history[str(action)]["pulls"] += 1
            agent.history[str(action)]["reward"] += reward
            agent.t += 1

            state["current_actions"][name] = action

        # Agents non LLM
        for state in states:

            for name in state["order"]:

                agent = state["agents"][name]

                cfg = state["cfg_by_name"][name]


                if cfg.get("class") != "LLM":

                    action = agent.getNextAction(None)

                    state["current_actions"][name] = action

        # Reward + courbes
        for state in states:
            for name in state["order"]:
                reward = state["agents"][name].reward
                state["cumulative"][name] += reward
                state["rewards_ts"][name].append(
                    state["cumulative"][name] /
                    (t+1)
                )

            state["opt_cum"] += np.max(
                state["bandit"].probs
            )

            state["opt_curve"].append(
                state["opt_cum"]/(t+1)
            )

            for name in exp.get(
                "track_other_actions_for",
                []
            ):

                action = state["current_actions"].get(
                    name
                )

                if action is not None:
                    state["other_action_counts"][action]+=1


            state["current_actions"] = {}


    # Format sortie identique à avant
    results=[]
    for state in states:
        out={}
        for name,data in state["rewards_ts"].items():
            out[name]=np.array(data)

        out["OPT"]=np.array(state["opt_curve"])
        results.append(out)


    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",required=True)

    args=parser.parse_args()

    config=load_config(
        args.config
    )

    if uses_llm(config):

        model_name=get_llm_model_name(
            config
        )
        print("Loading LLM once...")

        model=LLM(
            model=model_name,
            max_model_len=4096
        )

        results=run_batched_experiment(
            config,
            model
        )

    else:
        raise RuntimeError("This file is only for LLM experiments")

    save_multi_results(results,config)
    print(len(results))
    print(results[0]["llm_history"].shape)
    print(results[0]["ts"].shape)


if __name__=="__main__":

    main()