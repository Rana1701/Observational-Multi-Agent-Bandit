import argparse
import os
import sys
import random
import numpy as np
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.experiment_utils import load_config, run_seed, build_bandit, save_multi_results, get_llm_model_name
from agents.llmClique import LLMClique


def batch_generate(model, prompts):
    if not prompts:
        return []
    params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_p=0.9,
        stop=["</Answer>"]
    )
    outputs = model.generate(prompts, params)
    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("</Answer>"):
            text += "</Answer>"
        responses.append(text)
    return responses


def init_run(cfg, run_idx, model):
    seed = run_seed(cfg["experiment"].get("seed"), run_idx)
    random.seed(seed)
    np.random.seed(seed)

    bandit = build_bandit(cfg["environment"], seed)
    agent_cfg = next(
    a for a in cfg["agents"]
    if a["class"] == "LLMClique")

    # Créer une copie pour éviter de modifier la configuration globale
    clique_params = agent_cfg["params"].copy()
    # Supprimer la clé 'model' issue du YAML si elle existe
    clique_params.pop("model", None)

    clique = LLMClique(
        bandit,
        model=model,  # Utilise l'instance vLLM unique optimisée
        **clique_params
    )

    return {
        "bandit": bandit,
        "agent": clique,
        "reward": [],
        "regret": []
    }


def run_experiment(cfg, model):
    states = [
        init_run(cfg, i, model)
        for i in range(cfg["experiment"]["runs"])
    ]

    horizon = cfg["experiment"]["horizon"]

    for t in range(horizon):
        prompts = []
        refs = []

        for state in states:
            clique = state["agent"]

            for i in range(clique.clique_size):
                prompts.append(clique.getPrompt(i))
                refs.append((state, i))

        responses = batch_generate(model, prompts)

        actions_by_state = {}

        for (state, i), response in zip(refs, responses):
            action = state["agent"].agents[i].extract_reponse(response)
            actions_by_state.setdefault(id(state), {})[i] = action

        for state in states:
            actions = actions_by_state[id(state)]
            clique = state["agent"]

            clique.updateActions(actions)

            state["reward"].append(clique.reward)
            state["regret"].append(clique.history_regret[-1])

    results = []

    for state in states:
        rewards = np.array(state["reward"])

        results.append({
            "time_averaged_rewards": {
                "LLM clique": np.cumsum(rewards) / np.arange(1, len(rewards)+1)
            },
            "cumulated_regrets": {
                "LLM clique": np.array(state["regret"])
            }
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loading LLM once...")

    model = LLM(
        model=get_llm_model_name(cfg),
        max_model_len=4096,
        max_num_seqs=200,
        gpu_memory_utilization=0.95,
        enable_prefix_caching = True
    )

    results = run_experiment(cfg, model)

    save_multi_results(results, cfg)

    print(f"Saved {len(results)} runs")


if __name__ == "__main__":
    main()