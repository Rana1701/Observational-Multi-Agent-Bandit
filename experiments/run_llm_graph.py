import argparse
import os
import sys
import random
import numpy as np
import networkx as nx
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.experiment_utils import load_config, save_multi_results, run_seed
from utils.prompt_builder import request_cot
from environnement.bernoulli_bandit import BernoulliBandit
from agents.llm import LLMAgent
from agents.ucb import UCB
from agents.ucbClique import UCBClique


def batch_generate(model, prompts):
    if not prompts: return []
    params = SamplingParams(temperature=0, max_tokens=1024, top_p=0.9, stop=["</Answer>"])
    outputs = model.generate(prompts, params)
    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("</Answer>"): text += "</Answer>"
        responses.append(text)
    return responses


def generate_graph(n, graph_type):
    if graph_type == "clique": G = nx.complete_graph(n)
    elif graph_type == "chain": G = nx.path_graph(n)
    elif graph_type == "loop": G = nx.cycle_graph(n)
    elif graph_type == "random": G = nx.erdos_renyi_graph(n, p=0.5)
    elif graph_type == "small-world": G = nx.barabasi_albert_graph(n, m=1)
    return {i: list(G.neighbors(i)) for i in range(n)}


def build_random_bandit(cfg, seed):
    np.random.seed(seed)
    probs = np.random.uniform(0.1, 0.9, cfg["environment"]["n_arms"])
    return BernoulliBandit(probs=probs)


def init_run(cfg, run_idx, model):
    seed = run_seed(cfg["experiment"]["seed"], run_idx)
    random.seed(seed)
    np.random.seed(seed)
    bandit = build_random_bandit(cfg, seed)
    return {"bandit": bandit, "agents": {}, "reward": {}, "regret": {}}


def run_experiment(cfg, model):
    graph_types = ["clique", "chain", "loop", "random", "small-world"]
    states = []

    for i in range(cfg["experiment"]["runs"]):
        state = init_run(cfg, i, model)
        for graph in graph_types:
            neighbors = generate_graph(cfg["experiment"]["n_agents"], graph)
            agents = [LLMAgent(state["bandit"], model=model) for _ in range(cfg["experiment"]["n_agents"])]
            state["agents"][graph] = {
                "agents": agents,
                "neighbors": neighbors,
                "prev_actions": {k: [0 for _ in neighbors[k]] for k in neighbors}
            }
            state["reward"][graph] = []
            state["regret"][graph] = []
        states.append(state)

    horizon = cfg["experiment"]["horizon"]

    for t in range(horizon):
        prompts, refs = [], []
        for state in states:
            for graph, data in state["agents"].items():
                for i, agent in enumerate(data["agents"]):
                    other_actions = data["prev_actions"][i]
                    counts = {str(a): 0 for a in range(state["bandit"].n_arms)}
                    for a in other_actions: counts[str(a)] += 1

                    prompt = request_cot(agent.bandit, agent.t, agent.history, counts, horizon)
                    prompts.append(prompt)
                    refs.append((state, graph, i))

        responses = batch_generate(model, prompts)
        actions = {}

        for ref, response in zip(refs, responses):
            state, graph, i = ref
            action = state["agents"][graph]["agents"][i].extract_reponse(response)
            actions.setdefault((id(state), graph), {})[i] = action

        for state in states:
            for graph, data in state["agents"].items():
                acts = actions[(id(state), graph)]
                regret, reward = 0, 0

                for i, action in acts.items():
                    agent = data["agents"][i]
                    r = agent.getReward(action)
                    agent.history[str(action)]["pulls"] += 1
                    agent.history[str(action)]["reward"] += r
                    agent.t += 1

                    step_regret = state["bandit"].regret(action)
                    if len(agent.cumul_regret):
                        agent.cumul_regret.append(agent.cumul_regret[-1] + step_regret)
                    else:
                        agent.cumul_regret.append(step_regret)

                    regret += agent.cumul_regret[-1]
                    reward += r

                data["prev_actions"] = {i: [acts[j] for j in data["neighbors"][i]] for i in acts}
                state["regret"][graph].append(regret / cfg["experiment"]["n_agents"])
                state["reward"][graph].append(reward / cfg["experiment"]["n_agents"])

    results = []
    for state in states:
        results.append({
            "cumulated_regrets": {"LLM " + g: np.array(state["regret"][g]) for g in graph_types},
            "time_averaged_rewards": {
                "LLM " + g: np.cumsum(state["reward"][g]) / np.arange(1, len(state["reward"][g]) + 1) for g in graph_types
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
        model="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=4096,
        max_num_seqs=200,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True
    )

    results = run_experiment(cfg, model)
    save_multi_results(results, cfg)
    print(f"Saved {len(results)} runs")


if __name__ == "__main__":
    main()
