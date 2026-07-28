import numpy as np
import pandas as pd
import argparse
import os
import sys
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results
from utils.prompt_builder import request_cot
from agents.llm import LLMAgent


def load_human_clique_columns(filepath, n_agents=4):
    df_raw = pd.read_csv(filepath, sep=None, engine="python", header=None)
    start_row = 2
    for idx, row in df_raw.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if "arm" in vals:
            start_row = idx + 1
            break

    agent_sequences = []
    for col_idx in range(n_agents):
        raw = df_raw.iloc[start_row:, col_idx]
        mapped = raw.astype(str).str.strip().str.upper().map({"B": 1, "A": 0, "1": 1, "0": 0})
        agent_sequences.append(mapped.dropna().astype(int).values)
    return agent_sequences


def batch_generate(model, prompts):
    params = SamplingParams(temperature=0, max_tokens=1024, top_p=0.9, stop=["</Answer>"])
    outputs = model.generate(prompts, params)
    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("</Answer>"): text += "</Answer>"
        responses.append(text)
    return responses


def run_llm_clique_experiment():
    parser = argparse.ArgumentParser(description="Run LLM Human Clique Experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    exp, env_cfg = config["experiment"], config["environment"]
    n_runs, horizon, n_agents = exp.get("runs", 20), exp.get("horizon", 100), 4
    target_probs = env_cfg.get("probs", [0.6, 0.4])
    csv_folder = exp.get("input_dir", "Target-UCB/Human bandit dataset/cliques/")
    human_files = config["humanFiles"]["files"]

    print("Loading LLM once...")
    model = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=4096, gpu_memory_utilization=0.95, enable_prefix_caching=True)

    all_plots_data, human_trajectories = {}, []
    filename = human_files[0]
    filepath = os.path.join(csv_folder, filename)

    if not os.path.exists(filepath):
        print("Missing file:", filepath)
        sys.exit(1)

    human_agents = load_human_clique_columns(filepath, n_agents)
    bandit_ref = BernoulliBandit(probs=target_probs)

    for i in range(n_agents):
        regrets = [bandit_ref.regret(int(a)) for a in human_agents[i][:horizon]]
        cum = np.cumsum(regrets)
        all_plots_data[f"Single Human Player {i+1}"] = cum.tolist()
        human_trajectories.append(cum)

    all_plots_data["Human Clique Average"] = np.mean(human_trajectories, axis=0).tolist()
    llm_runs = np.zeros((n_runs, horizon))

    for run in range(n_runs):
        bandits = [BernoulliBandit(probs=target_probs) for _ in range(n_agents)]
        agents = [LLMAgent(bandits[i], model=model) for i in range(n_agents)]
        prev_actions = {i: [0, 0, 0] for i in range(n_agents)}
        regrets = np.zeros((n_agents, horizon))

        for t in range(horizon):
            prompts = []
            for i in range(n_agents):
                counts = {"0": 0, "1": 0}
                for a in prev_actions[i]: counts[str(a)] += 1
                prompt = request_cot(agents[i].bandit, agents[i].t, agents[i].history, counts, horizon)
                prompts.append(prompt)

            responses = batch_generate(model, prompts)
            current_actions = {}

            for i, response in enumerate(responses):
                action = agents[i].extract_reponse(response)
                reward = agents[i].getReward(action)
                agents[i].history[str(action)]["pulls"] += 1
                agents[i].history[str(action)]["reward"] += reward
                agents[i].t += 1

                step_regret = agents[i].bandit.regret(action)
                if len(agents[i].cumul_regret) > 0:
                    agents[i].cumul_regret.append(agents[i].cumul_regret[-1] + step_regret)
                else:
                    agents[i].cumul_regret.append(step_regret)

                regrets[i, t] = agents[i].cumul_regret[-1]
                current_actions[i] = action

            for i in range(n_agents):
                prev_actions[i] = [current_actions[j] for j in range(n_agents) if j != i]

        llm_runs[run] = np.mean(regrets, axis=0)

    all_plots_data["LLM Clique Average"] = np.mean(llm_runs, axis=0).tolist()
    payload = {"LLM_clique_vs_Human_clique": {k: v for k, v in all_plots_data.items()}}
    save_multi_results(payload.values(), config)
    print(f"Saved {n_runs} runs")


if __name__ == "__main__":
    run_llm_clique_experiment()
