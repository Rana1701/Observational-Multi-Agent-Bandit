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


def load_human_csv(filepath):
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
    header_idx = None

    for idx, row in df.iterrows():
        row_str = [str(val).strip().lower() for val in row.values]
        if 'arm' in row_str:
            header_idx = idx
            break

    if header_idx is None: raise KeyError("Arm column not found")
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    arm_col = [c for c in df.columns if 'arm' in c.lower()][0]

    df['Arm_num'] = df[arm_col].astype(str).str.strip().map({'A': 0, 'B': 1, '0': 0, '1': 1})
    df = df.dropna(subset=['Arm_num'])
    return df['Arm_num'].astype(int).values


def batch_generate(model, prompts):
    params = SamplingParams(temperature=0, max_tokens=1024, top_p=0.9, stop=["</Answer>"])
    outputs = model.generate(prompts, params)
    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("</Answer>"): text += "</Answer>"
        responses.append(text)
    return responses


def run_llm_human_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    print("Loading LLM once...")
    model = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=4096, gpu_memory_utilization=0.95)

    n_runs, horizon = 20, 100
    env_cfg = config["environment"]
    csv_folder = config['experiment'].get('intput_dir', 'Target-UCB/Human bandit dataset/single_humans/')
    human_files = config['humanFiles']['files']
    all_plots_data = {}

    for filename in human_files:
        filepath = os.path.join(csv_folder, filename)
        if not os.path.exists(filepath):
            print(f"Missing {filepath}")
            continue

        human_label = filename.replace("_plays.csv", "")
        human_arms = load_human_csv(filepath)

        bandit_ref = BernoulliBandit(
            n_arms=env_cfg["n_arms"], delta=env_cfg["delta"], 
            best_mean=env_cfg["best_mean"], probs=env_cfg["probs"]
        )

        human_regret = [bandit_ref.regret(int(a)) for a in human_arms[:horizon]]
        all_plots_data[f"Single {human_label}"] = np.cumsum(human_regret).tolist()

        llm_runs = np.zeros((n_runs, horizon))

        for run in range(n_runs):
            bandit = BernoulliBandit(
                n_arms=env_cfg["n_arms"], delta=env_cfg["delta"], 
                best_mean=env_cfg["best_mean"], probs=env_cfg["probs"]
            )
            agent = LLMAgent(bandit, model=model)

            for t in range(horizon):
                previous_action = int(human_arms[t])
                other_actions = {str(i): 0 for i in range(env_cfg["n_arms"])}
                other_actions[str(previous_action)] = 1

                prompt = request_cot(bandit, agent.t, agent.history, other_actions, horizon)
                response = batch_generate(model, [prompt])[0]
                action = agent.extract_reponse(response)
                reward = agent.getReward(action)

                agent.history[str(action)]["pulls"] += 1
                agent.history[str(action)]["reward"] += reward
                agent.t += 1

                regret = bandit.regret(action)
                if len(agent.cumul_regret) > 0:
                    agent.cumul_regret.append(agent.cumul_regret[-1] + regret)
                else:
                    agent.cumul_regret.append(regret)

                llm_runs[run, t] = agent.cumul_regret[-1]

        avg = np.mean(llm_runs, axis=0)
        all_plots_data[f"LLM ({human_label})"] = avg.tolist()

    payload = {"Human_vs_LLM": {k: v for k, v in all_plots_data.items()}}
    save_multi_results(payload.values(), config)


if __name__ == "__main__":
    run_llm_human_experiment()
