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
    df=pd.read_csv(filepath,sep=r"\s+",engine="python",header=None)
    header_idx=None
    for idx,row in df.iterrows():
        vals=[str(v).strip().lower() for v in row.values]
        if "arm" in vals:
            header_idx=idx
            break
    if header_idx is None:
        raise ValueError("Arm column not found")
    df=pd.read_csv(filepath,sep=r"\s+",engine="python",skiprows=header_idx)
    df.columns=[c.strip() for c in df.columns]
    arm_col=[c for c in df.columns if "arm" in c.lower()][0]
    df["Arm_num"]=df[arm_col].astype(str).str.strip().map({"A":0,"B":1,"0":0,"1":1})
    return df.dropna(subset=["Arm_num"])["Arm_num"].astype(int).values


def batch_generate(model,prompts):
    if not prompts:
        return []
    params=SamplingParams(temperature=0,max_tokens=1024,top_p=0.9,stop=["</Answer>"])
    outputs=model.generate(prompts,params)
    responses=[]
    for out in outputs:
        txt=out.outputs[0].text.strip()
        if not txt.endswith("</Answer>"):
            txt+="</Answer>"
        responses.append(txt)
    return responses


def run_llm_human_experiment():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=load_config(args.config)

    exp=config["experiment"]
    env=config["environment"]

    n_runs=exp.get("runs",20)
    horizon=exp.get("horizon",100)
    csv_folder=exp.get("input_dir","Target-UCB/Human bandit dataset/single_humans/")

    print("Loading LLM once...")

    model=LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        max_num_seqs=n_runs
    )

    all_regrets={}
    all_rewards={}

    for filename in config["humanFiles"]["files"]:
        filepath=os.path.join(csv_folder,filename)

        if not os.path.exists(filepath):
            print("Missing",filepath)
            continue

        label=filename.replace("_plays.csv","")
        human_actions=load_human_csv(filepath)

        ref_bandit=BernoulliBandit(probs=env["probs"])
        human_regret=np.cumsum([ref_bandit.regret(int(a)) for a in human_actions[:horizon]])

        all_regrets[f"Single {label}"]=human_regret
        all_rewards[f"Single {label}"]=np.zeros(horizon)

        states=[]

        for run in range(n_runs):
            bandit=BernoulliBandit(probs=env["probs"])
            agent=LLMAgent(bandit,model=model)
            states.append({"agent":agent,"prev_action":np.random.randint(env["n_arms"])})

        llm_runs=np.zeros((n_runs,horizon))

        for t in range(horizon):
            prompts=[]

            for state in states:
                agent=state["agent"]
                counts={i:0 for i in range(env["n_arms"])}
                counts[state["prev_action"]]=1
                prompts.append(request_cot(agent.bandit,agent.t,agent.history,counts,horizon))

            responses=batch_generate(model,prompts)

            for run,response in enumerate(responses):
                agent=states[run]["agent"]

                action=agent.extract_reponse(response)
                reward=agent.getReward(action)

                agent.history[str(action)]["pulls"]+=1
                agent.history[str(action)]["reward"]+=reward

                regret=agent.bandit.regret(action)

                if agent.cumul_regret:
                    agent.cumul_regret.append(agent.cumul_regret[-1]+regret)
                else:
                    agent.cumul_regret.append(regret)

                agent.t+=1
                llm_runs[run,t]=agent.cumul_regret[-1]
                states[run]["prev_action"]=int(human_actions[t])

        all_regrets[f"LLM ({label})"]=llm_runs.mean(axis=0)
        all_rewards[f"LLM ({label})"]=np.zeros(horizon)

    results=[{
        "cumulated_regrets":all_regrets,
        "time_averaged_rewards":all_rewards
    }]

    save_multi_results(results,config)
    print("Saved")


if __name__=="__main__":
    run_llm_human_experiment()