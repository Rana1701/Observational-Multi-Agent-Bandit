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


def load_human_clique_columns(filepath,n_agents=4):
    df_raw=pd.read_csv(filepath,sep=None,engine="python",header=None)
    start_row=2
    for idx,row in df_raw.iterrows():
        vals=[str(v).strip().lower() for v in row.values if pd.notna(v)]
        if "arm" in vals:
            start_row=idx+1
            break
    agents=[]
    for col in range(n_agents):
        raw=df_raw.iloc[start_row:,col]
        mapped=raw.astype(str).str.strip().str.upper().map({"B":1,"A":0,"1":1,"0":0})
        agents.append(mapped.dropna().astype(int).values)
    return agents


def batch_generate(model,prompts):
    if not prompts:return []
    params=SamplingParams(temperature=0,max_tokens=1024,top_p=0.9,stop=["</Answer>"])
    outputs=model.generate(prompts,params)
    return [o.outputs[0].text.strip()+("</Answer>" if not o.outputs[0].text.strip().endswith("</Answer>") else "") for o in outputs]


def run_llm_clique_experiment():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=load_config(args.config)

    exp,env_cfg=config["experiment"],config["environment"]
    n_runs=exp.get("runs",20)
    horizon=exp.get("horizon",100)
    n_agents=4
    target_probs=env_cfg.get("probs",[0.6,0.4])
    csv_folder=exp.get("input_dir","Target-UCB/Human bandit dataset/cliques/")
    human_file=config["humanFiles"]["files"][0]

    print("Loading LLM once...")
    model=LLM(model="Qwen/Qwen2.5-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.95,enable_prefix_caching=True,max_num_seqs=100)

    all_plots_data={}
    path=os.path.join(csv_folder,human_file)

    if not os.path.exists(path):
        print("Missing file:",path)
        sys.exit(1)

    humans=load_human_clique_columns(path,n_agents)
    ref_bandit=BernoulliBandit(probs=target_probs)
    human_reg=[]

    for i in range(n_agents):
        r=np.cumsum([ref_bandit.regret(int(a)) for a in humans[i][:horizon]])
        all_plots_data[f"Single Human Player {i+1}"]=r.tolist()
        human_reg.append(r)

    all_plots_data["Human Clique Average"]=np.mean(human_reg,axis=0).tolist()


    states=[]

    for run in range(n_runs):
        bandits=[BernoulliBandit(probs=target_probs) for _ in range(n_agents)]
        agents=[LLMAgent(bandits[i],model=model) for i in range(n_agents)]

        states.append({
            "agents":agents,
            "prev_actions":{i:[0,0,0] for i in range(n_agents)},
            "regret":np.zeros((n_agents,horizon))
        })


    for t in range(horizon):

        prompts=[]
        refs=[]

        for run,state in enumerate(states):
            for i,agent in enumerate(state["agents"]):

                counts={j:0 for j in range(agent.bandit.n_arms)}

                for a in state["prev_actions"][i]:
                    counts[a]+=1

                prompt=request_cot(agent.bandit,agent.t,agent.history,counts,horizon)

                prompts.append(prompt)
                refs.append((run,i))


        responses=batch_generate(model,prompts)

        actions={}

        for ref,response in zip(refs,responses):
            run,i=ref
            action=states[run]["agents"][i].extract_reponse(response)
            actions[(run,i)]=action


        for run,state in enumerate(states):

            current={}

            for i,agent in enumerate(state["agents"]):

                action=actions[(run,i)]
                reward=agent.getReward(action)

                agent.history[str(action)]["pulls"]+=1
                agent.history[str(action)]["reward"]+=reward
                agent.t+=1

                regret=agent.bandit.regret(action)

                if len(agent.cumul_regret):
                    agent.cumul_regret.append(agent.cumul_regret[-1]+regret)
                else:
                    agent.cumul_regret.append(regret)

                state["regret"][i,t]=agent.cumul_regret[-1]
                current[i]=action


            for i in range(n_agents):
                state["prev_actions"][i]=[current[j] for j in range(n_agents) if j!=i]


    llm_runs=np.zeros((n_runs,horizon))

    for r,state in enumerate(states):
        llm_runs[r]=np.mean(state["regret"],axis=0)

    all_plots_data["LLM Clique Average"]=np.mean(llm_runs,axis=0).tolist()

    payload=[{
        "cumulated_regrets":{
            k:np.array(v) for k,v in all_plots_data.items()
        },
        "time_averaged_rewards":{
            k:np.zeros(len(v)) for k,v in all_plots_data.items()
        }
    }]

    save_multi_results(payload,config)
    print("Saved",n_runs,"runs")


if __name__=="__main__":
    run_llm_clique_experiment()