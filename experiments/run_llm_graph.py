import numpy as np
import argparse
import os
import sys
import random
import networkx as nx
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results, get_llm_model_name
from utils.prompt_builder import request_cot
from agents.llm import LLMAgent
from agents.ucb import UCB
from agents.ucbClique import UCBClique


def batch_generate(model,prompts):
    if not prompts:
        return []
    params=SamplingParams(temperature=0,max_tokens=1024,top_p=0.9,stop=["</Answer>"])
    outputs=model.generate(prompts,params)
    return [o.outputs[0].text.strip()+"</Answer>" for o in outputs]


def generate_graph(n,graph_type):
    if graph_type=="clique":
        G=nx.complete_graph(n)
    elif graph_type=="chain":
        G=nx.path_graph(n)
    elif graph_type=="loop":
        G=nx.cycle_graph(n)
    elif graph_type=="random":
        G=nx.erdos_renyi_graph(n,p=0.5)
    elif graph_type=="small-world":
        G=nx.barabasi_albert_graph(n,m=1)
    else:
        raise ValueError(graph_type)
    return {i:list(G.neighbors(i)) for i in range(n)}


def run_single_simulation(run_idx,cfg,model):
    exp,env=cfg["experiment"],cfg["environment"]

    seed=exp["seed"]+run_idx
    np.random.seed(seed)
    random.seed(seed)

    n_agents=exp["n_agents"]
    horizon=exp["horizon"]
    n_arms=env["n_arms"]

    probs=np.random.uniform(0.1,0.9,n_arms)

    results={}

    # Single UCB
    bandit=BernoulliBandit(probs=probs)
    agent=UCB(bandit)
    for _ in range(horizon):
        agent.getNextAction()
    results["Single UCB"]=np.array(agent.cumul_regret)

    # UCB clique
    bandit=BernoulliBandit(probs=probs)
    agent=UCBClique(bandit,n_agents)
    for _ in range(horizon):
        agent.getNextAction()
    results["UCB clique"]=np.array(agent.history_regret)

    # LLM graphs
    graph_types=["clique","chain","loop","random","small-world"]

    for graph_type in graph_types:
        neighbors=generate_graph(n_agents,graph_type)

        bandit=BernoulliBandit(probs=probs)

        agents=[
            LLMAgent(bandit,model=model)
            for _ in range(n_agents)
        ]

        prev_actions={
            i:[np.random.randint(n_arms) for _ in neighbors[i]]
            for i in range(n_agents)
        }

        regret=np.zeros(horizon)

        for t in range(horizon):

            prompts=[]
            refs=[]

            for i in range(n_agents):

                counts=[0]*n_arms

                for a in prev_actions[i]:
                    counts[a]+=1

                prompt=request_cot(
                    bandit,
                    agents[i].t,
                    agents[i].history,
                    counts,
                    horizon
                )

                prompts.append(prompt)
                refs.append(i)

            responses=batch_generate(model,prompts)

            actions={}

            for i,response in zip(refs,responses):
                actions[i]=agents[i].extract_reponse(response)

            total_regret=0

            for i,action in actions.items():

                r=agents[i].getReward(action)

                agents[i].history[str(action)]["pulls"]+=1
                agents[i].history[str(action)]["reward"]+=r
                agents[i].t+=1

                step=bandit.regret(action)

                if agents[i].cumul_regret:
                    agents[i].cumul_regret.append(
                        agents[i].cumul_regret[-1]+step
                    )
                else:
                    agents[i].cumul_regret.append(step)

                total_regret+=agents[i].cumul_regret[-1]

            for i in range(n_agents):
                prev_actions[i]=[
                    actions[j]
                    for j in neighbors[i]
                ]

            regret[t]=total_regret/n_agents

        results["LLM "+graph_type]=regret

    return {
        "cumulated_regrets":results,
        "time_averaged_rewards":{
            k:np.zeros_like(v)
            for k,v in results.items()
        }
    }


def run_experiment():

    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()

    cfg=load_config(args.config)

    print("Loading LLM once...")

    model=LLM(
        model=get_llm_model_name(cfg),
        max_model_len=4096,
        max_num_seqs=200,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True
    )

    runs=cfg["experiment"]["runs"]

    results=[]

    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        results.append(
            run_single_simulation(i,cfg,model)
        )

    save_multi_results(results,cfg)

    print("Terminé")


if __name__=="__main__":
    run_experiment()