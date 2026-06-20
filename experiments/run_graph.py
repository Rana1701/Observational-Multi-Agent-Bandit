import numpy as np
import argparse
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.tucb_graph import TUCBGraph  # Ton agent TUCB adapté aux graphes
from utils.graph_gen import get_neighbors   # La fonction de génération de graphe
from environnement.bernoulli_bandit import BernoulliBandit

import yaml
from utils.common import load_config, save_multi_results

def run_experiment():

    parser = argparse.ArgumentParser(
        description="Run a multi-agent observational bandit experiment from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    
    exp = config['experiment']
    env_cfg = config['environment']
    
    # 1. Fix the Seed for reproducibility
    if 'seed' in exp and exp['seed'] is not None:
        np.random.seed(exp['seed'])
        random.seed(exp['seed'])
    
    # 2. Define structural properties from config
    n_runs = exp.get('runs', 1)
    # Prefer 'horizon' if defined, fallback to 'n_episodes'
    horizon = exp.get('horizon', exp.get('n_episodes', 1000))
    n_agents = exp['n_agents']
    graph_type = exp['graph_type']
    
    # Generate the network adjacency mapping
    adj = get_neighbors(n=n_agents, graph_type=graph_type)
    
    # Matrix to store average step regrets for each run over the time horizon
    # Shape: (n_runs, horizon)
    run_regrets = np.zeros((n_runs, horizon))
    
    # 3. Loop over the independent execution runs
    for run in range(n_runs):
        # Instantiate a clean environment per run
        bandit = BernoulliBandit(n_arms=env_cfg['n_arms'], delta=env_cfg['delta'], best_mean=env_cfg['best_mean'])
        
        # Instantiate clean agents per run
        agents = [TUCBGraph(bandit, i, adj[i]) for i in range(n_agents)]
        
        all_prev_actions = {i: None for i in range(n_agents)}
        
        # Run the multi-agent execution loop across the horizon steps
        for step in range(horizon):
            current_actions = {}
            step_regrets = []
            
            for i, agent in enumerate(agents):
                # Agent chooses an action based on observational network context
                action = agent.getNextAction(all_prev_actions)
                current_actions[i] = action
                step_regrets.append(agent.cumul_regret[-1])
                
            all_prev_actions = current_actions
            # Calculate the average regret across all agents at this specific time step
            run_regrets[run, step] = np.mean(step_regrets)
            
    # 4. Average the final trajectories across all historical simulation runs
    final_avg_regrets = np.mean(run_regrets, axis=0)
    
    # 5. Build standard dictionary structure
    all_regrets = {
        graph_type: {
            "Average": final_avg_regrets.tolist()  # Valid native list datatype
        }
    }
    
    # Bypass structural line 270 AttributeError by passing only the internal dictionary
    save_multi_results(all_regrets.values(), config)


if __name__ == "__main__" :
    run_experiment()
