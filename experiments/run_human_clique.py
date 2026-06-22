import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environnement.bernoulli_bandit import BernoulliBandit
from utils.common import load_config, save_multi_results
from agents.tucb import TUCB 

def load_human_clique_columns(filepath, n_agents=4):
    """
    Charge le fichier unique contenant les colonnes d'actions juxtaposées (Arm, Arm, Arm, Arm).
    """
    df_raw = pd.read_csv(filepath, sep=None, engine='python', header=None)
    
    # Finding the line 'Arm'
    start_row = 2 
    for idx, row in df_raw.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if 'arm' in vals:
            start_row = idx + 1
            break
            
    agent_sequences = []
    #  Parcourir les colonnes de 0 à 3 de manière linéaire
    for col_idx in range(n_agents):
        raw_series = df_raw.iloc[start_row:, col_idx]
        
        # Encoding B as 1, A as 0
        mapped_series = raw_series.astype(str).str.strip().str.upper().map({'B': 1, 'A': 0, '1': 1, '0': 0})
        
        clean_array = mapped_series.dropna().astype(int).values
        agent_sequences.append(clean_array)
        
    return agent_sequences

def run_clique_experiment():
    parser = argparse.ArgumentParser(description="Run Figure 6 Clique Experiment.")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    
    exp = config['experiment']
    env_cfg = config['environment']
    
    n_runs = exp.get("runs", 200)
    horizon = exp.get("horizon", 100)
    n_agents = 4  
    
    target_probs = env_cfg.get("probs", [0.6, 0.4])
    
    csv_folder = exp.get('input_dir', 'Target-UCB/Human bandit dataset/cliques/')
    clique_files = config['humanFiles'].get("files", ["human_clique_1_plays.csv"])
    
    all_plots_data = {}
    human_trajectories = []
    
    # Instanciation du bandit témoin
    bandit_ref = BernoulliBandit(probs=target_probs)
    
    if clique_files:
        filename = clique_files[0] 
        filepath = os.path.join(csv_folder, filename)
        
        if os.path.exists(filepath):
            
            human_agent_arms = load_human_clique_columns(filepath, n_agents=n_agents)
            print(len(human_agent_arms))
            for i,a in enumerate(human_agent_arms):
                print(i, len(a))
            # Traiter et stocker chaque humain individuellement
            for i in range(n_agents):
                arms = human_agent_arms[i]
                step_regrets = [bandit_ref.regret(int(arm)) for arm in arms[:horizon]]
                cum_regret = np.cumsum(step_regrets)
                
                all_plots_data[f"Single Human Player {i+1}"] = cum_regret.tolist()
                human_trajectories.append(cum_regret)
                
            # Calculer et ajouter la moyenne de la clique humaine
            if human_trajectories:
                all_plots_data["Human Clique Average"] = np.mean(human_trajectories, axis=0).tolist()
        else:
            print(f"[ERREUR] Le fichier de clique unique est introuvable au chemin : {filepath}")
            sys.exit(1)

    # Simulation de la clique d'agents Target-UCB
    target_ucb_runs = np.zeros((n_runs, horizon))
    
    for run in range(n_runs):
        clique_bandits = [BernoulliBandit(probs=target_probs) for _ in range(n_agents)]
        agents = [TUCB(bandit=clique_bandits[i], nbr_neighbours=3, delta=0.2) for i in range(n_agents)]
        
        # Initialisation à t=0 pour satisfaire len(prev_actions) == 3 de chaque agent
        prev_actions_matrix = {i: [0, 0, 0] for i in range(n_agents)}
        run_step_regrets = np.zeros((n_agents, horizon))
        
        for t in range(horizon):
            current_actions = {}
            
            # Sélection des actions par l'ensemble des agents
            for i in range(n_agents):
                action = agents[i].getNextAction(prev_actions_matrix[i])
                current_actions[i] = action
                run_step_regrets[i, t] = agents[i].cumul_regret[-1]
            
            # Diffusion des observations réciproques pour l'étape suivante t+1 (j != i)
            for i in range(n_agents):
                neighbors_actions = [current_actions[j] for j in range(n_agents) if j != i]
                prev_actions_matrix[i] = neighbors_actions
                
        # Stockage de la trajectoire moyenne de la clique pour ce run
        target_ucb_runs[run, :] = np.mean(run_step_regrets, axis=0)
        
    
    all_plots_data["Target-UCB Clique Average"] = np.mean(target_ucb_runs, axis=0).tolist()

    payload = {
        "Tucb_clique_VS_Human_clique": {
            label: regrets for label, regrets in all_plots_data.items()
        }
    }
    save_multi_results(payload.values(), config)


if __name__ == "__main__":
    run_clique_experiment()
