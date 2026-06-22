import numpy as np
import argparse
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environnement.bernoulli_bandit import BernoulliBandit
from utils.common import load_config, save_multi_results
from agents.tucb import TUCB 
from agents.ucbClique import UCBClique
from agents.ucb import UCB

def run_simulation():
    parser = argparse.ArgumentParser(description="Run Figure 3 Simulation Experiments.")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    
    exp = config['experiment']
    
    n_runs = exp.get("runs", 2000)
    horizon = exp.get("horizon", 1000)
    clique_size = exp.get("n_agents", 11)
    
    if 'seed' in exp:
        np.random.seed(exp['seed'])
        random.seed(exp['seed'])

    target_probs = [0.5, 0.4]
    
    single_ucb_history = np.zeros((n_runs, horizon))
    ucb_clique_history = np.zeros((n_runs, horizon))
    tucb_clique_history = np.zeros((n_runs, horizon))
    
    print(f"Lancement de la simulation : {n_runs} runs, clique de {clique_size} agents.")

    for run in range(n_runs):
        # 1. RUN SINGLE UCB
        bandit_single = BernoulliBandit(probs=target_probs)
        agent_single = UCB(bandit_single)
        for t in range(horizon):
            agent_single.getNextAction()
        single_ucb_history[run, :] = agent_single.cumul_regret

        # 2. RUN UCB CLIQUE
        bandit_u_clique = BernoulliBandit(probs=target_probs)
        agent_u_clique = UCBClique(bandit_u_clique, clique_size)
        for t in range(horizon):
            agent_u_clique.getNextAction()
        ucb_clique_history[run, :] = agent_u_clique.history_regret

        # 3. RUN TARGET-UCB CLIQUE
        clique_bandits = [BernoulliBandit(probs=target_probs) for _ in range(clique_size)]
        # nbr_neighbours = clique_size - 1 (Chaque agent observe ses 10 pairs)
        tucb_agents = [TUCB(bandit=clique_bandits[i], nbr_neighbours=clique_size-1, delta=0.1) for i in range(clique_size)]
        
        # Initialisation à t=0 pour satisfaire la taille attendue des voisins
        prev_actions_matrix = {i: [0] * (clique_size - 1) for i in range(clique_size)}
        run_step_regrets = np.zeros((clique_size, horizon))
        
        for t in range(horizon):
            current_actions = {}
            for i in range(clique_size):
                action = tucb_agents[i].getNextAction(prev_actions_matrix[i])
                current_actions[i] = action
                run_step_regrets[i, t] = tucb_agents[i].cumul_regret[-1]
            
            # Distribution des observations réciproques (j != i) pour le coup suivant
            for i in range(clique_size):
                neighbors_actions = [current_actions[j] for j in range(clique_size) if j != i]
                prev_actions_matrix[i] = neighbors_actions
                
        # Calculer le regret cumulé moyen par nœud de la clique Target-UCB pour ce run
        tucb_clique_history[run, :] = np.mean(run_step_regrets, axis=0)

    # Moyennage des courbes sur l'ensemble des runs pour stabiliser la variance
    all_plots_data = {
        "Single UCB": np.mean(single_ucb_history, axis=0).tolist(),
        "UCB clique": np.mean(ucb_clique_history, axis=0).tolist(),
        "Target-UCB clique": np.mean(tucb_clique_history, axis=0).tolist()
    }
    
    # Encapsulation finale pour forcer l'affichage propre dans votre framework commun
    payload = {
        "TUCB_vs_UCB_Clique": {
            label: regrets for label, regrets in all_plots_data.items()
        }
    }
    
    print("Simulation terminée. Sauvegarde des courbes en cours...")
    save_multi_results(payload.values(), config)


if __name__ == "__main__":
    run_simulation()
