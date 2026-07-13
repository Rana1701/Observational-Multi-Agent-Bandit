import numpy as np
import argparse
import os
import sys
import random
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results
from utils.graph_gen import get_neighbors
from agents.tucb import TUCB 
from agents.ucbClique import UCBClique
from agents.ucb import UCB

# --- FONCTION DE SÉLECTION POUR UN SEUL RUN INDÉPENDANT ---
def run_single_simulation_worker(args_tuple):
    """
    Exécute un seul run pour toutes les baselines et structures de graphes.
    Cette fonction est distribuée sur les différents cœurs du processeur.
    """
    run_idx, horizon, n_agents, n_arms, target_probs, delta, graph_types, seed = args_tuple
    
    # Initialiser une graine unique et distincte par processus pour éviter les doublons
    run_seed = seed + run_idx if seed is not None else None
    if run_seed is not None:
        np.random.seed(run_seed)
        random.seed(run_seed)

    local_results = {}

    # 1. Baseline : Single UCB
    bandit_single = BernoulliBandit(probs=target_probs)
    agent_single = UCB(bandit_single)
    for t in range(horizon):
        agent_single.getNextAction()
    local_results["Single UCB"] = agent_single.cumul_regret

    # 2. Baseline : UCB Clique
    bandit_u_clique = BernoulliBandit(probs=target_probs)
    agent_u_clique = UCBClique(bandit_u_clique, n_agents)
    for t in range(horizon):
        agent_u_clique.getNextAction()
    local_results["UCB clique"] = agent_u_clique.history_regret

    # 3. Structures de Graphes Target-UCB
    for g_type in graph_types:
        try:
            adj = get_neighbors(n=n_agents, graph_type=g_type)
        except Exception:
            mapped_name = g_type.replace("_", "-").title() if g_type == "small_world" else g_type.title()
            adj = get_neighbors(n=n_agents, graph_type=mapped_name)
            
        clique_bandits = [BernoulliBandit(probs=target_probs) for _ in range(n_agents)]
        agents = [TUCB(bandit=clique_bandits[i], nbr_neighbours=len(adj[i]), delta=delta) for i in range(n_agents)]
        
        prev_actions_matrix = {
            i: [int(np.random.choice(n_arms)) for _ in range(len(adj[i]))] 
            for i in range(n_agents)
        }
        run_step_regrets = np.zeros((n_agents, horizon))
        
        for t in range(horizon):
            current_actions = {}
            for i in range(n_agents):
                action = agents[i].getNextAction(prev_actions_matrix[i])
                current_actions[i] = action
                run_step_regrets[i, t] = agents[i].cumul_regret[-1]
            
            for i in range(n_agents):
                neighbors_actions = [current_actions[neighbor] for neighbor in adj[i]]
                prev_actions_matrix[i] = neighbors_actions
                
        display_name = g_type.replace("_", "-").title()
        local_results[display_name] = np.mean(run_step_regrets, axis=0)

    return local_results


def run_experiment():
    parser = argparse.ArgumentParser(description="Run Figure 4 Parallelized Graph Structure Experiments.")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    
    exp = config['experiment']
    env_cfg = config['environment']
    
    n_runs = exp.get("runs", 2000)
    horizon = exp.get("horizon", 1000)
    n_agents = exp.get("n_agents", 20)
    n_arms = env_cfg.get("n_arms", 10)
    n_jobs = exp.get("n_jobs", 4)  
    base_seed = exp.get("seed", 42)

    best_mean = env_cfg.get("best_mean", 0.5)
    delta = env_cfg.get("delta", 0.1)
    target_probs = [best_mean] + [best_mean - delta] * (n_arms - 1)
    
    graph_types = ["clique", "chain", "loop", "random", "small_world"]
    
    # Préparer les arguments pour chaque run parallèle
    worker_tasks = [
        (run_idx, horizon, n_agents, n_arms, target_probs, delta, graph_types, base_seed)
        for run_idx in range(n_runs)
    ]

    print(f"Lancement de la simulation parallélisée sur {n_jobs} cœurs...")
    print(f"Progression : 0/{n_runs} runs calculés...", end="\r")

    # Ouvrir le groupe de processus parallèles (Pool)
    all_runs_data = []
    with Pool(processes=n_jobs) as pool:
        # imap_unordered est plus performant et consomme moins de mémoire vive
        for idx, result_dict in enumerate(pool.imap_unordered(run_single_simulation_worker, worker_tasks)):
            all_runs_data.append(result_dict)
            if (idx + 1) % 50 == 0 or (idx + 1) == n_runs:
                print(f"Progression : {idx + 1}/{n_runs} runs calculés...", end="\r")
    print("\nCalculs parallèles terminés. Agrégation des courbes...")

    # Réunir et moyenner les trajectoires de regret récoltées
    all_plots_data = {}
    
    # Extraire les clés à partir du premier dictionnaire de résultat reçu
    keys = list(all_runs_data[0].keys())
    
    for key in keys:
        # Extraire le vecteur de regret de l'algorithme "key" pour chaque run
        curves_matrix = np.array([run_data[key] for run_data in all_runs_data])
        # Calculer la moyenne sur l'axe des lignes (axis=0) pour lisser la courbe
        all_plots_data[key] = np.mean(curves_matrix, axis=0).tolist()

    print("Sauvegarde des résultats...")
    payload = {
        "Figure4_Graph_Structures": {
            label: regrets for label, regrets in all_plots_data.items()
        }
    }
    save_multi_results(payload.values(), config)


if __name__ == "__main__":
    run_experiment()
