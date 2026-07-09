import numpy as np
import pandas as pd
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import load_config, save_multi_results
from agents.tucb import TUCB 

def load_human_csv(filepath):
    """
    Charge le fichier d'actions humain en gérant le formatage par espaces ou tabulations.
    """
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
    
    header_idx = None
    for idx, row in df.iterrows():
        row_str = [str(val).strip().lower() for val in row.values]
        if 'arm' in row_str:
            header_idx = idx
            break
            
    if header_idx is None:
        raise KeyError(f"Impossible de localiser la ligne d'en-tête 'Arm' dans le fichier {filepath}")
        
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', skiprows=header_idx)
    
    df.columns = [c.strip() for c in df.columns]
    
    arm_cols = [c for c in df.columns if 'arm' in c.lower()]
    if not arm_cols:
        raise KeyError(f"Colonne 'Arm' introuvable après alignement. Colonnes : {list(df.columns)}")
        
    arm_col = arm_cols[0]
    
    df['Arm_num'] = df[arm_col].astype(str).str.strip().map({'A': 0, 'B': 1, '0': 0, '1': 1})
    
    df = df.dropna(subset=['Arm_num'])
    
    return df['Arm_num'].astype(int).values


def run_human_experiment():
    parser = argparse.ArgumentParser(
        description="Run Target-UCB simulation matching Human data experiments."
    )
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    
    n_runs = 200
    horizon = 100  
    
    env_cfg = config.get('environment', {'n_arms': 2, 'delta': 0.2, 'best_mean': 0.6})

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Dossier contenant vos fichiers CSV (ex: results/figure5/)
    csv_folder = config['experiment'].get('intput_dir', 'Target-UCB/Human bandit dataset/single_humans/')
    human_files = config['humanFiles'].get('files', ["single_human_1_plays.csv", "single_human_2_plays.csv","single_human_3_plays.csv"])

    
    all_plots_data = {}
    
    for filename in human_files:
        filepath = os.path.join(csv_folder, filename)
        if not os.path.exists(filepath):
            print(f"Fichier manquant : {filepath}. Ignoré.")
            continue
            
        human_label = filename.replace("_full_results.csv", "").replace("_", " ").title()
        
        human_arms = load_human_csv(filepath)
        
        bandit_ref = BernoulliBandit(n_arms=env_cfg['n_arms'], delta=env_cfg['delta'], best_mean=env_cfg['best_mean'],probs=env_cfg['probs'])
        
        human_step_regrets = [bandit_ref.regret(int(arm)) for arm in human_arms[:horizon]]
        human_cum_regret = np.cumsum(human_step_regrets)
        
        all_plots_data[f"Single {human_label}"] = human_cum_regret.tolist()
        
        target_ucb_runs = np.zeros((n_runs, horizon))
        
        for run in range(n_runs):
            bandit = BernoulliBandit(n_arms=env_cfg['n_arms'], delta=env_cfg['delta'], best_mean=env_cfg['best_mean'], probs=env_cfg['probs'])
            

            agent = TUCB(bandit=bandit, nbr_neighbours=1, delta=env_cfg['delta'])
            
            # t=0 : Pour éviter l'erreur de taille, on initialise avec une action aléatoire fictive
            prev_human_action = int(np.random.choice(env_cfg['n_arms']))
            
            for t in range(horizon):
                prev_actions_list = [prev_human_action]
                
                agent.getNextAction(prev_actions_list)
                
                target_ucb_runs[run, t] = agent.cumul_regret[-1]
                
                prev_human_action = int(human_arms[t])
                
        avg_agent_regret = np.mean(target_ucb_runs, axis=0)
        all_plots_data[f"Target-UCB ({human_label})"] = avg_agent_regret.tolist()
        
    print("Calcul des regrets de l'expérience humaine terminé avec succès.")
    
    payload = {
        "Human_vs_TUCB": {
            label: regrets for label, regrets in all_plots_data.items()
        }
    }
    
    save_multi_results(payload.values(), config)


if __name__ == "__main__":
    run_experiment = run_human_experiment
    run_experiment()
