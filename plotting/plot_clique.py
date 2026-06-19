# plotting/plot_regret.py (version corrigée)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_single_regrets(result_dir):
    """Charge les regrets pour une expérience simple (e.g., ucb_single)."""
    result_dir = Path(result_dir)
    regrets_path = result_dir / "regrets.npy"

    if regrets_path.exists():
        return np.load(regrets_path) # shape: (runs, horizon)

    raise FileNotFoundError(f"No regrets.npy found in {result_dir}")


def load_clique_regrets(clique_dir):
    """Charge les regrets de TOUS les agents d'une clique et moyenne par agent."""
    clique_dir = Path(clique_dir)
    # On trie pour garantir l'ordre : clique, ucb_1, ucb_2, ..., tucb0, tucb1, ...
    agent_dirs = sorted([d for d in clique_dir.iterdir() if d.is_dir()])
    
    all_agent_regrets = []
    for agent_dir in agent_dirs:
        if (agent_dir / "regrets.npy").exists():
            regrets = np.load(agent_dir / "regrets.npy") # shape: (runs, horizon)
            all_agent_regrets.append(regrets)
    
    if not all_agent_regrets:
         raise FileNotFoundError(f"No agent data found in subdirectories of {clique_dir}")

    # On empile pour obtenir (agents, runs, horizon) et on moyenne sur l'axe des agents (axis=0)
    # Le résultat est de forme (runs, horizon) - le regret moyen de la clique par exécution.
    avg_clique_regrets = np.stack(all_agent_regrets, axis=0).mean(axis=0)
    return avg_clique_regrets


def plot_series(ax, regrets, label, alpha=0.2):
    regrets = np.asarray(regrets)
    if regrets.ndim == 1:
        regrets = regrets[None, :]

    mean = regrets.mean(axis=0)
    std = regrets.std(axis=0, ddof=1 if len(regrets) > 1 else 0)
    ci95 = 1.96 * std / np.sqrt(len(regrets))
    x = np.arange(len(mean))

    line, = ax.plot(x, mean, label=label, linewidth=2)
    ax.fill_between(
        x,
        mean - ci95,
        mean + ci95,
        color=line.get_color(),
        alpha=alpha,
        linewidth=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative regret from saved experiment results."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Result directory(ies). If multi-agent, averages over all agents.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (one per --input path).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output figure (png/pdf).",
    )
    parser.add_argument(
        "--title",
        default="Cumulative Regret (Clique Average)",
        help="Plot title",
    )
    args = parser.parse_args()

    # Styliser le plot (Optionnel, pour se rapprocher du rendu de la figure)
    plt.style.use('seaborn-v0_8-deep') # ou 'ggplot'
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = args.labels or []
    plotted = 0

    for i, input_path in enumerate(args.input):
        input_path = Path(input_path)
        
        # Définit le label : soit celui fourni, soit le nom du répertoire
        label = labels[i] if i < len(labels) else input_path.name

        try:
            # Vérifie si c'est une structure multi-agents (données dans des sous-répertoires)
            # e.g., tucb_clique/tucb0/regrets.npy, ...
            subdirs_with_data = [
                p for p in input_path.iterdir() 
                if p.is_dir() and (p / "regrets.npy").exists()
            ]

            if subdirs_with_data:
                print(f"Chargement et moyennage des données multi-agents de {input_path} ({len(subdirs_with_data)} agents)")
                # Charge TOUTES les données d'agents et moyenne entre eux pour obtenir la moyenne clique
                regrets = load_clique_regrets(input_path)
                plot_series(ax, regrets, label)
                plotted += 1
                
            # Structure simple ou à agent unique (données dans ce répertoire)
            # e.g., single_ucb/regrets.npy
            elif (input_path / "regrets.npy").exists():
                print(f"Chargement des données d'expérience simple de {input_path}")
                regrets = load_single_regrets(input_path)
                plot_series(ax, regrets, label)
                plotted += 1
                
            else:
                print(f"Attention : Aucune donnée valide trouvée dans {input_path}. Ignoré.")

        except FileNotFoundError as e:
            print(f"Erreur lors du traitement de {input_path} : {e}")

    if plotted == 0:
        raise SystemExit("Aucune donnée tracée. Vérifiez vos chemins --input.")

    ax.set_xlabel("Time (Episodes)")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(args.title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Figure sauvegardée sous {output_path}")


if __name__ == "__main__":
    main()