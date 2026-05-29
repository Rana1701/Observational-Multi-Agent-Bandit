import numpy as np
import matplotlib.pyplot as plt

# use the classes defined in your modules
from agents.ucb import UCB as UCBAgent
from agents.greedy import Greedy as GreedyAgent
from agents.tucb import TUCB as TUCBAgent
from utils.reward_function import reward_fn


def run_experiment(num_episodes=1000, num_runs=10, rate0=0.6, rate1=0.4):
    '''Run the experiment using the classes from your modules.'''

    target_regrets = []
    greedy_regrets = []
    target_ucb_regrets = []

    for run in range(num_runs):
        reward = reward_fn(rate0, rate1)
        delta = rate0 - rate1

        target = UCBAgent(reward_fn=reward, delta=delta)
        greedy = GreedyAgent(reward_fn=reward, delta=delta)
        target_ucb = TUCBAgent(1, reward_fn=reward, delta=delta)

        for t in range(num_episodes):
           
            target_action = target.getNextAction()
            greedy.getNextAction([target_action])

            target_ucb.getNextAction([target_action])

        target_regrets.append(target.cumul_regret)
        greedy_regrets.append(greedy.cumul_regret)
        target_ucb_regrets.append(target_ucb.cumul_regret)

    # Average across runs
    target_avg = np.mean(target_regrets, axis=0)
    greedy_avg = np.mean(greedy_regrets, axis=0)
    target_ucb_avg = np.mean(target_ucb_regrets, axis=0)

    return target_avg, greedy_avg, target_ucb_avg


def main():
    '''Run a simple comparison using your existing classes.'''

    num_episodes = 1000
    fig, ax = plt.subplots(figsize=(12, 8))
    
    deltas = []
    colors_target = ['#1f77b4', '#1f77b4', '#1f77b4']
    colors_greedy = ['#ff7f0e', '#ff7f0e', '#ff7f0e']
    colors_tucb = ['#2ca02c', '#2ca02c', '#2ca02c']
    linestyles = ['-', '--', ':']
    
    for idx, (a, b) in enumerate([(0.55, 0.45), (0.7, 0.3), (0.9, 0.1)]):
        print(f"Running experiment using module classes with rates {a:.1f}/{b:.1f}...")
        target_avg, greedy_avg, target_ucb_avg = run_experiment(
            num_episodes=num_episodes, rate0=a, rate1=b
        )
        
        delta = a - b
        deltas.append(delta)
        episodes = np.arange(num_episodes)
        
        ax.plot(episodes, target_avg, color=colors_target[idx], linestyle=linestyles[idx], 
                linewidth=2, label=f'Target (Δ_a={delta:.1f})')
        ax.plot(episodes, greedy_avg, color=colors_greedy[idx], linestyle=linestyles[idx], 
                linewidth=2, label=f'Greedy (Δ_a={delta:.1f})')
        ax.plot(episodes, target_ucb_avg, color=colors_tucb[idx], linestyle=linestyles[idx], 
                linewidth=2, label=f'Target-UCB (Δ_a={delta:.1f})')

    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Cumulative regret', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=11, loc='upper left', ncol=3)
    ax.set_title('Multi-Armed Bandit Algorithms with Different Δ_a Values', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figure1_experiment_combined.png', dpi=150)
    print(f"Figure saved as 'Figure1_experiment_combined.png'")
    plt.show()


if __name__ == "__main__":
    main()
