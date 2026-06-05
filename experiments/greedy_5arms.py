import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_function import reward_fn


class GreedyMultiArm:
    """Greedy agent for k-armed bandits."""
    
    def __init__(self, k=5, reward_fn=None, delta=0.1):
        self.k = k
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.delta = delta
        
        self.nb_plays = [0] * k
        self.t = 0
        self.cumul_regret = []
        self.values = [0.0] * k
        self.optimal_arm = 0  
    
    def _default_reward_fn(self, action):
        """Default reward function."""
        return np.random.rand()
    
    def getNextAction(self):
        """Greedy: exploit best estimated value."""
        self.t += 1
        
        # Play each arm at least once
        if self.t <= self.k:
            action = self.t - 1
        else:
            # Greedy: pick arm with highest estimated value
            action = np.argmax(self.values)
        
        # Observe reward
        reward = self.reward_fn(action)
        self.nb_plays[action] += 1
        n = self.nb_plays[action]
        
        # Incremental mean update
        self.values[action] += (reward - self.values[action]) / n
        
        # Regret: assume arm 0 is optimal
        step_regret = 0 if action == self.optimal_arm else self.delta
        
        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)
        
        return action


def reward_fn_5arms(arm):
    """Reward function for 5-arm bandit.
    Arm 0 (optimal): 0.7
    Arm 1: 0.6
    Arm 2: 0.5
    Arm 3: 0.4
    Arm 4: 0.3
    """
    means = [0.7, 0.6, 0.5, 0.4, 0.3]
    return np.random.binomial(1, means[arm])


def main():
    nb_runs = 10
    nb_plays = 1000
    
    greedy_runs = []
    
    print("Running 5-arm bandit experiment with Greedy...")
    for run in range(nb_runs):
        greedy = GreedyMultiArm(k=5, reward_fn=reward_fn_5arms, delta=0.1)
        
        for t in range(nb_plays):
            greedy.getNextAction()
        
        greedy_runs.append(greedy.cumul_regret)
    
    mean_greedy = np.mean(greedy_runs, axis=0)
    sem_greedy = np.std(greedy_runs, axis=0, ddof=1) / np.sqrt(nb_runs)
    
    episodes = np.arange(nb_plays)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    line, = ax.plot(episodes, mean_greedy, label="5-Arm Greedy", linewidth=2, color='#ff7f0e')

    
    ax.set_xlabel("Plays", fontsize=14)
    ax.set_ylabel("Cumulative regret", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=14)
    ax.set_title("5-Arm Bandit: Greedy Agent (10 runs)", fontsize=16)
    ax.grid(True, alpha=0.3)
    
    out_file = os.path.join(os.path.dirname(__file__), '../figs/greedy_5arms.png')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=150)
    print(f"Figure saved as '{out_file}'")
    
    print(f"Final cumulative regret: {mean_greedy[-1]:.2f} ± {1.96 * sem_greedy[-1]:.2f}")


if __name__ == "__main__":
    main()
