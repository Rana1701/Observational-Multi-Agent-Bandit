import numpy as np
import random

class E_Greedy():
    '''Epsilon-greedy that mimics the target's most frequent action.'''
    
    def __init__(self, reward_fn=None, delta=0.2, epsilon=0.1):
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.delta = delta
        self.epsilon = epsilon
        
        self.nb_plays = [0, 0]
        
        #step number
        self.t = 0
        
        self.cumul_regret = []
        
        # average rewards for each arm
        self.values = [0.0, 0.0]
    
    def getNextAction(self):
        """Epsilon-greedy: with prob. epsilon explore, else exploit estimated values."""

        self.t += 1
        # choose action: 
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            if self.values[0] > self.values[1]:
                action = 0
            elif self.values[1] > self.values[0]:
                action = 1
            else:
                action = random.choice([0, 1])

        # observe reward 
        reward = self.reward_fn(action)
        self.nb_plays[action] += 1
        n = self.nb_plays[action]
        # incremental mean update
        self.values[action] += (reward - self.values[action]) / n

        # regret: assume arm 0 optimal with gap delta
        step_regret = 0 if action == 0 else self.delta

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action
    
    def _default_reward_fn(self, arm_played):
        win_rate = [0.6, 0.4]
        pull = np.random.rand()
        if arm_played == 0 and pull < win_rate[0]:
            return 1
        elif arm_played == 1 and pull < win_rate[1]:
            return 1
        return 0

    def getReward(self, arm_played):
        return self.reward_fn(arm_played)


def main():
    import matplotlib.pyplot as plt
    
    nb_runs = 100
    nb_episodes = 1000
    epsilons = [0.1, 0.5, 0.01]
    results = {}

    '''Runs epsilon-greedy agents for 1000 episodes, averaged over 10 runs.'''

    for eps in epsilons:
        cumul_regrets = []
        for _ in range(nb_runs):
            agent = E_Greedy(epsilon=eps)
            for _ in range(nb_episodes):
                agent.getNextAction()
            cumul_regrets.append(agent.cumul_regret)
        results[eps] = np.mean(cumul_regrets, axis=0)
        
    plt.figure(figsize=(10, 6))
    for eps, mean_cumul_regret in results.items():
        plt.plot(mean_cumul_regret, label=f"Epsilon-Greedy Agent (ε={eps})")
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of Epsilon-Greedy Agent", fontsize=20)
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "E_Greedy_cumul_regret.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.show()


if __name__ == "__main__":
    main()
