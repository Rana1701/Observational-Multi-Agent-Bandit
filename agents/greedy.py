import numpy as np
import random

class Greedy():
    '''Greedy agent that always exploits the best estimated value.'''
    
    def __init__(self, reward_fn=None, delta=0.2):
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.delta = delta
        
        self.nb_plays = [0, 0]
        
        #step number
        self.t = 0
        
        self.cumul_regret = []
        
        # average rewards for each arm
        self.values = [0.0, 0.0]
    
    def getNextAction(self):
        """Greedy: always exploit the best estimated value."""

        self.t += 1

        # choose action: 
        if self.t < 2:  # play each arm at least once
            action = self.t - 1
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
    
    '''Runs a  Greedy agent for 100 plays'''
    agent = Greedy()
    
    for _ in range(100):
        agent.getNextAction()
    
    plt.figure(figsize=(10, 6))
    plt.plot(agent.cumul_regret, label="Greedy Agent")
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of Greedy Agent", fontsize=20)
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "Greedy_cumul_regret.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.show()


if __name__ == "__main__":
    main()
