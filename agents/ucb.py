import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

class UCB():
    '''Simple class implementing the UCB (Upper Confidence Bound) algorithm on a k-armed bandit.'''
    
    def __init__(self, reward_fn=None, delta=0.2):
        # injected reward function
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.delta = delta

        #number of own plays for arms 0 and 1
        self.nb_plays = [0, 0]
        
        #average reward
        self.avg_reward = [0, 0]
        
        #step (play) number
        self.t = 0
        
        #cumulative regret
        self.cumul_regret = []
    
    def getNextAction(self):
        '''Outputs the next action based on UCB strategy'''
        
        self.t += 1
        
        if self.nb_plays[0] == 0:
            #play arm 0 for the first time
            action = 0
        elif self.nb_plays[1] == 0:
            #play arm 1 for the first time
            action = 1
        else:
            #get UCB values for each arm
            ucb_values = [0, 0]
            ucb_values[0] = self.avg_reward[0] + np.sqrt(2 * np.log(self.t) / self.nb_plays[0])
            ucb_values[1] = self.avg_reward[1] + np.sqrt(2 * np.log(self.t) / self.nb_plays[1])
            
            #select arm with highest UCB value
            if ucb_values[0] != ucb_values[1]:
                action = np.argmax(ucb_values)
            else:
                #tie breaker
                action = random.choice([0, 1])
        
        step_reward = self.reward_fn(action)
        
        #update values
        if action == 0:
            self.nb_plays[0] += 1
            self.avg_reward[0] += (step_reward - self.avg_reward[0]) / self.nb_plays[0]
            
            step_regret = 0
        elif action == 1:
            self.nb_plays[1] += 1
            self.avg_reward[1] += (step_reward - self.avg_reward[1]) / self.nb_plays[1]
            
            step_regret = self.delta
        
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
    nb_runs = 100
    nb_plays = 100

    all_regrets = []

    # Plusieurs exécutions indépendantes
    for _ in range(nb_runs):
        agent = UCB()

        for _ in range(nb_plays):
            agent.getNextAction()

        all_regrets.append(agent.cumul_regret)

    all_regrets = np.array(all_regrets)

    # Moyenne
    mean_regret = np.mean(all_regrets, axis=0)

    # Intervalle de confiance 95 %
    std_regret = np.std(all_regrets, axis=0, ddof=1)
    ci95 = 1.96 * std_regret / np.sqrt(nb_runs)

    x = np.arange(1, nb_plays + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        mean_regret,
        label="UCB Mean Regret",
        linewidth=2
    )

    plt.fill_between(
        x,
        mean_regret - ci95,
        mean_regret + ci95,
        alpha=0.2,
        label="95% CI"
    )

    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("UCB Cumulative Regret with 95% Confidence Interval", fontsize=20)
    plt.legend(fontsize=14)

    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "UCB_cumul_regret_CI.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_file, bbox_inches="tight")
    plt.show()
    print(f"Figure saved as '{out_file}'")



def main2(): 
    import matplotlib.pyplot as plt
    
    '''Runs a UCB agent for 100 plays'''
    agent = UCB()
    
    for _ in range(100):
        agent.getNextAction()
    
    plt.figure(figsize=(10, 6))
    plt.plot(agent.cumul_regret, label="UCB Agent")
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of UCB Agent", fontsize=20)
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "UCB_cumul_regret.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.show()
    print(f"Figure saved as '{out_file}'")


if __name__ == "__main__":
    main()
    main2()