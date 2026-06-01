import numpy as np
import random

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
            
            #no added regret since arm 0 is optimal
            step_regret = 0
        elif action == 1:
            self.nb_plays[1] += 1
            self.avg_reward[1] += (step_reward - self.avg_reward[1]) / self.nb_plays[1]
            
            #add regret according to the expected gap
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
    plt.savefig("UCB_cumul_regret.png")
    plt.show()


if __name__ == "__main__":
    main()
