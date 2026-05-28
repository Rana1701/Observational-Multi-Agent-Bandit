import numpy as np
import random

class Greedy():
    '''Simple class implementing the Epsilon-Greedy algorithm on a 2-armed bandit.'''
    
    def __init__(self, epsilon=0.1):
        #exploration rate
        self.epsilon = epsilon
        
        #number of own plays for arms 0 and 1
        self.nb_plays = [0, 0]
        
        #average reward of each arm
        self.avg_reward = [0, 0]
        
        #step (play) number
        self.t = 0
        
        #cumulative regret
        self.cumul_regret = []
    
    def getNextAction(self):
        '''Outputs the next action based on epsilon-greedy strategy'''
        
        self.t += 1
        
        #exploration vs exploitation
        if np.random.rand() < self.epsilon:
            #explore: random arm selection
            action = random.choice([0, 1])
        else:
            #exploit: select best arm so far
            if self.nb_plays[0] == 0:
                #play arm 0 for the first time
                action = 0
            elif self.nb_plays[1] == 0:
                #play arm 1 for the first time
                action = 1
            else:
                #select arm with highest average reward
                if self.avg_reward[0] != self.avg_reward[1]:
                    action = np.argmax(self.avg_reward)
                else:
                    #tie breaker
                    action = random.choice([0, 1])
        
        step_reward = self.getReward(action)
        
        #update values
        if action == 0:
            self.nb_plays[0] += 1
            self.avg_reward[0] += (step_reward - self.avg_reward[0]) / self.nb_plays[0]
            step_regret = 0
            
        elif action == 1:
            self.nb_plays[1] += 1
            self.avg_reward[1] += (step_reward - self.avg_reward[1]) / self.nb_plays[1]
            
            #add regret 0.2 = the gap between arms
            step_regret = 0.2
        
        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)
        
        return action
    
    def getReward(self, arm_played):
        '''Returns a reward from a Bernoulli distribution associated with the arm'''
        
        win_rate = [0.6, 0.4]
        
        pull = np.random.rand()
        
        if arm_played == 0 and pull < win_rate[0]:
            return 1
        elif arm_played == 1 and pull < win_rate[1]:
            return 1
        else:
            
            return 0


def main():
    import matplotlib.pyplot as plt
    
    '''Runs a Greedy agent for 100 plays'''
    agent = Greedy(epsilon=0.1)
    
    for _ in range(100):
        agent.getNextAction()
    
    plt.figure(figsize=(10, 6))
    plt.plot(agent.cumul_regret, label="Greedy Agent (ε=0.1)")
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of Greedy Agent", fontsize=20)
    plt.savefig("Greedy_cumul_regret.png")
    plt.show()


if __name__ == "__main__":
    main()
