import numpy as np
import random

class Greedy():
    '''Greedy follower that mimics the target's most frequent action.'''
    
    def __init__(self, reward_fn=None, delta=0.2):
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.delta = delta
        
        #number of times target played each arm
        self.target_plays = [0, 0]
        
        #number of own plays for arms 0 and 1
        self.nb_plays = [0, 0]
        
        #step (play) number
        self.t = 0
        
        #cumulative regret
        self.cumul_regret = []
    
    def getNextAction(self, prev_actions=[]):
        '''Selects the action most frequently chosen by the target (greedy follower).
        
        Args:
            prev_actions: List of previous actions from the target (or neighbor)
        '''
        
        self.t += 1
        
        # Update target play counts from previous actions
        if len(prev_actions) > 0 and self.t > 1:
            for a in prev_actions:
                if a == 0:
                    self.target_plays[0] += 1
                elif a == 1:
                    self.target_plays[1] += 1
        
        # Select the action the target has played most often
        if self.target_plays[0] == 0 and self.target_plays[1] == 0:
            # First action: arbitrary choice
            action = 0
        elif self.target_plays[0] > self.target_plays[1]:
            action = 0
        elif self.target_plays[1] > self.target_plays[0]:
            action = 1
        else:
            # Tie: random choice
            action = random.choice([0, 1])
        
        step_reward = self.reward_fn(action)
        
        #update values
        if action == 0:
            self.nb_plays[0] += 1
            step_regret = 0
            
        elif action == 1:
            self.nb_plays[1] += 1
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
