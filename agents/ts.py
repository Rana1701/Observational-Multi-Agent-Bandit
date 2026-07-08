import numpy as np

class TS() :
    """
    Thompson Sampling algorithm for K-armed bandits 
    """
    def __init__(self, bandit):
        self.bandit = bandit
        self.n_arms = bandit.n_arms

        # Beta distribution parameters for each arm 
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

        self.t = 0 
        self.cumul_regret = []
        self.reward = 0
    
    def getNextAction(self, prev_actions=None):
        self.t += 1

        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)

        # Select the arm with the highest sample
        arm = int(np.argmax(samples))

        # get reward
        reward = self.bandit.pull(arm)
        self.reward = reward

        # update Beta parameters
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

        # regret
        step_regret = self.bandit.regret(arm)
        if self.t == 1:
            self.cumul_regret.append(step_regret)
        else:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)


        return arm