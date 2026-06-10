import numpy as np


class UCB:
    """
    UCB algorithm for K-armed bandits 
    """

    def __init__(self, bandit):
        self.bandit = bandit
        self.K = bandit.n_arms

        # empirical means
        self.empirical_mean = np.zeros(self.K)

        # number of pulls
        self.N = np.zeros(self.K)

        self.t = 0 
        self.cum_regret = []
     
    def select_arm(self):
        # Force each arm once (pure exploration phase)
        for i in range(self.K):
            if self.N[i] == 0:
                return i

        # UCB scores for each arm
        ucb = self.empirical_mean + np.sqrt(2 * np.log(self.t) / self.N)

        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.N[arm] += 1
        self.empirical_mean[arm] += (reward - self.empirical_mean[arm]) / self.N[arm]

    def getAction(self):
        self.t += 1

        arm = self.select_arm()
        reward = self.bandit.pull(arm)

        self.update(arm, reward)
 
        step_regret = self.bandit.regret(arm)

        if self.t == 1:
            self.cum_regret.append(step_regret)
        else:
            self.cum_regret.append(self.cum_regret[-1] + step_regret)

        return arm, reward