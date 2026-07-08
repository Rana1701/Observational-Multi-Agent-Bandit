import numpy as np

class alphaOptimal: 
    """
    Alpha-optimal policy:
    - with probability alpha -> play optimal arm
    - otherwise -> play a suboptimal arm uniformly
    """

    def __init__(self, bandit, alpha=0.8):
        self.bandit = bandit
        self.alpha = alpha

        self.n_arms = bandit.n_arms
        self.optimal_arm = int(np.argmax(bandit.probs)) 

        self.t = 0
        self.cumul_regret = []

    def getNextAction(self, previous_actions = None):
        self.t += 1

        if np.random.rand() < self.alpha:
            arm = self.optimal_arm
        else:
            suboptimal_arms = [a for a in range(self.n_arms) if a != self.optimal_arm]
            arm = np.random.choice(suboptimal_arms)

        reward = self.bandit.pull(arm)

        step_regret = self.bandit.regret(arm)
        if self.t == 1:
            self.cumul_regret.append(step_regret)
        else:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)

        return arm