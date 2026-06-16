import random


class Greedy:
    """
    Standalone greedy agent for K-armed bandits.
    Pulls each arm once, then always exploits the best empirical mean.
    """

    def __init__(self, bandit):
        self.bandit = bandit
        self.K = bandit.n_arms

        self.nb_plays = [0] * self.K
        self.values = [0.0] * self.K

        self.t = 0
        self.cumul_regret = []

    def getNextAction(self, prev_actions=None):
        self.t += 1

        if self.t <= self.K:
            action = self.t - 1
        else:
            max_val = max(self.values)
            candidates = [i for i in range(self.K) if self.values[i] == max_val]
            action = random.choice(candidates)

        reward = self.bandit.pull(action)

        self.nb_plays[action] += 1
        n = self.nb_plays[action]
        self.values[action] += (reward - self.values[action]) / n

        step_regret = self.bandit.regret(action)
        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action
