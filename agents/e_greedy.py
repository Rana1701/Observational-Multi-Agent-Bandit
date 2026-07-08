import numpy as np
import random


class EpsilonGreedy:
    """
    Epsilon-greedy algorithm for K-armed bandits.
    """

    def __init__(self, bandit, epsilon=0.1, delta=0.2):

        self.bandit = bandit
        self.K = bandit.n_arms

        self.epsilon = epsilon
        self.delta = delta

        self.nb_plays = [0] * self.K
        self.values = [0.0] * self.K

        self.t = 0
        self.cumul_regret = []

    def getNextAction(self):

        self.t += 1

        # exploration
        if random.random() < self.epsilon:
            action = random.randint(0, self.K - 1)

        # exploitation
        else:
            max_val = max(self.values)
            candidates = [
                i for i in range(self.K) if self.values[i] == max_val
            ]
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