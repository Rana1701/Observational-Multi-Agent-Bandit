import numpy as np
import random


class TUCB:
    """
    Target-UCB algorithm with injected bandit environment.
    """

    def __init__(self, bandit, nbr_neighbours, delta=0.2):

        if nbr_neighbours < 1:
            raise ValueError("At least one neighbour required")

        self.bandit = bandit
        self.delta = bandit.delta if hasattr(bandit, "delta") else delta

        self.neighbours = nbr_neighbours

        # stats (K-armed compatible)
        self.K = bandit.n_arms
        self.nb_plays = [0] * self.K
        self.avg_reward = [0.0] * self.K
        self.targets = [0.0] * self.K

        self.t = 0
        self.cumul_regret = []

    def getNextAction(self, prev_actions=None):

        if prev_actions is None:
            prev_actions = []

        self.t += 1

        if len(prev_actions) != self.neighbours:
            raise ValueError("Mismatch between neighbours and actions")

        # update targets
        if self.t > 1:
            self.targets = [0.0] * self.K  
            for a in prev_actions:
                if 0 <= a < self.K:
                    self.targets[a] += 1.0 / self.neighbours

        # forced exploration
        for i in range(self.K):
            if self.nb_plays[i] == 0:
                action = i
                break
        else:

            est_opt = np.sqrt(2 * np.log(self.t) / np.array(self.nb_plays))

            target_opt = np.zeros(self.K)

            for i in range(self.K):
                if self.targets[i] > 0:
                    diff = max(self.targets[i] - self.nb_plays[i], 0)
                    target_opt[i] = np.sqrt(diff / self.targets[i])

            Q = self.avg_reward + est_opt * target_opt

            action = int(np.argmax(Q))

        reward = self.bandit.pull(action)

        # update stats
        self.nb_plays[action] += 1
        self.avg_reward[action] += (
            reward - self.avg_reward[action]
        ) / self.nb_plays[action]

        step_regret = self.bandit.regret(action)

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action