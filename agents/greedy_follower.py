import numpy as np
import random


class Greedy:
    """
    Greedy follower that mimics the most frequent action of the target.
    """

    def __init__(self, bandit, delta=0.2):

        self.bandit = bandit
        self.delta = delta

        self.K = bandit.n_arms

        self.target_plays = [0] * self.K

        self.nb_plays = [0] * self.K

        self.t = 0
        self.cumul_regret = []

    def getNextAction(self, prev_actions=None):

        if prev_actions is None:
            prev_actions = []

        self.t += 1

        # update target counts
        if self.t > 1:
            for a in prev_actions:
                if 0 <= a < self.K:
                    self.target_plays[a] += 1

        # select greedy action
        if sum(self.target_plays) == 0:
            action = 0
        else:
            max_val = max(self.target_plays)
            candidates = [
                i for i in range(self.K) if self.target_plays[i] == max_val
            ]
            action = random.choice(candidates)

        # reward
        reward = self.bandit.pull(action)

        # update own stats
        self.nb_plays[action] += 1

        # regret (expected)
        step_regret = self.bandit.regret(action)

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action