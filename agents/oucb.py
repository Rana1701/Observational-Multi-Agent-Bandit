import numpy as np


class OUCB:
    """
    A simple social UCB variant that uses observed neighbor actions as a
    social signal in addition to its own empirical rewards.
    """

    def __init__(self, bandit, nbr_neighbours=1, c=1.0, social_weight=1.0):
        self.bandit = bandit
        self.K = bandit.n_arms
        self.nbr_neighbours = max(1, int(nbr_neighbours))
        self.c = float(c)
        self.social_weight = float(social_weight)

        self.empirical_mean = np.zeros(self.K)
        self.n_plays = np.zeros(self.K, dtype=int)
        self.social_counts = np.zeros(self.K, dtype=float)

        self.t = 0
        self.cumul_regret = []
        self.reward = 0

    def _update_social_counts(self, prev_actions):
        if not prev_actions:
            return

        weight = 1.0 / max(1, len(prev_actions))
        for action in prev_actions:
            if 0 <= action < self.K:
                self.social_counts[action] += weight

    def getNextAction(self, prev_actions=None):
        self.t += 1

        if prev_actions is None:
            prev_actions = []
        self._update_social_counts(prev_actions)

        if np.any(self.n_plays == 0):
            arm = int(np.where(self.n_plays == 0)[0][0])
        else:
            log_term = np.log(self.t + 1)
            exploration_bonus = self.c * np.sqrt(log_term / np.maximum(self.n_plays, 1))
            social_bonus = self.social_weight * np.sqrt(
                log_term / np.maximum(self.social_counts + 1e-8, 1e-8)
            )
            scores = self.empirical_mean + exploration_bonus + social_bonus
            arm = int(np.argmax(scores))

        reward = self.bandit.pull(arm)
        self.reward = reward

        self.n_plays[arm] += 1
        self.empirical_mean[arm] += (reward - self.empirical_mean[arm]) / self.n_plays[arm]

        step_regret = self.bandit.regret(arm)
        self.cumul_regret.append(step_regret if self.t == 1 else self.cumul_regret[-1] + step_regret)

        return arm
