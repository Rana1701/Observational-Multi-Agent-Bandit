import numpy as np


class OUCB:
    """
    Cooperative OUCB-inspired agent.

    This implementation is an approximation of the distributed cooperative
    UCB idea from Landgren et al. 2016. It uses neighbor action observations
    to bias exploration toward arms that appear promising in the social signal.
    """

    def __init__(self, bandit, nbr_neighbours=1, c=1.0, gamma=1.0):
        self.bandit = bandit
        self.K = bandit.n_arms
        self.nbr_neighbours = max(1, int(nbr_neighbours))
        self.c = float(c)
        self.gamma = float(gamma)

        self.empirical_mean = np.zeros(self.K)
        self.n_plays = np.zeros(self.K, dtype=int)
        self.social_counts = np.zeros(self.K, dtype=float)

        self.t = 0
        self.cumul_regret = []
        self.reward = 0

    def _update_social_counts(self, prev_actions):
        if not prev_actions:
            return

        for action in prev_actions:
            if 0 <= action < self.K:
                self.social_counts[action] += 1.0

    def getNextAction(self, prev_actions=None):
        self.t += 1

        if prev_actions is None:
            prev_actions = []
        self._update_social_counts(prev_actions)

        if np.any(self.n_plays == 0):
            arm = int(np.where(self.n_plays == 0)[0][0])
        else:
            log_term = np.log(self.t + 1)
            ucb_bonus = self.c * np.sqrt(log_term / self.n_plays)
            social_bonus = self.gamma * np.sqrt(
                log_term / np.maximum(self.social_counts + 1.0, 1.0)
            )
            scores = self.empirical_mean + ucb_bonus + social_bonus
            arm = int(np.argmax(scores))

        reward = self.bandit.pull(arm)
        self.reward = reward

        self.n_plays[arm] += 1
        self.empirical_mean[arm] += (reward - self.empirical_mean[arm]) / self.n_plays[arm]

        step_regret = self.bandit.regret(arm)
        self.cumul_regret.append(step_regret if self.t == 1 else self.cumul_regret[-1] + step_regret)

        return arm
