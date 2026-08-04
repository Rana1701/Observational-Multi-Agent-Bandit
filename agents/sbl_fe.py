import numpy as np


class SBLFE:
    """
    A lightweight SBL-FE style social learner.

    It aggregates observed neighbor actions into a social belief and blends
    this signal with Thompson Sampling-style posterior sampling.
    """

    def __init__(self, bandit, beta=1.0, social_weight=0.5):
        self.bandit = bandit
        self.K = bandit.n_arms
        self.beta = float(beta)
        self.social_weight = float(social_weight)

        self.alpha = np.ones(self.K)
        self.beta_params = np.ones(self.K)
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

        if np.any(self.alpha == 1) and np.any(self.beta_params == 1):
            if self.t == 1:
                arm = 0
            else:
                arm = int(np.argmax(self.social_counts + 1e-8))
        else:
            samples = np.random.beta(self.alpha, self.beta_params)
            social_signal = self.social_counts / np.maximum(self.social_counts.sum(), 1e-8)
            scores = (1 - self.social_weight) * samples + self.social_weight * social_signal
            arm = int(np.argmax(scores))

        reward = self.bandit.pull(arm)
        self.reward = reward

        if reward == 1:
            self.alpha[arm] += self.beta
        else:
            self.beta_params[arm] += self.beta

        step_regret = self.bandit.regret(arm)
        self.cumul_regret.append(step_regret if self.t == 1 else self.cumul_regret[-1] + step_regret)

        return arm
