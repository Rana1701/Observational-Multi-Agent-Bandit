import numpy as np


class SBLFE:
    """
    Simplified SBL-FE approximation.

    This implementation uses a TS posterior over arms and a social belief
    based on observed neighbor actions. The final action selection is made by
    combining those estimates, with an adaptive free-energy-like weighting.
    """

    def __init__(self, bandit, beta=1.0, lambda_param=0.5, epsilon=0.1):
        self.bandit = bandit
        self.K = bandit.n_arms
        self.beta = float(beta)
        self.lambda_param = float(lambda_param)
        self.epsilon = float(epsilon)

        self.alpha = np.ones(self.K)
        self.beta_params = np.ones(self.K)
        self.social_counts = np.ones(self.K, dtype=float)

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

        samples = np.random.beta(self.alpha, self.beta_params)
        social_prob = self.social_counts / np.sum(self.social_counts)

        # Free-energy-like mixture between TS and social belief
        scores = (1 - self.lambda_param) * samples + self.lambda_param * social_prob

        # Exploration bonus to ensure non-zero probability for all arms
        if np.random.rand() < self.epsilon:
            arm = np.random.randint(self.K)
        else:
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
