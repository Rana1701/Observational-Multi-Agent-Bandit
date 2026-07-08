import numpy as np

class BernoulliBandit:
    def __init__(self, probs=None, n_arms=None, best_mean=0.5, delta=None):

        if probs is not None:
            self.probs = np.array(probs, dtype=float)

        elif n_arms is not None and delta is not None:
            self.probs = np.full(n_arms, best_mean - delta)
            self.probs[0] = best_mean

        else:
            raise ValueError("Provide probs or (n_arms and delta)")

        self.n_arms = len(self.probs)
        self.best_mean = np.max(self.probs)

    def pull(self, arm):
        
        #Pull arm and return reward (0 or 1).
        
        return np.random.binomial(1, self.probs[arm])

    def expected_reward(self, arm):
        return self.probs[arm]

    def regret(self, arm):
        return self.best_mean - self.probs[arm]

    def __repr__(self):
        return (
            f"BernoulliBandit("
            f"n_arms={self.n_arms}, "
            f"probs={self.probs.tolist()})"
        )