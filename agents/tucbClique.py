import numpy as np

from agents.tucb import TUCB
from environnement.bernoulli_bandit import BernoulliBandit


class TUCBClique:
    """
    Clique of Target-UCB agents on a fully connected graph.

    Each node observes the actions of all other clique members and uses
    their empirical average as the target policy. Cumulative regret is
    averaged over all nodes, matching Figure 3 in Lupu et al.
    """

    def __init__(self, bandit, clique_size, delta=0.1):
        if clique_size < 2:
            raise ValueError("A clique requires at least 2 agents")

        self.bandit = bandit
        self.K = bandit.n_arms
        self.clique_size = clique_size
        self.delta = getattr(bandit, "delta", delta)

        probs = np.asarray(bandit.probs, dtype=float).tolist()
        nbr_neighbours = clique_size - 1

        self.agents = [
            TUCB(
                bandit=BernoulliBandit(probs=probs),
                nbr_neighbours=nbr_neighbours,
                delta=self.delta,
            )
            for _ in range(clique_size)
        ]

        self.prev_actions = [
            [0] * nbr_neighbours for _ in range(clique_size)
        ]

        self.t = 0
        self.cumul_regret = []
        self.history_regret = []
        self.reward = 0.0

    def getNextAction(self, prev_actions=None):
        del prev_actions  # observations are managed internally

        self.t += 1
        current_actions = {}
        step_regrets = []
        step_rewards = []

        for i in range(self.clique_size):
            action = self.agents[i].getNextAction(self.prev_actions[i])
            current_actions[i] = action
            step_regrets.append(self.agents[i].cumul_regret[-1])
            step_rewards.append(self.agents[i].reward)

        for i in range(self.clique_size):
            self.prev_actions[i] = [
                current_actions[j]
                for j in range(self.clique_size)
                if j != i
            ]

        avg_regret = float(np.mean(step_regrets))
        self.cumul_regret.append(avg_regret)
        self.history_regret.append(avg_regret)
        self.reward = float(np.mean(step_rewards))

        return current_actions[0]
