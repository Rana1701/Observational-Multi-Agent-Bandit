# agents/ucb_clique.py

from agents.ucb import UCB
import numpy as np

class UCBClique(UCB):

    def __init__(self, bandit, clique_size = 11):
        super().__init__(bandit)
        self.clique_size = clique_size

    def getNextAction(self, prev_actions=None):
        self.t += 1


        arm = self.select_arm()

        total_reward = 0
        for _ in range(self.clique_size):
            total_reward += self.bandit.pull(arm)
        
        for _ in range(self.clique_size):
            self.update(arm, total_reward / self.clique_size)

        step_regret = self.bandit.regret(arm)
        if len(self.cumul_regret) == 0:
            self.cumul_regret.append(step_regret)
        else:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)

        return arm