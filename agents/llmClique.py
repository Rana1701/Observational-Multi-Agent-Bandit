import numpy as np

from agents.llm import LLMAgent
from environnement.bernoulli_bandit import BernoulliBandit
from utils.prompt_builder import request_response , request_cot

class LLMClique:
    """
    Clique of LLM agents on a fully connected graph.

    Each node observes the actions of all other clique members and uses
    their empirical average as the target policy.
    """

    def __init__(self, bandit, clique_size, name_parameter="Qwen/Qwen2.5-7B-Instruct", model=None):
        if clique_size < 2:
            raise ValueError("A clique requires at least 2 agents")

        self.bandit = bandit
        self.K = bandit.n_arms
        self.clique_size = clique_size
        self.name_parameter = name_parameter
        self.model = model
    
        probs = np.asarray(bandit.probs, dtype=float).tolist()

        self.agents = [
            LLMAgent(
                bandit=BernoulliBandit(probs=probs), name_parameter=name_parameter, model=model
            )
            for _ in range(self.clique_size)
        ]

        self.prev_actions = [
            [0] * self.clique_size for _ in range(self.clique_size)
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
