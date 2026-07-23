import numpy as np

from agents.llm import LLMAgent
from environnement.bernoulli_bandit import BernoulliBandit
from utils.prompt_builder import request_cot


class LLMClique:
    """
    Clique of LLM agents on a fully connected graph.

    Every agent observes the previous actions of all other agents.
    All agents share the same vLLM model instance.
    """

    def __init__(self, bandit, clique_size, model):

        if clique_size < 2:
            raise ValueError("A clique requires at least 2 agents")

        self.bandit = bandit
        self.K = bandit.n_arms
        self.clique_size = clique_size
        self.model = model

        probs = np.asarray(bandit.probs, dtype=float).tolist()

        self.agents = [
            LLMAgent(
                bandit=BernoulliBandit(probs=probs),
                model=self.model,
            )
            for _ in range(clique_size)
        ]

        self.prev_actions = [
            [None] * (clique_size - 1)
            for _ in range(clique_size)
        ]

        self.t = 0
        self.cumul_regret = []
        self.history_regret = []
        self.reward = 0.0

    def getNextAction(self, prev_actions=None):
        del prev_actions

        self.t += 1

        current_actions = {}
        step_regrets = []
        step_rewards = []

        for i in range(self.clique_size):

            prompt = request_cot(
                self.agents[i].bandit,
                self.agents[i].t,
                self.agents[i].history,
                self.prev_actions[i],
                horizon=self.t,
            )

            action = self.agents[i].getNextAction(prompt)

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