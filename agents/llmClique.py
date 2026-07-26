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

    def getPrompt(self, i):
        agent = self.agents[i]

        return request_cot(
            agent.bandit,
            agent.t,
            agent.history,
            self.prev_actions[i],
            horizon=agent.t
        )

    def updateActions(self, actions):
        current_actions = {}

        for i, action in actions.items():
            current_actions[i] = action

        for i in range(self.clique_size):
            self.prev_actions[i] = [
                current_actions[j]
                for j in range(self.clique_size)
                if j != i
            ]

        rewards = []
        regrets = []

        for i, action in actions.items():

            agent = self.agents[i]

            reward = agent.getReward(action)

            agent.history[str(action)]["pulls"] += 1
            agent.history[str(action)]["reward"] += reward

            regret = agent.bandit.regret(action)

            if len(agent.cumul_regret) > 0:
                agent.cumul_regret.append(
                    agent.cumul_regret[-1] + regret
                )
            else:
                agent.cumul_regret.append(regret)

            agent.t += 1

            rewards.append(reward)
            regrets.append(agent.cumul_regret[-1])

        self.reward = float(np.mean(rewards))

        self.history_regret.append(
            float(np.mean(regrets))
        )

    def getNextAction(self):
        raise RuntimeError(
            "LLMClique uses batched execution. Use getPrompt/updateActions."
        )