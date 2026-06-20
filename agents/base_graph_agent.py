# agents/base_graph_agent.py
class BaseGraphAgent:
    def __init__(self, bandit, agent_id, neighbors_ids):
        self.bandit = bandit
        self.agent_id = agent_id
        self.neighbors_ids = neighbors_ids # Liste des IDs des voisins
        self.K = bandit.n_arms
        self.nb_plays = [0] * self.K
        self.avg_reward = [0.0] * self.K
        self.cumul_regret = []
        self.t = 0

    def update(self, action, reward):
        self.nb_plays[action] += 1
        self.avg_reward[action] += (reward - self.avg_reward[action]) / self.nb_plays[action]