import numpy as np

class UCBClique:
    def __init__(self, bandit, clique_size):
        self.bandit = bandit
        self.K = bandit.n_arms
        self.clique_size = clique_size
        self.nb_plays = [0] * self.K
        self.avg_reward = [0.0] * self.K
        self.t = 0
        self.cumul_regret = 0.0
        self.history_regret = []

    def getNextAction(self):
        self.t += 1
        # Exploration forcée initiale par l'ensemble de la clique
        for i in range(self.K):
            if self.nb_plays[i] == 0:
                action = i
                break
        else:
            # UCB basé sur le nombre total d'échantillons reçus par la clique
            bonus = np.sqrt(2 * np.log(self.t * self.clique_size) / np.array(self.nb_plays))
            Q = np.array(self.avg_reward) + bonus
            action = int(np.argmax(Q))

        # La clique entière tire le même bras sélectionné (Partage total d'information)
        for _ in range(self.clique_size):
            reward = self.bandit.pull(action)
            self.nb_plays[action] += 1
            self.avg_reward[action] += (reward - self.avg_reward[action]) / self.nb_plays[action]
            
        # Accumuler le regret moyen par nœud de la clique
        self.cumul_regret += self.bandit.regret(action)
        self.history_regret.append(self.cumul_regret)