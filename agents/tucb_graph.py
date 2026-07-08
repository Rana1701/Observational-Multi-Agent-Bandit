# agents/tucb_graph.py
import numpy as np
from agents.base_graph_agent import BaseGraphAgent

class TUCBGraph(BaseGraphAgent):
    def getNextAction(self, all_prev_actions):
        self.t += 1
        # Extraction actions des voisins uniquement
        neighbor_actions = [all_prev_actions[nid] for nid in self.neighbors_ids]
        
        # Logique de cible (moyenne des actions des voisins)
        target = np.zeros(self.K)
        for a in neighbor_actions:
            target[a] += 1.0 / len(self.neighbors_ids)

        # Calcul UCB avec rétrécissement
        exploration_term = np.sqrt(2 * np.log(self.t) / (np.array(self.nb_plays) + 1))
        # Formule de rétrécissement : sqrt(1 - Nb/t*target)
        shrinkage = np.sqrt(np.maximum(1.0 - (np.array(self.nb_plays) / (self.t * (target + 1e-6))), 0))
        
        Q = np.array(self.avg_reward) + exploration_term * shrinkage
        action = int(np.argmax(Q))
        
        reward = self.bandit.pull(action)
        self.update(action, reward)
        self.cumul_regret.append((self.cumul_regret[-1] if self.t > 1 else 0) + self.bandit.regret(action))
        return action