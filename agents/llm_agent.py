import random

class llm_agent():
    def __init__(self, nb_arms, reward_fn, delta=0.2):
        self.nb_arms = nb_arms
        self.reward_fn = reward_fn
        self.delta = delta
        self.t = 0
        self.cumul_regret = []
        self.prompt = "You are a multi-armed bandit agent. You have " + str(self.nb_arms) + " arms to choose from."
        " Each arm has a certain probability of giving a reward."
        " Your goal is to maximize your cumulative reward over time."
        " At each time step, you can observe the precedent actions of the other agents,"
        " but not their rewards."
        " Based on the actions of the other agents and your own experience,"
        " you need to decide which arm to pull next."

    def getNextAction(self, other_agents_actions):
        self.t += 1
        action = random.choice(range(self.nb_arms))
        step_reward = self.getReward(action)
        step_regret = 0 if action == 0 else self.delta
        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)
        return action

    def getReward(self, arm_played):
        return self.reward_fn(arm_played)
