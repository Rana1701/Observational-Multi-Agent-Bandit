import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    from vllm import LLM
except ImportError:
    LLM = None


class BanditEnv:
    def __init__(self, arm_means):
        self.arm_means = arm_means
        self.k = len(arm_means)

    def reward(self, action):
        p = self.arm_means[action]
        return 1 if np.random.rand() < p else 0


class Greedy5Arm:
    def __init__(self, k=5, optimal_arm=0, delta=0.2, reward_fn=None):
        self.k = k
        self.optimal_arm = optimal_arm
        self.delta = delta
        self.reward_fn = reward_fn
        self.counts = [0] * k
        self.values = [0.0] * k
        self.t = 0
        self.cumul_regret = []

    def getNextAction(self):
        self.t += 1
        if self.t <= self.k:
            action = self.t - 1
        else:
            action = int(np.argmax(self.values))

        reward = self.reward_fn(action)
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

        step_regret = 0 if action == self.optimal_arm else self.delta
        self.cumul_regret.append(step_regret if self.t == 1 else self.cumul_regret[-1] + step_regret)
        return action


class UCB5Arm:
    def __init__(self, k=5, optimal_arm=0, delta=0.2, reward_fn=None):
        self.k = k
        self.optimal_arm = optimal_arm
        self.delta = delta
        self.reward_fn = reward_fn
        self.counts = [0] * k
        self.values = [0.0] * k
        self.t = 0
        self.cumul_regret = []

    def getNextAction(self):
        self.t += 1
        if self.t <= self.k:
            action = self.t - 1
        else:
            ucb_values = [0.0] * self.k
            for a in range(self.k):
                if self.counts[a] == 0:
                    ucb_values[a] = float('inf')
                else:
                    bonus = np.sqrt(2 * np.log(self.t) / self.counts[a])
                    ucb_values[a] = self.values[a] + bonus
            action = int(np.argmax(ucb_values))

        reward = self.reward_fn(action)
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

        step_regret = 0 if action == self.optimal_arm else self.delta
        self.cumul_regret.append(step_regret if self.t == 1 else self.cumul_regret[-1] + step_regret)
        return action


class LLMAgent5Arm:
    def __init__(self, k=5, model_name="Qwen/Qwen2.5-7B-Instruct", reward_fn=None, optimal_arm=0, delta=0.2):
        self.k = k
        self.model_name = model_name
        self.reward_fn = reward_fn
        self.optimal_arm = optimal_arm
        self.delta = delta
        self.t = 0
        self.history = []
        self.cumul_regret = []
        self.model = self._load_model()

    def _load_model(self):
        if LLM is None:
            return None
        try:
            return LLM(model=self.model_name)
        except Exception:
            return None

    def _build_prompt(self):
        prompt = [
            "You are a multi-armed bandit agent choosing among 5 arms.",
            "Each arm has a fixed unknown probability of reward.",
            "Your goal is to maximize cumulative reward over time.",
            "At each step, decide which arm to pull.",
            "Answer only with a JSON object containing an integer action between 0 and 4.",
            "Example: {\"action\": 0}"
        ]
        if self.history:
            prompt.append("History of previous actions and rewards:")
            for entry in self.history[-20:]:
                prompt.append(f"Step {entry['t']}: action={entry['action']}, reward={entry['reward']}")
        prompt.append("Which arm should you choose next?")
        return "\n".join(prompt)

    def _parse_response(self, text):
        if not text:
            return None
        try:
            start = text.index('{')
            text = text[start:]
        except ValueError:
            pass
        try:
            response = json.loads(text)
            action = int(response.get('action', 0))
            if 0 <= action < self.k:
                return action
        except Exception:
            pass
        for token in text.replace(',', ' ').split():
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < self.k:
                    return idx
        return None

    def ask_model(self, prompt):
        if self.model is None:
            return None
        try:
            result = self.model.generate([prompt], max_new_tokens=64)
            return result[0].outputs[0].text
        except Exception:
            return None

    def getNextAction(self):
        self.t += 1
        prompt = self._build_prompt()
        response_text = self.ask_model(prompt)
        action = self._parse_response(response_text)
        if action is None:
            action = random.randrange(self.k)

        reward = self.reward_fn(action)
        self.history.append({"t": self.t, "action": action, "reward": reward})

        step_regret = 0 if action == self.optimal_arm else self.delta
        self.cumul_regret.append(step_regret if self.t == 1 else self.cumul_regret[-1] + step_regret)
        return action


def run_experiment(agent_class, reward_fn, nb_runs, nb_plays, **agent_kwargs):
    regrets = []
    for run in range(nb_runs):
        agent = agent_class(reward_fn=reward_fn, **agent_kwargs)
        for _ in range(nb_plays):
            agent.getNextAction()
        regrets.append(agent.cumul_regret)
    return np.array(regrets)


def main():
    np.random.seed(0)
    random.seed(0)

    k = 5
    delta = 0.2
    optimal_mean = 0.5 + delta / 2
    other_mean = 0.5 - delta / 2
    arm_means = [optimal_mean] + [other_mean] * (k - 1)
    env = BanditEnv(arm_means)

    def reward_fn(arm):
        return env.reward(arm)

    nb_runs = 20
    nb_plays = 500

    print("Running 5-arm regret experiment for Greedy, UCB and LLM...")

    greedy_regrets = run_experiment(Greedy5Arm, reward_fn, nb_runs, nb_plays, k=k, optimal_arm=0, delta=delta)
    ucb_regrets = run_experiment(UCB5Arm, reward_fn, nb_runs, nb_plays, k=k, optimal_arm=0, delta=delta)
    llm_regrets = run_experiment(LLMAgent5Arm, reward_fn, nb_runs, nb_plays, k=k, optimal_arm=0, delta=delta)

    mean_greedy = np.mean(greedy_regrets, axis=0)
    mean_ucb = np.mean(ucb_regrets, axis=0)
    mean_llm = np.mean(llm_regrets, axis=0)

    ci_greedy = 1.96 * np.std(greedy_regrets, axis=0, ddof=1) / np.sqrt(nb_runs)
    ci_ucb = 1.96 * np.std(ucb_regrets, axis=0, ddof=1) / np.sqrt(nb_runs)
    ci_llm = 1.96 * np.std(llm_regrets, axis=0, ddof=1) / np.sqrt(nb_runs)

    episodes = np.arange(1, nb_plays + 1)
    fig, ax = plt.subplots(figsize=(12, 8))

    line_greedy, = ax.plot(episodes, mean_greedy, label="Greedy", color="#ff7f0e", linewidth=2)
    ax.fill_between(episodes, mean_greedy - ci_greedy, mean_greedy + ci_greedy, color=line_greedy.get_color(), alpha=0.25)

    line_ucb, = ax.plot(episodes, mean_ucb, label="UCB", color="#1f77b4", linewidth=2)
    ax.fill_between(episodes, mean_ucb - ci_ucb, mean_ucb + ci_ucb, color=line_ucb.get_color(), alpha=0.25)

    line_llm, = ax.plot(episodes, mean_llm, label="LLM (Qwen)", color="#2ca02c", linewidth=2)
    ax.fill_between(episodes, mean_llm - ci_llm, mean_llm + ci_llm, color=line_llm.get_color(), alpha=0.25)

    ax.set_xlabel("Plays", fontsize=14)
    ax.set_ylabel("Cumulative regret", fontsize=14)
    ax.set_title("Mean Cumulative Regret + 95% CI", fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "Krishnamurthy_5arms.png")
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    print(f"Figure saved as '{out_file}'")


if __name__ == "__main__":
    main()
