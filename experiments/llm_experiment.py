import json
import os
import sys
import numpy as np
import random
from vllm import LLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.prompt_builder import build_prompt 
from agents.llm import LLMAgent

class LLMAgent1:
    def __init__(self, name_parameter="Qwen/Qwen2.5-7B-Instruct", model=None, reward_fn=None):
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn

        self.model = model if model is not None else self.charging_model(name_parameter)
        self.target = {}
        self.history = {"0": {"pulls": 0, "reward": 0}, "1": {"pulls": 0, "reward": 0}}
        self.default_prompt = (
                "You are a multi-armed bandit agent. You have 2 arms to choose from."
                " Each arm has a certain probability of giving a reward."
                " Your goal is to maximize your cumulative reward over time."
                " At each time step, you can observe the previous actions of the other agents,"
                " but not their rewards."
                " Based on the actions of the other agents and your own experience,"
                " you need to decide which arm to pull next."
                "Please answer in the following JSON format: "
                "{\"action\": 0 or 1, \"explication\": \"Your explanation here\"}"
            )

        self.cumul_regret = []
        self.t = 0
        self.delta = 0.2

    def ask(self, prompt):
        if self.model is None:
            print("Model loading failed. Using fallback action.")
            return {"action": random.choice([0, 1]), "explication": "no model loaded - fallback action"}

        try:
            result = self.model.generate([prompt], max_new_tokens=64)
            response = result[0].outputs[0].text
        except Exception:
            return {"action": random.choice([0, 1]), "explication": "model generation failed"}

        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"action": random.choice([0, 1]), "explication": "default fallback action"}

    def getNextAction(self, prompt=None):
        if prompt is None:
            prompt = self.default_prompt
        
        self.t += 1

        try:
            response = self.ask(prompt)
        except Exception:
            response = {"action": 0, "explication": "default fallback action"}

        action = response.get('action', 0)
        step_reward = self.getReward(action)

        self.history[str(action)]["pulls"] += 1
        self.history[str(action)]["reward"] += step_reward

        if action == 0:
            step_regret = 0
        else:
            step_regret = self.delta

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action


    def charging_model(self, name_parameter="Qwen/Qwen2.5-7B-Instruct"):
        try:
            return LLM(model=name_parameter)
        except Exception:
            return None

    def _default_reward_fn(self, arm_played):
        win_rate = [0.6, 0.4]
        pull = np.random.rand()
        if arm_played == 0 and pull < win_rate[0]:
            return 1
        elif arm_played == 1 and pull < win_rate[1]:
            return 1
        return 0

    def getReward(self, arm_played):
        return self.reward_fn(arm_played)


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from vllm import LLM

    nb_runs = 20
    nb_plays = 100
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    regrets_default_runs = []
    regrets_prompt_runs = []

    llm = LLM(model=model_name)

    for run in range(nb_runs):

        
        agent_default = LLMAgent(model=llm)
        agent_prompt = LLMAgent1(model=llm)

        for t in range(nb_plays):

            # default
            agent_default.getNextAction()

            # prompt-based 
            prompt = build_prompt(agent_prompt.t, agent_prompt.history)
            agent_prompt.getNextAction(prompt)

        regrets_default_runs.append(agent_default.cumul_regret)
        regrets_prompt_runs.append(agent_prompt.cumul_regret)

        print(f"Run {run + 1}/{nb_runs} completed")

    regrets_default_runs = np.array(regrets_default_runs)
    regrets_prompt_runs = np.array(regrets_prompt_runs)

    mean_default = np.mean(regrets_default_runs, axis=0)
    mean_prompt = np.mean(regrets_prompt_runs, axis=0)

    std_default = np.std(regrets_default_runs, axis=0, ddof=1)
    std_prompt = np.std(regrets_prompt_runs, axis=0, ddof=1)

    ci_default = 1.96 * std_default / np.sqrt(nb_runs)
    ci_prompt = 1.96 * std_prompt / np.sqrt(nb_runs)

    x = np.arange(nb_plays)

    plt.figure(figsize=(10, 6))

    l1, = plt.plot(x, mean_default, label="Default Prompt")
    plt.fill_between(x, mean_default - ci_default, mean_default + ci_default,
                     alpha=0.2, color=l1.get_color())

    l2, = plt.plot(x, mean_prompt, label="Summarized Prompt")
    plt.fill_between(x, mean_prompt - ci_prompt, mean_prompt + ci_prompt,
                     alpha=0.2, color=l2.get_color())

    plt.xlabel("Plays")
    plt.ylabel("Cumulative Regret")
    plt.title(f"LLM Bandit Comparison ({nb_runs} runs)")
    plt.legend()
    plt.grid(alpha=0.3)

    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "LLM_comparison.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_file)
    plt.show()
    print(f"Figure saved as '{out_file}'")

if __name__ == "__main__":
    main()