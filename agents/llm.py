import json
import random
from vllm import LLM
from utils.reward_function import reward_fn

class LLMAgent:
    def __init__(self, name_parameter="Qwen/Qwen2.5-7B-Instruct", model=None):
        self.model = model if model is not None else self.charging_model(name_parameter)
        self.target = {}
        self.history = ""
        self.default_prompt = ""
        self.cumul_regret = []
        self.t = 0
        self.delta = 0.2
        self.reward_fn = reward_fn()

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

    def getNextAction(self):
        self.t += 1

        if self.t == 1:
            self.default_prompt = (
                "You are a multi-armed bandit agent. You have 2 arms to choose from."
                " Each arm has a certain probability of giving a reward."
                " Your goal is to maximize your cumulative reward over time."
                " At each time step, you can observe the previous actions of the other agents,"
                " but not their rewards."
                " Based on the actions of the other agents and your own experience,"
                " you need to decide which arm to pull next."
                "Please answer in the following JSON format: "
                "{\"action\": 0, \"explication\": \"Your explanation here\"}"
            )

        prompt = self.default_prompt + f" Your history of actions and rewards is as follows:\n{self.history}\n"
        prompt += "Based on this history, and the previous actions of the other agents, which arm should you pull next? "


        try:
            response = self.ask(prompt)
        except Exception:
            response = {"action": 0, "explication": "default fallback action"}

        action = response.get('action', 0)
        step_reward = self.getReward(action)
        self.history += f"Action: {action}, Reward: {step_reward}\n"

        if action == 0:
            step_regret = 0
        else:
            step_regret = self.delta

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action

    def update_history(self, model, action):
        self.history += f"{model} - Action: {action}\n"

    def charging_model(self, name_parameter="Qwen/Qwen2.5-7B-Instruct"):
        try:
            return LLM(model=name_parameter)
        except Exception:
            return None

    def getReward(self, arm_played):
        return self.reward_fn(arm_played)

        
def main():
    import matplotlib.pyplot as plt
    
    # Test vllm simple
    print("Testing vLLM...")
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    outputs = llm.generate(["The capital of France is"])
    text = outputs[0].outputs[0].text
    print(f"vLLM output: {text}")
    
    '''Runs a LLM agent for 10 plays'''
    agent = LLMAgent(model=llm)  
    
    for _ in range(10):
        agent.getNextAction()
    
    plt.figure(figsize=(10, 6))
    plt.plot(agent.cumul_regret, label="LLM Agent ")
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of LLM Agent", fontsize=20)
    plt.savefig("LLM_cumul_regret.png")
    plt.show()


if __name__ == "__main__":
    main()

