import json
import numpy as np
import random
from vllm import LLM 
from vllm import SamplingParams 


class LLMAgent:
    def __init__(self, name_parameter="Qwen/Qwen2.5-7B-Instruct", model=None, reward_fn=None):
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        
        self.error = 0 # count parsing errors
        self.explanation = ""
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
                "{\"action\": 0 or 1, \"explanation\": \"Your explanation here\"}"
                "Return ONLY one valid JSON object."
                "Do not output multiple JSON objects."
                "Do not use markdown."
                "Do not write any text before or after."
            )

        self.cumul_regret = []
        self.t = 0
        self.delta = 0.2

    def ask(self, prompt):
        if self.model is None:
            print("Model loading failed. Using fallback action.")
            return {"action": random.choice([0, 1]), "explanation": "no model loaded - fallback action"}

        try:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=64
            )

            result = self.model.generate(
                [prompt],
                sampling_params
            )

            response = result[0].outputs[0].text

            #print("RAW RESPONSE:")
            #print(response)

        except Exception as e:
            print("GENERATION ERROR:", repr(e))
            return {"action": random.choice([0, 1]), "explanation": "model generation failed"}

        try:
            return self.extract_json(response)
        except:
            self.error += 1
            return {
                "action": random.choice([0, 1]),
                "explanation": "parse failed"
            }

    def getNextAction(self, prompt=None):
        if prompt is None:
            prompt = self.default_prompt
        
        self.t += 1

        try:
            response = self.ask(prompt)
        except Exception:
            response = {"action": 0, "explanation": "default fallback action"}

        action = response.get('action', 0)
        self.explanation = response.get('explanation', '')
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

    def extract_json(self, response):
        import json
        import re

  
        # 1. EXTRAIRE ACTION (JSON)
        start = response.find("{")
        action = None

        if start != -1:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(response[start:i+1])
                            if "action" in obj:
                                action = obj.get("action")
                                break
                        except:
                            pass

        if action is None:
            action = random.choice([0, 1])

   
        # 2. EXTRAIRE EXPLICATION (TEXT HEURISTIC)
        exp = ""

        match = re.search(r"(explication|explanation)\s*:\s*(.*)", response, re.IGNORECASE | re.DOTALL)

        if match:
            exp = match.group(2).strip()
        else:
            exp = "no explanation found"

        return {
            "action": action,
            "explication": exp
        }

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
    import matplotlib.pyplot as plt
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    
    '''Runs a LLM agent for 100 plays'''
    agent = LLMAgent(model=llm)  
    
    for _ in range(100):
        agent.getNextAction()
    print (f"Total parsing errors: {agent.error}")

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

