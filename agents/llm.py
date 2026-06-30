import json
import os
import sys
import numpy as np
import random
from vllm import LLM 
from vllm import SamplingParams 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environnement.bernoulli_bandit import BernoulliBandit
 

class LLMAgent:
    def __init__(self,bandit,  name_parameter="Qwen/Qwen2.5-7B-Instruct", model=None):

        self.bandit = bandit
        self.error = 0 # count parsing errors
        self.explanation = ""
        self.model = model if model is not None else self.charging_model(name_parameter)
        self.target = {}
        self.history = {str(i): {"pulls": 0, "reward": 0} for i in range(self.bandit.n_arms)}

        self.default_prompt = (
                f"You are a multi-armed bandit agent. You have {self.bandit.n_arms} arms to choose from."
                " Each arm has a certain probability of giving a reward."
                " Your goal is to maximize your cumulative reward over time."
                " At each time step, you can observe the previous actions of the other agents,"
                " but not their rewards."
                " Based on the actions of the other agents and your own experience,"
                " you need to decide which arm to pull next."
                "Please answer in the following JSON format: "
                "{\"action\": integer in [0, K-1] \"explanation\": \"Your explanation here\"}"
                "Return ONLY one valid JSON object."
                "Do not output multiple JSON objects."
                "Do not use markdown."
                "Do not write any text before or after."
            )

        self.cumul_regret = []
        self.t = 0

    def ask(self, prompt):
        if self.model is None:
            print("Model loading failed. Using fallback action.")
            return {"action": 0, "explanation": "no model loaded - fallback action"}

        try:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=32
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
            return {"action": 0, "explanation": "model generation failed"}

        try:
            return self.extract_json(response)
        except:
            self.error += 1
            return {
                "action": 0,
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

        step_regret = self.bandit.regret(action)

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action

    def extract_json(self, response):
        import json
        import re

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return {"action": 0, "explanation": "parse failed"}

        try:
            obj = json.loads(match.group(0))
            return {
                "action": obj.get("action", 0),
                "explanation": obj.get("explanation", "no explanation")
            }
        except:
            return {"action": 0, "explanation": "invalid json"}

    def charging_model(self, name_parameter="Qwen/Qwen2.5-7B-Instruct"):
        try:
            return LLM(model=name_parameter)
        except Exception as e:
            print("Erreur lors du chargement du modèle :")
            print(repr(e))
            return None


    def getReward(self, arm_played):
        return self.bandit.pull(arm_played)


        
def main():
    import matplotlib.pyplot as plt
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    
    '''Runs a LLM agent for 100 plays'''
    agent = LLMAgent(BernoulliBandit(n_arms=2, best_mean=0.5, delta=0.2), model=llm)  
    
    import time

    t0 = time.perf_counter()

    for _ in range(100):
        agent.getNextAction()

    t1 = time.perf_counter()

    print(f"TOTAL TIME: {t1 - t0:.3f}s")
    print(f"AVG TIME / STEP: {(t1 - t0)/100:.3f}s")
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

