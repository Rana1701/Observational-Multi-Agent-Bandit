import json
import os
import sys
import numpy as np
import random
from vllm import LLM 
from vllm import SamplingParams 
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.prompt_builder import request_response
 

class LLMAgent:
    def __init__(self,bandit,  name_parameter="Qwen/Qwen2.5-7B-Instruct", model=None):

        self.bandit = bandit
        self.reward = 0
        self.error = 0 
        self.explanation = ""
        self.model = model if model is not None else self.charging_model(name_parameter)
        self.target = {}
        self.history = {str(i): {"pulls": 0, "reward": 0} for i in range(self.bandit.n_arms)}

        self.default_prompt = (
                f"You are a multi-armed bandit agent with {self.bandit.n_arms} arms.\n"
                "Maximize cumulative reward. Base decisions on other agents' observed actions and your experience.\n"
                "Respond ONLY with valid JSON, nothing else:\n"
                f'{{\"action\": <int 0 to {self.bandit.n_arms-1}>, \"explication\": \"<reason>\"}}\n'
                "CRITICAL: Return only <Answer>Color</Answer> . No markdown, no extra text before/after."
            )

        self.cumul_regret = []
        self.t = 0

    # seperate the asking task from the response treating one
    def getNextActionFromResponse(self, response):
        """
        Parse a batched LLM response and update agent statistics.
        """

        self.explanation = response

        try:
            action = self.extract_reponse(response)

        except Exception:
            self.error += 1
            action = 0

        reward = self.getReward(action)

        self.history[str(action)]["pulls"] += 1
        self.history[str(action)]["reward"] += reward

        self.t += 1

        regret = self.bandit.regret(action)

        if len(self.cumul_regret) > 0:
            self.cumul_regret.append(
                self.cumul_regret[-1] + regret
            )
        else:
            self.cumul_regret.append(regret)

        return action

    #extract response and updates stats 
    def extract_reponse(self, response_text):
        response = None
        action = None
        colors = ["blue", "green", "red", "yellow", "purple", "orange", "black", "white"]

        self.explanation = response_text

        # 1. JSON format case
        try:
            response = self.extract_json(response_text)
            action = response.get("action", None)
        except Exception:
            pass

        # 2. <Answer>COLOR</Answer> format case
        if action is None:
            answer_content = None
            try:
                # Accept:
                # <Answer>blue</Answer>
                # <Answer>BLUE</Answer>
                # <Answer>Blue</Answer>
                # <Answer> bLuE </Answer>
                match = re.search(r"<Answer>(.*?)</Answer>", response_text, re.DOTALL | re.IGNORECASE)

                if match:
                    answer_content = match.group(1).strip().lower()
                    choices = colors[:self.bandit.n_arms]

                    # Convert color to index
                    if answer_content in choices:
                        action = choices.index(answer_content)

                    # If answer is a digit (ex: "3")
                    elif answer_content.isdigit():
                        action = int(answer_content)
                    else:
                        raise ValueError(f"Unknown color: {answer_content}")

                else:
                    raise ValueError("No <Answer> tags found")

            except Exception as e:
                self.error += 1
                action = 0
                print(f"Unrecognized answer content: {answer_content}. "
                    f"Defaulting to action 0. Parsing errors = {self.error}. Error: {e}")

        return action
        
    def ask(self, prompt):
        if self.model is None:
            print("Model loading failed. Using fallback action.")
            return {"action": 0, "explanation": "no model loaded - fallback action"}

        try:
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1024,
                top_p=0.9,
                stop=["</Answer>"]
            )

            result = self.model.generate(
                [prompt],
                sampling_params
            )

            response = result[0].outputs[0].text.strip()
            response += "</Answer>"
        except Exception as e:
            print("GENERATION ERROR:", repr(e))
            self.error += 1
            return "action: 0, explanation: model generation failed"

        try:
            return response
        except:
            self.error += 1
            return "action: 0, explanation: parsing failed"

    def getNextAction(self, prompt=None):
        self.t += 1

        try:
            #print("="*80)
            #print(prompt)
            #print("="*80)
            self.explanation = self.ask(prompt)
        except Exception:
            print ("action: 0, explanation: ask method failed" )
            return 0

        try:
            print("="*80)
            print(f"Explanation (reponse 1 ): {self.explanation}")
            print("="*80)

            """
            response = self.ask(request_response())
            
            print("="*80)
            print(f"Reponse 2 {response} ")
            print("="*80)
            """
            action = self.extract_reponse(self.explanation)
        except Exception:
            print ("action: 0, explanation: response extraction failed" )
            return 0            

        # Updating stats
        step_reward = self.getReward(action)

        self.history[str(action)]["pulls"] += 1
        self.history[str(action)]["reward"] += step_reward

        step_regret = self.bandit.regret(action)

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        print(f"parse errors: {self.error} ")
        return action

    def charging_model(self, name_parameter="Qwen/Qwen2.5-7B-Instruct"):
        try:
            return LLM(model=name_parameter)
        except Exception as e:
            print("Erreur lors du chargement du modèle :")
            print(repr(e))
            return None


    def getReward(self, arm_played):
        self.reward= self.bandit.pull(arm_played)
        return self.reward
    
    


        
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

