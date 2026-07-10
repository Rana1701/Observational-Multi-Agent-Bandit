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
                "CRITICAL: Return only ONE JSON object. No markdown, no extra text before/after."
            )

        self.cumul_regret = []
        self.t = 0

    #extract response and updates stats 
    def extract_reponse(self, response_text):
        self.t += 1
        response = None
        action = None
        colors = ["blue","green","red","yellow","purple","orange","black","white"]

        self.explanation = response_text

        # 1. JSON format case
        try:
            if not response_text.endswith("}"):
                response_text += "}"
            response = self.extract_json(response_text)
            action = response.get("action", 0)
        except Exception:
            pass

        # 2. <Answer><Answer> Format case
        if action is None:
            try:
                # Recherche du texte contenu entre <Answer> et </Answer>
                match = re.search(r"<Answer>(.*?)</Answer>", response_text, re.DOTALL)
                
                if match:
                    answer_content = match.group(1).strip()
                    
                    choices = colors[:self.bandit.n_arms]

                    # Convert color to its index
                    if answer_content in choices:
                        action = choices.index(answer_content)
                    
                    # Si la réponse est directement l'entier sous forme de texte (ex: "3")
                    elif answer_content.isdigit():
                        action = int(answer_content)
                    
                    # Si le texte de la balise ne correspond à rien de connu
                    else:
                        raise ValueError("Content inside tags does not match any valid action")
                else:
                    raise ValueError("No <Answer> tags found")

            except Exception:
                self.error += 1
                action = 2
                print(f"Unrecognized answer content: {answer_content}. Defaulting to action 0. Parsing errors = {self.error}")

        # 3. Traitement des récompenses et historique (inchangé)
        step_reward = self.getReward(action)

        self.history[str(action)]["pulls"] += 1
        self.history[str(action)]["reward"] += step_reward

        step_regret = self.bandit.regret(action)

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

        return action


    def ask(self, prompt):
        if self.model is None:
            print("Model loading failed. Using fallback action.")
            return {"action": 0, "explanation": "no model loaded - fallback action"}

        try:
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=512,
                top_p=0.9,
                stop=["}"]
            )

            result = self.model.generate(
                [prompt],
                sampling_params
            )

            response = result[0].outputs[0].text.strip()
            
            # Only add closing brace if not already present
            if not response.rstrip().endswith("}"):
                response += "}"
            
            if self.t < 10: # Print the first few responses for debugging
                print("RAW RESPONSE:")
                print(response)

        except Exception as e:
            print("GENERATION ERROR:", repr(e))
            self.error += 1
            return {"action": 0, "explanation": "model generation failed"}

        try:
            return self.extract_reponse(response)
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

        #print(f"Action {action} , Explanation: {self.explanation}")
        if self.t >498: 
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

