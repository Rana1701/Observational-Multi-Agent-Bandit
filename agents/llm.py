import json
from utils.reward_function import reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMAgent:
    def __init__(self, name_parameter="Qwen/Qwen2.5-7B-Instruct"):
        self.model, self.tokenizer = self.charging_model(name_parameter)
        self.target = {}
        self.history = ""
        self.default_prompt = ""
        self.cumul_regret = []
        self.t = 0
        self.delta = 0.2
        self.reward_fn = reward_fn()

    def ask(self, prompt):
        if self.model is None or self.tokenizer is None:
            return {"action": 0, "explication": "no model loaded - default action"}

        model, tokenizer = self.model, self.tokenizer

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        # generating the model's response
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        #Converting to json  
        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"action": 0, "explication": "default action"}

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
        model_name = name_parameter
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def getReward(self, arm_played):
        return self.reward_fn(arm_played)

        
def main():
    import matplotlib.pyplot as plt
    
    '''Runs a LLM agent for 10 plays'''
    agent = LLMAgent()  
    
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

