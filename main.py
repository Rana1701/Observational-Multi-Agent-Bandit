from agents.ucb import UCB as UCBAgent
from agents.greedy import Greedy as GreedyAgent
from agents.llm import LLMAgent
from utils.reward_function import reward_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    
    agents = []
    nb_agents = 3
    nb_plays = 50
    actions = [0,1]

    reward = reward_fn(0.6, 0.4)
    delta = 0.6 - 0.4

    ucb = UCBAgent(reward_fn=reward, delta=delta)
    greedy = GreedyAgent(reward_fn=reward, delta=delta)
    llm = LLMAgent()

    agents.append(ucb)
    agents.append(greedy)
    agents.append(llm)

    for t in range(nb_plays):
        ucb_action = ucb.getNextAction()
        greedy_action = greedy.getNextAction([ucb_action])
        llm_action = llm.getNextAction()
        llm.update_history("ucb", ucb_action)
        llm.update_history("greedy", greedy_action)

    #The display  
    plt.figure(figsize=(12,8))
    i=1

    plt.plot(ucb.cumul_regret, label = "UCB")
    plt.plot(greedy.cumul_regret, label = "Greedy")
    plt.plot(llm.cumul_regret, label = "LLM")

    plt.xlabel("Plays", fontsize = 14)
    plt.ylabel("Cumulative regret", fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14)
    plt.title("Cumulative Regret of 3 Agents in a Fully Connected Graph", fontsize = 20)
    plt.savefig("Agents_cumul_regret.png")
    print("Figure saved as 'Agents_cumul_regret.png'")

if __name__ == "__main__":
    main()