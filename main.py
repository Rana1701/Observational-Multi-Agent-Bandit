from vllm import LLM

from agents.ucb import UCB as UCBAgent
from agents.greedy import Greedy as GreedyAgent
from agents.llm import LLMAgent
from utils.reward_function import reward_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    import numpy as np
    from pathlib import Path

    nb_runs = 10
    nb_plays = 1000
    delta = 0.6 - 0.4

    ucb_runs = []
    greedy_runs = []
    llm_runs = []


    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    
    for _ in range(nb_runs):
        reward = reward_fn(0.6, 0.4)
        ucb = UCBAgent(reward_fn=reward, delta=delta)
        greedy = GreedyAgent(reward_fn=reward, delta=delta)
        llm = LLMAgent(model=llm)

        for t in range(nb_plays):
            ucb_action = ucb.getNextAction()
            greedy_action = greedy.getNextAction()
            llm_action = llm.getNextAction()
            llm.update_history("ucb", ucb_action)
            llm.update_history("greedy", greedy_action)

        ucb_runs.append(ucb.cumul_regret)
        greedy_runs.append(greedy.cumul_regret)
        llm_runs.append(llm.cumul_regret)

    mean_ucb = np.mean(ucb_runs, axis=0)
    mean_greedy = np.mean(greedy_runs, axis=0)
    mean_llm = np.mean(llm_runs, axis=0)

    plt.figure(figsize=(12, 8))
    plt.plot(mean_ucb, label="UCB")
    plt.plot(mean_greedy, label="Greedy")
    plt.plot(mean_llm, label="LLM")

    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Average Cumulative Regret of 3 Agents over 10 runs", fontsize=20)

    out_file = Path(__file__).resolve().parent / "figs" / "Agents_cumul_regret.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    print(f"Figure saved as '{out_file}'")

if __name__ == "__main__":
    main()