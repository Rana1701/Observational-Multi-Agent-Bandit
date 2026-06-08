import os
import sys
from vllm import LLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.ucb import UCB as UCBAgent
from agents.e_greedy import E_Greedy as GreedyAgent
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

    std_ucb = np.std(ucb_runs, axis=0, ddof=1)
    std_greedy = np.std(greedy_runs, axis=0, ddof=1)
    std_llm = np.std(llm_runs, axis=0, ddof=1)

    ci_ucb = 1.96 * std_ucb / np.sqrt(nb_runs)
    ci_greedy = 1.96 * std_greedy / np.sqrt(nb_runs)
    ci_llm = 1.96 * std_llm / np.sqrt(nb_runs)

    episodes = np.arange(nb_plays)
    fig, ax = plt.subplots(figsize=(12, 8))

    line_ucb, = ax.plot(episodes, mean_ucb, label="UCB", linewidth=2)
    ax.fill_between(
        episodes,
        mean_ucb - ci_ucb,
        mean_ucb + ci_ucb,
        color=line_ucb.get_color(),
        alpha=0.2,
        linewidth=0,
    )

    line_greedy, = ax.plot(episodes, mean_greedy, label="Greedy", linewidth=2)
    ax.fill_between(
        episodes,
        mean_greedy - ci_greedy,
        mean_greedy + ci_greedy,
        color=line_greedy.get_color(),
        alpha=0.2,
        linewidth=0,
    )

    line_llm, = ax.plot(episodes, mean_llm, label="LLM", linewidth=2)
    ax.fill_between(
        episodes,
        mean_llm - ci_llm,
        mean_llm + ci_llm,
        color=line_llm.get_color(),
        alpha=0.2,
        linewidth=0,
    )

    ax.set_xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Average Cumulative Regret of 3 Agents over 10 runs", fontsize=20)
    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "LLM vs UCB vs Greedy.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    print(f"Figure saved as '{out_file}'")

if __name__ == "__main__":
    main()