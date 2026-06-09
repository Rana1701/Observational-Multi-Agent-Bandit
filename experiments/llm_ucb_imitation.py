import os
import sys
import numpy as np
from vllm import LLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.prompt_builder import build_prompt , build_prompt2, build_prompt_ucb, build_prompt_ucb_history
from agents.llm import LLMAgent
from agents.ucb import UCB as UCBAgent
from utils.reward_function import reward_fn

def main():
    import matplotlib.pyplot as plt
    from pathlib import Path


    nb_runs = 20
    nb_plays = 100
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    regrets_history_runs = []
    regrets_ucb_llm_runs = []
    regrets_ucb_runs = []
    regrets_llm_runs = []

    llm = LLM(model=model_name)

    for run in range(nb_runs):

        other_actions = [0, 0]
        
        agent1 = LLMAgent(model=llm)
        agent2 = LLMAgent(model=llm)
        agent3 = UCBAgent(reward_fn=reward_fn())
        agent4 = LLMAgent(model=llm)

        for t in range(nb_plays):

            # --- LLM imitating ucb + using history ---
            prompt1 = build_prompt_ucb_history(agent1.t, agent1.history)
            agent1.getNextAction(prompt1)

            # --- LLM imitating ucb - without history ---
            prompt2 = build_prompt_ucb(agent2.t)
            agent2.getNextAction(prompt2)

            # --- UCB ---
            ucb_action = agent3.getNextAction()

            # LLM just using history (no UCB imitation)
            prompt3 = build_prompt(agent4.t, agent4.history, other_actions)
            agent4.getNextAction(prompt3)


            # Mettre à jour les actions observées
            other_actions[ucb_action] += 1

        regrets_history_runs.append(agent1.cumul_regret)
        regrets_ucb_llm_runs.append(agent2.cumul_regret)
        regrets_ucb_runs.append(agent3.cumul_regret)
        regrets_llm_runs.append(agent4.cumul_regret)

        print(f"Run {run + 1}/{nb_runs} completed")

    # convert to numpy
    regrets_history_runs = np.array(regrets_history_runs)
    regrets_ucb_llm_runs = np.array(regrets_ucb_llm_runs)
    regrets_ucb_runs = np.array(regrets_ucb_runs)
    regrets_llm_runs = np.array(regrets_llm_runs)

    # means
    mean_history = np.mean(regrets_history_runs, axis=0)
    mean_ucb_llm = np.mean(regrets_ucb_llm_runs, axis=0)
    mean_ucb = np.mean(regrets_ucb_runs, axis=0)
    mean_llm = np.mean(regrets_llm_runs, axis=0)

    # std
    std_history = np.std(regrets_history_runs, axis=0, ddof=1)
    std_ucb_llm = np.std(regrets_ucb_llm_runs, axis=0, ddof=1)
    std_ucb = np.std(regrets_ucb_runs, axis=0, ddof=1)
    std_llm = np.std(regrets_llm_runs, axis=0, ddof=1)

    # CI 95%
    ci_history = 1.96 * std_history / np.sqrt(nb_runs)
    ci_ucb_llm = 1.96 * std_ucb_llm / np.sqrt(nb_runs)
    ci_ucb = 1.96 * std_ucb / np.sqrt(nb_runs)
    ci_llm = 1.96 * std_llm / np.sqrt(nb_runs)

    x = np.arange(nb_plays)

    plt.figure(figsize=(10, 6))

    l1, = plt.plot(x, mean_history, label="LLM imitating UCB + History")
    plt.fill_between(x, mean_history - ci_history, mean_history + ci_history,
                     alpha=0.2, color=l1.get_color())

    l2, = plt.plot(x, mean_ucb_llm, label="LLM imitating UCB")
    plt.fill_between(x, mean_ucb_llm - ci_ucb_llm, mean_ucb_llm + ci_ucb_llm,
                     alpha=0.2, color=l2.get_color())

    l3, = plt.plot(x, mean_ucb, label="UCB")
    plt.fill_between(x, mean_ucb - ci_ucb, mean_ucb + ci_ucb,
                     alpha=0.2, color=l3.get_color())

    l4, = plt.plot(x, mean_llm, label="LLM with history (no UCB imitation)")
    plt.fill_between(x, mean_llm - ci_llm, mean_llm + ci_llm,
                     alpha=0.2, color=l4.get_color())

    plt.xlabel("Plays")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Bandit Comparison ({nb_runs} runs)")
    plt.legend()
    plt.grid(alpha=0.3)

    base_dir = Path(__file__).resolve().parent.parent
    out_file = base_dir / "figs" / "LLM_UCB_imitation.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_file)
    plt.show()

    print(f"Figure saved as '{out_file}'")


if __name__ == "__main__":
    main()