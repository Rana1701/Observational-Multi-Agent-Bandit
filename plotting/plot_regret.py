# plotting/plot_regret.py

import numpy as np
import matplotlib.pyplot as plt

regrets = np.load("results/ucb/regrets.npy")

mean = regrets.mean(axis=0)
std = regrets.std(axis=0)

ci95 = 1.96 * std / np.sqrt(len(regrets))

x = np.arange(len(mean))

plt.plot(x, mean, label="UCB")
plt.fill_between(
    x,
    mean - ci95,
    mean + ci95,
    alpha=0.2
)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.legend()

plt.savefig("results/ucb/regret_plot.png")
plt.show()