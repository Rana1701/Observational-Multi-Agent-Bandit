# experiments/run.py

import argparse
from pathlib import Path

import numpy as np
import yaml

from environnement.bernoulli_bandit import BernoulliBandit

from agents.ucb import UCB
from agents.tucb import TUCB
from agents.greedy import Greedy
from agents.e_greedy import EpsilonGreedy
from agents.llm import LLMAgent


AGENTS = {
    "UCB": UCB,
    "TUCB": TUCB,
    "Greedy": Greedy,
    "EpsilonGreedy": EpsilonGreedy,
    "LLM": LLMAgent,
}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(cfg):

    runs = cfg["experiment"]["runs"]
    horizon = cfg["experiment"]["horizon"]

    all_regrets = []

    for _ in range(runs):

        bandit = BernoulliBandit(
            **cfg["environment"]
        )

        agent_class = AGENTS[cfg["agent"]]

        agent = agent_class(
            bandit,
            **cfg.get("agent_params", {})
        )

        for _ in range(horizon):
            agent.getNextAction()

        all_regrets.append(agent.cumul_regret)

    return np.asarray(all_regrets)


def save_results(regrets, cfg):

    output_dir = Path(
        cfg["experiment"].get(
            "output_dir",
            f"results/{cfg['agent'].lower()}"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "regrets.npy",
        regrets
    )

    np.save(
        output_dir / "mean_regret.npy",
        regrets.mean(axis=0)
    )

    np.save(
        output_dir / "std_regret.npy",
        regrets.std(axis=0)
    )

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Results saved to {output_dir}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    regrets = run_experiment(cfg)

    save_results(regrets, cfg)


if __name__ == "__main__":
    main()