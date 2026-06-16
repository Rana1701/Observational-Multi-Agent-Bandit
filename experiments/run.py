# experiments/run.py

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.common import (
    create_agent,
    load_config,
    parallel_runs,
    run_single_solo,
    save_solo_results,
    uses_llm,
    AGENTS,
    build_bandit,
    run_seed,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-agent bandit experiment from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: from config or 1)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_jobs = args.jobs or cfg["experiment"].get("n_jobs", 1)

    if uses_llm(cfg):
        if n_jobs > 1:
            print("LLM experiments run sequentially (model cannot be shared across workers).")
            n_jobs = 1

        from vllm import LLM

        model_name = cfg.get("agent_params", {}).get("model", "Qwen/Qwen2.5-7B-Instruct")
        print(f"Loading LLM: {model_name}")
        shared_model = LLM(model=model_name)

        regrets = [
            run_single_solo(cfg, i, shared_model)
            for i in range(cfg["experiment"]["runs"])
        ]
    else:
        regrets = parallel_runs(run_single_solo, cfg, n_jobs)

    save_solo_results(regrets, cfg)


if __name__ == "__main__":
    main()
