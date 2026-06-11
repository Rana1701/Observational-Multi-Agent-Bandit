# experiments/run_multi.py

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.common import (
    load_config,
    parallel_runs,
    run_single_multi,
    save_multi_results,
    uses_llm,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-agent observational bandit experiment from a YAML config."
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

    shared_model = None
    if uses_llm(cfg):
        if n_jobs > 1:
            print("LLM experiments run sequentially (model cannot be shared across workers).")
            n_jobs = 1

        from vllm import LLM

        model_name = None
        for agent_cfg in cfg["agents"]:
            if agent_cfg.get("class") == "LLM":
                model_name = agent_cfg.get("params", {}).get("model")
                break

        if model_name is None:
            model_name = "Qwen/Qwen2.5-7B-Instruct"

        print(f"Loading LLM: {model_name}")
        shared_model = LLM(model=model_name)

        def run_with_model(config, run_idx):
            return run_single_multi(config, run_idx, shared_model=shared_model)

        all_regrets = parallel_runs(run_with_model, cfg, n_jobs)
    else:
        all_regrets = parallel_runs(run_single_multi, cfg, n_jobs)

    save_multi_results(all_regrets, cfg)


if __name__ == "__main__":
    main()
