# experiments/run_exp.py

import argparse
import os
import sys
from vllm import LLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_reward_batch import run_batched_experiment

from utils.experiment_utils import (
    get_llm_model_name,
    load_config,
    parallel_runs,
    run_single_multi,
    save_multi_results,
    uses_llm,
)


def main():
    parser = argparse.ArgumentParser(description="Run a multi-agent observational bandit experiment from a YAML config.")
    parser.add_argument("--config",required=True,help="Path to configuration file",)
    parser.add_argument("--jobs", type=int, default=None,help="Number of parallel workers (default: from config or 1)",)
    
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_jobs = args.jobs or cfg["experiment"].get("n_jobs", 1)

    shared_model = None
    if uses_llm(cfg):
        if n_jobs > 1:
            n_jobs = 1

        model_name = get_llm_model_name(cfg)
        shared_model = LLM(model=model_name, max_model_len=4096)
        all_regrets = run_batched_experiment(cfg, shared_model=shared_model)
    else:
        all_regrets = parallel_runs(run_single_multi, cfg, n_jobs)

    save_multi_results(all_regrets, cfg)


if __name__ == "__main__":
    main()
