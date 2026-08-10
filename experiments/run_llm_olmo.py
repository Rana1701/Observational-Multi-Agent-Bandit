import argparse
import os
import sys
import random
import numpy as np
import pickle
from multiprocessing import Pool

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.prompt_builder import request_response
from utils.experiment_utils import (
    load_config,
    run_seed,
    build_bandit,
    create_agent,
    AGENTS,
    build_llm_prompt,
    uses_llm,
    get_llm_model_name,
    save_multi_results,
)
from utils.checkpoint_utils import (
    build_checkpoint_entry,
    save_checkpoint,
    load_checkpoint,
    restore_states_from_checkpoint,
)


class SamplingParams:
    def __init__(self, temperature=0, max_tokens=1024, top_p=0.9, stop=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop or []


class HFGenerateResultOutput:
    def __init__(self, text):
        class O:
            def __init__(self, text):
                self.text = text

        self.outputs = [O(text)]


class HuggingFaceLLMWrapper:
    """Minimal wrapper that exposes a vLLM-like `generate(prompts, params)` API
    backed by `transformers` so existing `LLMAgent` code can reuse it.
    """
    def __init__(self, model_name, device=None, max_model_len=4096):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_model_len = max_model_len

        print(f"Loading HF model {model_name} on device {self.device} ...")

        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        def _retry_load(load_fn, *args, **kwargs):
            max_retries = 5
            delay = 1.0
            for attempt in range(max_retries + 1):
                try:
                    return load_fn(*args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if any(k in msg.lower() for k in ("timed out", "timeout", "connection", "socket")):
                        print(f"Attempt {attempt+1}/{max_retries+1} failed (network): {msg}")
                        if attempt == max_retries:
                            raise
                        time.sleep(delay)
                        delay *= 2
                        continue
                    raise

        tokenizer_kwargs = {"use_fast": True}
        if hf_token:
            tokenizer_kwargs["use_auth_token"] = hf_token

        try:
            self.tokenizer = _retry_load(AutoTokenizer.from_pretrained, model_name, **tokenizer_kwargs)
        except Exception:
            print("Failed to download tokenizer from HF. Trying local cache only...")
            try:
                tokenizer_kwargs_local = dict(tokenizer_kwargs)
                tokenizer_kwargs_local["local_files_only"] = True
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs_local)
            except Exception as e:
                raise RuntimeError(
                    "Unable to load tokenizer from Hugging Face (network error and no cached copy). "
                    "If the model is private, set HUGGINGFACE_HUB_TOKEN environment variable. "
                    f"Original error: {e}"
                )

        model_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        if hf_token:
            model_kwargs["use_auth_token"] = hf_token

        try:
            self.model = _retry_load(AutoModelForCausalLM.from_pretrained, model_name, **model_kwargs)
        except Exception:
            print("Failed to download model from HF. Trying local cache only...")
            try:
                model_kwargs_local = dict(model_kwargs)
                model_kwargs_local["local_files_only"] = True
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs_local)
            except Exception as e:
                raise RuntimeError(
                    "Unable to load model from Hugging Face (network error and no cached copy). "
                    "If the model is private, set HUGGINGFACE_HUB_TOKEN environment variable. "
                    f"Original error: {e}"
                )

        if self.device == "cpu":
            self.model.to("cpu")

    def generate(self, prompts, sampling_params: SamplingParams):
        results = []
        max_new_tokens = int(sampling_params.max_tokens)

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Deterministic generation for temperature=0, else sample
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(sampling_params.temperature > 0),
                    top_p=float(sampling_params.top_p),
                    temperature=float(sampling_params.temperature),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            # If stop token present in text, trim there (simple handling)
            for stop_tok in sampling_params.stop:
                idx = text.find(stop_tok)
                if idx != -1:
                    text = text[: idx + len(stop_tok)]
                    break

            results.append(HFGenerateResultOutput(text))

        return results


def batch_generate(model, prompts):
    """Generate multiple LLM responses in one call using the wrapper."""
    if not prompts:
        return []

    params = SamplingParams(temperature=0, max_tokens=1024, top_p=0.9, stop=["</Answer>"])
    outputs = model.generate(prompts, params)
    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("</Answer>"):
            text += "</Answer>"
        responses.append(text)

    return responses


def init_run(cfg, run_idx, shared_model=None):
    """Initialize one experiment run."""
    exp = cfg["experiment"]
    seed = run_seed(exp.get("seed"), run_idx)
    random.seed(seed)
    np.random.seed(seed)
    bandit = build_bandit(cfg["environment"], seed)

    order = exp.get("order") or [
        a["name"] for a in cfg["agents"]
    ]

    agents = {}
    cfg_by_name = {}

    for a in cfg["agents"]:
        name = a["name"]
        cfg_by_name[name] = a

        agents[name] = create_agent(
            AGENTS[a["class"]],
            bandit,
            a.get("params"),
            shared_model=shared_model,
        )

    best_reward = (
        np.max(bandit.probs)
        if hasattr(bandit, "probs")
        else cfg["environment"].get("best_mean", 0.9)
    )

    return {
        "bandit": bandit,
        "agents": agents,
        "cfg": cfg_by_name,
        "order": order,
        "history": {n: [] for n in order},
        "other_counts": [0] * bandit.n_arms,
        "reward": {n: 0.0 for n in order},
        "regret": {n: 0.0 for n in order},
        "rewards_ts": {n: [] for n in order},
        "regrets_ts": {n: [] for n in order},
        "best": best_reward,
    }


def run_single_rep(task, shared_model=None):
    """Run one experiment replica (used for non LLM experiments)."""
    run_idx, cfg = task
    state = init_run(cfg, run_idx, shared_model)
    horizon = cfg["experiment"]["horizon"]

    for t in range(horizon):
        actions = {}

        for name in state["order"]:
            agent = state["agents"][name]
            agent_cfg = state["cfg"][name]

            if agent_cfg.get("class") == "LLM":
                agent_cfg["_other_action_counts"] = state["other_counts"].copy()
                action = agent.getNextAction(build_llm_prompt(agent_cfg, agent))

            elif agent_cfg.get("observes"):
                obs = [
                    state["history"][o][t]
                    for o in agent_cfg["observes"]
                    if len(state["history"][o]) > t
                ]

                action = agent.getNextAction(obs or None)

            else:
                action = agent.getNextAction()

            actions[name] = action
            state["history"][name].append(action)

            reward = agent.reward

            state["reward"][name] += reward
            state["rewards_ts"][name].append(state["reward"][name] / (t + 1))

            state["regret"][name] += (
                state["bandit"].regret(action)
                if hasattr(state["bandit"], "regret")
                else state["best"] - state["bandit"].probs[action]
            )

            state["regrets_ts"][name].append(state["regret"][name])

        for name in cfg["experiment"].get("track_other_actions_for", []):
            if name in actions:
                state["other_counts"][actions[name]] += 1

    return format_result(state, cfg)


def init_batched_runs(cfg, model):
    """Initialize all runs sharing the same LLM."""
    states = []
    for run_idx in range(cfg["experiment"].get("runs", 20)):
        states.append(init_run(cfg, run_idx, model))

    return states



def run_batched_llm_experiment(cfg, model, resume=False):
    """Run multiple replicas with batched LLM inference."""
    save_dir = cfg["experiment"]["output_dir"]
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = os.path.join(save_dir, "checkpoint_results.pkl")
    horizon = cfg["experiment"]["horizon"]
    track = cfg["experiment"].get("track_other_actions_for", [])

    start_step = 0
    states = None

    if resume and os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        states = restore_states_from_checkpoint(cfg, checkpoint, model, init_run)
        start_step = checkpoint["step"]
        np.random.set_state(checkpoint["np_random_state"])
        print(f"Resuming from checkpoint at step {start_step}/{horizon}")

    if states is None:
        states = init_batched_runs(cfg, model)
        start_step = 0

    if start_step >= horizon:
        print("Checkpoint already contains completed experiment. Loading final results.")
        return [format_result(state, cfg) for state in states]

    for t in range(start_step, horizon):
        prompts = []
        refs = []

        # Collect all LLM requests from all runs
        for state in states:
            for name in state["order"]:
                agent_cfg = state["cfg"][name]

                if agent_cfg.get("class") == "LLM":
                    agent_cfg["_other_action_counts"] = state["other_counts"].copy()

                    prompts.append(build_llm_prompt(agent_cfg, state["agents"][name]))
                    refs.append((state, name))

        # One batched generation for all runs
        responses = batch_generate(model, prompts)
        actions_by_state = {}

        # Première génération : CoT
        for (state, name), response in zip(refs, responses):
            state["agents"][name].explanation = response

        # Deuxième génération : extraction
        extract_prompts = []
        for (state, name), response in zip(refs, responses):
            extract_prompts.append(request_response(state["bandit"].n_arms, response))

        extract_responses = batch_generate(model, extract_prompts)

        # Parsing des réponses finales
        for (state, name), answer in zip(refs, extract_responses):
            agent = state["agents"][name]
            action = agent.extract_reponse(answer)

            actions_by_state.setdefault(id(state), {})[name] = action

        # Compute non LLM actions
        for state in states:
            actions = actions_by_state.setdefault(id(state), {})

            for name in state["order"]:
                agent_cfg = state["cfg"][name]
                if agent_cfg.get("class") != "LLM":
                    agent = state["agents"][name]
                    if agent_cfg.get("observes"):
                        obs = [
                            state["history"][o][t]
                            for o in agent_cfg["observes"]
                            if len(state["history"][o]) > t
                        ]

                        action = agent.getNextAction(obs or None)
                    else:
                        action = agent.getNextAction()

                    actions[name] = action

        # Update all runs
        for state in states:
            actions = actions_by_state.get(id(state), {})
            for name, action in actions.items():
                agent = state["agents"][name]
                # LLM response already updates reward internally only partially
                # so update reward here for batched execution
                if state["cfg"][name].get("class") == "LLM":
                    reward = agent.getReward(action)
                    agent.history[str(action)]["pulls"] += 1
                    agent.history[str(action)]["reward"] += reward
                    agent.t += 1
                    if t == horizon - 1:
                        print(f"number of parsing errors : {agent.error}")
                else:
                    reward = agent.reward
                state["history"][name].append(action)
                state["reward"][name] += reward
                state["rewards_ts"][name].append(state["reward"][name] / (t + 1))

                state["regret"][name] += (
                    state["bandit"].regret(action)
                    if hasattr(state["bandit"], "regret")
                    else state["best"] - state["bandit"].probs[action]
                )
                state["regrets_ts"][name].append(state["regret"][name])

            for name in track:
                if name in actions:
                    state["other_counts"][actions[name]] += 1

        save_checkpoint(states, t + 1, checkpoint_file)
        print(f"Checkpoint saved at step {t + 1}/{horizon}")

    results = []

    for idx, state in enumerate(states):
        result = format_result(state, cfg)
        results.append(result)
        run_file = os.path.join(save_dir, f"run_{idx}.pkl")

        with open(run_file, "wb") as f:
            pickle.dump(result, f)

        print(f"Saved completed run {idx+1}/{len(states)}")

    return results


def format_result(state, cfg):
    """Convert state to experiment output format."""
    out = {
        "time_averaged_rewards": {},
        "cumulated_regrets": {},
    }

    for name in state["order"]:
        out["time_averaged_rewards"][name] = np.array(state["rewards_ts"][name])
        out["cumulated_regrets"][name] = np.array(state["regrets_ts"][name])

    if cfg["experiment"].get("add_opt", False):
        horizon = cfg["experiment"]["horizon"]
        out["time_averaged_rewards"]["OPT"] = np.ones(horizon) * state["best"]
        out["cumulated_regrets"]["OPT"] = np.zeros(horizon)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent experiment (Olmo/HF mode).")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from the last checkpoint if available",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    exp = cfg["experiment"]
    runs = exp.get("runs", 20)

    results = []

    if uses_llm(cfg):
        print("Loading Hugging Face model once for Olmo/other recent models...")
        model = HuggingFaceLLMWrapper(model_name=get_llm_model_name(cfg), max_model_len=4096)
        results = run_batched_llm_experiment(cfg, model, resume=args.resume)

    else:
        print(f"Running {runs} replicas...")
        tasks = [(i, cfg) for i in range(runs)]
        with Pool(processes=exp.get("n_jobs", 4)) as pool:
            results = pool.map(run_single_rep, tasks)

    save_multi_results(results, cfg)
    print(f"Saved {len(results)} runs")


if __name__ == "__main__":
    main()