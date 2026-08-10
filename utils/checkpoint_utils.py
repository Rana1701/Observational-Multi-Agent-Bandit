import os
import pickle
import numpy as np


def build_checkpoint_entry(state):
    agent_states = {}
    for name, agent in state["agents"].items():
        agent_states[name] = {
            "history": getattr(agent, "history", None),
            "t": getattr(agent, "t", None),
            "error": getattr(agent, "error", None),
            "explanation": getattr(agent, "explanation", None),
            "cumul_regret": getattr(agent, "cumul_regret", None),
        }

    return {
        "history": state["history"],
        "other_counts": state["other_counts"],
        "reward": state["reward"],
        "regret": state["regret"],
        "rewards_ts": state["rewards_ts"],
        "regrets_ts": state["regrets_ts"],
        "agent_states": agent_states,
    }


def save_checkpoint(states, step, checkpoint_file):
    checkpoint = {
        "step": step,
        "np_random_state": np.random.get_state(),
        "states": [build_checkpoint_entry(state) for state in states],
    }

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        return pickle.load(f)


def restore_states_from_checkpoint(cfg, checkpoint, shared_model, init_run_fn):
    runs = cfg["experiment"].get("runs", 20)
    if len(checkpoint["states"]) != runs:
        raise ValueError(
            f"Checkpoint contains {len(checkpoint['states'])} runs, "
            f"but current config expects {runs} runs."
        )

    states = []
    for run_idx, entry in enumerate(checkpoint["states"]):
        state = init_run_fn(cfg, run_idx, shared_model)

        state["history"] = entry["history"]
        state["other_counts"] = entry["other_counts"]
        state["reward"] = entry["reward"]
        state["regret"] = entry["regret"]
        state["rewards_ts"] = entry["rewards_ts"]
        state["regrets_ts"] = entry["regrets_ts"]

        for name, agent in state["agents"].items():
            agent_state = entry["agent_states"].get(name, {})

            if agent_state.get("history") is not None:
                agent.history = agent_state["history"]
            if agent_state.get("t") is not None:
                agent.t = agent_state["t"]
            if agent_state.get("error") is not None:
                agent.error = agent_state["error"]
            if agent_state.get("explanation") is not None:
                agent.explanation = agent_state["explanation"]
            if agent_state.get("cumul_regret") is not None:
                agent.cumul_regret = agent_state["cumul_regret"]

        states.append(state)

    return states
