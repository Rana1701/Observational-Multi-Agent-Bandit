import numpy as np
import argparse
import os
import sys
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnement.bernoulli_bandit import BernoulliBandit
from utils.experiment_utils import (
    load_config,
    save_multi_results,
    create_agent,
    AGENTS,
    build_llm_prompt,
    uses_llm,
    get_llm_model_name,
    run_seed,
)

def batch_generate(model, prompts):
    if not prompts:
        return []
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        top_p=0.9,
        stop=["}"],
    )
    outputs = model.generate(prompts, sampling_params)
    responses = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        if not text.endswith("}"):
            text += "}"
        responses.append(text)
    return responses

def init_runs(config, shared_model):
    exp = config["experiment"]
    runs = exp.get("runs", 20)
    states = []
    probs = config["environment"].get("probs")

    for run_id in range(runs):
        seed = run_seed(exp.get("seed"), run_id)
        
        # CORRECTION : Utilisation d'un générateur d'aléa local par simulation
        rng = np.random.RandomState(seed)

        bandit = BernoulliBandit(probs=probs)
        # Injection du rng local dans le bandit si sa classe le permet
        if hasattr(bandit, 'rng'):
            bandit.rng = rng

        agents = {}
        cfg_by_name = {}

        for a in config["agents"]:
            name = a["name"]
            cfg_by_name[name] = a
            agents[name] = create_agent(
                AGENTS[a["class"]],
                bandit,
                a.get("params"),
                shared_model=shared_model,
            )
            # Injection du rng local dans l'agent pour stabiliser TS / UCB
            if hasattr(agents[name], 'rng'):
                agents[name].rng = rng

        state = {
            "rng": rng, # Sauvegarde du générateur
            "bandit": bandit,
            "agents": agents,
            "cfg_by_name": cfg_by_name,
            "order": exp.get("order", [a["name"] for a in config["agents"]]),
            "cumulative": {name: 0.0 for name in agents},
            "rewards_ts": {name: [] for name in agents},
            "other_action_counts": [0] * bandit.n_arms,
            "current_actions": {},
            "opt_curve": [],
            "opt_cum": 0.0,
        }
        states.append(state)
    return states

def run_batched_experiment(config, model):
    exp = config["experiment"]
    horizon = exp["horizon"]
    interaction = exp.get("interaction", "sequential")
    states = init_runs(config, model)

    for t in range(horizon):
        
        # ----------------------------------------------------
        # ÉTAPE 1 : AGENTS NON-LLM EN PREMIER (Si Séquentiel)
        # ----------------------------------------------------
        if interaction == "sequential":
            for state in states:
                # Configuration temporaire du seed global pour ce run précis
                np.random.set_state(state["rng"].get_state())
                
                for name in state["order"]:
                    cfg = state["cfg_by_name"][name]
                    agent = state["agents"][name]
                    
                    if cfg.get("class") != "LLM":
                        # En mode séquentiel, l'agent observe les actions précédentes du pas t
                        observed = [
                            state["current_actions"][o]
                            for o in cfg.get("observes", [])
                            if o in state["current_actions"]
                        ]
                        action = agent.getNextAction(observed or None)
                        state["current_actions"][name] = action
                
                state["rng"].set_state(np.random.get_state())

        # ----------------------------------------------------
        # ÉTAPE 2 : COLLECTE ET EXÉCUTION DU BATCH LLM
        # ----------------------------------------------------
        prompts = []
        llm_refs = []

        for state in states:
            for name in state["order"]:
                cfg = state["cfg_by_name"][name]
                agent = state["agents"][name]
                
                if cfg.get("class") == "LLM":
                    cfg["_other_action_counts"] = state["other_action_counts"]
                    prompt = build_llm_prompt(cfg, agent)
                    prompts.append(prompt)
                    llm_refs.append((state, name, agent))

        responses = batch_generate(model, prompts)

        # Application des réponses LLM
        for (state, name, agent), response in zip(llm_refs, responses):
            action = agent.getNextActionFromResponse(response)
            
            # CORRECTION : On s'assure d'appeler getReward pour déclencher le bandit
            reward = agent.getReward(action) 
            
            agent.history[str(action)]["pulls"] += 1
            agent.history[str(action)]["reward"] += reward
            agent.t += 1
            state["current_actions"][name] = action

            if t > 98 : 
                print("parsing errors for agent", name, ":", agent.error)

        # ----------------------------------------------------
        # ÉTAPE 3 : AGENTS NON-LLM (Si Simultané)
        # ----------------------------------------------------
        if interaction != "sequential":
            for state in states:
                np.random.set_state(state["rng"].get_state())
                for name in state["order"]:
                    cfg = state["cfg_by_name"][name]
                    agent = state["agents"][name]
                    if cfg.get("class") != "LLM":
                        action = agent.getNextAction(None)
                        state["current_actions"][name] = action
                state["rng"].set_state(np.random.get_state())

        # ----------------------------------------------------
        # ÉTAPE 4 : CALCUL DES RECOMPENSES ET ADVERSAIRES
        # ----------------------------------------------------
        for state in states:
            # CORRECTION : Forcer la mise à jour des récompenses manquantes pour les non-LLM
            for name in state["order"]:
                cfg = state["cfg_by_name"][name]
                agent = state["agents"][name]
                action = state["current_actions"][name]
                
                if cfg.get("class") != "LLM":
                    # Déclenche le pull du bandit pour l'agent standard et met à jour son score
                    if hasattr(agent, 'update'): 
                        # Dépend de votre implémentation de create_agent, souvent géré par bandit.pull
                        reward = state["bandit"].pull(action)
                        agent.reward = reward
                        # Si votre agent a besoin d'apprendre du reward :
                        if hasattr(agent, 'update'): agent.update(action, reward)
                
                reward = agent.reward
                state["cumulative"][name] += reward
                state["rewards_ts"][name].append(state["cumulative"][name] / (t + 1))

            # Courbe optimale
            state["opt_cum"] += np.max(state["bandit"].probs)
            state["opt_curve"].append(state["opt_cum"] / (t + 1))

            # Suivi des actions des autres agents
            for name in exp.get("track_other_actions_for", state["order"]):
                action = state["current_actions"].get(name)
                if action is not None and 0 <= action < len(state["other_action_counts"]):
                    state["other_action_counts"][action] += 1

            # Reset pour le prochain pas de temps
            state["current_actions"] = {}

    # Formatage identique des résultats
    results = []
    for state in states:
        out = {name: np.array(data) for name, data in state["rewards_ts"].items()}
        out["OPT"] = np.array(state["opt_curve"])
        results.append(out)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    if uses_llm(config):
        model_name = get_llm_model_name(config)
        print(f"Loading LLM {model_name} once...")
        model = LLM(model=model_name, max_model_len=4096)
        results = run_batched_experiment(config, model)
    else:
        raise RuntimeError("This file is only for LLM experiments")

    save_multi_results(results, config)

if __name__ == "__main__":
    main()
