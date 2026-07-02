from os import times

def build_prompt_krishnamurthy(
    bandit,
    nb_plays,
    arm_stats,
    other_actions=None,
    scenario="B",
    framing="N",
    history="R",
    cot="N",
    return_distribution="0",
):
    # L1: B/A = buttons / advertisements
    # L2: N/S = neutral / suggestive
    # L3: R/S = raw history / summarized history
    # L4: N/C/Ce = no CoT / chain-of-thought / reinforced CoT
    # L5: 0/1/D = temperature 0 / temperature 1 / distribution (with temp 0)
    scenario = str(scenario).upper()
    if scenario not in {"B", "A"}:
        scenario = "B"

    framing = str(framing).upper()
    if framing not in {"N", "S"}:
        framing = "N"

    history = str(history).upper()
    if history not in {"R", "S"}:
        history = "R"

    cot = str(cot)
    if cot.lower() == "ce":
        cot = "Ce"
    else:
        cot = cot.upper()
    if cot not in {"N", "C", "Ce"}:
        cot = "N"

    return_distribution = str(return_distribution)
    if return_distribution not in {"0", "1", "D"}:
        return_distribution = "0"

    if other_actions is None:
        other_actions = [0] * bandit.n_arms
    elif len(other_actions) < bandit.n_arms:
        other_actions = list(other_actions) + [0] * (bandit.n_arms - len(other_actions))

    system = f"""
[SYSTEM]
You are solving a {bandit.n_arms}-armed Bernoulli bandit problem.
Scenario: {scenario}.
Framing: {framing}.
"""
    if cot == "C":
        system += "Use chain-of-thought reasoning.\n"
    elif cot == "Ce":
        system += "Use chain-of-thought reasoning and explain step-by-step.\n"
    if return_distribution:
        system += "Return a distribution over actions instead of a single arm.\n"

    user = f"""
[USER]
So far you have played {nb_plays} times.
"""

    if history == "raw":
        user += "\nYour history so far:\n"
        for i in range(bandit.n_arms):
            pulls = arm_stats[str(i)]["pulls"]
            reward = arm_stats[str(i)]["reward"]
            avg = reward / pulls if pulls > 0 else 0.0
            user += f"- Arm {i}: pulled {pulls} times, average reward {avg:.3f}.\n"
    else:
        total_pulls = sum(arm_stats[str(i)]["pulls"] for i in range(bandit.n_arms))
        total_reward = sum(arm_stats[str(i)]["reward"] for i in range(bandit.n_arms))
        avg_reward = total_reward / total_pulls if total_pulls > 0 else 0.0
        user += f"\nSummarized history: {total_pulls} pulls so far, average reward {avg_reward:.3f}.\n"
        user += "Include which arms have been pulled more often and which have higher observed reward.\n"

    if other_actions is not None:
        user += "\nObserved actions by other agents:\n"
        for i in range(bandit.n_arms):
            user += f"- Arm {i} selected {other_actions[i]} times by other agents.\n"

    if return_distribution:
        user += "\nReturn a probability distribution over actions as a JSON array."
        expected_format = "{\n    \"distribution\": [0.0, ..., 1.0],\n    \"explication\": \"your reasoning\"\n}"
    else:
        user += "\nWhich arm should be selected next?"
        expected_format = "{\n    \"action\": 0,\n    \"explication\": \"your reasoning\"\n}"

    if cot == "Ce":
        user += "\nPlease reason step-by-step before answering.\n"
    else:
        user += "\nThink carefully before answering.\n"

    user += f"\nRemember:\nReturn ONLY ONE JSON object with keys:\n"
    if return_distribution:
        user += f"- distribution (list of {bandit.n_arms} probabilities)\n"
    else:
        user += f"- action (0 ... {bandit.n_arms - 1})\n"
    user += "- explication (string)\nDo not add any text before or after. Do not use markdown.\n"
    user += "\nExpected format:\n" + expected_format + "\n"

    return system + "\n" + user


def build_prompt_history(bandit, nb_plays, arm_stats, other_actions = None):
    prompt = ""
    prompt += f"""
[SYSTEM]

You are a multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Each arm has an unknown but fixed probability of returning reward 1.
Whenever an arm is pulled, the reward is either 0 or 1.

Your objective is to maximize cumulative reward over time.

You can observe:
1. Your own history of actions and rewards.
2. The actions selected by other agents (but NOT their rewards).

Use all available information to balance exploration and exploitation.

You MUST return a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

[USER]

So far you have played {nb_plays} times.

Your personal observations:

    """
    for i in range(bandit.n_arms):
        prompt += f"""
    Arm {i}:
    - Pulled {arm_stats[str(i)]["pulls"]} times
    - Average reward: {arm_stats[str(i)]["reward"] / arm_stats[str(i)]["pulls"] if arm_stats[str(i)]["pulls"] > 0 else 0:.3f}

    """
    if other_actions is not None:
        prompt += f"Observed actions of other agent(s) :"
        for i in range(bandit.n_arms):
            prompt += f"""
        - Arm {i} selected {other_actions[i]} times
        """
        
    prompt += f"""
    Which arm should be selected next?

    Think step-by-step before answering.

    Remember:
    Return ONLY ONE JSON object with keys:
    - action (0 ... {bandit.n_arms - 1})
    - explication (string)
    Do not add any text before or after.
    Do not use markdown.
    """
    return prompt
# Prompt without history
def build_prompt_noHistory(bandit, nb_plays, arm_stats= None, other_actions = None):

    return f"""
[SYSTEM]

You are a multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Each arm has an unknown but fixed probability of returning reward 1.
Whenever an arm is pulled, the reward is either 0 or 1.

Your objective is to maximize cumulative reward over time.

Think carefully about which arm is most promising.
Reason step-by-step before making a decision.

You MUST return a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

The action must be between 0 and {bandit.n_arms - 1}

[USER]

So far you have played {nb_plays} times.

Think step-by-step before answering.

Remember:
Return ONLY ONE JSON object with keys:
- action (0 ... {bandit.n_arms - 1})
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""

# Prompt to imitate UCB
def build_prompt_ucb_noHistory(bandit, nb_plays, arm_stats= None, other_actions = None):

    return f"""
[SYSTEM]

You are a UCB multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Each arm has an unknown but fixed probability of returning reward 1.
Whenever an arm is pulled, the reward is either 0 or 1.

Your objective is to maximize cumulative reward over time.

You should act exactly like a UCB algorithm.

You MUST return you answer as a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

The action must be between 0 and {bandit.n_arms - 1}.

[USER]

So far you have played {nb_plays} times.

Think as a UCB algorithm before answering.

Remember:
Return ONLY ONE JSON object with keys:
- action (0 ... {bandit.n_arms - 1})
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""

def build_prompt_exploit(bandit, nb_plays = None, arm_stats = None, other_actions = None):
    return f"""
[SYSTEM]

You are a greedy multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Whenever an arm is pulled, the reward is either 0 or 1.

You should try all arms then stick to the arm with the highest observed average reward.

You MUST return you answer as a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

[USER]

Remember:
Return ONLY ONE JSON object with keys:
- action (0 ... {bandit.n_arms - 1})
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""

def build_prompt_explore(bandit, nb_plays = None, arm_stats = None, other_actions = None):
    return f"""
[SYSTEM]

You are an exploring multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Whenever an arm is pulled, the reward is either 0 or 1.

You should explore frequently and avoid sticking to a single arm.

You MUST return you answer as a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

[USER]

Remember:
Return ONLY ONE JSON object with keys:
- action (0 ... {bandit.n_arms - 1})
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""

def build_prompt_ucb_history(bandit, nb_plays, arm_stats, other_actions = None):
    prompt = ""
    prompt += f"""
[SYSTEM]

You are a UCB multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1} 

Each arm has an unknown but fixed probability of returning reward 1.
Whenever an arm is pulled, the reward is either 0 or 1.

Your objective is to maximize cumulative reward over time by playing according to the UCB strategy.

Act like you are a UCB algorithm.

You MUST return a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

[USER]

So far you have played {nb_plays} times.

Your personal observations:
"""
    for i in range(bandit.n_arms):
        prompt += f"""
Arm {i}:
- Pulled {arm_stats[str(i)]["pulls"]} times
- Average reward: {arm_stats[str(i)]["reward"] / arm_stats[str(i)]["pulls"] if arm_stats[str(i)]["pulls"] > 0 else 0:.3f}
"""
    prompt += f"""

Which arm should be selected next?

Think step-by-step before answering.

Remember:
Return ONLY ONE JSON object with keys:
- action (0 ... {bandit.n_arms - 1})
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""
    return prompt