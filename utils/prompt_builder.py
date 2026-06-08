def build_prompt(nb_plays, arm_stats, other_actions = None):

    return f"""
[SYSTEM]

You are a multi-armed bandit algorithm solving a 2-armed Bernoulli bandit problem.

There are two arms:
- Arm 0
- Arm 1

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

Arm 0:
- Pulled {arm_stats["0"]["pulls"]} times
- Average reward: {arm_stats["0"]["reward"] / arm_stats["0"]["pulls"] if arm_stats["0"]["pulls"] > 0 else 0:.3f}

Arm 1:
- Pulled {arm_stats["1"]["pulls"]} times
- Average reward: {arm_stats["1"]["reward"] / arm_stats["1"]["pulls"] if arm_stats["1"]["pulls"] > 0 else 0:.3f}

Observed actions of other agent :

- Arm 0 selected {other_actions[0]} times
- Arm 1 selected {other_actions[1]} times

Which arm should be selected next?

Think step-by-step before answering.

Remember:
Return ONLY ONE JSON object with keys:
- action (0 or 1)
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""


# Prompt without history
def build_prompt2(nb_plays):

    return f"""
[SYSTEM]

You are a multi-armed bandit algorithm solving a 2-armed Bernoulli bandit problem.

There are two arms:
- Arm 0
- Arm 1

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

The action must be either 0 or 1.

[USER]

So far you have played {nb_plays} times.

Think step-by-step before answering.

Remember:
Return ONLY ONE JSON object with keys:
- action (0 or 1)
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""

# Prompt to imitate UCB
def build_prompt_ucb(nb_plays):

    return f"""
[SYSTEM]

You are a UCB multi-armed bandit algorithm solving a 2-armed Bernoulli bandit problem.

There are two arms:
- Arm 0
- Arm 1

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

The action must be either 0 or 1.

[USER]

So far you have played {nb_plays} times.

Think as a UCB algorithm before answering.

Remember:
Return ONLY ONE JSON object with keys:
- action (0 or 1)
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""

def build_prompt_exploit(nb_plays):
    return f"""
[SYSTEM]

You are a multi-armed bandit algorithm solving a 2-armed Bernoulli bandit problem.

There are two arms:
- Arm 0
- Arm 1

Whenever an arm is pulled, the reward is either 0 or 1.

You should always select the arm number 0 because it is the best arm.

You MUST return you answer as a valid JSON object and nothing else.

Expected format:

{{
    "action": 0,
    "explication": "your reasoning"
}}

The action must ALWAYS be 0 

[USER]

Remember:
Return ONLY ONE JSON object with keys:
- action (0 or 1)
- explication (string)
Do not add any text before or after.
Do not use markdown.
"""