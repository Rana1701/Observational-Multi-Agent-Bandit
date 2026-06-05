def build_prompt(nb_plays, arm_stats):

    return f"""
[SYSTEM]

You are a multi-armed bandit algorithm solving a 2-armed Bernoulli bandit problem.

There are two arms:
- Arm 0
- Arm 1

Each arm has an unknown but fixed probability of returning reward 1.
Whenever an arm is pulled, the reward is either 0 or 1.

Your objective is to maximize cumulative reward over time.

At every step, you are shown a summary of your past experience.
Use this information to balance exploration and exploitation.

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

Summary of past observations:

Arm 0:
- Pulled {arm_stats["0"]["pulls"]} times
- Average reward: {arm_stats["0"]["reward"] / arm_stats["0"]["pulls"] if arm_stats["0"]["pulls"] > 0 else 0:.3f}

Arm 1:
- Pulled {arm_stats["1"]["pulls"]} times
- Average reward: {arm_stats["1"]["reward"] / arm_stats["1"]["pulls"] if arm_stats["1"]["pulls"] > 0 else 0:.3f}

Which arm should be selected next?

Think step-by-step before answering.

Remember:
- Return ONLY a valid JSON object.
- The action must be either 0 or 1.
"""