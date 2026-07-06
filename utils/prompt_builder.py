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

    return_distribution = str(return_distribution).upper()
    distribution_mode = return_distribution == "D"

    if other_actions is None:
        other_actions = [0] * bandit.n_arms
    elif len(other_actions) < bandit.n_arms:
        other_actions = list(other_actions) + [0] * (bandit.n_arms - len(other_actions))

    if bandit.n_arms == 5:
        if scenario == "B":
            choices = ["blue", "green", "red", "yellow", "purple"]
            action_unit = "button"
            answer_type = "COLOR"
        else:
            choices = ["A", "B", "C", "D", "E"]
            action_unit = "advertisement"
            answer_type = "DIST" if distribution_mode else "CHOICE"
    else:
        choices = [f"arm {i}" for i in range(bandit.n_arms)]
        action_unit = "arm"
        answer_type = "DIST" if distribution_mode else "ACTION"

    choice_list = ", ".join(choices)

    system = "[SYSTEM]\n"
    if scenario == "B":
        if framing == "N":
            system += (
                f"You are in a room with {bandit.n_arms} buttons labeled {choice_list}.\n"
                "Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the buttons could be different.\n"
                "For each button, when you press it, you will get a reward that is sampled from the button's associated distribution.\n"
                "At each time step, you can choose any button and receive the reward. Your goal is to maximize the total reward over the time steps.\n"
                "At each time step, I will show you your past choices and rewards.\n"
                f"Then you must make the next choice, which must be exactly one of {choice_list}.\n"
            )
        else:
            system += (
                f"You are a bandit algorithm in a room with {bandit.n_arms} buttons labeled {choice_list}.\n"
                "Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the buttons could be different.\n"
                "For each button, when you press it, you will get a reward that is sampled from the button's associated distribution.\n"
                "At each time step, you can choose any button and receive the reward. Your goal is to maximize the total reward over the time steps.\n"
                "At each time step, I will show you your past choices and rewards.\n"
                f"Then you must make the next choice, which must be exactly one of {choice_list}.\n"
            )
    else:
        system += (
            f"You are a recommendation engine that chooses advertisements to display to users when they visit your webpage.\n"
            f"There are {bandit.n_arms} advertisements you can choose from, named {choice_list}.\n"
            "When a user visits the webpage you can choose an advertisement to display and you will observe whether the user clicks on the ad or not.\n"
            "You model this by assuming that each advertisement has a certain click rate and users click on advertisements with their corresponding rates.\n"
            "You have a budget of 10 users to interact with and your goal is to maximize the total number of clicks during this process.\n"
            "A good strategy to optimize for clicks in these situations requires balancing exploration and exploitation.\n"
            "When each user visits the webpage, I will show you a summary of the data you have collected so far.\n"
            "Then you must choose which advertisement to display.\n"
        )

    if distribution_mode:
        if scenario == "B":
            format_example = ",".join([f"{name}:n" for name in choices])
            system += f"You may output a distribution over the {bandit.n_arms} choices formatted EXACTLY like \"{format_example}\".\n"
        else:
            format_example = ",".join([f"{name}:n" for name in choices])
            system += f"You may output a distribution over the {bandit.n_arms} choices formatted EXACTLY like \"{format_example}\".\n"
    else:
        if scenario == "B":
            system += (
                f"You must provide your final answer immediately within the tags <Answer>{answer_type}</Answer> where {answer_type} is one of {choice_list} and with no text explanation.\n"
            )
        else:
            system += (
                f"You must provide your final answer immediately within the tags <Answer>{answer_type}</Answer> where {answer_type} is one of {choice_list} and with no text explanation.\n"
            )

    user = "[USER]\n"
    if history == "R":
        if scenario == "B":
            user += f"So far you have played {nb_plays} times with the following choices and rewards:\n"
            for i, name in enumerate(choices):
                pulls = arm_stats[str(i)]["pulls"]
                reward = arm_stats[str(i)]["reward"]
                if pulls > 0:
                    avg = reward / pulls
                    user += f"{name} {action_unit}, total reward {reward}, average reward {avg:.2f}\n"
        else:
            user += f"So far you have interacted with {nb_plays} users. Here is the data you have collected:\n"
            for i, name in enumerate(choices):
                pulls = arm_stats[str(i)]["pulls"]
                reward = arm_stats[str(i)]["reward"]
                if pulls > 0:
                    ctr = reward / pulls
                    user += f"Advertisement {name} was shown to {pulls} users with an estimated click rate of {ctr:.2f}\n"
                else:
                    user += f"Advertisement {name} has not been shown\n"
    else:
        if scenario == "B":
            user += f"So far you have played {nb_plays} times with your past choices and rewards summarized as follows:\n"
            for i, name in enumerate(choices):
                pulls = arm_stats[str(i)]["pulls"]
                reward = arm_stats[str(i)]["reward"]
                if pulls > 0:
                    avg = reward / pulls
                    user += f"{name} {action_unit}: pressed {pulls} times with average reward {avg:.2f}\n"
                else:
                    user += f"{name} {action_unit}: pressed 0 times\n"
        else:
            user += f"So far you have interacted with {nb_plays} users. Here is a summary of the data you have collected:\n"
            for i, name in enumerate(choices):
                pulls = arm_stats[str(i)]["pulls"]
                reward = arm_stats[str(i)]["reward"]
                if pulls > 0:
                    ctr = reward / pulls
                    user += f"Advertisement {name} was shown to {pulls} users with an estimated click rate of {ctr:.2f}\n"
                else:
                    user += f"Advertisement {name} has not been shown\n"

    if other_actions is not None and any(count > 0 for count in other_actions):
        user += "\nOther agents have selected actions as follows:\n"
        for i, count in enumerate(other_actions):
            if count > 0:
                other_name = choices[i] if i < len(choices) else f"arm {i}"
                user += f"- {other_name}: selected {count} times by other agents.\n"

    if distribution_mode:
        user += f"\nWhich {action_unit} will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>DIST</Answer> where DIST is formatted like \"{format_example}\".\n"
    else:
        user += f"\nWhich {action_unit} will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>{answer_type}</Answer> where {answer_type} is one of {choice_list}.\n"

    if cot in {"C", "Ce"}:
        user += "Let’s think step by step to make sure we make a good choice.\n"

    return system + "\n" + user


def build_prompt_history(bandit, nb_plays, arm_stats, other_actions = None):
    prompt = ""
    prompt += f"""
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
    "action": x,
    "explication": "your reasoning step by step"
}}

So far you have played {nb_plays} times.

Statistics : here is your personal history of actions and rewards:

    """
    for i in range(bandit.n_arms):
        prompt += f"""
    Arm {i}:
    - Pulled {arm_stats[str(i)]["pulls"]} times
    - Average reward: {arm_stats[str(i)]["reward"] / arm_stats[str(i)]["pulls"] if arm_stats[str(i)]["pulls"] > 0 else 0:.3f}

    """
    if other_actions is not None:
        prompt += f"Total observed actions of other agent(s) :"
        for i in range(bandit.n_arms):
            prompt += f"""
        - Arm {i} selected {other_actions[i]} times
        """
        
    prompt += f"""
    Which arm should be selected next?

    Remember:
    Return ONLY ONE JSON object with keys:
    - action (0 ... {bandit.n_arms - 1})
    - explication (string)
    Do not return the prompt in your answer
    """
    prompt += f"""
    ---
    FINAL INSTRUCTION: Generate EXACTLY ONE JSON object below and STOP immediately after the closing }}.
    Do NOT generate multiple responses, do NOT use markdown code blocks.
    Your single response:"""
    return prompt

# Prompt without history
def build_prompt_noHistory(bandit, nb_plays, arm_stats= None, other_actions = None):

    return f"""
You are a multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Each arm has an unknown but fixed probability of returning reward 1.
Whenever an arm is pulled, the reward is either 0 or 1.
Your objective is to maximize cumulative reward over time.

You MUST return a valid JSON object and nothing else. Expected format:

{{
    "action": x,
    "explication": "your reasoning step by step"
}}

The action must be between 0 and {bandit.n_arms - 1}
So far you have played {nb_plays} times.

Remember:
- Return ONLY ONE JSON object with keys:
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
    "action": x,
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
    "action": x,
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
    prompt = ""
    prompt += f"""

You are an exploring multi-armed bandit algorithm solving a {bandit.n_arms}-armed Bernoulli bandit problem.

There are {bandit.n_arms} arms:
- Arm 0 ... Arm {bandit.n_arms - 1}

Whenever an arm is pulled, the reward is either 0 or 1.

You should explore frequently and avoid sticking to a single arm.
Never replay the same arm more than 2 times in a row.
Even if an arm has a high observed average reward, you should still explore other arms.

You MUST return you answer as a valid JSON object and nothing else.

Expected format:

{{
    "action": x,
    "explication": "your reasoning"
}}
"""
    prompt += f"""
So far you have played {nb_plays} times.

Statistics : here is your personal history of actions and rewards:

    """
    for i in range(bandit.n_arms):
        prompt += f"""
    Arm {i}:
    - Pulled {arm_stats[str(i)]["pulls"]} times
    - Average reward: {arm_stats[str(i)]["reward"] / arm_stats[str(i)]["pulls"] if arm_stats[str(i)]["pulls"] > 0 else 0:.3f}

    """
    prompt += f"""
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
    "action": x,
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