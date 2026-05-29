import numpy as np


def reward_fn(rate0=0.6, rate1=0.4):
    '''Returns a reward function for a Bernoulli bandit with two arms.'''
    def _reward(arm_played):
        win_rate = [rate0, rate1]
        pull = np.random.rand()
        if arm_played == 0 and pull < win_rate[0]:
            return 1
        elif arm_played == 1 and pull < win_rate[1]:
            return 1
        return 0
    return _reward

