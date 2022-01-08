import numpy as np


class NoisyPolicyWrapper:
    def __init__(self, policy, noise_std=0.1):
        self.policy = policy
        self.noise_std = noise_std

    def get_action(self, *args, **kwargs):
        action = self.policy.get_action(*args, **kwargs)
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        return action + noise

    def reset(self):
        return self.policy.reset()
