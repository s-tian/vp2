import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from hydra.utils import instantiate

from vp2.mpc.utils import (
    extract_image,
    stack_into_dict,
    concat_into_dict,
)


def np_split(a, size):
    return np.split(a, np.arange(size, len(a), size))


class SimulatorModel:
    def __init__(self, env_cfg, num_envs, multiprocess=True):
        self.multiprocess = multiprocess
        if self.multiprocess:
            print("Multiprocessing using SubprocVecEnv!")
            self.env = SubprocVecEnv([lambda: instantiate(env_cfg)] * num_envs)
            self.a_dim = self.env.get_attr("action_dimension")[0]
        else:
            self.env = instantiate(env_cfg)
            self.a_dim = self.env.action_dimension
        self.num_envs = num_envs
        self.base_prediction_modality = "rgb"
        self.num_context = 2

    def reset_to(self, state):
        if self.multiprocess:
            self.env.env_method("reset_to", state)
        else:
            self.env.reset_to(state)

    def __call__(self, batch):
        # input:
        # dictionary with keys:
        # 'state_obs': list of state observation history
        # 'actions': action sequence to execute
        reset_state = {"states": batch["state_obs"][0]}
        # gather up action batches
        actions = batch["actions"][:, :]

        action_batches = np_split(actions, self.num_envs)
        # [n_batches, num_envs, T, num_actions]
        all_images = []
        for batch in action_batches:
            if self.multiprocess:
                batch = np.transpose(batch, (1, 0, 2))
                minibatch_size = batch.shape[1]
                # If this is the last batch and the number of samples doesn't evenly divide num_envs, pad to make it so.
                if minibatch_size < self.num_envs:
                    dummy_batch_remainder = np.zeros(
                        (batch.shape[0], self.num_envs - minibatch_size, batch.shape[2])
                    )
                    batch = np.concatenate((batch, dummy_batch_remainder), axis=1)
                # reset each worker env
                self.env.env_method("reset_state", reset_state)
                self.env.step(np.zeros((self.num_envs, self.a_dim)))
                batch_images = list()
                for action_step in batch:
                    obs, _, _, _ = self.env.step(action_step)
                    images = stack_into_dict(obs, axis=0)
                    batch_images.append(images)
                # stack over time
                batch_images = stack_into_dict(batch_images, axis=1)
                if (
                    minibatch_size < self.num_envs
                ):  # If we padded this batch, only take the first n which are actually real
                    batch_images = {
                        k: v[:minibatch_size] for k, v in batch_images.items()
                    }
            else:
                # Non-multiprocessed logic
                batch_images = list()
                for action_sequence in batch:
                    self.env.reset_state(reset_state)
                    self.env.step(np.zeros(self.a_dim))
                    sequence = list()
                    for action_step in action_sequence:
                        obs, _, _, _ = self.env.step(action_step)
                        sequence.append(obs)
                    batch_images.append(stack_into_dict(sequence, axis=0))
                batch_images = stack_into_dict(batch_images, axis=0)

            all_images.append(batch_images)
        all_images = concat_into_dict(all_images, axis=0)
        all_images = {k: v / 255.0 for k, v in all_images.items()}
        return all_images
