import numpy as np
import time

from vp2.mpc.optimizer import Optimizer
from vp2.mpc.utils import ObservationList, write_moviepy_gif


class MPPIOptimizer(Optimizer):
    def __init__(
        self,
        sampler,
        model,
        objective,
        a_dim,
        horizon,
        num_samples,
        gamma,
        init_std=0.5,
        log_every=1,
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.num_samples = num_samples
        self.gamma = gamma
        self.init_std = np.array(init_std)
        self.log_every = log_every
        self._model_prediction_times = list()

    def update_dist(self, samples, scores):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]
        scaled_rews = self.gamma * (scores - np.max(scores))
        # exponentiated scores
        exp_rews = np.exp(scaled_rews)
        mu = np.sum(exp_rews * samples, axis=0) / (np.sum(exp_rews, axis=0) + 1e-10)
        return mu, self.init_std

    def plan(
        self,
        t,
        log_dir,
        obs_history,
        state_history,
        action_history,
        goal,
        init_mean=None,
    ):
        assert init_mean is not None or t == 1, "init_mean must be provided for MPPI"
        n_ctxt = self.model.num_context
        action_history = action_history[-(n_ctxt - 1) :]
        obs_history = obs_history[-n_ctxt:]
        state_history = state_history[-n_ctxt:]
        context_actions = np.tile(
            np.array(action_history)[None], (self.num_samples, 1, 1)
        )
        if init_mean is not None:
            mu = np.zeros((self.horizon, self.a_dim))
            mu[: len(init_mean)] = init_mean
            mu[len(init_mean) :] = init_mean[-1]
        else:
            mu = np.zeros((self.horizon, self.a_dim))
        std = self.init_std[None].repeat(self.horizon, axis=0)

        new_action_samples = self.sampler.sample_actions(self.num_samples, mu, std)
        new_action_samples = np.clip(new_action_samples, -1, 1)
        action_samples = np.concatenate((context_actions, new_action_samples), axis=1)

        batch = {
            "video": np.tile(
                np.array(obs_history[self.model.base_prediction_modality])[None],
                (self.num_samples, 1, 1, 1, 1),
            ),
            "actions": action_samples,
            "state_obs": state_history,
        }

        pred_start_time = time.time()
        predictions = self.model(batch)
        prediction_time = time.time() - pred_start_time
        self._model_prediction_times.append(prediction_time)
        print(f"Prediction time {prediction_time}")
        print(
            f"Out of {len(self._model_prediction_times)}, Median prediction time {np.median(self._model_prediction_times)}"
        )

        rewards = self.obj_fn(predictions, goal)
        best_prediction_inds = np.argsort(-rewards.flatten())[:10]
        best_rewards = [rewards[i] for i in best_prediction_inds]
        vis_preds = list()
        for i in best_prediction_inds:
            if len(predictions["rgb"].shape) == 6:
                vis_preds.append(
                    ObservationList({k: v[0, i] for k, v in predictions.items()})
                )
            else:
                vis_preds.append(
                    ObservationList({k: v[i] for k, v in predictions.items()})
                )
        # best_actions = [action_samples[i] for i in best_predictions[:3]]
        print("best rewards:", best_rewards)
        # print('best actions:', best_actions)
        if t % self.log_every == 0:
            self.log_best_plans(
                f"{log_dir}/step_{t}_best_plan", vis_preds, goal, best_rewards
            )

        mu, std = self.update_dist(action_samples[:, n_ctxt - 1 :], rewards)
        print(f"mu shape = {mu.shape}: {mu}")

        return mu
        # return action_samples[np.argmax(rewards), n_ctxt:]
