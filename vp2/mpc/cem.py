import numpy as np
import time
from hydra.utils import instantiate

from vp2.mpc.optimizer import Optimizer
from vp2.mpc.utils import ObservationList, write_moviepy_gif


class CEMOptimizer(Optimizer):
    # Based on https://github.com/google-research/pddm/blob/06b88cdbafa7fc35451b3feb3e01aa6726113f72/pddm/policies/cem.py

    def __init__(
        self,
        sampler,
        model,
        objective,
        a_dim,
        horizon,
        num_samples,
        elites_frac,
        opt_iters,
        log_every=1,
        init_std=0.5,
        init_mean=None,
        alpha=0,
        verbose=False,
        round_gripper_action=False,
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.num_samples = num_samples
        self.elites_frac = elites_frac
        self.opt_iters = opt_iters
        self.log_every = log_every
        self.alpha = alpha  # Controls mean/variance update rate
        self.init_std = np.array(init_std)
        self.init_mean = np.array(init_mean)
        self.lower_bound, self.upper_bound = -1, 1
        self.verbose = verbose
        self.round_gripper_action = round_gripper_action

    def update_dist(self, samples, scores, mu, var):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]
        num_elites = int(self.elites_frac * self.num_samples)
        indices = np.argsort(-scores.flatten())[:num_elites]
        elite_samples = samples[indices]
        n_mu = np.mean(elite_samples, axis=0)
        n_var = np.var(elite_samples, axis=0)
        new_mu = self.alpha * mu + (1 - self.alpha) * n_mu
        new_var = self.alpha * var + (1 - self.alpha) * n_var
        return new_mu, new_var

    def rounded_gripper_action(self, actions):
        # Gripper action is assumed to be in the last dimension
        actions[..., -1] = np.where(actions[..., -1] < 0, -1, 1)
        return actions

    def score_trajectories(
        self,
        new_action_samples,
        obs_history,
        state_history,
        action_history,
        goal,
        requires_grad=False,
    ):

        n_ctxt = self.model.num_context
        action_history = action_history[-(n_ctxt - 1) :]
        obs_history = obs_history[-n_ctxt:]
        state_history = state_history[-n_ctxt:]
        context_actions = np.tile(
            np.array(action_history)[None], (new_action_samples.shape[0], 1, 1)
        )

        if requires_grad:
            import torch

            new_action_samples = torch.clip(new_action_samples, -1, 1)
            action_samples = torch.cat(
                (
                    torch.from_numpy(context_actions).to(new_action_samples),
                    new_action_samples,
                ),
                axis=1,
            )
        else:
            new_action_samples = np.clip(new_action_samples, -1, 1)
            action_samples = np.concatenate(
                (context_actions, new_action_samples), axis=1
            )

        if self.round_gripper_action:
            action_samples = self.rounded_gripper_action(action_samples)

        if self.verbose:
            print("action samples", action_samples)

        batch = {
            "video": np.tile(
                np.array(obs_history[self.model.base_prediction_modality])[None],
                (new_action_samples.shape[0], 1, 1, 1, 1),
            ),
            "actions": action_samples,
            "state_obs": state_history,
        }

        pred_start_time = time.time()
        predictions = self.model(batch, grad_enabled=requires_grad)
        print(f"Prediction time {time.time() - pred_start_time}")
        if requires_grad:
            import torch

            goal = dict(rgb=torch.from_numpy(goal["rgb"]).to(self.model.device))
        rewards = self.obj_fn(predictions, goal)
        return predictions, rewards, action_samples

    def perform_cem(
        self,
        t,
        log_dir,
        obs_history,
        state_history,
        action_history,
        goal,
        init_mean=None,
    ):
        if init_mean is not None:
            mu = np.zeros((self.horizon, self.a_dim))
            mu[: len(init_mean)] = init_mean
            mu[len(init_mean) :] = init_mean[-1]
        else:
            mu = np.zeros((self.horizon, self.a_dim))
        var = np.tile((self.init_std**2)[None], (self.horizon, 1))

        for iter in range(self.opt_iters):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var
            )
            new_action_samples = self.sampler.sample_actions(
                self.num_samples, mu, np.sqrt(constrained_var)
            )
            predictions, rewards, action_samples = self.score_trajectories(
                new_action_samples,
                obs_history,
                state_history,
                action_history,
                goal,
            )

            best_prediction_inds = np.argsort(-rewards.flatten())[:10]

            vis_preds = list()
            for i in best_prediction_inds:
                vis_preds.append(
                    ObservationList({k: v[i] for k, v in predictions.items()})
                )

            best_rewards = [rewards[i] for i in best_prediction_inds]
            print("best rewards:", best_rewards)
            if t % self.log_every == 0:
                self.log_best_plans(
                    f"{log_dir}/step_{t}_itr_{iter}_best_plan",
                    vis_preds,
                    goal,
                    best_rewards,
                )

            n_ctxt = self.model.num_context
            mu, var = self.update_dist(
                action_samples[:, n_ctxt - 1 :], rewards, mu, var
            )
            if self.verbose:
                print(f"itr {iter}: shape = {mu.shape}: {mu}")

        return action_samples, rewards

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
        n_ctxt = self.model.num_context
        action_samples, rewards = self.perform_cem(
            t, log_dir, obs_history, state_history, action_history, goal, init_mean
        )
        return action_samples[np.argmax(rewards), n_ctxt - 1 :]
