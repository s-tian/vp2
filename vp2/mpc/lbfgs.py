import numpy as np
import torch
import time

from vp2.mpc.optimizer import Optimizer
from vp2.mpc.utils import ObservationList, write_moviepy_gif, dict_to_numpy


class LBFGSOptimizer(Optimizer):
    def __init__(
        self,
        model,
        objective,
        a_dim,
        horizon,
        num_opt_steps,
        lr,
        tolerance_change=1e-4,
        line_search_fn=None,
        log_every=1,
    ):
        self.obj_fn = objective
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.log_every = log_every
        self.num_opt_steps = num_opt_steps
        self.lr = lr
        self.tolerance_change = tolerance_change
        self.line_search_fn = line_search_fn
        self._model_prediction_times = list()

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
        action_history = action_history[-(n_ctxt - 1) :]
        obs_history = obs_history[-n_ctxt:]
        state_history = state_history[-n_ctxt:]
        context_actions = np.array(action_history)[None]

        if init_mean is not None:
            mu = np.zeros((self.horizon, self.a_dim))
            mu[: len(init_mean)] = init_mean
            mu[len(init_mean) :] = init_mean[-1]
        else:
            mu = np.zeros((self.horizon, self.a_dim))

        actions_leaf = torch.from_numpy(mu)
        actions_leaf.requires_grad = True

        num_steps = 0
        loss_list = list()
        vis_preds = list()

        optimizer = torch.optim.LBFGS(
            [actions_leaf],
            lr=self.lr,
            tolerance_change=self.tolerance_change,
            line_search_fn=self.line_search_fn,
        )

        def closure():
            nonlocal num_steps
            nonlocal loss_list
            nonlocal vis_preds
            nonlocal context_actions
            nonlocal goal
            nonlocal actions_leaf

            actions = actions_leaf[None]
            actions = actions.float().to(self.model.device)

            optimizer.zero_grad()
            torch_context_actions = (
                torch.from_numpy(context_actions).float().to(actions)
            )
            action_samples = torch.cat((torch_context_actions, actions), dim=1)
            torch_goal = dict(rgb=torch.from_numpy(goal["rgb"]).to(self.model.device))

            batch = {
                "video": np.array(obs_history[self.model.base_prediction_modality])[
                    None
                ],
                "actions": action_samples,
                "state_obs": state_history,
            }

            pred_start_time = time.time()
            predictions = self.model(batch, grad_enabled=True)
            prediction_time = time.time() - pred_start_time
            self._model_prediction_times.append(prediction_time)

            # print(f"Prediction time {prediction_time}")

            rewards = self.obj_fn(predictions, torch_goal)
            loss = -rewards
            loss_list.append(loss.item())
            loss.backward()
            num_steps += 1

            vis_preds = list()

            predictions = dict_to_numpy(predictions)

            for i in range(rewards.shape[0]):
                if len(predictions["rgb"].shape) == 6:
                    vis_preds.append(
                        ObservationList({k: v[0, i] for k, v in predictions.items()})
                    )
                else:
                    vis_preds.append(
                        ObservationList({k: v[i] for k, v in predictions.items()})
                    )
            return loss

        for grad_step in range(self.num_opt_steps):
            optimizer.step(closure)

        print(
            f"Out of {len(self._model_prediction_times)}, Median prediction time {np.median(self._model_prediction_times)}"
        )
        if t % self.log_every == 0:
            self.log_best_plans(
                f"{log_dir}/step_{t}_best_plan", vis_preds, goal, [loss_list[-1]]
            )
        actions_leaf = torch.clip(actions_leaf, -1, 1)

        print("Starting loss value", loss_list[0])
        print("Final loss value", loss_list[-1])

        return actions_leaf.detach().cpu().numpy()
