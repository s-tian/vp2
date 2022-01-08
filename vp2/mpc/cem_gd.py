import numpy as np
from copy import deepcopy

from vp2.mpc.cem import CEMOptimizer

import torch
import math
from torch.optim.optimizer import Optimizer


def clamp(x, low, high):
    return torch.max(torch.min(x, high), low)


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, factor=None, lr_force=None, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            if "factor" not in group:
                group["factor"] = 1
            div = group["factor"] if factor is None else factor

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            lr = group["lr"] / div if lr_force is None else lr_force
            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

            low_bounds, high_bounds = group["action_bounds"]

            # Project
            for p in group["params"]:
                low = low_bounds.repeat(p.shape[0], 1)
                high = high_bounds.repeat(p.shape[0], 1)
                p.data = clamp(p.data, low, low)
        return loss


def adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float
):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class CEMGDOptimizer(CEMOptimizer):
    # Implementation of CEM-GD planner as described by https://arxiv.org/pdf/2112.07746.pdf.
    # Code is heavily based on https://github.com/KevinHuang8/CEM-GD/blob/main/mbrl/planning/gradient_optimizer.py

    def __init__(
        self,
        sampler,
        model,
        objective,
        a_dim,
        horizon,
        num_samples_init,
        num_samples_replan,
        elites_frac,
        opt_iters,
        log_every=1,
        init_std=0.5,
        init_mean=None,
        alpha=0,
        verbose=False,
        round_gripper_action=False,
        num_grad_opt_seqs=1,
        start_lr=1e-2,
        factor_shrink=0.5,
        max_tries=7,
        max_iterations=15,
    ):
        super().__init__(
            sampler,
            model,
            objective,
            a_dim,
            horizon,
            num_samples_init,
            elites_frac,
            opt_iters,
            log_every,
            init_std,
            init_mean,
            alpha,
            verbose,
            round_gripper_action,
        )

        self.num_samples_init = num_samples_init
        self.num_samples_replan = num_samples_replan
        self.num_grad_opt_seqs = num_grad_opt_seqs
        self.start_lr = start_lr
        self.factor_shrink = factor_shrink
        self.max_tries = max_tries
        self.max_iterations = max_iterations

    def gradient_optimization(
        self, action_sequences_list, action_history, obs_history, state_history, goal
    ):

        for action_sequences in action_sequences_list:
            action_sequences.requires_grad = True

        n = len(action_sequences_list)

        optimizer = Adam(
            [
                {
                    "params": act_seq,
                    "factor": 1,
                    "action_bounds": (torch.tensor(-1).cuda(), torch.tensor(1).cuda()),
                }
                for act_seq in action_sequences_list
            ],
            lr=self.start_lr,
        )
        optimizer.zero_grad()

        saved_parameters = [None] * n
        saved_opt_states = [None] * n
        current_iteration = np.zeros(n)
        done = np.zeros(n, dtype=bool)

        action_sequences_batch = torch.stack(action_sequences_list).float()
        _, rewards_all, _ = self.score_trajectories(
            action_sequences_batch,
            obs_history,
            state_history,
            action_history,
            goal,
            requires_grad=True,
        )
        objective_all = -rewards_all

        current_objective = [objective_all[i] for i in range(n)]

        for i in range(n):
            action_sequences = action_sequences_list[i]
            saved_parameters[i] = action_sequences.detach().clone()
            saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
            objective_all[i].backward(retain_graph=(i != n - 1))

        while not np.all(done):
            optimizer.step()

            # Compute objectives of all trajectories after stepping
            action_sequences_batch = torch.stack(action_sequences_list)
            _, rewards_all, _ = self.score_trajectories(
                action_sequences_batch.float(),
                obs_history,
                state_history,
                action_history,
                goal,
                requires_grad=True,
            )
            objective_all = -rewards_all

            backwards_pass = []

            for i in range(n):
                if done[i]:
                    continue
                action_sequences = action_sequences_list[i]
                if objective_all[i] > current_objective[i]:
                    # If after the step, the cost is higher, then undo
                    action_sequences.data = saved_parameters[i].data.clone()
                    optimizer.state[action_sequences] = deepcopy(saved_opt_states[i])
                    optimizer.param_groups[i]["factor"] *= self.factor_shrink

                    if (
                        optimizer.param_groups[i]["factor"]
                        > self.factor_shrink**self.max_tries
                    ):
                        # line search failed, mark action sequence as done
                        action_sequences.grad = None
                        done[i] = True
                else:
                    # successfully completed step.
                    # Save current state, and compute gradients
                    saved_parameters[i] = action_sequences.detach().clone()
                    saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
                    current_objective[i] = objective_all[i]
                    optimizer.param_groups[i]["factor"] = 1
                    action_sequences.grad = None
                    backwards_pass.append(i)

                    current_iteration[i] += 1
                    if current_iteration[i] > self.max_iterations:
                        action_sequences.grad = None
                        done[i] = True

            to_compute = [objective_all[i] for i in backwards_pass]
            grads = [
                (torch.empty_like(objective_all[i]) * 0 + 1).to(self.model.device)
                for i in backwards_pass
            ]
            torch.autograd.backward(to_compute, grads)

        return [traj.detach() for traj in action_sequences_list]

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
        if t == 0 or t == 1:
            self.num_samples = self.num_samples_init
        else:
            self.num_samples = self.num_samples_replan

        action_samples, rewards = self.perform_cem(
            t, log_dir, obs_history, state_history, action_history, goal, init_mean
        )
        # get top k best action samples
        top_k = self.num_grad_opt_seqs
        top_k_indices = np.argsort(-rewards.flatten())[:top_k]
        top_k_action_samples = list(action_samples[top_k_indices])
        top_k_action_samples = [
            torch.tensor(action_sample[n_ctxt - 1 :]).to(self.model.device)
            for action_sample in top_k_action_samples
        ]

        top_k_grad_optimized = self.gradient_optimization(
            top_k_action_samples, action_history, obs_history, state_history, goal
        )
        top_k_grad_optimized = torch.stack(top_k_grad_optimized).detach().cpu().numpy()
        _, top_k_grad_optimized_rewards, _ = self.score_trajectories(
            top_k_grad_optimized,
            obs_history,
            state_history,
            action_history,
            goal,
            requires_grad=False,
        )
        print("Top k grad opt rewards: ", top_k_grad_optimized_rewards)

        best_sequence = top_k_grad_optimized[
            np.argmax(top_k_grad_optimized_rewards), n_ctxt - 1 :
        ]
        return best_sequence
