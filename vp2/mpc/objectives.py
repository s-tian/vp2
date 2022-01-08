import abc
import numpy as np
import torch
import piq
from hydra.utils import to_absolute_path

from vp2.mpc.utils import slice_dict


def sum(x, dim):
    # torch and np agnostic sum
    if isinstance(x, torch.Tensor):
        return x.sum(dim=dim)
    elif isinstance(x, np.ndarray):
        return x.sum(axis=dim)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def stack(x, dim=0):
    # torch and np agnostic stack
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=dim)
    elif isinstance(x[0], np.ndarray):
        return np.stack(x, axis=dim)
    else:
        raise ValueError(f"Unknown type {type(x)}")


class Objective(metaclass=abc.ABCMeta):
    def __init__(self, weight=1):
        self.weight = weight

    def compute_reward(self, prediction, goals):
        pass

    def __call__(self, predictions, goal, **kwargs):
        # IMPORTANT: All Objective subclasses compute REWARDS, not costs/losses.
        # This means that for any implemented objective, higher is better.
        return self.compute_reward(predictions, goal) * self.weight


class SquaredError(Objective):
    def __init__(self, key, weight):
        super().__init__(weight)
        self.key = key

    def compute_reward(self, prediction, goal):
        cost = (prediction[self.key] - goal[self.key]) ** 2
        # sum works much better than mean -- mean has small magnitudes (and floating point errors?)
        reward = -sum(cost, dim=(1, 2, 3, 4))
        return reward[:, None, None]


class CombinedObjective(Objective):
    def __init__(self, objectives, combine_method="sum"):
        super().__init__(weight=1)
        self.objectives = objectives
        self.combine_method = combine_method

    def compute_reward(self, prediction, goal):
        results = list()
        for name, objective in self.objectives.items():
            results.append(objective(prediction, goal))
        return sum(stack(results), dim=0)


class LPIPSError(Objective):
    def __init__(self, weight, key):
        super().__init__(weight)
        self.lpips = piq.LPIPS(reduction="none")
        self.key = key

    def flatten_image(self, im):
        leading_dims = im.shape[:-3]
        im = im.reshape(-1, *im.shape[-3:])
        return im, leading_dims

    def compute_reward(self, prediction, goal):
        prediction = prediction[self.key]
        goal = np.repeat(goal[self.key][None], prediction.shape[0], axis=0)
        goal = torch.tensor(goal).float().cuda()
        goal = torch.moveaxis(goal, -1, -3)
        prediction = torch.tensor(prediction).float().cuda()
        prediction = torch.moveaxis(prediction, -1, -3)
        lpips = []
        with torch.no_grad():
            for t in range(prediction.shape[1]):
                lpips_t = self.lpips(prediction[:, t], goal[:, t])
                lpips.append(lpips_t[..., None])
        lpips = torch.stack(lpips, dim=1)
        # [B, T, 1]
        lpips = lpips.mean(dim=(-1, -2), keepdim=True).cpu().detach().numpy()
        return -lpips


class ClassifierReward(Objective):
    def __init__(
        self,
        checkpoint_directory,
        weight,
        key,
        max_batch_size=1024,
        use_probs=False,
        use_gpu=True,
    ):
        """
        :param checkpoint_directory: directory containing classifier checkpoint
        :param weight: weight on this cost
        :param key: key containing image to classify (e.g. "rgb")
        :param max_batch_size: maximum batch size for each classifier forward pass.
        If the input has more samples than max_batch_size, the samples are split into
        batches of at most size max_batch_size.
        :param use_probs: if True, use sigmoid(logits) as the score, otherwise, directly use the logits
        :param use_gpu:
        """
        from vp2.scripts.train_task_success_classifier import (
            ConvPredictor,
        )

        super().__init__(weight)
        self.checkpoint_directory = to_absolute_path(checkpoint_directory)
        self.key = key
        self.max_batch_size = max_batch_size
        self.use_probs = use_probs
        self.model = ConvPredictor()
        self.use_gpu = use_gpu
        print(f"Loading classifier reward predictor from {checkpoint_directory}")
        self.model.load_state_dict(torch.load(checkpoint_directory))
        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

    def compute_reward(self, prediction, goal):
        prediction = torch.tensor(prediction[self.key], dtype=torch.float32)
        if self.use_gpu:
            prediction = prediction.cuda()
        flattened_predictions = prediction.reshape(-1, *prediction.shape[-3:])
        # convert from BHWC to BCHW
        flattened_predictions = flattened_predictions.permute(0, 3, 1, 2)
        scores = []
        num_batches = int(np.ceil(flattened_predictions.shape[0] / self.max_batch_size))
        with torch.no_grad():
            for batch_num in range(num_batches):
                logits = self.model(
                    flattened_predictions[
                        batch_num
                        * self.max_batch_size : (batch_num + 1)
                        * self.max_batch_size
                    ]
                )
                if self.use_probs:
                    score = torch.sigmoid(logits)
                else:
                    score = logits
                scores.append(score)
        scores = torch.cat(scores, dim=0)
        scores = scores.view(prediction.shape[0], prediction.shape[1])
        scores = scores.sum(dim=1)
        scores = scores.cpu().numpy()
        return np.expand_dims(scores, (1, 2))


class PolicyFeatureDistance(Objective):
    def __init__(self, policy_feature_metrics, weight, image_key="rgb"):
        super().__init__(weight)
        if not isinstance(policy_feature_metrics, torch.nn.ModuleList):
            policy_feature_metrics = [policy_feature_metrics]
        self.policy_feature_metrics = policy_feature_metrics
        self.image_key = image_key

    def compute_features(self, imgs):
        """
        :param imgs: tensor of shape [..., C, H, W]
        :return: tensor of shape [..., D] where D is the shape of the feature dimension
        """
        imgs = torch.tensor(np.moveaxis(imgs, -1, -3)).float().cuda()
        features = [
            f.get_feature_activations(imgs) for f in self.policy_feature_metrics
        ]
        features = torch.cat(features, dim=-1)
        return features

    def compute_reward(self, prediction, goal):
        with torch.no_grad():
            pred_feats, goal_feats = (
                self.compute_features(prediction[self.image_key]),
                self.compute_features(goal[self.image_key]),
            )
            # both shapes are [B, T, D]
            cost = (pred_feats - goal_feats) ** 2
            # sum works much better than mean -- mean has small magnitudes (and floating point errors?)
            cost = cost.detach().cpu().numpy()
        reward = -np.sum(cost, axis=(1, 2))
        return np.expand_dims(reward, (1, 2))


class EnsembleObjective(Objective):

    # Compute objectives across an ensemble of models and aggregate.
    def __init__(self, objective, agg="mean", weight=1.0, lamb=0.0):
        super().__init__(weight=weight)
        self.objective = objective
        assert agg in [
            "mean",
            "min",
            "penalize_disagreement",
        ], "Only mean, min objective aggregation are supported!"
        self.agg = agg
        self.lamb = lamb

    def compute_reward(self, prediction, goal):
        rewards = list()
        ensemble_count = prediction[list(prediction.keys())[0]].shape[0]
        for i in range(ensemble_count):
            rew = self.objective.compute_reward(
                slice_dict(prediction, i, i + 1, squeeze=True), goal
            )
            rewards.append(rew)
        rewards = np.stack(rewards, axis=0)
        if self.agg == "mean":
            rewards = np.mean(rewards, axis=0)
        elif self.agg == "min":
            rewards = np.amin(rewards, axis=0)
        elif self.agg == "penalize_disagreement":
            # rewards = np.mean(rewards, axis=0)
            indices = np.random.randint(0, ensemble_count, size=(rewards.shape[1]))
            rewards = rewards[indices, np.arange(rewards.shape[1])]
            for key in prediction:
                mean = np.mean(prediction[key], axis=0)
                disagreements = np.abs(prediction[key] - mean).sum(axis=(2, 3, 4, 5))
                disagreements = np.amax(disagreements, axis=0)
                rewards -= self.lamb * np.expand_dims(disagreements, (1, 2))
        return rewards
