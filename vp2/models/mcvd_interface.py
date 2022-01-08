import os
import yaml
from math import ceil
from functools import partial
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from hydra.utils import to_absolute_path
from mcvd.datasets import get_dataset, data_transform, inverse_data_transform
from mcvd.main import dict2namespace
from mcvd.models.ema import EMAHelper
from mcvd.models import ddpm_sampler
from mcvd.runners.ncsn_runner import get_model, conditioning_fn

from vp2.models.model import VideoPredictionModel


class MCVDInterface(VideoPredictionModel):
    def __init__(
        self,
        checkpoint_dir,
        n_past,
        planning_modalities,
        num_diffusion_steps=None,
        verbose=True,
        epoch=None,
        device="cuda",
    ):
        self.checkpoint_file = self.get_checkpoint_file(checkpoint_dir, epoch)
        self.scorenet, self.config = self.load_model(self.checkpoint_file, device)
        self.sampler = partial(ddpm_sampler, config=self.config)
        self.num_context = n_past
        self.planning_modalities = planning_modalities
        self.base_prediction_modality = "rgb"
        if num_diffusion_steps:
            self.num_diffusion_steps = num_diffusion_steps
        else:
            self.num_diffusion_steps = getattr(self.config.sampling, "subsample", None)
        self.verbose = verbose

    def format_model_epoch_filename(self, epoch):
        return f"checkpoint_{epoch}.pt"

    def generate_initial_noise_sample(self, net, batch_size):
        init_samples_shape = (
            batch_size,
            self.config.data.channels * self.config.data.num_frames,
            self.config.data.image_size,
            self.config.data.image_size,
        )
        if getattr(self.config.model, "gamma", False):
            used_k, used_theta = net.k_cum[0], net.theta_t[0]
            z = (
                Gamma(
                    torch.full(init_samples_shape, used_k),
                    torch.full(init_samples_shape, 1 / used_theta),
                )
                .sample()
                .to(self.config.device)
            )
            z = z - used_k * used_theta
        else:
            z = torch.randn(init_samples_shape, device=self.config.device)
        return z

    def make_batch_prediction(self, batch):
        """
        Given a batch of context frames and actions, output predicted RGB frames the DDPM sampler.
        :param batch: A dictionary containing the following keys:
            - 'video': A tensor of shape (B, #context, C, H, W) containing the context frames.
            - 'actions': A tensor of shape (B, T, A) containing the actions.
        :return: A tensor of shape (B, T, C, H, W) containing the predicted RGB frames.
        """
        video, actions = batch["video"], batch["actions"]
        batch_size, num_actions = actions.shape[0], actions.shape[1]
        num_pred_frames = num_actions - self.config.data.num_frames_cond + 1
        num_iterations = ceil(num_pred_frames / self.config.data.num_frames)
        init_samples = self.generate_initial_noise_sample(self.scorenet, batch_size)
        num_conditioning_actions = (
            self.config.data.num_frames + self.config.data.num_frames_cond - 1
        )
        context_frames = data_transform(self.config, video)

        for i_frame in tqdm(range(num_iterations), desc="Generating video frames"):
            if i_frame == 0:
                x_mod = init_samples
            else:
                x_mod = gen_samples

            conditioning_frames = context_frames[:, -self.config.data.num_frames_cond :]
            start_action_idx = i_frame * self.config.data.num_frames
            real, cond, cond_mask = conditioning_fn(
                self.config,
                conditioning_frames,
                num_frames_pred=self.config.data.num_frames,
                prob_mask_cond=0.0,
                prob_mask_future=0.0,
                conditional=True,
                actions=actions[
                    :, start_action_idx : start_action_idx + num_conditioning_actions
                ],
            )
            gen_samples = self.sampler(
                x_mod,
                self.scorenet,
                cond=cond,
                cond_mask=cond_mask,
                n_steps_each=self.config.sampling.n_steps_each,
                step_lr=self.config.sampling.step_lr,
                verbose=self.verbose,
                final_only=True,
                denoise=self.config.sampling.denoise,
                subsample_steps=self.num_diffusion_steps,
                clip_before=getattr(self.config.sampling, "clip_before", True),
                t_min=getattr(self.config.sampling, "init_prev_t", -1),
                log=False,
                gamma=getattr(self.config.model, "gamma", False),
            )
            gen_samples = gen_samples[-1].reshape(
                gen_samples[-1].shape[0],
                self.config.data.channels * self.config.data.num_frames,
                self.config.data.image_size,
                self.config.data.image_size,
            )
            gen_samples_sep_channel = gen_samples.reshape(
                gen_samples.shape[0],
                self.config.data.num_frames,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
            )
            context_frames = torch.cat((context_frames, gen_samples_sep_channel), dim=1)
        context_frames = inverse_data_transform(self.config, context_frames)
        return context_frames[:, 1:]

    def load_model(self, ckpt_path, device):
        # Parse config file
        with open(os.path.join(os.path.dirname(ckpt_path), "config.yml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # Load config file
        config = dict2namespace(config)
        config.device = device
        # Load model
        scorenet = get_model(config)
        if config.device != torch.device("cpu"):
            scorenet = torch.nn.DataParallel(scorenet)
            states = torch.load(ckpt_path, map_location=config.device)
        else:
            states = torch.load(ckpt_path, map_location="cpu")
            states[0] = OrderedDict(
                [(k.replace("module.", ""), v) for k, v in states[0].items()]
            )
        scorenet.load_state_dict(states[0], strict=True)
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(scorenet)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(scorenet)
        scorenet.eval()
        scorenet = scorenet.module if hasattr(scorenet, "module") else scorenet

        assert (
            config.model.version.upper() == "DDPM"
        ), "Only the DDPM sampler is supported"
        return scorenet, config

    def prepare_batch(self, xs):
        keys = ["video", "actions"]
        batch = {
            k: torch.from_numpy(x).cuda().float() for k, x in xs.items() if k in keys
        }
        batch["video"] = torch.permute(batch["video"], (0, 1, 4, 2, 3))
        return batch

    def __call__(self, batch):
        batch = self.prepare_batch(batch)
        prediction = self.make_batch_prediction(batch)
        prediction = prediction.permute(0, 1, 3, 4, 2)
        prediction = prediction.cpu().numpy()
        return dict(rgb=prediction)


import hydra


@hydra.main(config_path="../scripts/configs/model", config_name="mcvd")
def test_model(model_cfg):
    from hydra.utils import instantiate

    model = instantiate(model_cfg)
    batch = dict(
        video=np.random.rand(100, 2, 64, 64, 3),
        actions=np.random.rand(100, 11, 4),
    )
    prediction = model(batch)
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    test_model()
