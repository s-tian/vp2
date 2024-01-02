import os
import numpy as np
import torch
from contextlib import ExitStack
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from hydra.utils import instantiate

try:
    from struct_vrnn.model.keypoint_vrnn import StructVRNN
except ModuleNotFoundError:
    raise ModuleNotFoundError(f"Failed to load StructVRNN model. This is installed separately from the VP2 benchmark. "
                              f"Please follow the package installation instructions in the VP2 README to clone/install.")

from vp2.models.model import VideoPredictionModel
from vp2.mpc.utils import dict_to_float_tensor, dict_to_cuda


class KeypointVRNNInterface(VideoPredictionModel):
    def __init__(
        self,
        checkpoint_dir,
        n_past,
        planning_modalities,
        max_batch_size=200,
        epoch=None,
        device="cuda:0",
    ):
        self.checkpoint_file = self.get_checkpoint_file(checkpoint_dir, epoch)
        saved_model = torch.load(self.checkpoint_file)
        config_file = to_absolute_path(os.path.join(checkpoint_dir, "config.yaml"))
        config = OmegaConf.load(config_file)
        self.model = instantiate(config.model)
        self.model.load_state_dict(saved_model["model_state_dict"])
        self.model.eval()
        self.model.to(device)
        self.planning_modalities = planning_modalities
        self.base_prediction_modality = "rgb"
        self.num_context = n_past
        self.max_batch_size = max_batch_size
        self.device = device

    def format_model_epoch_filename(self, epoch):
        return f"checkpoint-{epoch}.pt"

    def prepare_batch(self, xs):
        keys = ["video", "actions"]
        batch = {k: x for k, x in xs.items() if k in keys}
        batch = dict_to_float_tensor(batch)
        batch = dict_to_cuda(batch, device=self.device)
        batch["video"] = torch.permute(batch["video"], (0, 1, 4, 2, 3))
        return batch["video"], batch["actions"]

    def __call__(self, batch, grad_enabled=False):
        all_preds = list()
        with torch.no_grad() if not grad_enabled else ExitStack():
            video, actions = self.prepare_batch(batch)
            for compute_batch_idx in range(
                0, batch["video"].shape[0], self.max_batch_size
            ):
                predictions = self.model(
                    video[
                        compute_batch_idx : compute_batch_idx + self.max_batch_size
                    ],
                    actions[
                        compute_batch_idx : compute_batch_idx + self.max_batch_size
                    ],
                )
                predictions = torch.permute(predictions, (0, 1, 3, 4, 2))
                all_preds.append(predictions)
        predictions = torch.cat(all_preds, dim=0)
        if not grad_enabled:
            predictions = predictions.cpu().numpy()
        return dict(rgb=predictions)
