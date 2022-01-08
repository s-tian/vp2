import os
import json
import time
import pprint
import torch
import numpy as np
import tqdm

from vp2.mpc.utils import dict_to_numpy
from mfm.models.framegpt.wgpt import WindowGPT
import mfm.utils.file as fu
import mfm.utils.video as vu
import mfm.utils.ckpt as cku
import pytorch_lightning as pl
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue, Queue, JoinableQueue
from torch.cuda.amp import autocast
from contextlib import ExitStack
import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import OmegaConf


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def worker(rank, size, cfg, in_queue, out_queue):
    cfg = dotdict(cfg)
    if rank == 0:
        # print(vars(cfg))
        print(cfg)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{cfg.ddp_port}",
        world_size=size,
        rank=rank,
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print("use device", torch.cuda.current_device(), "from rank", rank)
    torch.set_grad_enabled(False)

    pl.seed_everything(rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sweep_name = cfg.sweep_name
    task_num = cfg.task_num
    if sweep_name and task_num:
        # load model path and cfg path from sweep name
        model_path, cfg_path = fu.get_model_and_cfg_path(sweep_name, task_num)
    else:
        model_path, cfg_path = cfg.model_path, cfg.cfg_path
    print("Loading from ", cfg_path)
    model = cku.load_model(cfg_path, model_path, gpu_num=rank)
    model = model.to(device)
    print("Started process -- rank is ", rank)

    while True:
        msg = in_queue.get()
        if msg == "DONE":
            break
        else:
            batch = msg
            idx = batch["batch_index"]
            batch.pop("batch_index")
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["video"] = pad_video_length(batch["video"], cfg.total_pred_len)
            with autocast() if cfg.fp16 else ExitStack() as ac:
                outputs = model.sample(
                    cfg.total_pred_len - cfg.n_cond_frames,
                    batch,
                    n_cond_frames=cfg.n_cond_frames,
                    num_iters=cfg.num_iters,
                    temperature=cfg.temperature,
                )

            # print("Batch video shape", batch["video"].shape)
            # print("Sampled video shape", outputs["sampled_video"].shape)
            # output_video = torch.cat((batch["video"][:, :, :cfg.n_cond_frames], outputs["sampled_video"]), dim=2)
            output_video = outputs["sampled_video"]
            output_video = output_video.detach().cpu()
            output_video = output_video[:, :, 1:]
            out_queue.put((idx, output_video))


class WGPTModel:
    def __init__(self, cfg, n_past, a_dim, epoch):
        self.cfg = cfg
        self.num_context = n_past
        self.a_dim = a_dim
        self.n_past = n_past
        self.base_prediction_modality = "rgb"

        print("Loading WGPT model with hyperparameters:")
        pprint.pprint(cfg)
        ngpus = torch.cuda.device_count()
        self.ngpus = ngpus
        print(f"Loading model onto {self.ngpus} GPUs:")

        smp = mp.get_context("spawn")
        to_child_queue, result_queue = smp.SimpleQueue(), smp.SimpleQueue()
        self.context = mp.spawn(
            worker,
            nprocs=ngpus,
            args=(ngpus, OmegaConf.to_container(cfg), to_child_queue, result_queue),
            join=False,
        )
        self.to_child_queue, self.result_queue = to_child_queue, result_queue

    def prepare_batch(self, xs):
        keys = ["video", "actions"]
        batch_size = xs["video"].shape[0]
        batch = {k: torch.from_numpy(x).float() for k, x in xs.items() if k in keys}
        # Normalize video to -1, 1
        batch["video"] = (batch["video"] - 0.5) * 2
        # Video initially is [B, T, H, W, C]
        # Convert to [B, C, T, H, W]
        batch["video"] = torch.permute(batch["video"], (0, 4, 1, 2, 3))

        assert (
            self.cfg.total_pred_len == batch["actions"].shape[1] + 1
        ), f"Action length is {batch['actions'].shape[1]}, should be {self.cfg.total_pred_len-1}"

        # Pad action with one zero action at the end
        dummy_action = torch.zeros(batch_size, 1, self.a_dim)
        batch["actions"] = torch.cat((batch["actions"], dummy_action), dim=1)

        assert batch["actions"].shape[1] == self.cfg.total_pred_len
        print(batch["actions"].shape)
        print(batch["video"].shape)

        return batch

    def mp_forward(self, batch):
        batch_size = batch["video"].shape[0]
        num_batches = batch_size // self.cfg.batch_size
        assert (
            batch_size % self.cfg.batch_size == 0
        ), "Num samples must be divisible by batch size"
        print(f"Performing prediction and queueing {num_batches} gpu-batches...")
        for i in range(num_batches):
            minibatch = {
                k: v[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size]
                for k, v in batch.items()
            }
            minibatch["batch_index"] = i
            self.to_child_queue.put(minibatch)
        batch_results = dict()
        for _ in tqdm.tqdm(range(num_batches)):
            result = self.result_queue.get()
            batch_results[result[0]] = result[1]
        results = torch.cat([batch_results[i] for i in range(num_batches)])
        print("result shape is ", results.shape)
        return results

    def __call__(self, batch, grad_enabled=False):
        batch = self.prepare_batch(batch)
        results = self.mp_forward(batch)
        # Result of mp_forward is in shape [B, C, T, H, W] -> [B, T, H, W, C]
        preds = dict(
            # rgb=torch.permute(results, (0, 2, 3, 4, 1))[:, self.n_past:]
            rgb=torch.permute(results, (0, 2, 3, 4, 1))
        )
        return dict_to_numpy(preds)

    def close(self):
        for _ in range(self.ngpus):
            self.to_child_queue.put("DONE")
        self.context.join()


def pad_video_length(video, target_len):
    video_len = video.shape[2]
    to_add = target_len - video_len
    new_shape = list(video.shape)
    new_shape[2] = to_add
    padding = torch.zeros(
        new_shape, dtype=video.dtype, layout=video.layout, device=video.device
    )
    return torch.cat((video, padding), dim=2)


@hydra.main(config_path="../scripts/configs", config_name="policy_server_config")
def test_policy_server_model(cfg):
    model_cfg = dotdict(
        sweep_name=None,
        task_num=None,
        model_path="/viscam/u/stian/mfm/mfm/mfm_vp2_outputs/robosuite_wgpt_f16/last.ckpt",
        cfg_path="/viscam/u/stian/mfm/mfm/mfm_vp2_outputs/robosuite_wgpt_f16/tb/version_1/hparams.yaml",
        ddp_port=23452,
        batch_size=8,
    )
    test_batch = {
        "video": np.random.normal(size=(48, 2, 64, 64, 3)),
        "actions": np.random.normal(size=(48, 12, 4)),
    }
    model_path, cfg_path = model_cfg.model_path, model_cfg.cfg_path
    print("Loading from ", cfg_path)
    model = cku.load_model(cfg_path, model_path)
    model = model.to("cuda")
    test_batch = {k: torch.from_numpy(v).float().cuda() for k, v in test_batch.items()}
    test_batch["video"] = torch.permute(test_batch["video"], (0, 4, 1, 2, 3))
    test_batch["video"] = pad_video_length(test_batch["video"], 12)
    # output = model.sample(14, test_batch, n_cond_frames=2, num_iters=5, temperature=4.5)
    output = model.sample(10, test_batch, n_cond_frames=2, num_iters=5, temperature=4.5)


if __name__ == "__main__":
    test_policy_server_model()
