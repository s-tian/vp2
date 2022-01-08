import os
import torch
import json
import numpy as np
import tqdm
import random
import piq
import lpips
import hydra
from hydra.utils import instantiate
from fitvid.utils.fvd.fvd import get_fvd_logits, frechet_distance
from fitvid.utils.utils import dict_to_cuda
from fitvid.data.robomimic_data import load_dataset_robomimic_torch
from vp2.models.torch_fitvid_interface import FitVidTorchModel
from fitvid.utils.pytorch_metrics import flatten_image
from torchmetrics.functional import structural_similarity_index_measure as ssim

N_BATCHES = 32


@hydra.main(config_path="configs", config_name="config")
def compute_fvd(cfg):
    lpips_official = lpips.LPIPS(net="alex").cuda()
    model = instantiate(cfg.model)

    # set up logging to json
    model_checkpoint_file = model.checkpoint_file
    model_checkpoint_dir = os.path.dirname(model_checkpoint_file)
    save_metrics_path = os.path.join(model_checkpoint_dir, "metrics.json")
    print(f"Save metrics path: {save_metrics_path}")
    # if already exists
    if os.path.isfile(save_metrics_path):
        with open(save_metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = dict()

    # format: is a dictionary where the keys are epoch numbers
    # and each entry is a dictionary with the format metric:value
    if "epoch" in cfg.model:
        epoch = cfg.model.epoch
    else:
        epoch = int(
            "".join(filter(str.isdigit, os.path.basename(model_checkpoint_file)))
        )

    dataset_name = "/".join(cfg.dataset.dataset_file.split("/")[-2:])
    key = str(epoch) + "_" + dataset_name

    if key in metrics:
        print(
            f"Key {key} in metrics already, check for results in {save_metrics_path}!"
        )

    real_embeddings = []
    predicted_embeddings = []
    if "robosuite" in cfg.env._target_:
        view = "agentview_shift_2"
    elif "robodesk" in cfg.env._target_:
        view = "camera"
    else:
        raise ValueError(f"Unknown environment being used")

    test_data_loader = load_dataset_robomimic_torch(
        [cfg.dataset.dataset_file],
        32,
        12,
        (64, 64),
        "valid",
        False,
        False,
        view=view,
        cache_mode=None,
        seg=False,
        only_depth=False,
        augmentation=None,
    )
    mse = []
    lpips_all = []
    ssim_all = []
    pbar = tqdm.tqdm(total=N_BATCHES)
    for i, batch in enumerate(test_data_loader):
        batch = dict_to_cuda(batch)
        batch["actions"] = batch["actions"].float()
        if isinstance(model, FitVidTorchModel):
            with torch.no_grad():
                _, eval_preds = model.model.evaluate(batch, compute_metrics=False)
        else:
            with torch.no_grad():
                batch_prepped = dict()
                batch_prepped["video"] = (
                    batch["video"].permute(0, 1, 3, 4, 2).cpu().numpy()[:, :2]
                )
                batch_prepped["actions"] = batch["actions"].cpu().numpy()
                outputs = model(batch_prepped)
                outputs["rgb"] = (
                    torch.tensor(outputs["rgb"][:, 1:]).permute(0, 1, 4, 2, 3).cuda()
                )
                eval_preds = dict(ag=outputs)
        gt_video = (batch["video"][:, 1:] * 255).to(torch.uint8)
        pred_video = (eval_preds["ag"]["rgb"] * 255).to(torch.uint8)
        with torch.no_grad():

            lpips_official_score = lpips_official(
                flatten_image(gt_video / 255.0) * 2 - 1,
                flatten_image(pred_video / 255.0) * 2 - 1,
            )

        lpips_all.append(lpips_official_score.mean())
        ssim_all.append(
            piq.ssim(flatten_image(gt_video / 255.0), flatten_image(pred_video / 255.0))
        )
        mse.append(
            torch.mean(
                ((batch["video"][:, 1:] - eval_preds["ag"]["rgb"]) ** 2),
                dim=(1, 2, 3, 4),
            )
        )
        real_embeddings.append(get_fvd_logits(gt_video).detach().cpu())
        predicted_embeddings.append(get_fvd_logits(pred_video).detach().cpu())
        pbar.update(1)
        if i == N_BATCHES:
            break
    pbar.close()

    real_embeddings = torch.cat(real_embeddings, dim=0)
    predicted_embeddings = torch.cat(predicted_embeddings, dim=0)
    result = frechet_distance(real_embeddings, predicted_embeddings)
    mse = torch.cat(mse, dim=0).mean()
    ssim_all = torch.stack(ssim_all).mean()
    lpips_all = torch.stack(lpips_all, dim=0).mean()
    print("FVD: {}".format(result.item()))
    print("MSE: {}".format(mse.item()))
    print("LPIPS: {}".format(lpips_all.item()))
    print("SSIM: {}".format(ssim_all.item()))

    metrics[key] = dict(
        fvd=result.item(),
        mse=mse.item(),
        lpips=lpips_all.item(),
        ssim=ssim_all.item(),
    )

    with open(save_metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    compute_fvd()
