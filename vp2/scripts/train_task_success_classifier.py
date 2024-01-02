import argparse
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vp2.mpc.utils import dict_to_cuda, write_moviepy_gif
from vp2.util.conv_predictor import ConvPredictor
from fitvid.data.robomimic_data import load_dataset_robomimic_torch


def load_model(model_name, path):
    # assert model_name in DEFAULT_WEIGHT_LOCATIONS.keys(), f"Model name was {model_name} but must be in {list(DEFAULT_LOCATIONS.keys())}"
    depth_model = ConvPredictor(num_linear_layers=1)
    if os.path.exists(path):
        print(f"Loading model from {path}")
        depth_model.load_state_dict(torch.load(path))
    return depth_model


def get_dataloaders(dataset_files, bs, dims, view):
    train_load = load_dataset_robomimic_torch(
        dataset_files,
        batch_size=bs,
        video_len=10,
        video_dims=dims,
        phase="train",
        depth=False,
        normal=False,
        view=view,
        # cache_mode="all_nogetitem",
        cache_mode="low_dim",
        seg=False,
    )
    val_load = load_dataset_robomimic_torch(
        dataset_files,
        batch_size=bs,
        video_len=10,
        video_dims=dims,
        phase="valid",
        depth=False,
        normal=False,
        view=view,
        # cache_mode="all_nogetitem",
        cache_mode="low_dim",
        seg=False,
    )
    return train_load, val_load


def flatten_dims(img):
    shape = img.shape
    final_shape = [-1] + list(shape[-3:])
    img = torch.reshape(img, final_shape)
    return img


def loss_fn(pred, actual):
    return nn.BCEWithLogitsLoss()(pred, actual)


def prep_batch(batch, reward_success_threshold):
    images = flatten_dims(batch["video"])
    labels = (
        (batch["rewards"] >= reward_success_threshold)
        .to(torch.uint8)
        .float()
        .view((-1, 1))
    )
    return images, labels


def log_preds(folder, rgb_images, true_images, preds, epoch, phase):
    preds = preds.detach().cpu().numpy()
    preds = np.transpose(preds, (0, 1, 3, 4, 2)) * 255
    true_images = true_images.detach().cpu().numpy()
    true_images = np.transpose(true_images, (0, 1, 3, 4, 2)) * 255
    rgb_images = rgb_images.detach().cpu().numpy()
    rgb_images = np.transpose(rgb_images, (0, 1, 3, 4, 2)) * 255
    for i, (rgb_image, pred, timg) in enumerate(zip(rgb_images, preds, true_images)):
        if i > 10:
            continue
        # depth_video = depth_to_rgb_im(pred)
        # true_image = depth_to_rgb_im(timg)
        video = np.concatenate([pred, timg, rgb_image], axis=-2)
        save_moviepy_gif(
            list(video), os.path.join(folder, f"{phase}_epoch_{epoch}_pred_{i}")
        )


def get_accuracy(preds, labels):
    # Compute sigmoid of preds and count how many match the labels
    preds = torch.sigmoid(preds)
    return (preds > 0.5).eq(labels).squeeze().sum().item() / len(labels)


def log_preds_images(preds, images, labels):
    from vp2.mpc.utils import generate_text_square, save_np_img

    preds = torch.sigmoid(preds)
    preds, images, labels = (
        preds.detach().cpu().numpy(),
        images.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
    )
    indices = np.arange(len(preds))
    np.random.shuffle(indices)
    indices = indices[:32]
    out_images = []
    for idx in indices:
        pred, image, label = preds[idx], images[idx], labels[idx]
        pred_text = generate_text_square(str(np.round(pred, 3).item()))
        label_text = generate_text_square(str(np.round(label, 3).item()))
        image = np.concatenate([image, pred_text, label_text], axis=-2)
        out_images.append(np.moveaxis(image, 0, -1))
    out_images = np.concatenate(out_images, axis=-2)
    save_np_img(out_images, "preds.png")


def main(args):
    model = load_model(args.model_type, args.checkpoint)
    model = model.cuda()
    train_loader, val_loader = get_dataloaders(
        args.dataset_files,
        args.batch_size,
        (args.image_size, args.image_size),
        args.view,
    )
    print(f"Train loader has length {len(train_loader)}")

    train_steps_per_epoch = 300
    val_steps_per_epoch = 24

    optimizer = torch.optim.Adam(model.parameters())

    output_folder = os.path.dirname(args.output_file)
    if not os.path.exists(output_folder):
        print(f"Creating output folder {output_folder}")
        os.makedirs(output_folder)

    for epoch in range(args.epochs):
        model.eval()
        print("Running validation...")

        if args.eval:
            for i, batch in enumerate(val_loader):
                batch = dict_to_cuda(batch)
                images, labels = prep_batch(batch, args.reward_success_threshold)
                with torch.no_grad():
                    preds = model(images)
                log_preds_images(preds, images, labels)
                exit()
        val_losses, pred_accs = list(), list()
        for i, batch in enumerate(val_loader):
            # save_torch_img(batch['obs']['agentview_shift_2_image'][0][0], 'test_image')
            batch = dict_to_cuda(batch)
            traj_length = batch["video"].shape[1]
            images, labels = prep_batch(batch, args.reward_success_threshold)
            with torch.no_grad():
                preds = model(images)
            val_loss = loss_fn(preds, labels)
            val_losses.append(val_loss)
            prediction_accuracy = get_accuracy(preds, labels)
            pred_accs.append(prediction_accuracy)
            if i > val_steps_per_epoch:
                break
        print(f"Epoch {epoch} validation loss: {torch.stack(val_losses).mean()}")
        print(f"Epoch {epoch} validation accuracy: {np.stack(pred_accs).mean()}")
        # log_preds(output_folder, batch['video'], normal_images, preds, epoch, 'val')

        model.train()
        train_acc = []
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            # save_torch_img(batch['obs']['agentview_shift_2_image'][0][0], 'test_image')
            batch = dict_to_cuda(batch)
            shape = batch["video"].shape
            images, labels = prep_batch(batch, args.reward_success_threshold)
            preds = model(images)
            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
            train_acc.append(get_accuracy(preds, labels))
            if i % 100 == 0:
                print(f"Train loss: {loss}")
            if i > train_steps_per_epoch:
                break
        # log_preds(output_folder, batch['video'], normal_images, preds, epoch, 'train')
        print(f"Epoch {epoch} training loss: {loss}")
        print(f"Epoch {epoch} training accuracy: {np.stack(train_acc).mean()}")
        torch.save(model.state_dict(), args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune MiDaS depth model.")
    parser.add_argument("--checkpoint", default="", help="Model checkpoint to load")
    parser.add_argument(
        "--output_file",
        default="",
        required=True,
        help="Where to save final model params",
    )
    parser.add_argument(
        "--view",
        default="camera",
        required=True,
        help="Camera view to use for training",
    )
    parser.add_argument("--image_size", default=64, help="image dimension")
    parser.add_argument(
        "--dataset_files",
        nargs="+",
        required=True,
        help="number of trajectories to run for complete eval",
    )
    parser.add_argument(
        "--model_type", default="", required=False, help="which MiDaS model to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of finetuning epochs"
    )
    parser.add_argument(
        "--reward_success_threshold",
        type=float,
        default=45.0,
        help="success threshold for a particular obseravtion to be considered a success",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batchsize")
    parser.add_argument("--eval", action="store_true", help="evaluate model")
    args = parser.parse_args()
    main(args)
