import numpy as np
import torch
import cv2

from torchvision.transforms import Resize, InterpolationMode


class ObservationList:

    """
    Represents a list of observations with possibly multiple modalities.
    The internal representation is a dictionary of numpy arrays, where the keys are the modalities
    and the values are numpy arrays of shape [T, H, W, C] where T is the number of time steps,
    H and W are the height and width of the image, and C is the number of channels.
    """

    POSSIBLE_MODALITIES = ["rgb", "depth", "normal", "policy"]

    def __init__(self, data_dict, add_time_dimension=False, image_shape=(64, 64)):
        self.data_dict = data_dict
        self.image_shape = image_shape
        for k, v in self.data_dict.items():
            assert (
                k in self.POSSIBLE_MODALITIES
            ), f"observation key {k} not in possible modalities list!"
            if len(v.shape) > 2:
                assert v.shape[-2] == v.shape[-3], "Image is not square!"
                if v.shape[-3:-1] != self.image_shape:
                    self.data_dict[k] = resize_np_image_aa(v, self.image_shape)
        if add_time_dimension:
            for (
                k,
                v,
            ) in self.data_dict.items():
                self.data_dict[k] = self.data_dict[k][None]

    def __len__(self):
        return len(next(iter(self.data_dict.items()))[1])

    @classmethod
    def from_obs(cls, obs, cfg):
        data_dict = dict()
        for modality in cfg.planning_modalities:
            if modality == "depth":
                img = obs[modality].copy()
                if cfg.env.renderer == "igibson":
                    img = 1.0 / (img + 1e-10)
                    img = clip_and_norm(img, -4.5, -0.2)[..., None]
                elif cfg.env.renderer == "mujoco":
                    img = (img - 0.9717689) / (0.9963344 - 0.9717689)
                else:
                    raise NotImplementedError("Depth not supported for this renderer")
            elif modality == "rgb":
                img = obs[modality].copy() / 255.0
            elif modality == "normal":
                img = obs[modality].copy() / 255.0
            data_dict[modality] = img[None]  # Add time dimension
        return ObservationList(data_dict)

    @classmethod
    def from_observations_list(cls, l, axis=0):
        """
        :param l: Python list of ObservationList objects
        :return: Single observationlist with the objects concatenated
        """
        return l[0].append(*l[1:], axis=axis)

    def append(self, *obs, axis=0):
        for k in self.data_dict.keys():
            to_cat = [self.data_dict[k]] + [o.data_dict[k] for o in obs]
            self.data_dict[k] = np.concatenate(to_cat, axis=axis)
        return self

    def repeat(self, times, axis=0):
        for k, item in self.data_dict.items():
            self.data_dict[k] = np.repeat(item, times, axis=axis)
        return self

    def to_image_list(self):
        """
        :return: a numpy array of shape [T, H, W, C] which contains the image representations of all
        observation modalities, concatenated in the vertical direction.
        """
        imgs = []
        for key, val in self.data_dict.items():
            if key == "depth":
                imgs.append(depth_to_rgb_im(val) / 255.0)
            elif key in ["rgb", "normal"]:
                imgs.append(val)
        img = (np.concatenate(imgs, axis=-3) * 255).astype(np.uint8)
        return img

    def log_gif(self, name, fps=5):
        obs_list = self.to_image_list()
        write_moviepy_gif(list(obs_list), name, fps=fps)

    def save_image(self, fname, filetype="png", index=0):
        img_list = self.to_image_list()
        save_np_img(img_list[index], fname, filetype=filetype)

    def __getitem__(self, key):
        """
        Enable standard numpy slicing behavior
        """
        if isinstance(key, slice):
            return ObservationList({k: v[key] for k, v in self.data_dict.items()})
        if isinstance(key, str):
            return self.data_dict[key]
        # Note that if we select only a single index, we maintain the time dimension
        return ObservationList({k: v[key][None] for k, v in self.data_dict.items()})


def write_moviepy_gif(obs_list, name, fps=5):
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(obs_list, fps=fps)
    if not name.endswith(".gif"):
        name = f"{name}.gif"
    clip.write_gif(f"{name}", fps=fps)


# TODO: Replace extract_image in simulator_model and deprecate
def extract_image(obs, args):
    if args.depth_only:
        img = obs[f"{args.camera_names[0]}_depth"].copy()
        img = 1.0 / (img + 1e-10)
        img = clip_and_norm(img, -4.5, -0.2)[..., None] * 255
    else:
        img = obs[f"{args.camera_names[0]}_image"].copy()
        if args.depth:
            depth_img = obs[f"{args.camera_names[0]}_depth"].copy()
            depth_img = 1.0 / (depth_img + 1e-10)
            depth_img = clip_and_norm(depth_img, -4.5, -0.2)[..., None] * 255
            img = np.concatenate((img, depth_img), axis=-1)
        if args.normal:
            normal_img = obs[f"{args.camera_names[0]}_normal"].copy()
            img = np.concatenate((img, normal_img), axis=-1)
    if tuple(img.shape[-2:]) != (args.camera_height, args.camera_width):
        img = resize_np_image_aa(img, (args.camera_height, args.camera_width)).astype(
            np.uint8
        )
    return img


def generate_text_square(text, size=(64, 64), fontscale=2.5):
    img = np.ones(shape=(512, 512, 3), dtype=np.int16)
    cv2.putText(
        img=img,
        text=text,
        org=(50, 250),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=fontscale,
        color=(255, 255, 255),
        thickness=3,
    )
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = np.moveaxis(img, 2, 0)
    return img


def save_np_img(arr, fname, filetype="png"):
    """
    Write the given nd.ndarray to a image file using PIL
    """
    from PIL import Image

    if isinstance(arr, np.floating) or arr.dtype == "float" or arr.dtype == "float32":
        arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:  # depth image
        arr = depth_to_rgb_im(arr)
    im = Image.fromarray(arr)
    if filetype not in fname:
        fname = f"{fname}.{filetype}"
    im.save(fname)


def save_torch_img(tensor, fname, filetype="png"):
    assert isinstance(
        tensor, torch.Tensor
    ), "input to save_torch_img was not a torch tensor!"
    tensor = torch.permute(tensor, (1, 2, 0)).detach().cpu().numpy()
    save_np_img(tensor, fname, filetype=filetype)


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))


def chw_to_hwc(t):
    return torch.movedim(t, -3, -1)


def hwc_to_chw(t):
    return torch.movedim(t, -1, -3)


def resize_np_image_aa(img, dims):
    img = torch.tensor(img)
    img = hwc_to_chw(img)
    interpolation_mode = InterpolationMode.BILINEAR
    img = Resize(dims, interpolation=interpolation_mode, antialias=True)(img)
    img = chw_to_hwc(img)
    return img.numpy()


def clip_and_norm(d, depth_min, depth_max):
    d = np.clip(d, depth_min, depth_max)
    d = (d - depth_min) / (depth_max - depth_min)
    return d


def stack_into_dict(iterable_of_dicts, axis=0):
    # Given a list of dicts with the same keys, perform np.stack on the values to form a single dict.
    # The axis argument specifies the axis to stack on.
    new_dict = dict()
    for k in iterable_of_dicts[0]:
        to_stack = [d[k] for d in iterable_of_dicts]
        new_dict[k] = np.stack(to_stack, axis=axis)
    return new_dict


def concat_into_dict(iterable_of_dicts, axis=0):
    # Given a list of dicts with the same keys, perform np.stack on the values to form a single dict.
    # The axis argument specifies the axis to stack on.
    new_dict = dict()
    for k in iterable_of_dicts[0]:
        to_cat = [d[k] for d in iterable_of_dicts]
        new_dict[k] = np.concatenate(to_cat, axis=axis)
    return new_dict


def dict_to_numpy(d):
    return {
        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in d.items()
    }


def dict_to_float_tensor(d):
    return {
        k: torch.from_numpy(v).float() if not torch.is_tensor(v) else v
        for k, v in d.items()
    }


def dict_to_cuda(d, device="cuda"):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


def slice_dict(d, start_idx, end_idx, squeeze=False):
    if squeeze:
        return {k: v[start_idx:end_idx].squeeze() for k, v in d.items()}
    return {k: v[start_idx:end_idx] for k, v in d.items()}


def stack_dicts(dicts):
    out = dict()
    for k in dicts[0].keys():
        if isinstance(dicts[0][k], np.ndarray):
            out[k] = np.stack([d[k] for d in dicts])
        else:
            out[k] = torch.stack([d[k] for d in dicts], dim=0)
    return out


def cat_dicts(dicts):
    out = dict()
    for k in dicts[0].keys():
        if isinstance(dicts[0][k], np.ndarray):
            out[k] = np.concatenate([d[k] for d in dicts])
        else:
            out[k] = torch.cat([d[k] for d in dicts], dim=0)
    return out
