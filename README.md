# VP<sup>2</sup>

VP<sup>2</sup> is a benchmark for video prediction models for robotic manipulation via model-predictive control. 
This code accompanies the paper [A Control-Centric Benchmark for Video Prediction](https://arxiv.org/abs/2304.13723) 
(ICLR 2023).
 
## Installation

This repository comes with an `environment.yml` file for creating a conda environment with required dependencies. 

After cloning this repository to your machine and installing conda, use `conda env create -f environment.yml` to generate a conda environment called `vp2`.

You will also need to download the data containing task instance specifications (initial states and goals). 
This can be found at the following link: https://purl.stanford.edu/qf310mj0842 (1.5GB).

After downloading, you should point the `goals_dataset` variables (currently with dummy values) in 
`scripts/configs/env/robodesk.yaml` and `scripts/configs/env/robosuite.yaml` such that they point to the location of 
the `goals.hdf5` file for each environment. Note that these paths are relative to directory that the script is run from, 
not the output folder as per Hydra defaults.

Lastly, video prediction training datasets for both the robosuite and robodesk environments can be found here:
https://purl.stanford.edu/cw843qn4148 (182GB total).

Included are video data for the robosuite environment (5k trajectories, with the remaining trajectories renderable 
from low-dimensional state, see [below](#rendering-low-dimensional-datasets)) and full video training data for all robodesk tasks.

Note that while there are for example separate data files for different robodesk tasks, we always train models
on all of the robodesk data (or all the robosuite data) at once.

## Running control experiments
Configs in this repo are handled using [Hydra](https://hydra.cc/). 
The entry point for control is `python scripts/run_control.py`. 
This will run a control benchmark using the configuration specified in `scripts/configs/config.yaml`.


To run a control experiment with a different configuration, you can specify a different configuration file or
use command line overrides (see Hydra documentation for more details). The high level configuration choices
are as follows:
- `model`: the video prediction model to use. By default, this is set to `fitvid`, which loads the FitVid model config
from [the config folder.](https://github.com/s-tian/vp2/blob/master/vp2/scripts/configs/model/fitvid.yaml)
As an example, to use the SVG' model, you can set `model=svg_prime`.
- `agent`: By default, this is the MPC agent `planning_agent`. `random_agent` is a random baseline.
- `env`: The environment to run the control experiment in. By default, this is `robosuite` to run in 
robosuite environments. `robodesk` contains the remainder of tasks.

Lower level configuration choices include:
- `agent/optimizer`: The optimizer to use for the MPC agent. By default, this is `mppi`.
We implement `cem`, `mppi`, `cem-gd`, and `lbfgs`. The latter two require the model and cost function to be differentiable.

## Example control experiment commands
We provide the experiment commands used in the paper in the `experiments` folder.

## Adding new models
To add a new model, you can add a new config file to the `perceptual_metrics/scripts/configs/model` folder.
The config file should use `_target_` to specify the model class to use. The remainder of config items will be 
passed to the model class as kwargs.

The model class should implement a function `__call__` that takes as input a dictionary with two keys:
- `video`: context frames from the environment, a torch tensor of shape `(B, T, C, H, W)` in range `[0, 1]`
- `actions`: the history of actions taken by the agent a torch tensor of shape `(B, T, A)` in range `[-1, 1]`
where `B` is the batch size, `T` is the number of frames in the video, `C` is the number of channels, `H` and `W` are the height and width, 
and A is the action dimension.

It should then return a dictionary with one key:
- `rgb`: a torch tensor of shape `(B, T+n_pred, C, H, W)` in range `[0, 1]` containing the predicted frames.

## Adding new environments
Although we provide code and configs for the robosuite and robodesk environments, it is relatively straightforward to add new environments.
To add a new environment, you should do the following:
- Create a new environment class implementing the BaseEnv interface in `perceptual_metrics/envs/base_env.py`.
- Add a new config file to the `perceptual_metrics/scripts/configs/env` folder. This should specify the environment class to use and any parameters to pass to the environment class.
- Use the `env` config to specify the new environment from the `env` config group in the control experiment config or via command override. 

## Rendering low-dimensional datasets
The full (50k trajectory) training dataset for the robosuite environment is provided as `robosuite_demo_1` through `robosuite_demo_5`. 
Each dataset contains 10000 trajectories (so there are 50000 in total). Note that we perform most experiments on just 5000 trajectories, which is provided with full RGB image data. 
The download for the full 50k datasets only contains the raw environment observations. 
To perform video prediction training, they must be rendered into images, using the following command after cloning 
[this branch](https://github.com/s-tian/robomimic/tree/depth_images) of robomimic:

(This renders the data at `256x256` resolution as we do in the paper, but you can specify any resolution.)

```
python robomimic/robomimic/scripts/dataset_states_to_obs.py --dataset /PATH/HERE/demo.hdf5 --output_name rendered_256.hdf5 --done_mode 2 --camera_names agentview_shift_2 --camera_depths 1 --camera_segmentations instance --camera_height 256 --camera_width 256 --renderer igibson
```

## FAQ / Troubleshooting

#### MuJoCo rendering fails with the message `Offscreen framebuffer is not complete, error 0x8cdd`:
This seems to be related to EGL driver issues with the `dm_control` package. See [this thread](https://github.com/deepmind/dm_control/issues/370) for more details. 
For a workaround, try setting the `dm_control` rendering environment variable via `export MUJOCO_GL=osmesa`. 
Note that this unfortunately does not support GPU rendering, but this is usually not the bottleneck for visual foresight experiments.

## Citation
If you find this code useful, please cite the following paper:

```
@inproceedings{tian2023vp2,
  title={A Control-Centric Benchmark for Video Prediction},
  author={Tian, Stephen and Finn, Chelsea and Wu, Jiajun},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
