# VP<sup>2</sup>

VP<sup>2</sup> is a benchmark for video prediction models for robotic manipulation via model-predictive control. 
This code accompanies the paper [A Control-Centric Benchmark for Video Prediction](https://arxiv.org/abs/2304.13723) 
(ICLR 2023). The project page containing more details can be found [here](https://s-tian.github.io/projects/vp2/).
 
## Installation

This repository comes with an `environment.yml` file for creating a conda environment with required dependencies. 

After cloning this repository to your machine and installing conda, use `conda env create -f environment.yml` to generate a conda environment called `vp2`.

You will also need to download the data containing task instance specifications (initial states and goals) as
well as the classifier weights for the robodesk environment. Pretrained models for
SVG', FitVid, MCVD, and StructVRNN are also included.

These can all be found at the following link: https://purl.stanford.edu/qf310mj0842 (19.6GB).

The easiest way to get started is to download all files and extract them to the `vp2` folder in this repository.
That is, the `vp2` folder should contain the following directories:
- `envs`
- `models`
- `mpc`
- `scripts`
- `util`
- `cost_classifiers`
- `robodesk_benchmark_tasks`
- `robosuite_benchmark_tasks`
- `pretrained_models`

If you want to place any of the files in a different location, you should point the corresponding variables 
in the Hydra configs to those paths. For instance, if you want to change the location of your goal datasets, point
`goals_dataset` in `scripts/configs/env/robodesk.yaml` and `scripts/configs/env/robosuite.yaml` such that they point to the location of 
the `goals.hdf5` file for each environment. Note that these paths are relative to directory that the script is run from, 
not the output folder as per Hydra defaults.

Lastly, video prediction training datasets for both the robosuite and robodesk environments can be found here:
https://purl.stanford.edu/cw843qn4148 (182GB total).

Included are video data for the robosuite environment (5k trajectories, with the remaining trajectories renderable 
from low-dimensional state, see [below](#rendering-low-dimensional-datasets)) and full video training data for all robodesk tasks.

Note that while there are separate data files for different robodesk tasks, we always train models
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
We provide the experiment commands used in the case study in the paper in the `experiments` folder.


## Set up video prediction models 
Here we provide pretrained code for the experiments in the paper. Each model should be installed through a separate
Python module and interfaced using the code in `vp2/models`. Please install the modules for the models you would like,
and then download the pretrained weights if you have not already (please see "Installation" above for the download link). 

### SVG'
SVG' is implemented in the repository [here](https://github.com/s-tian/svg-prime).
To install,
```
git clone git@github.com:s-tian/svg-prime.git
cd svg-prime 
pip install -e .
```

Example commands for control experiments can be found in `experiments/case_study.txt`.

### FitVid
FitVid is implemented in the repository [here](https://github.com/s-tian/fitvid).
To install, 
```
git clone git@github.com:s-tian/fitvid.git
cd fitvid
pip install -e .
```

Example commands for control experiments can be found in `experiments/case_study.txt`.
Unfortunately the exact model weights for the FitVid model used in the paper case study are not available, but the weights 
provided in the pretrained download are trained in the same way as the models in the paper.

### MCVD
MCVD is implemented in the repository [here](https://github.com/s-tian/mcvd-pytorch), forked from the authors' original implementation.
To install, 
```
git clone git@github.com:s-tian/mcvd-pytorch.git
cd mcvd-pytorch
pip install -e .
```

Example command to run control with MCVD on the robodesk environment (`push_red` task):

```
python scripts/run_control.py hydra.job.name=test_mcvd planning_modalities=[rgb] seed=0 env=robodesk agent.optimizer.init_std=[0.5,0.5,0.5,0.1,0.1] env.task=push_red model=mcvd model_name=mcvd agent.optimizer.objective.objectives.rgb.weight=0.5 agent.optimizer.objective.objectives.classifier.weight=10 agent/optimizer/objective=combined_classifier_mse agent.optimizer.log_every=5 model.checkpoint_dir=pretrained_models/mcvd/rdall_base/checkpoint_330000.pt 
```

### Struct-VRNN
Struct-VRNN is implemented in the repository [here](https://github.com/s-tian/struct-vrnn-pytorch).
To install, 
```
git clone git@github.com:s-tian/struct-vrnn-pytorch.git
cd struct-vrnn-pytorch
pip install -e .
```

Example command to run control with Struct-VRNN on the robodesk environment (`push_red` task):

```
python scripts/run_control.py hydra.job.name=test_structvrnn model_name=structvrnn_robodesk model.checkpoint_dir=pretrained_models/struct-vrnn/rdall_base/ model.epoch=210000 model=keypoint_vrnn planning_modalities=[rgb] seed=0  env=robodesk agent.optimizer.init_std=[0.5,0.5,0.5,0.1,0.1] env.task=push_red  agent.optimizer.objective.objectives.rgb.weight=0.5 agent.optimizer.objective.objectives.classifier.weight=10 agent/optimizer/objective=combined_classifier_mse agent.optimizer.log_every=5
```

Note that some large-valued pixel "specks" appear in the Struct-VRNN predictions. This is a so far unexplained artifact of the model, that may be due to my reimplementation.

### MaskViT
Please stay tuned for the MaskViT code release [here](https://github.com/agrimgupta92/maskvit).


## Adding new models
To add a new model, you can add a new config file to the `vp2/scripts/configs/model` folder.
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
- Create a new environment class implementing the BaseEnv interface in `vp2/envs/base_env.py`.
- Add a new config file to the `vp2/scripts/configs/env` folder. This should specify the environment class to use and any parameters to pass to the environment class.
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

#### hydra.errors.InstantiationException: Error locating target 'vp2.models...', see chained exception above.
This error occurs when the model class specified in the config file cannot be found. Make sure that the model class is installed as a Python module,
as described above in the [model implementations](#model-implementations-and-pre-trained-weights) section.

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
