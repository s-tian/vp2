import numpy as np
from PIL import Image

from hydra.utils import to_absolute_path

import gym.spaces as spaces
from dm_control import mujoco
from vp2.envs.base import BaseEnv
from robodesk import RoboDesk

ALL_STATE_KEYS = [
    "qpos_robot",
    "qvel_robot",
    "qpos_objects",
    "qvel_objects",
    "end_effector",
]
STATE_KEYS = ["qpos_objects", "end_effector"]
JOINT_NAMES = [
    "flat_block",
    "ball",
    "upright_block",
    "slide_joint",
    "drawer_joint",
    "red_button",
    "blue_button",
    "green_button",
]
OBJECT_NAMES = [
    "desk",
    "drawer",
    "slide",
    "red_button",
    "blue_button",
    "green_button",
    "flat_block",
    "upright_block",
]


class RoboDeskWrapper(RoboDesk):

    CROP_BOX_120 = [16.75, 25.0, 105.0, 88.75]

    def render(self, mode="rgb_array", resize=True, crop=True):
        render_size = max(120, self.image_size)
        params = {
            "distance": 1.8,
            "azimuth": 90,
            "elevation": -60,
            "size": render_size,
            "crop_box": [d / 120.0 * render_size for d in self.CROP_BOX_120],
        }
        camera = mujoco.Camera(
            physics=self.physics,
            height=params["size"],
            width=params["size"],
            camera_id=-1,
        )
        camera._render_camera.distance = params[
            "distance"
        ]  # pylint: disable=protected-access
        camera._render_camera.azimuth = params[
            "azimuth"
        ]  # pylint: disable=protected-access
        camera._render_camera.elevation = params[
            "elevation"
        ]  # pylint: disable=protected-access
        camera._render_camera.lookat[:] = [
            0,
            0.535,
            1.1,
        ]  # pylint: disable=protected-access

        image = camera.render(depth=False, segmentation=False)
        camera._scene.free()  # pylint: disable=protected-access

        if resize:
            image = Image.fromarray(image)
            if crop:
                image = image.crop(box=params["crop_box"])
            image = image.resize(
                [self.image_size, self.image_size], resample=Image.Resampling.LANCZOS
            )
            image = np.asarray(image)
        return image

    def get_state(self):
        """
        Returns a np array containing the low-dimensional state of the environment, concatenated together.
        :return:
        """
        state_dict = self.get_state_dict()
        obs_list = [state_dict[k] for k in ALL_STATE_KEYS]
        return np.concatenate(obs_list)

    def reset(self):
        ret_val = super().reset()
        # The original RoboDesk environment doesn't correctly create copied objects for the final object positions,
        # which makes the rewards for the push off table tasks incorrect (and misleading).
        self._save_initial_block_positions()
        return ret_val

    def _save_initial_block_positions(self):
        self.original_pos["ball"] = self.physics.named.data.xpos["ball"].copy()
        self.original_pos["upright_block"] = self.physics.named.data.xpos[
            "upright_block"
        ].copy()
        self.original_pos["flat_block"] = self.physics.named.data.xpos[
            "flat_block"
        ].copy()

    def _get_obs(self):
        # qpos_objects is bugged due to a typo in the original RoboDesk and just returns qvel_objects
        obs_dict = super()._get_obs()
        obs_dict["qpos_objects"] = self.physics.data.qpos[self.num_joints :].copy()
        return obs_dict

    def get_state_dict(self):
        # Return just the state portion of the observation dict, not images
        obs_dict = RoboDeskWrapper._get_obs(self)
        states = {k: obs_dict[k] for k in ALL_STATE_KEYS}
        return states

    def get_named_object_states(self):
        object_states = dict()
        object_states.update(
            {f"{name}_qpos": self.physics.named.data.qpos[name] for name in JOINT_NAMES}
        )
        object_states.update(
            {f"{name}_qvel": self.physics.named.data.qvel[name] for name in JOINT_NAMES}
        )
        object_states.update(
            {
                f"{name}_xpos": self.physics.named.data.xpos[name]
                for name in OBJECT_NAMES
            }
        )
        object_states.update(
            {"slide_handle_xpos": self.physics.named.data.site_xpos["slide_handle"]}
        )
        object_states.update(
            {"drawer_handle_xpos": self.physics.named.data.geom_xpos["drawer_handle"]}
        )
        return object_states


class RoboDeskEnv(RoboDeskWrapper, BaseEnv):
    def __init__(self, **kwargs):
        self.env_hparams = kwargs
        self.env_hparams["goals_dataset"] = to_absolute_path(
            self.env_hparams["goals_dataset"]
        )
        self.image_key_name = kwargs.get("image_key_name", "rgb")
        wrapper_kwargs = [
            "task",
            "reward",
            "action_repeat",
            "episode_length",
            "image_size",
        ]
        wrapper_params = {k: v for k, v in kwargs.items() if k in wrapper_kwargs}
        super().__init__(**wrapper_params)

    def _get_obs(self):
        """
        In RoboDeskEnv, only image observations are returned (not the low-dim state).
        """
        obs = super()._get_obs()
        obs[self.image_key_name] = obs["image"]
        del obs["image"]
        for k in ALL_STATE_KEYS:
            del obs[k]
        return obs

    def reset_to(self, state):
        state = state["states"]
        ###
        # State shapes:
        # qvel_robot = 9
        # qpos_robot = 9
        # qpos_objects = 29
        # qvel_objects = 26
        ###
        self.success = False
        self.num_steps = 0
        self.physics.reset()
        self.physics.data.qpos[: self.num_joints] = state[: self.num_joints]
        self.physics.data.qvel[: self.num_joints] = state[
            self.num_joints : 2 * self.num_joints
        ]
        self.physics.data.qpos[self.num_joints :] = state[
            2 * self.num_joints : 2 * self.num_joints + 29
        ]
        self.physics.data.qvel[self.num_joints :] = state[
            2 * self.num_joints + 29 : 2 * self.num_joints + 29 + 26
        ]
        self.physics.forward()

        # Perform state updates from robodesk reset function (L234-241)
        # Copy physics state into IK simulation.
        self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]
        self._save_initial_block_positions()
        self.drawer_opened = False

        return self._get_obs()

    def compute_score(self, state, goal_state):
        # Reset to the given state
        self.reset_to({"states": state})
        # Compute task success reward and invert (lower should be better)
        return -1 * self._get_task_reward(self.task, "success")

    def reset_state(self, state):
        # Since the robodesk environment always has the same model, we just need to reset the model state
        return self.reset_to(state)

    @property
    def action_dimension(self):
        return super().action_space.shape[0]

    @property
    def observation_space(self):
        return super().observation_space["image"]

    @property
    def observation_shape(self):
        return self.observation_space.low.size

    def goal_generator(self):
        print(f"Using goals dataset {self.env_hparams['goals_dataset']}")
        yield from self.goal_generator_from_robosuite_hdf5(
            self.env_hparams["goals_dataset"], "camera"
        )


class RoboDeskEnvState(RoboDeskWrapper):
    @property
    def observation_space(self):
        env_obs_space = super().observation_space
        final_shape = 0
        for key in STATE_KEYS:
            final_shape += env_obs_space[key].low.size
        if "qpos_objects" in STATE_KEYS:
            final_shape += 3  # The qpos is actually of shape 29, but due to the bug in original robodesk it said 26
        return spaces.Box(np.float32(-np.inf), np.float32(np.inf), shape=(final_shape,))

    def _get_obs(self):
        return self.get_state()

    def render(self, mode="rgb_array", resize=True):
        """
        Disable rendering when we only use the state version of the env
        """
        pass


if __name__ == "__main__":
    # import robodesk

    # env = robodesk.RoboDesk()
    env = RoboDeskEnv()
    obs = env.reset()
    import ipdb

    ipdb.set_trace()
    # assert obs["image"].shape == (64, 64, 3)

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
