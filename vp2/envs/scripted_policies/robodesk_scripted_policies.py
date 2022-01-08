import numpy as np


class RoboDeskScriptedPolicy:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compute_prop_action(self, start, end, k=5, min_magnitude=0.1):
        direction_vector = end - start
        magnitude = np.linalg.norm(direction_vector)
        direction_vector = direction_vector * k
        if np.linalg.norm(direction_vector) < min_magnitude:
            direction_vector = direction_vector / magnitude * min_magnitude
        return direction_vector

    def reset(self):
        pass

    def log(self, *strings):
        if self.verbose:
            print(*strings)


class ButtonPushingPolicy(RoboDeskScriptedPolicy):
    def __init__(self, color, verbose=False):
        super().__init__(verbose)
        self.color = color

    def get_action(self, env):
        end_effector_position = env.get_state_dict()["end_effector"]
        button_position = env.get_named_object_states()[f"{self.color}_button_xpos"]
        xyz_control = self.compute_prop_action(end_effector_position, button_position)
        action = np.zeros(env.action_dimension)
        action[:3] = xyz_control
        return action


class SlidePolicy(RoboDeskScriptedPolicy):
    def __init__(self, goal="open", verbose=False):
        super().__init__(verbose)
        assert goal in ["open", "close"], "Goal must be either 'open' or 'close'"
        self.goal = goal

    def reset(self):
        self.reached_slide = False
        # self.slide_target_shift = np.array([np.random.uniform(-0.1, 0.1, size=1), 0, 0])
        self.slide_target_x = np.random.uniform(-0.1, 0.1, size=1)[0]
        self.slide_target_z = np.random.uniform(-0.05, 0.05, size=1)[0]
        self.slide_target_shift = np.array(
            [self.slide_target_x, 0, self.slide_target_z]
        )
        self.log("Slide target shift is ", self.slide_target_shift)

    def get_action(self, env):
        env_state_dict = env.get_state_dict()
        named_object_states = env.get_named_object_states()
        slide_position = named_object_states["slide_handle_xpos"] - np.array(
            [0.15, 0, 0]
        )
        slide_position = slide_position + self.slide_target_shift
        end_effector_position = env_state_dict["end_effector"]
        if not self.reached_slide and (
            np.linalg.norm(slide_position - end_effector_position) > 0.1
            or np.linalg.norm(slide_position[1] - end_effector_position[1]) > 0.05
        ):
            # move to slide handle
            xyz_control = self.compute_prop_action(
                end_effector_position, slide_position, k=[5, 5 * 0.75, 5]
            )
            self.log("Move to slide")
        else:
            self.reached_slide = True
            # slide the slide over
            slide_qpos = named_object_states["slide_joint_qpos"][0]
            self.log(slide_qpos)
            slide_goal = 1 if self.goal == "open" else 0
            sliding_magnitude = (slide_goal - slide_qpos) * 2
            xyz_control = np.array([sliding_magnitude, 0.1, 0])
            self.log("Pushing slide")

        action = np.zeros(env.action_dimension)
        action[:3] = xyz_control
        return action


class DrawerPolicy(RoboDeskScriptedPolicy):
    def __init__(self, goal="open", verbose=False):
        super().__init__(verbose)
        assert goal in ["open", "close"], "Goal must be either 'open' or 'close'"
        self.goal = goal

    def reset(self):
        self.phase = 0
        self.i = 0
        # self.slide_target_shift = np.array([np.random.uniform(-0.1, 0.1, size=1), 0, 0])
        # self.slide_target_x = np.random.uniform(-0.1, 0.1, size=1)[0]
        # self.slide_target_z = np.random.uniform(-0.05, 0.05, size=1)[0]
        # self.slide_target_shift = np.array([self.slide_target_x, 0, self.slide_target_z])
        # self.log("Slide target shift is ", self.slide_target_shift)

    def get_action(self, env):
        env_state_dict = env.get_state_dict()
        named_object_states = env.get_named_object_states()
        if abs(named_object_states["drawer_joint_qpos"][0]) > 0.05:
            self.log("Open drawer!")
            drawer_position = named_object_states["drawer_handle_xpos"] - np.array(
                [-0.05, named_object_states["drawer_joint_qpos"][0] / 2, 0.02]
            )
        else:
            drawer_position = named_object_states["drawer_handle_xpos"] - np.array(
                [0.05, 0.05, 0.02]
            )
        # drawer_position = named_object_states["drawer_handle_xpos"]
        # slide_position = slide_position + self.slide_target_shift
        end_effector_position = env_state_dict["end_effector"]
        if (
            self.phase == 0
            and np.linalg.norm((drawer_position - end_effector_position)[:2]) < 0.05
        ):
            self.phase += 1
        elif (
            self.phase == 1
            and np.linalg.norm(drawer_position[2] - end_effector_position[2]) < 0.03
        ):
            self.phase += 1
        theta_control = 0
        if self.phase == 0:
            xyz_control = self.compute_prop_action(
                end_effector_position, drawer_position, k=[3, 3, 1.5]
            )
            theta_control = (np.pi - env_state_dict["qpos_robot"][6]) * 0.5
            self.log("Move to drawer XY")
        elif self.phase == 1:
            # move to slide handle
            xyz_control = self.compute_prop_action(
                end_effector_position, drawer_position, k=[3, 3, 3], min_magnitude=0.5
            )
            self.log("Move to drawer")
            self.log(np.linalg.norm(drawer_position - end_effector_position))
        else:
            # slide the slide over
            drawer_qpos = named_object_states["drawer_joint_qpos"][0]
            self.log(drawer_qpos)
            drawer_goal = -0.3 if self.goal == "open" else 0
            sliding_magnitude = drawer_goal - drawer_qpos
            if self.i == 0:
                xyz_control = np.array([0.00, -0.1, -0.06])
            else:
                xyz_control = np.array([0.00, -0.2, -0.06])
            self.i += 1
            self.log("Pushing drawer")
        self.log("xyz_control", xyz_control)
        action = np.zeros(env.action_dimension)
        action[:3] = xyz_control
        action[3] = theta_control
        return action


class PushObjectOffPolicy(RoboDeskScriptedPolicy):
    def __init__(self, target_object="flat_block", verbose=False):
        super().__init__(verbose)
        self.target_object = target_object

    def reset(self):
        self.reached_block = False

        self.target_offset_x = np.random.uniform(-0.2, -0.1, size=1)[0]
        # self.target_offset_y = np.random.uniform(-0.1, 0, size=1)[0]
        self.target_offset_x = -0.15
        self.target_offset_y = 0
        self.target_offset_z = -0.005 if self.target_object == "flat_block" else -0.05
        self.target_offset = np.array(
            [self.target_offset_x, self.target_offset_y, self.target_offset_z]
        )

    def get_action(self, env):
        env_state_dict = env.get_state_dict()
        named_object_states = env.get_named_object_states()
        object_position = named_object_states[f"{self.target_object}_xpos"]
        target_position = object_position + self.target_offset
        end_effector_position = env_state_dict["end_effector"]
        if not self.reached_block and (
            np.linalg.norm(target_position - end_effector_position) > 0.1
            or np.linalg.norm(target_position[2] - end_effector_position[2]) > 0.05
        ):
            # move to slide handle
            xyz_control = self.compute_prop_action(
                end_effector_position, target_position, k=[5, 5 * 0.75, 5]
            )
            self.log("Move to object")
        else:
            self.reached_block = True
            self.log(object_position)
            object_goal = 1
            if object_position[2] > 0.1:
                # pushing_magnitude = (object_goal - object_position[0]) * 2
                # xyz_control = np.array([pushing_magnitude, 0, 0])
                xyz_control = self.compute_prop_action(
                    end_effector_position,
                    object_position - self.target_offset * 2,
                    k=[3, 3, 3],
                )
            else:
                xyz_control = np.zeros(3)
            self.log("Pushing object")

        action = np.zeros(env.action_dimension)
        action[:3] = xyz_control
        return action


TASK_TO_POLICY = dict(
    push_red=ButtonPushingPolicy("red"),
    push_blue=ButtonPushingPolicy("blue"),
    push_green=ButtonPushingPolicy("green"),
    open_slide=SlidePolicy("open"),
    # close_slide = SlidePolicy("close"),
    open_drawer=DrawerPolicy("open"),
    flat_block_off_table=PushObjectOffPolicy("flat_block"),
    upright_block_off_table=PushObjectOffPolicy("upright_block"),
)
