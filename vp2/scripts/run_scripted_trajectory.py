import argparse

from vp2.envs.robodesk_env import RoboDeskEnv
from vp2.envs.scripted_policies.robodesk_scripted_policies import *
from vp2.envs.scripted_policies.noisy_policy_wrapper import (
    NoisyPolicyWrapper,
)


def main(args):
    env = RoboDeskEnv(action_repeat=50, image_size=256, task=args.task, reward="dense")
    policy = TASK_TO_POLICY[args.task]
    policy = NoisyPolicyWrapper(policy, noise_std=0.4)
    run_traj(env, policy)


def run_traj(env, policy):
    env.reset()
    policy.reset()
    vis_frames = []
    rewards = []
    for i in range(30):
        action = policy.get_action(env)
        obs, rew, done, info = env.step(action)
        vis_frames.append(obs["rgb"])
        print(rew)
        rewards.append(rew)
    print(f"Total reward is {sum(rewards)}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=str, default="push_red")
    args = args.parse_args()
    main(args)
