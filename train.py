# -*- coding: utf-8 -*-
# @Description: train XRouting model
# @author: victor
# @create time: 2022-07-27-10:53
import argparse
import os

import ray
from ray import tune
from rl.env.dynamic_rerouting_env import DynamicRerouteEnv
from utils.registry import create_env
from ray.tune.registry import register_env
from rl.model_config import ModelConfig
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

DIR = os.getcwd()


def get_arguments():
    parser = argparse.ArgumentParser(description="Dynamic Routing in Virtue of RL")
    parser.add_argument("--run",
                        type=str,
                        default="XRouting",
                        choices=["XRouting", "PPO", "DQN"],
                        help="The sumo-RL registered algorithm to use.")
    parser.add_argument("--stop-iters",
                        type=int,
                        default=10000,
                        help="Number of iterations to train")
    parser.add_argument("--num-cpus",
                        type=int,
                        default=3,
                        help="Number of CPU used when training")
    parser.add_argument("--num-workers",
                        type=int,
                        default=5,
                        help="Number of workers used for sampling")
    parser.add_argument("--stop-timesteps",
                        type=int,
                        default=3000000,
                        help="Number of timesteps to train.")
    parser.add_argument("--stop-reward",
                        type=float,
                        default=500000.0,
                        help="Reward at which we stop training.")
    parser.add_argument("--no-tune",
                        action="store_true",
                        help="Train without Tune")
    parser.add_argument("--sumo-home",
                        type=str,
                        help="The directory of SOMO_HOME environment")
    parser.add_argument("--sumo-env-directory",
                        type=str,
                        help="The local directory of sumo environment")
    parser.add_argument("--sumo-conf-directory",
                        type=str,
                        help="The local directory of sumo configuration")
    parser.add_argument("--sumo-net-directory",
                        type=str,
                        help="The local directory of sumo network file")
    parser.add_argument("--sumo-trace-directory",
                        type=str,
                        help="The local directory of sumo trace file")
    parser.add_argument("--edge-coordinates-dir",
                        type=str,
                        help="The local directory of edge coordinates excel file")
    parser.add_argument("--tripinfo-dir",
                        type=str,
                        help="The local directory of tripinfo files")
    parser.add_argument("--training-results-dir",
                        type=str,
                        help="The local directory of the training results files")

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_arguments()

    ray.init(num_cpus=args.num_cpus or None, local_mode=False)

    # Create the sumo environment
    env, env_name = create_env(params=dict(env_name=DynamicRerouteEnv,
                                           version=0,
                                           reward_threshold=-200,
                                           max_episode_steps=args.stop_timesteps,
                                           observation_size=38,
                                           action_size=4,
                                           initial_edge="right0D0",
                                           destination="A2left2",
                                           work_dir=DIR,
                                           model=args.run
                                           ))
    # Register as gym env
    register_env(env_name, env)

    # determine the algorithm to use
    configuration = ModelConfig(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                                num_cpus=args.num_cpus, num_workers=6, num_cpus_per_worker=1)

    if args.run == "XRouting":
        config = configuration.XRouting_config(env_name=env_name)
        training_results_dir = DIR + "/training_result/XRouting"
        name = "PPO"
    elif args.run == "PPO":
        config = configuration.ppo_config(env_name=env_name)
        name = "PPO"
        training_results_dir = DIR + "/training_result/PPO"
    elif args.run == "DQN":
        config = configuration.DQN_config(env_name=env_name)
        name = "DQN"
        training_results_dir = DIR + "/training_result/DQN"
    else:
        config = configuration.XRouting_config(env_name=env_name)
        name = "PPO"
        training_results_dir = DIR + "/training_result/XRouting"

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # training in virtue of ray.tune.run
    tune.run(
        name,
        config=config,
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=training_results_dir
    )

    ray.shutdown()
