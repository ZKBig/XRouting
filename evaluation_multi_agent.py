# -*- coding: utf-8 -*-
# @Description: evaluate the model in multi-agent scenario
# @author: victor
# @create time: 2022-07-27-17:58

import argparse
import os

import ray
from rl.env.multi_agent_rerouting_env import MultiAgentReroutingEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from rl.env.dynamic_rerouting_env import DynamicRerouteEnv
from utils.registry import create_env
from rl.model.XRouting_model import XRoutingModel
from ray.rllib.agents.ppo.ppo import PPOTrainer
from utils.store_observation import store_observation_action
from ray.tune.registry import register_env

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
    parser.add_argument("--trip-dir",
                        type=str,
                        help="The local directory of vehicle trip files")
    parser.add_argument("--evaluation-results-dir",
                        type=str,
                        help="The absolute directory of the evaluation excel results")
    parser.add_argument("--checkpoint-dir",
                        type=str,
                        default=DIR+"/trained_models/XRouting/checkpoint_1/checkpoint-50",
                        help="The absolute directory of the model checkpoint")
    parser.add_argument("--training-results-dir",
                        type=str,
                        default=os.path.abspath('sumo_rerouting_results'),
                        help="The local directory of the training results files")

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_arguments()
    ray.init(num_cpus=3 or None, local_mode=False)
    ModelCatalog.register_custom_model("XRouting_model", XRoutingModel)

    env_name = "MultiAgentReroutingEnv-V0"

    def create_env1():
        return MultiAgentReroutingEnv(observation_size=38, action_size=4,
                                      work_dir=DIR, model=args.run,
                                      destination="A2left2", initial_edge="right0D0")

    # Create the sumo environment
    env1, env_name1 = create_env(params=dict(env_name=DynamicRerouteEnv,
                                             version=0,
                                             reward_threshold=-200,
                                             max_episode_steps=args.stop_timesteps,
                                             observation_size=38,
                                             action_size=4,
                                             work_dir=DIR,
                                             initial_edge="right0D0",
                                             destination="A2left2",
                                             model=args.run
                                             ))
    # Register as gym env
    register_env(env_name1, env1)

    agent1 = PPOTrainer(config={
        "entropy_coeff": 0.01,
        "evaluation_interval": 1,
        "env": "DynamicRerouteEnv-v0",
        "framework": "tf",
        "gamma": 0.99,
        "lr": 0.0004,
        "model": {
            "custom_model": "XRouting_model",
            "custom_model_config": {
                "attention_dim": 64,
                "head_dim": 32,
                "mlp_dim": 100,
                "num_heads": 4,
                "observation_dim": (38, 6),
                "pos_encoding_dim": (38, 1),
            }
        },
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "num_gpus": 0,
        "num_workers": 1,
        "evaluation_num_workers": 1,
        "evaluation_num_episodes": 1,
        "num_sgd_iter": 4,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_loss_coeff": 1e-5
    })

    agent1.restore(args.checkpoint_dir)

    env = create_env1()

    observations = env.reset()
    all_done = False
    flag = False
    rl_car_num = 75

    dones = dict()
    for i in range(rl_car_num):
        dones["rl_{0}".format(i)] = False

    agent_ids = ["rl_{0}".format(i) for i in range(rl_car_num)]
    agent_action_index = dict()

    for i in range(rl_car_num):
        agent_action_index["rl_{0}".format(i)] = 0

    no_car = False

    observation_action_data = dict()
    for i in range(rl_car_num):
        observation_action_data[agent_ids[i]] = dict()

    while not all_done and not no_car:
        actions = {}
        all_done = True
        legal_rl_ids = []
        for agent in agent_ids:
            if not dones[agent]:
                observation = observations[agent]
                if len(observation["action_mask"]) != 0:
                    action = agent1.compute_single_action(observation)
                    actions[agent] = action

                    observation_action_data[agent][agent_action_index[agent]] = {
                        "observation": observation["real_observation"],
                        "action": action,
                        "edge": "initial_edge",
                        "current_edge": "initial_edge",
                        "position_encoding": observation["position"],
                        "sorted_edges": [1 for i in range(38)]
                    }

                    legal_rl_ids.append(agent)
                    agent_action_index[agent] += 1
                else:
                    actions[agent] = 1
            else:
                actions[agent] = 0

        observations, rewards, dones, infos = env.step(actions)
        rl_edges = infos["rl_edges"]
        rl_current_edges = infos["rl_current_edges"]
        sorted_edges = infos["sorted_edges"]

        for k, v in rl_edges.items():
            if len(observation_action_data[k]) != 0:
                for i, j in observation_action_data[k].items():
                    count = i
                observation_action_data[k][count]["edge"] = v
                observation_action_data[k][count]["current_edge"] = rl_current_edges[k]
                observation_action_data[k][count]["sorted_edges"] = sorted_edges[k]

        no_car = infos["no_car"]
        for k, v in dones.items():
            if not v:
                all_done = False

    final_data_list = []

    for k, v in observation_action_data.items():
        temp_observation_sequence = v
        # sort the obtained action-observation sequence
        sorted_temp = dict(sorted(temp_observation_sequence.items(), key=lambda x: x[0], reverse=False))

        temp_list = []

        for number, info in sorted_temp.items():
            temp_dict = info
            temp_dict["rl_id"] = k
            temp_list.append(temp_dict)

        if len(temp_list) > 0:
            final_data_list.append(temp_list)

    store_observation_action(final_data_list, DIR+"/evaluation_results/")

    ray.shutdown()







