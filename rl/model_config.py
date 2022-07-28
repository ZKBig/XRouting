# -*- coding: utf-8 -*-
# @Description: model configuration
# @author: victor
# @create time: 2022-07-27-10:32

from ray.rllib.models import ModelCatalog
from rl.model.XRouting_model import XRoutingModel
from rl.model.normal_PPO_model import PPOModel
from rl.model.DQN_model import DQN

class ModelConfig:
    """
    This class is used to define configurations of rl models
    """

    def __init__(self,
                 num_gpus,
                 num_cpus=4,
                 num_workers=5,
                 num_cpus_per_worker=3,
                 ):
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker

    def XRouting_config(self, env_name):
        """
        Configuration of XRouting actor
        """
        ModelCatalog.register_custom_model("XRouting_model", XRoutingModel)

        return {
            "env": env_name,
            "num_gpus": self.num_gpus,
            "num_envs_per_worker": 1,
            "gamma": 0.99,
            "train_batch_size": 4096,
            "sgd_minibatch_size": 256,
            "lr": 4e-4,
            "vf_loss_coeff": 1e-5,
            "model": {
                "custom_model": "XRouting_model",
                "custom_model_config": {
                    "attention_dim": 64,
                    "num_heads": 4,
                    "head_dim": 32,
                    "mlp_dim": 100,
                    "observation_dim": (38, 6),
                    "pos_encoding_dim": (38, 1),
                }
            },
            "entropy_coeff": 0.01,
            "num_sgd_iter": 4,
            "framework": "tf",
            "num_cpus_per_worker": self.num_cpus_per_worker,
        }

    def ppo_config(self, env_name):
        """
        Configuration of ppo with GTrXL transformer actor
        """
        # register the custom model for action mask
        ModelCatalog.register_custom_model("PPO_model", PPOModel)

        return {
            "env": env_name,
            "num_gpus": self.num_gpus,
            "num_envs_per_worker": 1,
            "gamma": 0.99,
            "train_batch_size": 4096,
            "sgd_minibatch_size": 256,
            "lr": 4e-4,
            "vf_loss_coeff": 1e-5,
            "model": {
                "custom_model": "PPO_model",
                "custom_model_config": {
                    "true_obs_shape": (80,)
                }
            },
            "entropy_coeff": 0.01,
            "num_sgd_iter": 5,
            "framework": "tf",
            "num_cpus_per_worker": self.num_cpus_per_worker,
        }

    def DQN_config(self, env_name, timesteps_per_iteration=4096,
                   min_iter_time_s=1):
        """
        configuration of Dueling DQN for sumo rerouting
        """
        # register the custom model for Dueling DQN
        ModelCatalog.register_custom_model("DQN_model", DQN)

        cfg = {
            "hiddens": [],
            "dueling": False,
        }

        return dict({
            "env": env_name,
            "num_gpus": self.num_gpus,
            "num_envs_per_worker": 1,
            "noisy": False,
            "double_q": True,
            "lr": 0.003,
            "learning_starts": 1000,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 1000
            },
            "model": {
                "custom_model": "DQN_model",
                "custom_model_config": {
                    "true_obs_shape": (80, )
                }
            },
            "target_network_update_freq": 3000,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "final_prioritized_replay_beta": 1.0,
            "n_step": 1,
            "replay_buffer_config": {
                "capacity": 10000,
            },
            "framework": "tf",
            "prioritized_replay_beta_annealing_timesteps": 200000,
            "train_batch_size": 256,
            "num_cpus_per_worker": self.num_cpus_per_worker,
            "timesteps_per_iteration": timesteps_per_iteration,
            "min_iter_time_s": min_iter_time_s
            },
            **cfg)

