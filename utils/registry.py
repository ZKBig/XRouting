# -*- coding: utf-8 -*-
# @Description: register the custom environment in OpenAI gym
# @author: victor
# @create time: 2022-07-27-10:38
import gym
from gym.envs.registration import register

ENV_PARAMS = {
    "env_name": None,
    "version": 3,
    "max_episode_steps": 1000,
    "reward_threshold": -200,
    # "conf_dir": None,
    # "net_dir": None,
    # "trace_dir": None,
    # "coor_dir": None,
    # "tripinfo_dir": None,
    "work_dir": None,
    "observation_size": 38,
    "action_size": 4,
    "initial_edge": None,
    "destination": None
}


def create_env(params):
    """
    Create a parametrized sumo environment with OpenAI gym.

    This environment creation method allows for the specification of several
    key parameters when creating any sumo environment, including the requested
    environment.

    :param params: dict
        OpenAI gym-related parameters:
            - env_name: environment class of the sumo-related environment the
                        experiment is running on. (note: it must be in an
                        importable module.)
            - version: int, the version number of the input environment which
                       is used to define id in gym registry.
            - reward_threshold: the reward threshold which is custom by user
            - max_episode_steps: the max episode steps defined by user

    :return: env instance, name of the created gym environment
    """

    for key in ENV_PARAMS.keys():
        if key not in params:
            raise KeyError('Env parameter "{}" not supplied'.format(key))

    # obtain the base env name from the input parameters
    if isinstance(params["env_name"], str):
        raise TypeError("Please pass the Env instance instead.")
    else:
        base_env_name = params["env_name"].__name__

    # obtain the version number of the obtained env name
    version_number = params["version"]

    # deal with multiple environment being created under the same base name
    envs = gym.envs.registry.all()
    env_ids = [env.id for env in envs]
    while "{}-v{}".format(base_env_name, version_number) in env_ids:
        version_number += 1
    env_name = "{}-v{}".format(base_env_name, version_number)
    print("The new created environment name is ", env_name)

    def env(*_):
        # obtain the entry_point
        entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__
        print(entry_point)

        # obtain the max episode steps
        max_episode_steps = params["max_episode_steps"]

        # obtain the reward threshold
        reward_threshold = params["reward_threshold"]

        # register the environment with OpenAI gym
        register(
            id=env_name,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            reward_threshold=reward_threshold
        )

        # obtain the needed parameters
        # sumo_configuration_file = params["conf_dir"]
        # sumo_network_file = params["net_dir"]
        # sumo_trace_file = params["trace_dir"]
        # edges_position_file = params["coor_dir"]
        # trip_info_file = params["tripinfo_dir"]
        observation_size = params["observation_size"]
        action_size = params["action_size"]
        initial_edge = params["initial_edge"]
        destination = params["destination"]
        model = params["model"]
        work_dir = params["work_dir"]

        # make the env
        return gym.envs.make(env_name, observation_size=observation_size, model=model,
                             action_size=action_size, initial_edge=initial_edge,
                             destination=destination, work_dir=work_dir)

    return env, env_name
