# -*- coding: utf-8 -*-
# @Description: DQN model in [1]
# @author: victor
# @create time: 2022-07-27-11:33

"""
[1] - S. Koh, B. Zhou, H. Fang, P. Yang, Z. Yang, Q. Yang, L. Guan, and
      Z. Ji, “Real-time deep reinforcement learning based vehicle navigation,”
      Applied Soft Computing, vol. 96, p. 106694, 2020.
"""

from ray.rllib.agents.dqn.distributional_q_tf_model import \
    DistributionalQTFModel
from ray.rllib.utils.framework import try_import_tf
from tensorflow import keras

tf1, tf, tfv = try_import_tf()


class DQN(DistributionalQTFModel):
    """
    Model Implementation of Dueling DQN
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape,
                 **kw):
        """see the parents"""

        super(DQN, self).__init__(
            obs_space, action_space, num_outputs,
            model_config, name, **kw)

        # define the core model layers which will be used by the output head of
        # DistributedQModel
        self.input_layer = keras.layers.Input(shape=true_obs_shape, name="observations")
        self.layer_1 = keras.layers.Dense(150, name="layer_1", activation='relu')(self.input_layer)
        self.layer_2 = keras.layers.Dense(100, name="layer_2", activation='relu')(self.layer_1)
        self.A_value = keras.layers.Dense(num_outputs, activation=None)(self.layer_2)
        self.V_value = keras.layers.Dense(1, activation=None)(self.layer_2)

        self.q_value = self.get_q_value()

        self.base_model = keras.Model(self.input_layer, self.q_value)

        self.base_model.summary()

    def get_q_value(self):
        """
        calculate q value according to the following equation:
                                q_value = v_value + advantage
        :return: q value with shape [batch, 1]
        """
        advantages_mean = tf.reduce_mean(self.A_value, 1)
        advantages_centered = self.A_value - tf.expand_dims(advantages_mean, 1)
        return self.V_value + advantages_centered

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits = self.base_model(input_dict["obs"]["real_observation"])

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        return masked_logits, state
