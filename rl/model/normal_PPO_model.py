# -*- coding: utf-8 -*-
# @Description: PPO model with normal actor structure in [1]
# @author: victor
# @create time: 2022-07-27-11:33

"""
[1] - Z. Wang and S. Wang, "XRouting: Explainable Vehicle Rerouting for Urban
      Road Congestion Avoidance using Deep Reinforcement Learning,"
      2022 IEEE International Smart Cities Conference (ISC2).
"""

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow import keras
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.typing import TensorType, List

tf1, tf, tfv = try_import_tf()


class PPOModel(TFModelV2):
    """
    This model is used to handle the discrete action masking requirement
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape,
                 **kwargs):
        super().__init__(obs_space, action_space,
                         num_outputs, model_config, name)

        self.input_layer = keras.layers.Input(shape=true_obs_shape, name="observations")
        self.layer_1 = keras.layers.Dense(150, name="layer_1", activation='relu',
                                          kernel_initializer=normc_initializer(1.0))(self.input_layer)
        self.layer_out = keras.layers.Dense(100, name="layer_2", activation='relu',
                                            kernel_initializer=normc_initializer(1.0))(self.layer_1)

        self._logits = keras.layers.Dense(num_outputs, activation=None)(self.layer_out)
        self.value_output = keras.layers.Dense(1, activation=None)(self.layer_out)

        outs = [self._logits, self.value_output]

        self.values = None

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=outs)

        self.model.summary()

    def forward(self, input_dict, state, seq_lens) \
            -> (TensorType, List[TensorType]):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, self.values = self.model(input_dict["obs"]["real_observation"])

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return tf.reshape(self.values, [-1])
