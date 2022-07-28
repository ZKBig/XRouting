# -*- coding: utf-8 -*-
# @Description: The structure of XRouting model in [1]
# @author: victor
# @create time: 2022-07-27-09:43

"""
[1] - Z. Wang and S. Wang, "XRouting: Explainable Vehicle Rerouting for Urban
      Road Congestion Avoidance using Deep Reinforcement Learning,"
      2022 IEEE International Smart Cities Conference (ISC2).
"""

import gym
from typing import Any, Optional
from rl.model.multi_head_attention import MultiHeadAttention
from ray.rllib.models.tf.layers import GRUGate
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow import keras
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List

tf1, tf, tfv = try_import_tf()


class FeedforwardMLP(tf.keras.layers.Layer if tf else object):
    """
    MLP layer in the XRouting Structure
    """

    def __init__(self,
                 out_dim: int,
                 hidden_dim: int,
                 output_activation: Optional[Any] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self._hidden_layer = keras.layers.Dense(
            hidden_dim,
            activation=tf.nn.relu,
        )

        self._output_layer = keras.layers.Dense(
            out_dim, activation=output_activation)

    def call(self, inputs: TensorType, **kwargs) -> TensorType:
        del kwargs
        output = self._hidden_layer(inputs)
        return self._output_layer(output)


class XRoutingModel(TFModelV2):
    """
    Definition of XRouting model structure in [1]
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: Optional[int],
                 model_config: ModelConfigDict,
                 name: str,
                 *,
                 attention_dim: int = 64,
                 num_heads: int = 2,
                 head_dim: int = 32,
                 mlp_dim: int = 32,
                 observation_dim,
                 pos_encoding_dim,
                 init_gru_gate_bias: float = 2.0):
        super().__init__(observation_space, action_space, num_outputs,
                         model_config, name)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.obs_dim = observation_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.num_outputs = num_outputs

        # Observation input layer (observation dim)
        input_layer = keras.layers.Input(
            shape=self.obs_dim, name="observation"
        )

        # position encoding input layer (position encoding dim)
        pos_encoding_input = keras.layers.Input(
            shape=self.pos_encoding_dim, name="pos_encoding"
        )

        # observation embedding layer
        observation_embedding_out = keras.layers.Dense(self.attention_dim)(input_layer)

        # position embedding layer
        pos_encoding_out = keras.layers.Dense(self.attention_dim / 2)(pos_encoding_input)

        # concat[observation embedding, position embedding]
        E_out = tf.concat([observation_embedding_out, pos_encoding_out], 2)

        # MHA layer (norm layer is defined in MHA)
        self.MHA = MultiHeadAttention(
            output_dim=self.attention_dim * (3 / 2),
            num_heads=num_heads,
            head_dim=head_dim, )
        output_1, self.attention_out = self.MHA(E_out)

        # GRU layer 1
        self.GRU_1 = GRUGate(init_gru_gate_bias)
        outputs = self.GRU_1((E_out, E_out + output_1))

        # norm layer
        self._layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        output_2 = self._layernorm(outputs)

        # MLP layer
        self.mlp = FeedforwardMLP(out_dim=self.attention_dim * (3 / 2),
                                  hidden_dim=mlp_dim,
                                  output_activation=tf.nn.relu)
        output_3 = self.mlp(output_2)

        # GRU layer 2
        self.GRU_2 = GRUGate(init_gru_gate_bias)
        E_out = self.GRU_2((outputs, outputs + output_3))

        self._logits = None
        self._value_out = None

        # flatten layer
        flatten_layer = tf.keras.layers.Flatten()(E_out)

        # dense layer before outputting
        out = tf.keras.layers.Dense(
            64, activation=tf.nn.relu, name="final_linear")(flatten_layer)

        # policy probability distribution
        self._logits = tf.keras.layers.Dense(
            self.num_outputs, activation=None, name="logits")(out)

        # state value output
        values_out = tf.keras.layers.Dense(
            1, activation=None, name="values")(out)

        outs = [self._logits, values_out]

        self.model = tf.keras.Model(
            inputs=[input_layer, pos_encoding_input], outputs=outs)

        self.model.summary()

    def forward(self, input_dict, state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        observations = input_dict["obs"]["real_observation"]
        position = input_dict["obs"]["position"]
        action_mask = input_dict["obs"]["action_mask"]

        out = self.model([observations, position])[0]

        self._value_out = self.model([observations, position])[1]

        out = tf.reshape(out, [-1, self.num_outputs])

        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        masked_logits = out + inf_mask

        return masked_logits, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])

    def get_attention_weights(self) -> TensorType:
        return self.attention_out
