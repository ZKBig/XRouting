# -*- coding: utf-8 -*-
# @Description: multi head attention layer definition
# @author: victor
# @create time: 2022-07-27-09:12

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType
from tensorflow import keras

tf1, tf, tfv = try_import_tf()


class MultiHeadAttention(tf.keras.layers.Layer if tf else object):
    """
    The MultiHeadAttention layer
    """

    def __init__(self,
                 num_heads: int,
                 head_dim: int,
                 output_dim: int,
                 **kwargs):
        """
        Initialize a MultiHeadAttention keras Layer object.

        :param num_heads: The number of attention heads defined in this model.
        :param head_dim: The dimension of a single attention head.
        :param output_dim:  The output dimensions of the multi-head attention unit.
        :param kwargs: other parameters
        """
        super().__init__(**kwargs)

        self._num_heads = num_heads
        self._head_dims = head_dim
        self._qkv_layer = keras.layers.Dense(3 * num_heads * head_dim, use_bias=False)
        self._linear_layer = keras.layers.TimeDistributed(
            keras.layers.Dense(output_dim, use_bias=False, activation=tf.nn.relu)
        )

        self._layernorm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs: TensorType) -> TensorType:
        seq_len = tf.shape(inputs)[1]

        # norm layer
        inputs = self._layernorm(inputs)

        qkv = self._qkv_layer(inputs)

        Qs, Ks, Vs = tf.split(qkv, 3, -1)

        """
        shape: [batch_size, num_heads, seq_length, head_dims]
        """
        Qs = tf.reshape(Qs, [-1, self._num_heads, seq_len, self._head_dims])
        Ks = tf.reshape(Ks, [-1, self._num_heads, seq_len, self._head_dims])
        Vs = tf.reshape(Vs, [-1, self._num_heads, seq_len, self._head_dims])

        attention_logits = tf.matmul(Qs, Ks, transpose_b=True)
        scaled_attention_logits = attention_logits / self._head_dims ** 0.5

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        """
        shape: [batch_size, num_heads, seq_length, head_dims]
        """
        out = tf.matmul(attention_weights, Vs)

        """
        shape: [batch_size, seq_length, num_heads, head_dims]
        """
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, tf.concat((tf.shape(out)[:2], [self._head_dims * self._num_heads]), axis=0))

        return self._linear_layer(out), attention_weights

