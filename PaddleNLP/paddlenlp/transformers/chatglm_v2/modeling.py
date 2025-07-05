# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.utils import map_structure
from paddle.distributed import fleet
from paddlenlp.transformers.long_sequence_strategies import LongSequenceStrategies
import time
from ...utils.converter import StateDictNameMapping, init_name_mappings
from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from .configuration import CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP, ChatGLMv2Config
import numpy as np
import logging
#paddle.set_default_dtype("float16")
logger = logging.getLogger('auto_tuner')
logger.setLevel(logging.INFO)

__all__ = [
    "ChatGLMv2Model",
    "ChatGLMv2PretrainedModel",
    "ChatGLMv2ForCausalLM",
]


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, original_impl=True):
        super().__init__()
        self.default_dtype = paddle.get_default_dtype()
        #self.default_dtype = 'float16'
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = True

    def forward_impl(self, seq_len: int, n_elem: int, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (paddle.arange(0, n_elem, 2, dtype="float32") / n_elem))
        ####logger.info(f"theta :{theta}")
        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = paddle.arange(0, seq_len, dtype="float32")
        ####logger.info(f"seq_idx :{seq_idx}")
        # Calculate the product of position index and $\theta_i$
        #wrong
        idx_theta = paddle.outer(seq_idx, theta).astype(self.default_dtype)
        ####logger.info(f"idx_theta :{idx_theta}")
        cache = paddle.stack([paddle.cos(idx_theta), paddle.sin(idx_theta)], axis=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        #if self.default_dtype in (paddle.float16, paddle.bfloat16, paddle.int8):
        if self.default_dtype in (paddle.float16, paddle.bfloat16, paddle.int8):
            #cache = cache.astype(self.default_dtype)
            #cache = cache.bfloat16() if dtype == paddle.bfloat16 else cache.astype("float16")
            cache = cache.astype("bfloat16") if self.default_dtype == "bfloat16" else cache.astype("float16")
        ####logger.info(f"cache :{cache}")
        return cache

    def forward(self, max_seq_len, offset=0):
        output = self.forward_impl(seq_len=max_seq_len, n_elem=self.dim)
        #logger.info(f"RotaryEmbedding")
        #logger.info(f"RotaryEmbedding_output:{output}")

        return output


# @paddle.jit.script
def apply_rotary_pos_emb(x: paddle.Tensor, rope_cache: paddle.Tensor) -> paddle.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape([sq, -1, np, rot_dim // 2, 2])
    rope_cache = rope_cache.reshape([sq, -1, 1, xshaped.shape[3], 2])
    x_out2 = paddle.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return paddle.concat((x_out2, x_pass), axis=-1)


class RMSNorm(nn.Layer):
    def __init__(self, hidden_size, epsilon=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.epsilon = 1e-5 if epsilon is None else epsilon

    def forward(self, hidden_states):
        #######logger.info(f"RMSNorm")
        #######logger.info(f"input:{hidden_states}")
        input_dtype = hidden_states.dtype
        variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.epsilon) 
        output = (self.weight * hidden_states).astype(input_dtype)

        #logger.info(f"RMSNorm")
        #logger.info(f"RMSNorm_output:{output}")
        return output


class CoreAttention(nn.Layer):
    def __init__(self, config: ChatGLMv2Config, layer_number):
        super(CoreAttention, self).__init__()

        self.default_dtype = paddle.get_default_dtype()
        #self.default_dtype = 'float16'
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        self.num_attention_heads_per_partition = config.num_attention_heads
        self.hidden_size_per_partition = config.kv_channels * self.num_attention_heads_per_partition
        self.hidden_size_per_attention_head = self.hidden_size_per_partition // self.num_attention_heads_per_partition

        self.tensor_parallel_degree = config.tensor_parallel_degree
        if self.tensor_parallel_degree > 1:
            assert (
                self.hidden_size_per_partition % self.tensor_parallel_degree == 0
            ), "hidden_size_per_partition % tensor_parallel_degree must be zero."
            self.hidden_size_per_partition = self.hidden_size_per_partition // self.tensor_parallel_degree

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)


    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        # Raw attention scores
        # [batch_size, num_heads, query_length, key_length]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [query_length, batch_size, num_heads, hidden] -> [query_length, batch_size * num_heads, hidden]
        query_layer = query_layer.reshape([output_size[2], output_size[0] * output_size[1], -1])
        # [key_length, batch_size, num_heads, hidden] -> [key_length, batch_size * num_heads, hidden]
        key_layer = key_layer.reshape([output_size[3], output_size[0] * output_size[1], -1])

        # Raw attention scores. [batch_size * num_heads, query_length, key_length]
        matmul_result = paddle.bmm(query_layer.transpose([1, 0, 2]), key_layer.transpose([1, 2, 0])) * (
            1.0 / self.norm_factor
        )

        # change view to [batch_size, num_heads, query_length, key_length]
        attention_scores = matmul_result.reshape(output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [batch_size, num_heads, query_length, key_length]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.astype("float32")
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff

        full_attention_mask = attention_mask
        # if attention_mask is not None:
        #     full_attention_mask = self.get_masks(attention_mask, output_size[2], output_size[1])

        if full_attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            # Create a lower triangular mask with ones
            tril_attention_mask = paddle.ones(
                shape=[output_size[0], 1, output_size[2], output_size[3]],
                dtype='bool'
            )
            # Make the mask lower triangular
            tril_attention_mask = paddle.tril(tril_attention_mask)
            # Invert the mask (logical NOT)
            tril_attention_mask = ~tril_attention_mask
            full_attention_mask = paddle.where(tril_attention_mask, paddle.full_like(attention_scores, float("-inf")), paddle.zeros_like(attention_scores))
        elif full_attention_mask is not None and attention_scores.shape[2] == attention_scores.shape[3]:
            # Create a lower triangular mask with ones
            tril_attention_mask = paddle.ones(
                shape=[output_size[0], 1, output_size[2], output_size[3]],
                dtype='bool'
            )
            # Make the mask lower triangular
            tril_attention_mask = paddle.tril(tril_attention_mask)
            # Invert the mask (logical NOT)
            tril_attention_mask = ~tril_attention_mask
            tril_attention_mask = paddle.where(tril_attention_mask, paddle.full_like(attention_scores, float("-inf")), paddle.zeros_like(attention_scores))
            full_attention_mask = full_attention_mask + tril_attention_mask

        if full_attention_mask is not None:
            attention_scores = attention_scores + full_attention_mask#这一句没用要修改，要把

        attention_probs = F.softmax(attention_scores.astype("float32"), axis=-1)
        attention_probs = attention_probs.astype(self.default_dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # [batch_size, num_heads, query_length, key_length]

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape([value_layer.shape[0], output_size[0] * output_size[1], -1])
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape([output_size[0] * output_size[1], output_size[2], -1])
        # matmul: [b * np, sq, hn]
        context_layer = paddle.bmm(attention_probs, value_layer.transpose([1, 0, 2]))
        # change view [b, np, sq, hn]
        context_layer = context_layer.reshape(output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.transpose([2, 0, 1, 3])
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context_layer.shape[:-2] + [self.hidden_size_per_partition]
        context_layer = context_layer.reshape(new_context_shape)

        return context_layer

class SelfAttention(nn.Layer):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMv2Config, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        # assert (
        #     config.kv_channels * config.num_attention_heads == config.hidden_size
        # ), "`kv_channels` * `num_attention_heads` must equal to `hidden_size`"

        # Per attention head and per partition values.

        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
        self.core_attention = CoreAttention(config, self.layer_number)
        self.num_multi_query_groups_per_partition = config.multi_query_group_num

        self.multi_query_group_num = config.multi_query_group_num
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.tensor_parallel_degree = config.tensor_parallel_degree

        if config.tensor_parallel_degree > 1:
            self.query_key_value = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size, config.hidden_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num, has_bias=config.add_bias_linear or config.add_qkv_bias, gather_output=False
            )
            self.dense = fleet.meta_parallel.RowParallelLinear(
                config.hidden_size, config.hidden_size, input_is_parallel=True, has_bias=False
            )
            self.num_attention_heads_per_partition = config.num_attention_heads // config.tensor_parallel_degree
            assert (
                self.num_multi_query_groups_per_partition % self.tensor_parallel_degree == 0
            ), "`multi_query_group_num` % `tensor_parallel_degree` must equal to `0`"
            self.num_multi_query_groups_per_partition = (
                self.num_multi_query_groups_per_partition // self.tensor_parallel_degree
            )
        else:
            self.query_key_value = nn.Linear(
                config.hidden_size,
                config.hidden_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num,
                bias_attr=config.add_bias_linear or config.add_qkv_bias,
            )
            self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=config.add_bias_linear)


    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        #logger.info(f"self_atten_hidden_states:{hidden_states}")
        ###logger.info(f"query_key_value.weight:{self.query_key_value.weight}")
        seq_length, batch_size, hidden_size = hidden_states.shape

        mixed_x_layer = self.query_key_value(hidden_states)
        #logger.info(f"self_atten_query_key_value:{mixed_x_layer}")

        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head * self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head * self.num_multi_query_groups_per_partition,
            ],
            axis=-1,
        )

        
        query_layer = query_layer.reshape(
            [seq_length, batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        )
        key_layer = key_layer.reshape([seq_length, batch_size, -1, self.hidden_size_per_attention_head])
        value_layer = value_layer.reshape([seq_length, batch_size, -1, self.hidden_size_per_attention_head])

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
        #logger.info(f"rotary_pos_emb——query_layer:{query_layer}")
        #logger.info(f"rotary_pos_emb——key_layer:{key_layer}")
        # adjust key and value for inference
        if use_cache:
            if kv_cache is not None:
                #logger.info(f"kv_cache is not None")
                cache_k, cache_v = kv_cache
                # #######logger.info(f"cache{cache_k}")
                # #######logger.info(f"key{key_layer}")
                # time.sleep(10)
                key_layer = paddle.concat((cache_k, key_layer), axis=0)
                value_layer = paddle.concat((cache_v, value_layer), axis=0)
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None
        ########logger.info(f"self.num_attention_heads_per_partition:{self.num_attention_heads_per_partition}")
        #self.num_multi_query_groups_per_partition = 1
        multiplier = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
        ########logger.info(f"key_layer.shape:{key_layer.shape}")
        key_layer = key_layer.unsqueeze(-2).tile([1, 1, 1, multiplier, 1])
        ########logger.info(f"key_layer.shape2:{key_layer.shape}")
        key_layer = key_layer.reshape(
            key_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        )
        value_layer = value_layer.unsqueeze(-2).tile([1, 1, 1, multiplier, 1])
        value_layer = value_layer.reshape(
            value_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        )
        #logger.info(f"core_before——query_layer:{query_layer}")
        #logger.info(f"core_before——key_layer:{key_layer}")
        #logger.info(f"core_before——value_layer:{value_layer}")
        #logger.info(f"attention_mask{attention_mask}")
        # ==================================
        # core attention computation
        # ==================================
        
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        ########logger.info(f"context_layer:{context_layer}")

        # =================
        # Output. [seq_length, b, h]
        # =================

        output = self.dense(context_layer)
        #####logger.info(f"SelfAttention")
        #####logger.info(f"output:{output}")
        #####logger.info(f"kv_cache:{kv_cache}")
        return output, kv_cache


class MLP(nn.Layer):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMv2Config):
        super(MLP, self).__init__()
   
        self.add_bias = config.add_bias_linear

        if config.tensor_parallel_degree > 1:
            self.dense_h_to_4h = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size, 2* config.ffn_hidden_size, has_bias=self.add_bias, gather_output=False
            )
            
            self.dense_4h_to_h = fleet.meta_parallel.RowParallelLinear(
                config.ffn_hidden_size, config.hidden_size, input_is_parallel=True, has_bias=self.add_bias
            )
        else:
            # Project to 4h due to swiglu doubling the output width, see https://arxiv.org/pdf/2002.05202.pdf
            self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias_attr=self.add_bias)
            # Project back to h.
            self.dense_4h_to_h = nn.Linear(
                config.ffn_hidden_size,
                config.hidden_size,
                bias_attr=self.add_bias,
            )

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        dim_size = intermediate_parallel.shape[-1]
        ffn_fc = intermediate_parallel[..., : dim_size // 2]
        gate = intermediate_parallel[..., dim_size // 2 :]
        intermediate_parallel = F.silu(ffn_fc) * gate
        output = self.dense_4h_to_h(intermediate_parallel)
        #####logger.info(f"MLP_output:{output}")
        return output


class GLMBlock(nn.Layer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMv2Config, layer_number,layerwise_recompute: bool = False,kv_caches = None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config)
        #pp添加
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity
        self.config = config
        self.kv_caches = kv_caches

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):   
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )


        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = F.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.

        mlp_output = self.mlp(layernorm_output)


        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = F.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output
        #logger.info(f"GLMBlock")
        #logger.info(f"output:{output}")
        #logger.info(f"kv_cache:{kv_cache}")
        return output, kv_cache


class GLMTransformer(nn.Layer):
    """Transformer class."""

    def __init__(self, config: ChatGLMv2Config):
        super(GLMTransformer, self).__init__()
        self.config = config
        self.enable_recompute = False
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm
        # Number of layers.
        self.num_hidden_layers = config.num_hidden_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number)

        self.layers = nn.LayerList([build_layer(i + 1) for i in range(self.num_hidden_layers)])
        #self.layers = nn.LayerList([build_layer(i + 1) for i in range(1)])
        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        rotary_embeds: paddle.Tensor,
        kv_cache: paddle.Tensor,
        use_cache: bool,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states, kv_cache = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            rotary_embeds,
            kv_cache,
            use_cache,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states, kv_cache

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):  
        ##logger.info(f"transformer :")
        ##logger.info(f"hidden_states :{hidden_states}")
        ##logger.info(f"attention_mask:{attention_mask}")
        ##logger.info(f"rotary_pos_emb:{rotary_pos_emb}")
        ##logger.info(f"kv_caches:{kv_caches}")
        ##logger.info(f"output_hidden_states:{output_hidden_states}")
        ##logger.info(f"use_cache:{use_cache}")
        
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_hidden_layers)]
        presents = () if use_cache else None
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None

        # zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
        # neg_inf = paddle.full_like(attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        # attention_mask = paddle.where(attention_mask, zero, neg_inf)#12.2
        #for index in range(2):
        for index in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

            if self.enable_recompute and not hidden_states.stop_gradient:
                hidden_states, kv_cache = self.recompute_training(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache,
                )
            else:
                hidden_states, kv_cache = layer(
                    hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_caches[index], use_cache=use_cache
                )

            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMv2PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = ChatGLMv2Config
    pretrained_resource_files_map = CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "chatglm_v2"


    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        # Initialize the full attention mask with ones and set the lower triangular mask
        full_attention_mask = paddle.ones((batch_size, seq_length, seq_length), dtype='float32')
        full_attention_mask = paddle.tril(full_attention_mask)
        
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        
        if past_length:
            # Concatenate ones for past attention
            past_attention_mask = paddle.ones((batch_size, seq_length, past_length), dtype='float32')
            full_attention_mask = paddle.concat([past_attention_mask, full_attention_mask], axis=-1)
        
        if padding_mask is not None:
            # Apply padding mask
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1).astype('float32')
        
        if not past_length and padding_mask is not None:
            # Adjust full_attention_mask based on padding mask
            full_attention_mask -= (padding_mask.unsqueeze(-1).astype('float32') - 1)
        
        # Convert to boolean mask (similar to < 0.5 in PyTorch)
        full_attention_mask = (full_attention_mask < 0.5).astype('bool')
        
        # Add an extra dimension to full_attention_mask
        full_attention_mask = full_attention_mask.unsqueeze(1)
        
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
        return position_ids

    @classmethod
    def _get_name_mappings(cls, config: ChatGLMv2Config) -> List[StateDictNameMapping]:
        ########logger.info(f"确实调用我了")
        mappings = [
            "embedding.word_embeddings.weight",
            "rotary_pos_emb.inv_freq",
            "encoder.final_layernorm.weight",
        ]

        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"encoder.layers.{layer_index}.input_layernorm.weight",
                    f"encoder.layers.{layer_index}.input_layernorm.weight",
                ],
                [
                    f"encoder.layers.{layer_index}.self_attention.query_key_value.weight",
                    f"encoder.layers.{layer_index}.self_attention.query_key_value.weight",
                    "transpose",
                ],
                [
                    f"encoder.layers.{layer_index}.self_attention.query_key_value.bias",
                    f"encoder.layers.{layer_index}.self_attention.query_key_value.bias",
                ],
                [
                    f"encoder.layers.{layer_index}.self_attention.dense.weight",
                    f"encoder.layers.{layer_index}.self_attention.dense.weight",
                    "transpose",
                ],
                [
                    f"encoder.layers.{layer_index}.post_attention_layernorm.weight",
                    f"encoder.layers.{layer_index}.post_attention_layernorm.weight",
                ],
                [
                    f"encoder.layers.{layer_index}.mlp.dense_h_to_4h.weight",
                    f"encoder.layers.{layer_index}.mlp.dense_h_to_4h.weight",
                    "transpose",
                ],
                [
                    f"encoder.layers.{layer_index}.mlp.dense_4h_to_h.weight",
                    f"encoder.layers.{layer_index}.mlp.dense_4h_to_h.weight",
                    "transpose",
                ],
            ]
            mappings.extend(layer_mappings)

        init_name_mappings(mappings)

        if config.architectures is not None:
            if "ChatGLMv2ForCausalLM" in config.architectures:
                mappings.extend(
                    [
                        [
                            "output_layer.weight",
                            "output_layer.weight",
                            "transpose",
                        ]
                    ]
                )

        for mapping in mappings:
            mapping[0] = "transformer." + mapping[0]
            if len(mapping) > 1 and mapping[1] is not None:
                mapping[1] = "chatglm_v2." + mapping[1]
        init_name_mappings(mappings)
        ########logger.info(f"cuowu{f1}")
        ########logger.info(f"确实调用我了：{mappings}")
        ########logger.info(f"确实调用我了：{[StateDictNameMapping(*mapping) for mapping in mappings]}")

        f1 = mappings
        return [StateDictNameMapping(*mapping) for mapping in mappings]
    #我写的
    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        def split_or_merge_qkv_weights(tensor_parallel_degree, tensor_parallel_rank, hidden_size, is_split, tensor):
            if is_split:
                return split_qkv_weights(tensor_parallel_degree, tensor_parallel_rank, hidden_size, tensor)
            else:
                assert (
                    len(tensor) == tensor_parallel_degree
                ), "The length of tensor_list must match tensor_parallel_degree"
                return merge_qkv_weights(tensor_parallel_degree, hidden_size, tensor)

        def split_qkv_weights(tensor_parallel_degree, tensor_parallel_rank, hidden_size, tensor):
            split_query_size = hidden_size // tensor_parallel_degree
            split_kv_size = (tensor.shape[-1] - hidden_size) // (2 * tensor_parallel_degree)

            query = tensor[..., :hidden_size]
            key = tensor[..., hidden_size : hidden_size + split_kv_size * tensor_parallel_degree]
            value = tensor[..., tensor.shape[-1] - split_kv_size * tensor_parallel_degree :]

            key_part = key[..., tensor_parallel_rank * split_kv_size : (tensor_parallel_rank + 1) * split_kv_size]
            value_part = value[..., tensor_parallel_rank * split_kv_size : (tensor_parallel_rank + 1) * split_kv_size]
            query_part = query[
                ..., tensor_parallel_rank * split_query_size : (tensor_parallel_rank + 1) * split_query_size
            ]

            return paddle.concat([query_part, key_part, value_part], axis=-1)

        def merge_qkv_weights(tensor_parallel_degree, hidden_size, tensor):
            split_query_size = hidden_size // tensor_parallel_degree
            split_kv_size = (tensor[0].shape[-1] - split_query_size) // 2
            merge_q = tensor[0][..., :split_query_size]
            merge_k = tensor[0][..., split_query_size : split_query_size + split_kv_size]
            merge_v = tensor[0][..., split_query_size + split_kv_size :]
            is_ndarry = isinstance(tensor[0], np.ndarray)
            for i in range(1, tensor_parallel_degree):
                if is_ndarry:
                    merge_q = np.concatenate([merge_q, tensor[i][..., :split_query_size]], axis=-1)
                    merge_k = np.concatenate(
                        [merge_k, tensor[i][..., split_query_size : split_query_size + split_kv_size]], axis=-1
                    )
                    merge_v = np.concatenate([merge_v, tensor[i][..., split_query_size + split_kv_size :]], axis=-1)
                else:
                    merge_q = paddle.concat([merge_q, tensor[i][..., :split_query_size]], axis=-1)
                    merge_k = paddle.concat(
                        [merge_k, tensor[i][..., split_query_size : split_query_size + split_kv_size]], axis=-1
                    )
                    merge_v = paddle.concat([merge_v, tensor[i][..., split_query_size + split_kv_size :]], axis=-1)
            if is_ndarry:
                return np.concatenate([merge_q, merge_k, merge_v], axis=-1)
            else:
                return paddle.concat([merge_q, merge_k, merge_v], axis=-1)

        def split_or_merge_mlp_weights(tensor_parallel_degree, tensor_parallel_rank, is_split, tensor):
            if is_split:
                return split_mlp_weights(tensor_parallel_degree, tensor_parallel_rank, tensor)
            else:
                assert (
                    len(tensor) == tensor_parallel_degree
                ), "The length of tensor_list must match tensor_parallel_degree"
                return merge_mlp_weights(tensor_parallel_degree, tensor)

        def split_mlp_weights(tensor_parallel_degree, tensor_parallel_rank, tensor):
            split_size = tensor.shape[-1] // tensor_parallel_degree // 2
            ffn_fc = tensor[..., : tensor.shape[-1] // 2]
            gate = tensor[..., tensor.shape[-1] // 2 :]
            ffn_fc_part = ffn_fc[..., tensor_parallel_rank * split_size : (tensor_parallel_rank + 1) * split_size]
            gate_part = gate[..., tensor_parallel_rank * split_size : (tensor_parallel_rank + 1) * split_size]
            return paddle.concat([ffn_fc_part, gate_part], axis=-1)

        def merge_mlp_weights(tensor_parallel_degree, tensor):
            split_size = tensor[0].shape[-1] // 2
            merge_ffn_fc = tensor[0][..., :split_size]
            merge_gate = tensor[0][..., split_size:]
            is_ndarry = isinstance(tensor[0], np.ndarray)
            for i in range(1, tensor_parallel_degree):
                if is_ndarry:
                    merge_ffn_fc = np.concatenate([merge_ffn_fc, tensor[i][..., :split_size]], axis=-1)
                    merge_gate = np.concatenate([merge_gate, tensor[i][..., split_size:]], axis=-1)
                else:
                    merge_ffn_fc = paddle.concat([merge_ffn_fc, tensor[i][..., :split_size]], axis=-1)
                    merge_gate = paddle.concat([merge_gate, tensor[i][..., split_size:]], axis=-1)
            if is_ndarry:
                return np.concatenate([merge_ffn_fc, merge_gate], axis=-1)
            else:
                return paddle.concat([merge_ffn_fc, merge_gate], axis=-1)


        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,#这个要注意
        )

        def get_tensor_parallel_split_mappings(num_hidden_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                #"output_layer.weight": partial(fn, is_column=True),
                "encoder.layers.0.mlp.dense_h_to_4h.weight": partial(
                    split_or_merge_mlp_weights, config.tensor_parallel_degree, config.tensor_parallel_rank, is_split
                ),
                "encoder.layers.0.self_attention.query_key_value.bias": partial(
                    split_or_merge_qkv_weights, config.tensor_parallel_degree, config.tensor_parallel_rank, config.hidden_size, is_split
                ),
                "encoder.layers.0.self_attention.query_key_value.weight": partial(
                    split_or_merge_qkv_weights, config.tensor_parallel_degree, config.tensor_parallel_rank, config.hidden_size, is_split
                ),
                # Row Linear
                "embedding.word_embeddings.weight": partial(fn, is_column=False),
                "encoder.layers.0.self_attention.dense.weight": partial(fn, is_column=False),
                "encoder.layers.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),

            }
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)


        return mappings


class Embedding(nn.Layer):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMv2Config):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        if self.config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, 
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
                )

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        # Data format change to avoid explicit tranposes
        # [batch_size, seq_length, hidden_size] --> [seq_length, batch_size, hidden_size].
        embeddings = embeddings.transpose([1, 0, 2]).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.astype("float32")
        #logger.info(f"Embedding_input:{input_ids}")
        #logger.info(f"Embedding_output{embeddings}")
        return embeddings
    
def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()

    if world_size > 1:
        # _c_identity is backwards is reduce
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        # _c_concat has not grad backwards
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits
    
class ChatGLMv2Head(nn.Layer):
    def __init__(self, config, embedding_weights=None):
        super(ChatGLMv2Head, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[config.vocab_size, config.hidden_size], dtype='float16')
            if embedding_weights is None
            else embedding_weights
        )
        self.config = config

    def forward(self, hidden_states):
        # if self.config.tensor_parallel_degree > 1:
        #     logits = parallel_matmul(hidden_states, self.decoder_weight, self.config.tensor_parallel_output)
        # else:
        logits = F.linear(hidden_states, self.decoder_weight.T)
        return logits


@register_base_model
class ChatGLMv2Model(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config, empty_init=True):
        super().__init__(config)
        self.embedding = Embedding(config)
        self.num_layers = config.num_hidden_layers
        # Rotary positional embeddings
        self.max_sequence_length = 2048
        #self.max_sequence_length = config.max_sequence_length
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        if config.use_long_sequence_strategies:
            self.config = config
            self.rotary_pos_emb = LongSequenceStrategies.build_long_sequence_strategy(
                config.long_sequence_strategy_type,
                config.long_sequence_strategy_name,
                **config.long_sequence_init_args,
            )
        else:
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2)
        self.encoder = GLMTransformer(config)
        # if config.tensor_parallel_degree > 1:
        #     self.output_layer = fleet.meta_parallel.ColumnParallelLinear(
        #         config.hidden_size, config.padded_vocab_size, has_bias=False, gather_output=True
        #     )
        # else:
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias_attr=False)

        

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value


    def forward(
        self,
        input_ids,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        full_attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor, paddle.Tensor], ...]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):  
        ########logger.info(f"1_input_ids:{input_ids}")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = input_ids.shape
        ##logger.info(f"Rotary_full_mask{full_attention_mask}")
        ##logger.info(f"Rotary_mask{attention_mask}")
        ##logger.info(f"Rotary_past_key_values{past_key_values}")
        ##logger.info(f"seq_length{seq_length}")
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if full_attention_mask is None:
                    if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                        full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
                        ##logger.info(f"after_Rotary_full_mask{full_attention_mask}")

        # Rotary positional embeddings
        if self.config.use_long_sequence_strategies:
            cos, sin = self.rotary_pos_emb(seq_len=self.max_sequence_length)
            cos, cos = paddle.chunk(cos, 2, axis=-1)
            sin, sin = paddle.chunk(sin, 2, axis=-1)
            rotary_pos_emb = paddle.stack([cos, sin], axis=-1)
        else:
            rotary_pos_emb = self.rotary_pos_emb(self.max_sequence_length)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3]).contiguous()
        #logger.info(f"rotary_pos_emb{rotary_pos_emb}")


        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )


class ChatGLMv2PretrainingCriterion(nn.Layer):
    """
    Criterion for ChatGLMv2. It calculates the final loss.
    """

    def __init__(self, config):
        super(ChatGLMv2PretrainingCriterion, self).__init__()
        self.config = config
        # if config.tensor_parallel_degree > 1 and config.tensor_parallel_output:
        #     self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        # else:
        self.loss_func = paddle.nn.CrossEntropyLoss()
        ##logger.info(f"init ChatGLMv2PretrainingCriterion")

    def forward(self, prediction_scores, masked_lm_labels):
        """
        Args:
            prediction_scores(Tensor):
                The logits of masked token prediction. Its data type should be float32 and
                its shape is [batch_size, sequence_length, vocab_size].
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, the dimensionality of `masked_lm_labels`
                is equal to `prediction_scores`. Its data type should be int64 and
                its shape is [batch_size, sequence_length, 1].
            loss_mask(Tensor):
                Mask used for calculating the loss of the masked language modeling to avoid
                calculating some unwanted tokens.
                Its data type should be float32 and its shape is [batch_size, sequence_length, 1].

        Returns:
            Tensor: The pretraining loss. Its data type should be float32 and its shape is [1].

        """
        with paddle.amp.auto_cast(False):
            ##logger.info(f"forward ChatGLMv2PretrainingCriterion")
            shift_logits = prediction_scores[:, :-1, :].contiguous()
            shift_labels = masked_lm_labels[:, 1:].contiguous()
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            shift_labels = shift_labels.reshape([-1])
            shift_logits = shift_logits[shift_labels != -100]
            shift_labels = shift_labels[shift_labels != -100]
            #logger.info(f"shift_logits{shift_logits}")
            #logger.info(f"shift_labels{shift_labels}")
            loss = self.loss_func(shift_logits, shift_labels)            
            ########logger.info(F"loss{loss}")
        return loss
    
class ChatGLMv2ForCausalLM(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)
        self.max_sequence_length = config.max_sequence_length
        self.chatglm_v2 = ChatGLMv2Model(config)
        # self.lm_head = ChatGLMv2Head(config, self.chatglm_v2.embedding.word_embeddings.weight)
        self.criterion = ChatGLMv2PretrainingCriterion(config)

    def reorder_cache(self, cache: paddle.Tensor, beam_idx):
        cache = map_structure(lambda x: paddle.index_select(x, beam_idx, axis=1), cache)
        return cache

    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs[1] if isinstance(outputs, tuple) else outputs["past_key_values"]

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            new_attention_mask = paddle.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
            model_kwargs["attention_mask"] = paddle.concat([attention_mask, new_attention_mask], axis=-1)

        # update position ids
        if model_kwargs.get("position_ids", None) is not None:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = paddle.concat([position_ids, new_position_id], axis=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        past_key_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        is_first_forward: bool = True,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": True,
        }

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
    ):  
        ##logger.info(f"input_ids :{input_ids}")
        ##logger.info(f"position_ids:{position_ids}")
        ##logger.info(f"attention_mask:{attention_mask}")
        ##logger.info(f"past_key_values:{past_key_values}")
        ##logger.info(f"inputs_embeds:{inputs_embeds}")
        ##logger.info(f"labels:{labels}")
        ##logger.info(f"use_cache:{use_cache}")
        ##logger.info(f"output_hidden_states:{output_hidden_states}")
        ##logger.info(f"return_dict:{return_dict}")
        ##logger.info(f"return_last_logit:{return_last_logit}")
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.chatglm_v2(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.chatglm_v2.output_layer(hidden_states)

        lm_logits = lm_logits.transpose([1, 0, 2]).contiguous().astype("float32")



        loss = None
        if labels is not None:
            loss = self.criterion(lm_logits, labels)
            lm_logits = lm_logits.astype(hidden_states.dtype)
            loss = loss.astype(hidden_states.dtype)
        # lm_logits = lm_logits.astype("float16")
        # loss = loss.astype("float16")
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    