# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.fleet.utils import recompute
from paddlenlp.transformers.long_sequence_strategies import LongSequenceStrategies
from paddlenlp.transformers.model_utils import PipelinePretrainedModel

from .modeling import (
    ChatGLMv2Config,
    GLMBlock,
    ChatGLMv2PretrainedModel,
    RotaryEmbedding,
    ChatGLMv2PretrainingCriterion,
)
from paddlenlp.utils.log import logger
from ..model_outputs import BaseModelOutputWithPastAndCrossAttentions
import importlib


def __repr__(self):
    return self.layer_func.__name__


# hack LayerDesc for showing to much config
LayerDesc.__repr__ = __repr__

__all__ = [
    "ChatGLMv2ForCausalLMPipe",
]


def parse_args(args):
    hidden_states = None
    full_attention_mask = None
    rotary_pos_emb = None
    if isinstance(args, tuple):
        if len(args) == 3:
            hidden_states, rotary_pos_emb, full_attention_mask = args
            kv_cache = None
        elif len(args) == 2:
            hidden_states, rotary_pos_emb = args
            full_attention_mask = None
    else:
        hidden_states = args
        full_attention_mask, rotary_pos_emb = None, None

    if full_attention_mask is not None:
        full_attention_mask.stop_gradient = True

    if rotary_pos_emb is not None:
        rotary_pos_emb.stop_gradient = True

    return hidden_states, rotary_pos_emb , full_attention_mask


def return_args(hidden_states, rotary_pos_emb=None, full_attention_mask=None):
    ret = (hidden_states,)


    if rotary_pos_emb is not None:
        ret += (rotary_pos_emb.clone(),)

    if full_attention_mask is not None:
        ret += (full_attention_mask.clone(),)

    if len(ret) == 1:
        ret = ret[0]

    return ret

def parse_args1(args):
    input_ids = None
    attention_mask = None
    position_ids = None
    past_key_values = None

    if isinstance(args, tuple):
        if len(args) == 4:
            input_ids, attention_mask, position_ids, past_key_values = args
        if len(args) == 3:
            input_ids, attention_mask, position_ids = args
            past_key_values = None
        elif len(args) == 2:
            input_ids, attention_mask = args
            position_ids = None
            past_key_values = None
    else:
        input_ids = args
        attention_mask, position_ids, past_key_values = None, None, None

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True


    return input_ids, attention_mask, position_ids, past_key_values


def return_args1(hidden_states, attention_mask=None, position_ids=None, alibi=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if alibi is not None:
        ret += (alibi.clone(),)

    if len(ret) == 1:
        ret = ret[0]

    return ret


class ChatGLMv2EmbeddingPipe(nn.Layer):
    """Extends Embeddings to forward attention_mask through the pipeline."""

    def __init__(self, config: ChatGLMv2Config):
        super(ChatGLMv2EmbeddingPipe, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        if config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)


    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        ##logger.info(f"forward ChatGLMv2EmbeddingPipe")
        ##logger.info(f"args{args}")
        input_ids, attention_mask, position_ids, past_key_values = parse_args1(args)

        hidden_states = self.word_embeddings(input_ids)
        hidden_states = hidden_states.transpose([1, 0, 2]).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            hidden_states = hidden_states.astype("float32")

        #logger.info(f"EmbeddingPipe")
        #logger.info(f"hidden_states{hidden_states}")
        #logger.info(f"input_ids{input_ids}")
        #logger.info(f"attention_mask{attention_mask}")
        #logger.info(f"position_ids{position_ids}")
        #logger.info(f"past_key_values{past_key_values}")
        return (hidden_states,input_ids,attention_mask, position_ids,past_key_values)
    

class LongSequenceStrategiesPipe(nn.Layer):
    """Extends Embeddings to forward attention_mask through the pipeline."""

    def __init__(self, config: ChatGLMv2Config , strategy_type=None, stratety_name=None, **init_args):
        super(LongSequenceStrategiesPipe, self).__init__()
        self.config = config
        self.max_sequence_length = config.max_sequence_length
        all_strategy_types = ["embedding_strategies", "attention_strategies"]
        try:
            import_class = importlib.import_module(f"paddlenlp.transformers.long_sequence_strategies.{strategy_type}")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Wrong strategy type {strategy_type}. module only supports the following types: "
                + ", ".join(m for m in all_strategy_types)
            )
        try:
            strategy_class = getattr(import_class, stratety_name)
        except:
            all_strategy_classes = import_class.__all__
            raise LookupError(
                f"module '{import_class.__name__}' only supports the following classes: "
                + ", ".join(m for m in all_strategy_classes)
            )
        strategy_instance = strategy_class(**init_args)
        self.strategy_instance = strategy_instance


    def forward(self, args):

        # input_ids,embeddings,attention_mask, position_ids, past_key_values = parse_args(args)
        hidden_states,input_ids,attention_mask, position_ids,past_key_values = args
        #full_attention_mask = RotaryEmbeddingPipe.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        full_attention_mask = None
        batch_size, seq_length = input_ids.shape
        # Rotary positional embeddings
        if self.config.use_long_sequence_strategies:
            cos, sin = self.strategy_instance(seq_len=self.max_sequence_length)
            cos, cos = paddle.chunk(cos, 2, axis=-1)
            sin, sin = paddle.chunk(sin, 2, axis=-1)
            rotary_pos_emb = paddle.stack([cos, sin], axis=-1)
        else:
            rotary_pos_emb = self.strategy_instance(self.max_sequence_length)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

        #transformer
        zero = paddle.zeros(full_attention_mask.shape, dtype=hidden_states.dtype)
        neg_inf = paddle.full_like(full_attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        full_attention_mask = paddle.where(full_attention_mask, zero, neg_inf)

        return return_args(hidden_states,rotary_pos_emb,full_attention_mask)
    
class RotaryEmbeddingPipe(RotaryEmbedding):
    """Extends Embeddings to forward attention_mask through the pipeline."""

    def __init__(self, config, dim, original_impl=False):
        super().__init__(dim, original_impl)
        self.config = config
        # h
        self.max_sequence_length = 2048
        #config.max_sequence_length
    
    # def get_masks(self, input_ids, past_key_values, padding_mask=None):
    #     batch_size, seq_length = input_ids.shape

    #     # casual mask
    #     casual_mask = paddle.tril(paddle.ones([batch_size, 1, seq_length, seq_length])).astype("bool")
    #     past_length = 0
    #     if past_key_values:
    #         past_length = past_key_values[0][0].shape[0]
    #     if past_length:
    #         casual_mask = paddle.concat(
    #             [paddle.ones([batch_size, 1, seq_length, past_length], dtype="bool"), casual_mask], axis=-1
    #         )

    #     # seq_mask
    #     if padding_mask is None:
    #         padding_mask = paddle.ones((batch_size, 1, seq_length, seq_length + past_length), dtype="bool")
    #     if len(padding_mask.shape) == 2:
    #         # from Tokenizer
    #         padding_mask = (
    #             padding_mask.unsqueeze(axis=[1, 2])
    #             .expand([batch_size, 1, seq_length, seq_length + past_length])
    #             .astype("bool")
    #         )
    #     elif len(padding_mask.shape) == 3:
    #         # [batch_size,tgt_length, src_length] -> [batch_size, 1, tgt_length, src_length]
    #         padding_mask = padding_mask.unsqueeze(1).astype("bool")
    #     elif len(padding_mask.shape) == 4:
    #         padding_mask = padding_mask.astype("bool")

    #     casual_mask = casual_mask & padding_mask

    #     return casual_mask
    def get_masks(self,valid_length_mask, seq_length, num_heads):
        """
        Generate attention mask based on valid length mask.
        
        Args:
            valid_length_mask (Tensor): Tensor of shape [1, seq_length] indicating valid positions (1) and invalid positions (0).
            seq_length (int): The total sequence length after padding.
            num_heads (int): The number of attention heads.
        
        Returns:
            attention_mask (Tensor): Tensor of shape [1, num_heads, seq_length, seq_length] for attention computation.
        """
        # Expand valid_length_mask to [1, seq_length, seq_length]
        valid_length_mask = valid_length_mask.unsqueeze(1)
        valid_length_mask = valid_length_mask.expand([1, seq_length, seq_length])
        
        # Create attention mask by setting invalid positions to a large negative value
        attention_mask = (1.0 - valid_length_mask) * -1e9
        
        # Expand to [1, num_heads, seq_length, seq_length]
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.expand([1, num_heads, seq_length, seq_length])
        
        return attention_mask

    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """

        # input_ids,embeddings,attention_mask, position_ids, past_key_values = parse_args(args)
        hidden_states,input_ids,attention_mask, position_ids,past_key_values = args
        ##logger.info(f"forward RotaryEmbeddingPipe")
        batch_size, seq_length = input_ids.shape
        full_attention_mask = None
        if full_attention_mask is None:
                    if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                        full_attention_mask = self.get_masks(attention_mask,seq_length,32)
        #full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        
        # Rotary positional embeddings
        if self.config.use_long_sequence_strategies:
            cos, sin = super().forward(seq_len=self.max_sequence_length)
            cos, cos = paddle.chunk(cos, 2, axis=-1)
            sin, sin = paddle.chunk(sin, 2, axis=-1)
            rotary_pos_emb = paddle.stack([cos, sin], axis=-1)
        else:
            rotary_pos_emb = super().forward(self.max_sequence_length)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3]).contiguous()

        #transformer
        # zero = paddle.zeros(full_attention_mask.shape, dtype=hidden_states.dtype)
        # neg_inf = paddle.full_like(full_attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        # full_attention_mask = paddle.where(full_attention_mask, zero, neg_inf)
        #logger.info(f"RotaryEmbeddingPipe")
        #logger.info(f"hidden_states{hidden_states}")
        #logger.info(f"full_attention_mask{full_attention_mask}")
        #logger.info(f"rotary_pos_emb{rotary_pos_emb}")

        return return_args(hidden_states,rotary_pos_emb,full_attention_mask)

        


class GLMBlockPipe(GLMBlock):
    # def __init__(self, config,layer_number,layerwise_recompute: bool = False):
    #     super().__init__(config, layer_number,layerwise_recompute)1
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

    def forward(self, args):
        ##logger.info(f"forward GLMBlockPipe")
        kv_caches = self.kv_caches
        use_cache = self.config.use_cache#重要


        hidden_states,  rotary_pos_emb ,full_attention_mask = parse_args(args)#很多问题
        if not kv_caches:
            kv_caches = [None for _ in range(self.config.num_hidden_layers)]

        ##logger.info(f"self.layer_number:{self.layer_number}")
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)
        if self.enable_recompute and not hidden_states.stop_gradient:
            hidden_states, kv_cache = self.recompute_training(
                super(),
                hidden_states,
                attention_mask=full_attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                kv_cache=kv_caches[self.layer_number],
                use_cache=use_cache,
                #use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states, kv_cache = super().forward(
                hidden_states = hidden_states, attention_mask = full_attention_mask, rotary_pos_emb=rotary_pos_emb,kv_cache=kv_caches[self.layer_number],use_cache=use_cache
            )
        #logger.info(f"GLMBlockPipe:")
        #logger.info(f"hidden_states{hidden_states}")
        #logger.info(f"full_attention_mask{full_attention_mask}")
        #logger.info(f"rotary_pos_emb{rotary_pos_emb}")
        return return_args(hidden_states, rotary_pos_emb , full_attention_mask)


class RMSNormPipe(nn.Layer):
    def __init__(self, config:ChatGLMv2Config,hidden_size, epsilon=None,return_last_logit=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.epsilon = 1e-5 if epsilon is None else epsilon
        self.return_last_logit = return_last_logit
        self.config = config

    def forward(self, args):
        ##logger.info(f"forward RMSNormPipe")
        hidden_states, full_attention_mask, rotary_pos_emb= parse_args(args)
        input_dtype = hidden_states.dtype
        variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
        hidden_states = paddle.rsqrt(variance + self.epsilon) * hidden_states
        outputs = (hidden_states * self.weight).astype(input_dtype)
        #logger.info(f"RMSNormPipe")
        #logger.info(f"outputs{outputs}")
        return outputs
    
class OutputLayerPipe(nn.Linear):

    def forward(self, hidden_states):
        ##logger.info(f"forward OutputLayerPipe")
        lm_logits = super().forward(
                hidden_states
            )
        lm_logits = lm_logits.transpose([1, 0, 2])
        #logger.info(f"OutputLayerPipe")
        #logger.info(f"lm_logits{lm_logits}")
        return lm_logits
    
class TpOutputLayerPipe(fleet.meta_parallel.ColumnParallelLinear):

    def forward(self, hidden_states):

        lm_logits = super().forward(
                hidden_states
            )
        lm_logits = lm_logits.transpose([1, 0, 2])
        return lm_logits


class ChatGLMv2ForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """ChatGLMv2ForPretraining adapted for pipeline parallelism.

    The largest change is flattening the ChatGLMv2Model class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = ChatGLMv2Config

    _get_tensor_parallel_mappings = ChatGLMv2PretrainedModel._get_tensor_parallel_mappings
    _init_weights = ChatGLMv2PretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = ChatGLMv2PretrainedModel._keys_to_ignore_on_load_unexpected

    # DONOT Add base_model_prefix !!!!

    def __init__(self, config:ChatGLMv2Config):
        self.config = config

        self.use_recompute = self.config.use_recompute
        self.recompute_granularity = self.config.recompute_granularity
        self.pp_recompute_interval = self.config.pp_recompute_interval
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        if self.recompute_granularity == "full":
            assert len(self.no_recompute_layers) == 0, "for pp with full recompute, no_recompute_layers is not support"

        virtual_pp_degree = getattr(self.config, "virtual_pp_degree", 1)

        def get_hcg():
            return fleet.get_hybrid_communicate_group()

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        # TODO: fix tensor_parallel_degree rewrite in here
        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(LayerDesc(ChatGLMv2EmbeddingPipe, config=config), "chatglm_v2.embedding")
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        if config.use_long_sequence_strategies:
            self.add_sequential_layer(LayerDesc(LongSequenceStrategiesPipe,
                config.long_sequence_strategy_type,
                config.long_sequence_strategy_name,
                **config.long_sequence_init_args,
            ),"rotary_pos_emb")
        else:
            self.add_sequential_layer(LayerDesc(RotaryEmbeddingPipe, config=config ,dim = rotary_dim // 2), "chatglm_v2.rotary_pos_emb")
            
        for i in range(config.num_hidden_layers):
        #for i in range(2):
            self.add_sequential_layer(
                LayerDesc(GLMBlockPipe, config=config,layer_number=i, layerwise_recompute=i not in self.no_recompute_layers,kv_caches = None),
                f"chatglm_v2.encoder.layers.{i}",
            )
        self.post_layer_norm = config.post_layer_norm
        if self.post_layer_norm:
            LayerNormFunc = RMSNormPipe if config.rmsnorm else nn.LayerNorm
            self.add_sequential_layer(LayerDesc(LayerNormFunc,config=config,hidden_size=config.hidden_size, epsilon=config.layernorm_epsilon), "chatglm_v2.encoder.final_layernorm")
        if config.tensor_parallel_degree > 1:
            self.add_sequential_layer(LayerDesc(TpOutputLayerPipe,in_features=config.hidden_size,out_features=config.padded_vocab_size,has_bias=False), "chatglm_v2.output_layer")
        else:
            self.add_sequential_layer(LayerDesc(OutputLayerPipe,in_features=config.hidden_size,out_features=config.padded_vocab_size,bias_attr=False), "chatglm_v2.output_layer")

        recompute_interval = 0

        seg_method = "layer:GLMBlock"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"
        ## ##logger.info(f"self.get_sequential_layers():{self.get_sequential_layers()}")
        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=ChatGLMv2PretrainingCriterion(config=config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=virtual_pp_degree,
        )

        self.apply(self._init_weights)