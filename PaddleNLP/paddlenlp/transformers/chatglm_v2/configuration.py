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

from ..configuration_utils import PretrainedConfig

CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "THUDM/chatglm2-6b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/chatglm2-6b/model_state.pdparams",
    }
}


class ChatGLMv2Config(PretrainedConfig):
    model_type = "chatglm_v2"
    attribute_map = {
        "num_layers": "num_hidden_layers",
        "padded_vocab_size": "vocab_size",
        "seq_length": "max_sequence_length",
    }

    def __init__(
        self,
        num_hidden_layers=28,
        vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        max_sequence_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        use_cache=False,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        interleaved_qkv=False,
        bias_dropout_fusion=True,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        eos_token_id=2,
        pad_token_id=0,
        use_flash_attention=False,
        long_sequence_strategy_type=None,
        long_sequence_strategy_name=None,
        long_sequence_init_args=None,
        use_long_sequence_strategies=False,
        use_recompute=False,
        recompute_granularity="full",
        pp_recompute_interval=1,
        no_recompute_layers=None,
        fuse_attention_qkv=False,
        fuse_attention_ffn=False,
        use_fused_rms_norm=False,
        use_fused_rope=False,
        alibi=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.use_cache = use_cache
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.use_flash_attention = use_flash_attention
        self.long_sequence_strategy_type = long_sequence_strategy_type
        self.long_sequence_strategy_name = long_sequence_strategy_name
        self.long_sequence_init_args = {} if long_sequence_init_args is None else long_sequence_init_args
        self.use_long_sequence_strategies = use_long_sequence_strategies
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.pp_recompute_interval = pp_recompute_interval
        self.no_recompute_layers= no_recompute_layers
        self.fuse_attention_qkv = fuse_attention_qkv

        self.fuse_attention_ffn = fuse_attention_ffn
        self.use_fused_rms_norm = use_fused_rms_norm
        self.alibi = alibi
        self.dtype = "float32"

        self.use_fused_rope = use_fused_rope