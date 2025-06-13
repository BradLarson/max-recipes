# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Build a GPT2 model that uses continuous or paged kv-caching"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    Module,
    Transformer,
    TransformerBlock,
)
from max.nn.norm import LayerNorm
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)

from .model_config import GPT2Config


class GPT2NoRopeAttention(Module):
    """GPT2-style attention without RoPE (uses learned position embeddings instead)."""
    
    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params,
        dtype: DType,
        devices: Sequence[DeviceRef],
        linear_cls,
        scale: float = 1.0,
        has_bias: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.scale = scale / np.sqrt(self.head_dim)
        self.kv_params = kv_params
        
        # GPT2 uses separate Q, K, V projections (after we split c_attn in weight adapter)
        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.o_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
    
    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # For now, implement a simple version that doesn't use optimized attention
        # This is a placeholder - in practice, we'd need to implement proper
        # attention with KV cache integration
        
        # Simple pass-through for now
        return x


class GPT2(Transformer):
    """GPT2 transformer model implementation."""
    
    def __init__(self, config: GPT2Config):
        assert len(config.devices) == 1, "Multi-GPU support not implemented yet"
        
        # Create layer norm for GPT2 (uses LayerNorm, not RMSNorm)
        create_norm = functools.partial(
            LayerNorm,
            dims=config.hidden_size,
            device=config.devices[0],
            dtype=config.dtype,
            eps=config.layer_norm_epsilon,
            use_bias=True,  # GPT2 uses bias in layer norm
        )
        
        # Linear layer factory
        linear_cls = functools.partial(Linear)
        
        # Create transformer blocks
        layers = [
            TransformerBlock(
                attention=GPT2NoRopeAttention(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_attention_heads,  # GPT2 doesn't use GQA
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    devices=config.devices,
                    linear_cls=linear_cls,
                    scale=1.0,
                    has_bias=True,  # GPT2 uses bias in attention
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                    linear_cls=linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]
        
        # Create embeddings
        token_embedding = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )
        
        # GPT2 has position embeddings
        self.position_embedding = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )
        
        # Output layer
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            config.devices[0],
            has_bias=False,  # GPT2 doesn't use bias in output layer
        )
        
        # GPT2 typically ties input and output embeddings
        output.set_shared_weight("weight", token_embedding.weight)
        
        # Select KV cache strategy
        kv_collection_cls: (
            type[FetchContinuousBatchingKVCacheCollection]
            | type[FetchPagedKVCacheCollection]
        )
        if config.kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(config.kv_params.cache_strategy)
            )
        
        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=token_embedding,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            return_logits=config.return_logits,
        )