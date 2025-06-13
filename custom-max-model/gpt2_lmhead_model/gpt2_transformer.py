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

from __future__ import annotations

import functools
from typing import Callable

from max.dtype import DType
from max.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    LayerNorm,
    Module,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)

from .model_config import GPT2Config


class GPT2Transformer(Transformer):
    """GPT2 implementation using MAX's built-in Transformer infrastructure."""
    
    def __init__(self, config: GPT2Config):
        assert len(config.devices) == 1, "Single device only for now"
        
        # Create rotary embedding (even though GPT2 originally uses learned positions)
        # We'll add learned position embeddings separately
        rope = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            n_heads=config.num_attention_heads,
            theta=10000.0,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0],
        )
        
        # Create layer norm function for GPT2 (uses LayerNorm with bias)
        create_norm: Callable[..., Module] = functools.partial(
            LayerNorm,
            dims=config.hidden_size,
            eps=config.layer_norm_epsilon,
            use_bias=True,
            device=config.devices[0],
            dtype=config.dtype,
        )
        
        # Create linear layer function
        linear_cls: Callable[..., Linear] = functools.partial(
            Linear,
            dtype=config.dtype,
            device=config.devices[0],
        )
        
        # Create MLP function
        mlp_cls = functools.partial(
            MLP,
            float8_config=None,  # No float8 for now
        )
        
        # Create attention function
        attention_cls: Callable[..., AttentionWithRope] = functools.partial(
            AttentionWithRope,
            stacked_qkv=False,
            scale=None,
            clip_qkv=None,
            has_bias=True,  # GPT2 uses bias in attention
            float8_config=None,
        )
        
        # Create transformer layers
        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_attention_heads,  # GPT2 uses same for K,V
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    None,  # quantization_encoding
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),  # pre-attention norm
                mlp_norm=create_norm(),       # pre-mlp norm
                residual_multiplier=1.0,      # Standard residual
            )
            for i in range(config.num_hidden_layers)
        ]
        
        # Create embedding layer (for token embeddings)
        embedding_layer = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )
        
        # Create output layer (language modeling head)
        output_layer = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            has_bias=False,  # GPT2 doesn't use bias in output layer
            dtype=config.dtype,
            device=config.devices[0],
        )
        
        # Create KV cache collection
        kv_collection_cls = (
            FetchContinuousBatchingKVCacheCollection
            if config.kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS
            else FetchPagedKVCacheCollection
        )
        
        # Initialize parent Transformer class
        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),  # Final layer norm
            output=output_layer,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            return_logits=config.return_logits,
        )
        
        # Note: GPT2 position embeddings will be handled in the weight loading
        # by adding them to token embeddings during inference
        
        print(f"DEBUG: Created GPT2Transformer with {config.num_hidden_layers} layers")
    
    def __call__(self, *args, **kwargs):
        """Override to add position embeddings like GPT2."""
        # Get the standard transformer inputs
        input_ids = args[0] if args else kwargs.get('input_ids')
        
        # Add position embeddings
        seq_len = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
        
        # Call parent transformer with position embedding addition
        # Note: We'll need to modify this to properly integrate position embeddings
        # For now, let the parent handle everything and we'll adjust the weights
        return super().__call__(*args, **kwargs)