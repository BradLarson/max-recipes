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

from max.nn import (
    Module,
    Embedding,
    Linear,
    LayerNorm,
    AttentionWithRope,
    MLP,
    RotaryEmbedding,
)
from max.graph import ops
# Remove unused import

from .model_config import GPT2Config


class GPT2Block(Module):
    """Single GPT2 transformer block with LayerNorm and attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        kv_params,
        device,
        dtype,
    ):
        super().__init__()
        
        # Pre-attention layer norm (GPT2 uses LayerNorm before attention)
        self.input_layernorm = LayerNorm(
            dims=hidden_size,
            eps=layer_norm_eps,
            use_bias=True,
            device=device,
            dtype=dtype,
        )
        
        # Create a dummy rope for GPT2 (which doesn't actually use RoPE)
        rope = RotaryEmbedding(
            dim=hidden_size // num_heads,
            n_heads=num_heads,
            theta=10000.0,
            max_seq_len=1024,
            device=device,
        )
        
        # Multi-head self-attention
        self.self_attn = AttentionWithRope(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # GPT2 uses same for K and V
            hidden_size=hidden_size,
            kv_params=kv_params,
            has_bias=True,  # GPT2 uses bias in attention
            devices=[device],  # Convert single device to list
            dtype=dtype,
            rope=rope,
        )
        
        # Pre-MLP layer norm
        self.post_attention_layernorm = LayerNorm(
            dims=hidden_size,
            eps=layer_norm_eps,
            use_bias=True,
            device=device,
            dtype=dtype,
        )
        
        # MLP (Feed Forward Network)
        self.mlp = MLP(
            dtype,
            None,  # quantization_encoding
            hidden_size,
            intermediate_size,
            [device],  # devices as list
            Linear,  # linear_cls
        )
    
    def __call__(self, hidden_states, kv_cache=None):
        """Forward pass through the transformer block."""
        # Pre-norm attention
        normed = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(normed, kv_cache)
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Pre-norm MLP
        normed = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states


class GPT2Full(Module):
    """Full GPT2 implementation using MAX components."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        device, dtype = config.devices[0], config.dtype
        
        print(f"DEBUG: Creating full GPT2 with {config.num_hidden_layers} layers")
        
        # Token embeddings
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            device=device,
            dtype=dtype,
        )
        
        # Position embeddings (GPT2 uses learned position embeddings)
        self.embed_positions = Embedding(
            vocab_size=config.max_position_embeddings,
            hidden_dim=config.hidden_size,
            device=device,
            dtype=dtype,
        )
        
        # Transformer blocks
        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            layer = GPT2Block(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                layer_norm_eps=config.layer_norm_epsilon,
                kv_params=config.kv_params,
                device=device,
                dtype=dtype,
            )
            self.layers.append(layer)
            # Register each layer as an attribute
            setattr(self, f"layers_{layer_idx}", layer)
        
        # Final layer norm
        self.norm = LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_epsilon,
            use_bias=True,
            device=device,
            dtype=dtype,
        )
        
        # Language modeling head
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            has_bias=False,  # GPT2 doesn't use bias in the output layer
            device=device,
            dtype=dtype,
        )
        
        print("DEBUG: Full GPT2 created successfully")
    
    def __call__(
        self,
        tokens,  # TensorValueLike
        kv_cache_inputs,  # Sequence[TensorValue]
        return_n_logits,  # TensorValue
        input_row_offsets,  # TensorValue
    ):
        """Forward pass following MAX's standard Transformer signature."""
        print(f"DEBUG: GPT2Full.__call__ - tokens shape: {tokens.shape}")
        
        # Get sequence length for position embeddings
        seq_len = tokens.shape[0]  # total_seq_len
        
        # Create position indices (0, 1, 2, ..., seq_len-1)
        position_ids = ops.range(0, seq_len, 1, dtype=tokens.dtype)
        
        # Token embeddings
        token_embeddings = self.embed_tokens(tokens)
        print(f"DEBUG: token_embeddings shape: {token_embeddings.shape}")
        
        # Position embeddings
        position_embeddings = self.embed_positions(position_ids)
        print(f"DEBUG: position_embeddings shape: {position_embeddings.shape}")
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        print(f"DEBUG: combined embeddings shape: {hidden_states.shape}")
        
        # Pass through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            print(f"DEBUG: Processing layer {layer_idx}")
            # For now, we'll ignore KV cache in this implementation
            hidden_states = layer(hidden_states, kv_cache=None)
            print(f"DEBUG: Layer {layer_idx} output shape: {hidden_states.shape}")
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        print(f"DEBUG: After final norm shape: {hidden_states.shape}")
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        print(f"DEBUG: Final logits shape before rebind: {logits.shape}")
        
        # Explicitly rebind to ensure consistent symbolic dimensions
        logits = ops.rebind(logits, shape=("total_seq_len", "vocab_size"))
        print(f"DEBUG: Final logits shape after rebind: {logits.shape}")
        
        # Return tuple as expected by MAX pipeline
        return (logits,)