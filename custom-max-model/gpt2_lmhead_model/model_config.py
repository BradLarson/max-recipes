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
"""Config for GPT2 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    upper_bounded_default,
)
from transformers import AutoConfig


@dataclass
class GPT2Config:
    """Configuration class for GPT2 models.
    
    This handles the translation between Hugging Face's config.json format
    and GPT2 model's internal parameter requirements for MAX graph building.
    """
    
    # Core model parameters
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_position_embeddings: int
    
    # Model-specific parameters
    n_embd: int  # GPT2 uses n_embd instead of hidden_size
    n_head: int  # GPT2 uses n_head instead of num_attention_heads
    n_layer: int  # GPT2 uses n_layer instead of num_hidden_layers
    n_positions: int  # GPT2 uses n_positions instead of max_position_embeddings
    
    # Additional parameters
    intermediate_size: int
    
    # MAX-specific parameters (no defaults)
    dtype: DType
    devices: Sequence[DeviceRef]
    kv_params: KVCacheParams
    
    # Parameters with defaults (must come after non-default parameters)
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    
    @classmethod
    def from_huggingface_config(
        cls,
        hf_config: AutoConfig,
        dtype: DType,
        devices: Sequence[DeviceRef],
        kv_params: KVCacheParams,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> GPT2Config:
        """Create GPT2Config from Hugging Face AutoConfig."""
        # GPT2 uses different naming conventions, handle both
        hidden_size = getattr(hf_config, "n_embd", getattr(hf_config, "hidden_size", 768))
        num_attention_heads = getattr(hf_config, "n_head", getattr(hf_config, "num_attention_heads", 12))
        num_hidden_layers = getattr(hf_config, "n_layer", getattr(hf_config, "num_hidden_layers", 12))
        max_position_embeddings = getattr(hf_config, "n_positions", getattr(hf_config, "max_position_embeddings", 1024))
        
        # Calculate intermediate size (GPT2 uses 4 * hidden_size)
        intermediate_size = getattr(hf_config, "n_inner", 4 * hidden_size) if hasattr(hf_config, "n_inner") else 4 * hidden_size
        
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            n_embd=hidden_size,
            n_head=num_attention_heads,
            n_layer=num_hidden_layers,
            n_positions=max_position_embeddings,
            intermediate_size=intermediate_size,
            activation_function=getattr(hf_config, "activation_function", "gelu_new"),
            layer_norm_epsilon=getattr(hf_config, "layer_norm_epsilon", 1e-5),
            initializer_range=getattr(hf_config, "initializer_range", 0.02),
            scale_attn_weights=getattr(hf_config, "scale_attn_weights", True),
            use_cache=getattr(hf_config, "use_cache", True),
            attn_pdrop=getattr(hf_config, "attn_pdrop", 0.1),
            embd_pdrop=getattr(hf_config, "embd_pdrop", 0.1),
            resid_pdrop=getattr(hf_config, "resid_pdrop", 0.1),
            dtype=dtype,
            devices=devices,
            kv_params=kv_params,
            return_logits=return_logits,
        )
    
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters for GPT2."""
        hidden_size = getattr(huggingface_config, "n_embd", getattr(huggingface_config, "hidden_size", 768))
        num_attention_heads = getattr(huggingface_config, "n_head", getattr(huggingface_config, "num_attention_heads", 12))
        head_dim = hidden_size // num_attention_heads
        
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=num_attention_heads,  # GPT2 doesn't use GQA
            head_dim=head_dim,
            cache_strategy=kv_cache_config.cache_strategy,
            n_devices=n_devices,
        )
    
    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Get number of layers from Hugging Face config."""
        return getattr(huggingface_config, "n_layer", getattr(huggingface_config, "num_hidden_layers", 12))
    
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length."""
        max_position_embeddings = getattr(
            huggingface_config, "n_positions", 
            getattr(huggingface_config, "max_position_embeddings", 1024)
        )
        try:
            return upper_bounded_default(
                upper_bound=max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for GPT2, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings ({max_position_embeddings})."
            )
            raise ValueError(msg) from e