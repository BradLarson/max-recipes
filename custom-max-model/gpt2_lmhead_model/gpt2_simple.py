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

from max.nn import Module, Embedding, Linear
from max.graph import ops

from .model_config import GPT2Config


class GPT2Simple(Module):
    """Minimal GPT2 model for testing weight loading."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        device = config.devices[0]
        dtype = config.dtype
        
        print(f"DEBUG: Creating minimal GPT2Simple with device={device}, dtype={dtype}")
        
        # Just create the bare minimum components to test weight loading
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            device=device,
            dtype=dtype,
        )
        
        # Create a simple linear layer for language modeling head
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            has_bias=False,
            device=device,
            dtype=dtype,
        )
        
        print("DEBUG: Minimal GPT2Simple created successfully")
    
    def __call__(
        self,
        tokens,  # TensorValueLike
        kv_cache_inputs,  # Sequence[TensorValue] - not used in minimal model
        return_n_logits,  # TensorValue
        input_row_offsets,  # TensorValue
    ):
        """Forward pass following MAX's standard Transformer signature."""
        print(f"DEBUG: GPT2Simple.__call__ - tokens shape: {tokens.shape}")
        print(f"DEBUG: GPT2Simple.__call__ - return_n_logits shape: {return_n_logits.shape}")
        print(f"DEBUG: GPT2Simple.__call__ - input_row_offsets shape: {input_row_offsets.shape}")
        
        # For minimal model, we just use tokens and ignore KV cache
        embeddings = self.embed_tokens(tokens)
        print(f"DEBUG: embeddings shape: {embeddings.shape}")
        
        logits = self.lm_head(embeddings)
        print(f"DEBUG: logits shape before rebind: {logits.shape}")
        
        # Explicitly rebind to ensure consistent symbolic dimensions
        # Use "total_seq_len" to match MAX's standard pattern for ragged sequences
        logits = ops.rebind(logits, shape=("total_seq_len", "vocab_size"))
        print(f"DEBUG: logits shape after rebind: {logits.shape}")
        
        # Return tuple as expected by MAX pipeline
        return (logits,)