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

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import GPT2Config


# Maps from GPT2 SafeTensor to MAX weight names
GPT2_SAFETENSOR_MAPPING = {
    "transformer.": "",  # Remove transformer prefix like Llama3 removes "model."
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: GPT2Config,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert GPT2 SafeTensors weights following Llama3's exact pattern.
    
    Use Llama3's approach: keep weights as WeightData, just rename keys.
    Handle GPT2's specific quirks (combined QKV weights).
    """
    
    new_state_dict: dict[str, WeightData] = {}
    hidden_size = huggingface_config.n_embd
    
    print("DEBUG: Starting GPT2 weight conversion (Llama3 pattern)")
    print(f"DEBUG: Hidden size: {hidden_size}, Layers: {huggingface_config.n_layer}")
    
    # First pass: Handle GPT2's combined QKV weights
    processed_keys = set()
    
    for safetensor_name, weights in state_dict.items():
        # Handle combined QKV weights first
        if "attn.c_attn" in safetensor_name:
            print(f"DEBUG: Processing combined QKV: {safetensor_name}")
            
            # Extract layer number and convert base name
            base_name = safetensor_name.replace("transformer.", "")
            layer_part = base_name.split(".attn.c_attn")[0]  # e.g., "h.0"
            
            if "weight" in safetensor_name:
                # Split weight: [768, 2304] -> 3 x [768, 768]
                weight_data = weights.data()
                weight = weight_data.data
                if weight.shape == (hidden_size, 3 * hidden_size):
                    q_weight = weight[:, :hidden_size]
                    k_weight = weight[:, hidden_size:2*hidden_size]
                    v_weight = weight[:, 2*hidden_size:]
                    
                    # Use standard attention naming: layers.{i}.self_attn.{q,k,v}_proj.weight
                    layer_idx = layer_part.replace("h.", "")
                    print(f"DEBUG: QKV split shapes: q={q_weight.shape}, k={k_weight.shape}, v={v_weight.shape}")
                    
                    q_data = WeightData.from_numpy(
                        np.ascontiguousarray(q_weight), f"layers.{layer_idx}.self_attn.q_proj.weight"
                    )
                    k_data = WeightData.from_numpy(
                        np.ascontiguousarray(k_weight), f"layers.{layer_idx}.self_attn.k_proj.weight"
                    )
                    v_data = WeightData.from_numpy(
                        np.ascontiguousarray(v_weight), f"layers.{layer_idx}.self_attn.v_proj.weight"
                    )
                    
                    print(f"DEBUG: QKV WeightData created successfully for layer {layer_idx}")
                    new_state_dict[f"layers.{layer_idx}.self_attn.q_proj.weight"] = q_data
                    new_state_dict[f"layers.{layer_idx}.self_attn.k_proj.weight"] = k_data
                    new_state_dict[f"layers.{layer_idx}.self_attn.v_proj.weight"] = v_data
                    
            elif "bias" in safetensor_name:
                # Split bias: [2304] -> 3 x [768]
                weight_data = weights.data()
                weight = weight_data.data
                if weight.shape == (3 * hidden_size,):
                    q_bias = weight[:hidden_size]
                    k_bias = weight[hidden_size:2*hidden_size]
                    v_bias = weight[2*hidden_size:]
                    
                    layer_idx = layer_part.replace("h.", "")
                    new_state_dict[f"layers.{layer_idx}.self_attn.q_proj.bias"] = WeightData.from_numpy(
                        np.ascontiguousarray(q_bias), f"layers.{layer_idx}.self_attn.q_proj.bias"
                    )
                    new_state_dict[f"layers.{layer_idx}.self_attn.k_proj.bias"] = WeightData.from_numpy(
                        np.ascontiguousarray(k_bias), f"layers.{layer_idx}.self_attn.k_proj.bias"
                    )
                    new_state_dict[f"layers.{layer_idx}.self_attn.v_proj.bias"] = WeightData.from_numpy(
                        np.ascontiguousarray(v_bias), f"layers.{layer_idx}.self_attn.v_proj.bias"
                    )
            
            processed_keys.add(safetensor_name)
            continue
    
    # Second pass: Handle all other weights with simple mapping (EXACT Llama3 pattern)
    for safetensor_name, weights in state_dict.items():
        if safetensor_name in processed_keys:
            continue
            
        # Apply basic GPT2 -> MAX mapping
        max_name = safetensor_name
        for before, after in GPT2_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        
        # Skip GPT2's attention bias (causal mask) - not needed for MAX Transformer
        if "attn.bias" in max_name:
            print(f"DEBUG: Skipping attention bias: {safetensor_name}")
            continue
            
        # Skip GPT2 position embeddings for now (will handle differently later)  
        if "wpe." in max_name:
            print(f"DEBUG: Skipping position embeddings: {safetensor_name}")
            continue
            
        # Convert GPT2 specific names to TransformerBlock naming
        max_name = max_name.replace("h.", "layers.")  # layer indexing
        max_name = max_name.replace("wte.", "embed_tokens.")  # token embeddings
        max_name = max_name.replace("ln_f.", "norm.")  # final norm
        max_name = max_name.replace("ln_1.", "input_layernorm.")  # attention norm (TransformerBlock uses this)
        max_name = max_name.replace("ln_2.", "post_attention_layernorm.")  # mlp norm (TransformerBlock uses this)
        max_name = max_name.replace("attn.c_proj.", "self_attn.o_proj.")  # attention output
        max_name = max_name.replace("mlp.c_fc.", "mlp.gate_proj.")  # mlp input 
        max_name = max_name.replace("mlp.c_proj.", "mlp.down_proj.")  # mlp output
        
        # Store the weight using exact Llama3 pattern with shape preservation
        weight_data = weights.data()
        print(f"DEBUG: Weight {safetensor_name} has shape {weight_data.data.shape if hasattr(weight_data, 'data') else 'unknown'}")
        new_state_dict[max_name] = weight_data
        print(f"DEBUG: Mapped {safetensor_name} -> {max_name}")
        
        # Handle MLP duplication for gated architecture  
        if "mlp.gate_proj" in max_name:
            # GPT2 has 2-layer MLP but MAX expects 3-layer gated MLP
            # Duplicate gate_proj as up_proj for compatibility
            up_proj_name = max_name.replace("gate_proj", "up_proj")
            new_state_dict[up_proj_name] = weight_data  # Reuse same weight data
            print(f"DEBUG: Duplicated {max_name} as {up_proj_name}")
    
    # Add tied embeddings if needed
    if "lm_head.weight" not in new_state_dict and "embed_tokens.weight" in new_state_dict:
        new_state_dict["lm_head.weight"] = new_state_dict["embed_tokens.weight"]
        print("DEBUG: Added tied embeddings for lm_head")
    
    print(f"DEBUG: Final weight count: {len(new_state_dict)}")
    return new_state_dict