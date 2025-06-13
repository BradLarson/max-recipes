# MAX Model Bringup Guide: A Complete Playbook for Custom Architecture Implementation

*Based on lessons learned from implementing GPT2LMHeadModel for Modular's MAX serving infrastructure*

## Table of Contents

1. [Prerequisites and Environment Setup](#prerequisites)
2. [Phase 1: Architecture Scaffolding](#phase-1)
3. [Phase 2: Weight Mapping Strategy](#phase-2)
4. [Phase 3: Model Implementation](#phase-3)
5. [Phase 4: Testing and Debugging](#phase-4)
6. [Common Pitfalls and Solutions](#pitfalls)
7. [Advanced Troubleshooting](#troubleshooting)
8. [Best Practices and Patterns](#best-practices)

---

## Prerequisites and Environment Setup {#prerequisites}

### Essential References
- **MAX Python API Documentation**: https://docs.modular.com/llms-python.txt
- **Tutorial**: `serve-custom-model-architectures.mdx` in your project
- **Examples**: `modular/max/pipelines/architectures/`

### Development Environment
```bash
# Ensure MAX is available via pixi
pixi run max --version

# Always work within a pixi environment with MAX dependencies
pixi add max  # if not already present
```

### Key Learning: Study Existing Architectures First
**⚠️ Critical Step:** Before writing any code, thoroughly examine 2-3 existing MAX architectures:
```bash
ls modular/max/pipelines/architectures/
# Study: Llama3 (standard transformer), MPNet (simpler), Gemma3 (similar patterns)
```

---

## Phase 1: Architecture Scaffolding {#phase-1}

### 1.1 File Structure Setup

Create the required directory structure:
```
your_model/
├── __init__.py          # Architecture exports
├── arch.py              # SupportedArchitecture registration
├── model.py             # Main PipelineModel implementation
├── model_config.py      # Configuration translation
├── weight_adapters.py   # Weight format conversion
└── simple_model.py      # Neural network implementation
```

### 1.2 Architecture Registration (`arch.py`)

**Key Learning:** The `name` field must **exactly** match the HuggingFace model class name.

```python
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import PipelineTask
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import YourModel

your_model_arch = SupportedArchitecture(
    name="YourModelForCausalLM",  # Must match HF exactly!
    example_repo_ids=[
        "your-org/your-model",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.PAGED],
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=YourModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
)
```

### 1.3 Module Exports (`__init__.py`)

```python
from .arch import your_model_arch

# MAX looks for this variable when loading custom architectures
ARCHITECTURES = [your_model_arch]

__all__ = ["your_model_arch", "ARCHITECTURES"]
```

---

## Phase 2: Weight Mapping Strategy {#phase-2}

### 2.1 Understanding the Challenge

**Key Insight:** Different model architectures use different naming conventions and weight organizations. You must map between:
- HuggingFace model's weight names → MAX's expected names
- HuggingFace config parameters → MAX's internal config format

### 2.2 Weight Name Mapping Pattern

Study your target model's weights first:
```python
# Add this debug code temporarily to understand weight structure
def convert_safetensor_state_dict(state_dict, huggingface_config, pipeline_config, **kwargs):
    print("DEBUG: Analyzing weight structure:")
    for name, weight in state_dict.items():
        print(f"  {name}: {weight.data().shape}")
    
    # Then implement actual conversion...
```

### 2.3 Common Weight Mapping Patterns

#### Standard Transformer Mappings:
```python
COMMON_MAPPINGS = {
    # Remove model prefixes
    "model.": "",
    "transformer.": "",
    
    # Layer indexing
    "layers.": "layers.",  # Usually consistent
    "h.": "layers.",       # GPT-style to standard
    
    # Embeddings
    "embed_tokens.": "embed_tokens.",     # Standard
    "wte.": "embed_tokens.",              # GPT-style
    "embeddings.word_embeddings.": "embed_tokens.",  # BERT-style
    
    # Layer norms
    "ln_1.": "input_layernorm.",          # GPT pre-attention
    "ln_2.": "post_attention_layernorm.", # GPT pre-MLP
    "ln_f.": "norm.",                     # GPT final norm
    "layer_norm.": "input_layernorm.",    # Generic
    
    # Attention components
    "attn.": "self_attn.",
    "attention.": "self_attn.",
    "self_attention.": "self_attn.",
    
    # MLP components
    "mlp.": "mlp.",                       # Usually consistent
    "feed_forward.": "mlp.",
}
```

### 2.4 Special Cases to Handle

#### Combined QKV Projections (GPT-style):
```python
if "c_attn" in weight_name:
    # GPT stores Q, K, V as single combined weight
    # Shape: [hidden_size, 3 * hidden_size] -> 3 x [hidden_size, hidden_size]
    weight = weight_data.data
    hidden_size = config.hidden_size
    
    if weight.shape == (hidden_size, 3 * hidden_size):
        q_weight = weight[:, :hidden_size]
        k_weight = weight[:, hidden_size:2*hidden_size]
        v_weight = weight[:, 2*hidden_size:]
        
        # Create separate weights
        new_weights[f"{base}.q_proj.weight"] = WeightData.from_numpy(
            np.ascontiguousarray(q_weight), f"{base}.q_proj.weight"
        )
        # ... similar for k_proj, v_proj
```

#### MLP Architecture Differences:
```python
# Some models use 2-layer MLP, MAX expects 3-layer gated MLP
if "gate_proj" in weight_name and model_has_2layer_mlp:
    # Duplicate gate weights as up_proj for compatibility
    new_weights[weight_name] = weight_data
    up_proj_name = weight_name.replace("gate_proj", "up_proj")
    new_weights[up_proj_name] = weight_data
```

### 2.5 Weight Adapter Template

```python
def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: YourConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert weights from HuggingFace format to MAX format."""
    
    new_weights: dict[str, WeightData] = {}
    
    # Phase 1: Handle special cases first (e.g., combined projections)
    for name, weights in state_dict.items():
        weight_data = weights.data()
        
        if "special_weight_pattern" in name:
            # Handle special case
            continue
    
    # Phase 2: Apply standard mappings
    for name, weights in state_dict.items():
        if name in processed_special_cases:
            continue
            
        max_name = name
        for pattern, replacement in WEIGHT_MAPPINGS.items():
            max_name = max_name.replace(pattern, replacement)
        
        new_weights[max_name] = weights.data()
    
    # Phase 3: Add tied embeddings if needed
    if "lm_head.weight" not in new_weights and "embed_tokens.weight" in new_weights:
        new_weights["lm_head.weight"] = new_weights["embed_tokens.weight"]
    
    return new_weights
```

---

## Phase 3: Model Implementation {#phase-3}

### 3.1 Configuration Translation (`model_config.py`)

**Key Learning:** Different models use different parameter names. Create a translation layer:

```python
@dataclass
class YourModelConfig:
    # Standard MAX parameters
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_position_embeddings: int
    
    # Model-specific parameters
    intermediate_size: int = 11008
    layer_norm_epsilon: float = 1e-6
    
    @classmethod
    def from_huggingface_config(cls, hf_config: AutoConfig) -> "YourModelConfig":
        return cls(
            # Handle different naming conventions
            vocab_size=hf_config.vocab_size,
            hidden_size=getattr(hf_config, 'hidden_size', getattr(hf_config, 'n_embd', 768)),
            num_attention_heads=getattr(hf_config, 'num_attention_heads', getattr(hf_config, 'n_head', 12)),
            num_hidden_layers=getattr(hf_config, 'num_hidden_layers', getattr(hf_config, 'n_layer', 6)),
            max_position_embeddings=getattr(hf_config, 'max_position_embeddings', getattr(hf_config, 'n_positions', 1024)),
            # Add other mappings as needed
        )
```

### 3.2 Neural Network Implementation

**Strategy: Start Minimal, Then Expand**

#### Phase 3.2.1: Minimal Model (for testing)
```python
class YourModelSimple(Module):
    """Minimal model for testing weight loading."""
    
    def __init__(self, config: YourModelConfig):
        super().__init__()
        device, dtype = config.devices[0], config.dtype
        
        # Just embeddings and output layer
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            device=device,
            dtype=dtype,
        )
        
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            has_bias=False,
            device=device,
            dtype=dtype,
        )
    
    def __call__(self, *args, **kwargs):
        input_ids = args[0]
        embeddings = self.embed_tokens(input_ids)
        logits = self.lm_head(embeddings)
        return logits
```

#### Phase 3.2.2: Full Model Implementation
Once basic loading works, expand to full architecture:

```python
# Use MAX's built-in components when possible
from max.nn import (
    TransformerModel,  # or Module for custom
    TransformerBlock,
    AttentionWithRope,
    MLP,
    LayerNorm,
    Embedding,
    Linear,
)

class YourModelFull(TransformerModel):
    def __init__(self, config: YourModelConfig):
        # Build layers using MAX components
        # Follow patterns from existing architectures
        pass
```

### 3.3 Critical Component Parameters

**⚠️ Common Error:** Using wrong parameter names for MAX components.

#### Correct Parameter Names:
```python
# Embedding
Embedding(
    vocab_size=config.vocab_size,    # NOT num_embeddings
    hidden_dim=config.hidden_size,   # NOT embedding_dim
    device=device,
    dtype=dtype,
)

# Linear
Linear(
    in_dim=input_size,     # NOT input_dims or in_features
    out_dim=output_size,   # NOT output_dims or out_features
    has_bias=True,         # NOT bias
    device=device,
    dtype=dtype,
)

# LayerNorm
LayerNorm(
    dims=hidden_size,      # NOT normalized_shape
    eps=1e-6,
    use_bias=True,         # Model-dependent
    device=device,
    dtype=dtype,
)
```

---

## Phase 4: Testing and Debugging {#phase-4}

### 4.1 Incremental Testing Strategy

**Step 1: Test Architecture Registration**
```bash
pixi run max generate --model-path your-model --custom-architectures your_module --max-new-tokens 1
# Should find and load your architecture
```

**Step 2: Test Weight Loading**
Add debug prints in `weight_adapters.py`:
```python
print(f"DEBUG: Processing {weight_name} with shape {weight.shape}")
print(f"DEBUG: Mapped to {max_name}")
```

**Step 3: Test Model Creation**
Add debug prints in your model `__init__`:
```python
print(f"DEBUG: Creating model with {config.num_hidden_layers} layers")
```

**Step 4: Test Graph Building**
Monitor for successful compilation:
```
[INFO] Building and compiling YourModel model...
```

### 4.2 Common Testing Commands

```bash
# Basic architecture test
pixi run max generate --model-path your-org/your-model --custom-architectures your_module --max-new-tokens 1

# Serve for debugging
pixi run max serve --model-path your-org/your-model --custom-architectures your_module

# Test with different encodings
pixi run max generate --model-path your-org/your-model --custom-architectures your_module --quantization-encoding bfloat16
```

---

## Common Pitfalls and Solutions {#pitfalls}

### 5.1 Weight Loading Issues

#### Problem: "Unsupported dimension type None"
```
TypeError: Unsupported dimension type None (<class 'NoneType'>)
```

**Root Cause:** WeightData objects with invalid shapes.

**Solution:**
1. Ensure `np.ascontiguousarray()` on all numpy arrays
2. Check that split operations produce valid shapes
3. Verify no `None` values in tensor dimensions

```python
# Bad
weight_split = weight[:, :hidden_size]  # May not be contiguous

# Good
weight_split = np.ascontiguousarray(weight[:, :hidden_size])

# Add debugging
if None in weight_split.shape:
    raise ValueError(f"Invalid shape: {weight_split.shape}")
```

#### Problem: Weight Shape Mismatches
**Symptoms:** Model loads but crashes during state dict loading.

**Solution:** Debug weight expectations vs. reality:
```python
# Add to weight adapter
try:
    new_weights[max_name] = weight_data
    print(f"SUCCESS: {max_name} -> {weight_data.shape}")
except Exception as e:
    print(f"ERROR: {max_name} -> {e}")
    # Fall back or skip problematic weights
```

### 5.2 Model Architecture Issues

#### Problem: Parameter Name Errors
```
TypeError: YourLayer.__init__() got an unexpected keyword argument 'wrong_param'
```

**Solution:** Always verify parameter names by checking existing implementations:
```bash
grep -r "Embedding(" /path/to/modular/max/pipelines/architectures/
grep -r "Linear(" /path/to/modular/max/pipelines/architectures/
```

#### Problem: Call Signature Mismatches
```
TypeError: YourModel.__call__() takes 2 positional arguments but 5 were given
```

**Solution:** Use flexible signatures:
```python
def __call__(self, *args, **kwargs):
    input_ids = args[0]  # Extract what you need
    # Handle variable arguments gracefully
```

### 5.3 Graph Building Issues

#### Problem: Graph Output Type Errors
```
TypeError: 'TensorValue' object is not iterable
```

**Solution:** Check if output is single tensor or tuple:
```python
# Bad
graph.output(*outputs)  # When outputs is single tensor

# Good
if isinstance(outputs, (list, tuple)):
    graph.output(*outputs)
else:
    graph.output(outputs)
```

### 5.4 Architecture Registration Issues

#### Problem: Model Not Found
```
Architecture 'YourModel' not supported
```

**Solutions:**
1. Verify `name` in `SupportedArchitecture` exactly matches HuggingFace model class
2. Check `ARCHITECTURES` list in `__init__.py`
3. Ensure module is properly importable

---

## Advanced Troubleshooting {#troubleshooting}

### 6.1 Debugging Weight Mappings

Create a weight mapping visualization:
```python
def debug_weight_mapping(state_dict, converted_dict):
    print("=== WEIGHT MAPPING DEBUG ===")
    print("HuggingFace -> MAX mappings:")
    
    hf_keys = set(state_dict.keys())
    max_keys = set(converted_dict.keys())
    
    print(f"HF weights: {len(hf_keys)}")
    print(f"MAX weights: {len(max_keys)}")
    
    # Show mappings
    for hf_key in sorted(hf_keys):
        print(f"  {hf_key} -> [converted to MAX format]")
    
    print("\nFinal MAX weights:")
    for max_key in sorted(max_keys):
        print(f"  {max_key}")
```

### 6.2 Batch Dimension Issues

If you encounter batch dimension mismatches in sampling:
```python
# In your model's forward pass, ensure consistent batch dimensions
def __call__(self, *args, **kwargs):
    input_ids = args[0]  # Shape: [batch_size, seq_len]
    
    # Ensure outputs maintain batch dimension
    embeddings = self.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]
    logits = self.lm_head(embeddings)          # [batch_size, seq_len, vocab_size]
    
    return logits  # Maintain [batch_size, seq_len, vocab_size]
```

### 6.3 Memory and Performance Issues

Monitor memory usage during development:
```bash
# Watch memory consumption
watch -n 1 'nvidia-smi | grep -A 5 "GPU Memory Usage"'

# Profile with smaller models first
pixi run max generate --model-path distilbert/distilgpt2 --custom-architectures your_module
```

---

## Best Practices and Patterns {#best-practices}

### 7.1 Development Workflow

1. **Start Simple:** Implement minimal model with just embeddings + output layer
2. **Test Early:** Verify each component works before adding complexity
3. **Debug Incrementally:** Add one layer type at a time
4. **Use Existing Patterns:** Copy successful patterns from similar models
5. **Document Issues:** Keep notes on solutions for future reference

### 7.2 Code Organization

```python
# Separate concerns clearly
class YourModel(PipelineModel):
    """Main pipeline integration."""
    pass

class YourModelImpl(Module):
    """Neural network implementation."""
    pass

class YourModelConfig:
    """Configuration translation."""
    pass

def convert_weights():
    """Weight format conversion."""
    pass
```

### 7.3 Error Handling

```python
def robust_weight_conversion(state_dict):
    converted = {}
    errors = []
    
    for name, weight in state_dict.items():
        try:
            max_name = convert_weight_name(name)
            converted[max_name] = weight.data()
        except Exception as e:
            errors.append(f"Failed to convert {name}: {e}")
            # Continue with other weights
    
    if errors:
        print("Weight conversion warnings:")
        for error in errors:
            print(f"  {error}")
    
    return converted
```

### 7.4 Testing Checklist

- [ ] Architecture loads without import errors
- [ ] HuggingFace model downloads successfully
- [ ] Weight conversion completes without crashes
- [ ] Model creation succeeds
- [ ] Graph compilation passes
- [ ] Basic generation produces output (even if nonsensical)
- [ ] Memory usage is reasonable

### 7.5 Performance Optimization

Once basic functionality works:
1. **Profile bottlenecks:** Use MAX's profiling tools
2. **Optimize weight formats:** Test different quantization levels
3. **Tune batch sizes:** Find optimal batch size for your hardware
4. **Cache optimizations:** Ensure KV cache strategy is appropriate

---

## Conclusion

Model bringup is an iterative process. Start with the minimal viable implementation and gradually add complexity. The key is systematic debugging and leveraging MAX's existing patterns rather than reinventing the wheel.

**Remember:** Most issues are configuration or naming problems, not fundamental algorithmic issues. When in doubt, study working examples and copy their patterns.

**Success Metrics:**
- Model loads and compiles
- Generates coherent text (quality comes later)
- Memory usage is reasonable
- No crashes during inference

Good luck with your model implementation!
