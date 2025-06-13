# GPT2LMHeadModel Custom Architecture for MAX

## Project Structure

```
new_model_architecture2/
├── gpt2_lmhead_model/          # Custom architecture implementation
│   ├── __init__.py             # Architecture exports
│   ├── arch.py                 # SupportedArchitecture registration
│   ├── model.py                # Main PipelineModel implementation
│   ├── model_config.py         # GPT2 configuration handling
│   ├── weight_adapters.py      # Weight format conversion
│   └── gpt2_simple.py         # Simplified GPT2 model implementation
├── serve-custom-model-architectures.mdx  # Tutorial reference
├── PLAN.txt                    # Original implementation plan
└── CLAUDE.md                   # This file
```

## Key API References

### MAX Python APIs
- **Documentation Index**: https://docs.modular.com/llms-python.txt
- **Graph API**: Used for building neural network computation graphs
- **nn module**: Contains layers like `EmbeddingV1`, `LinearV1`, `LayerNormV1`
- **PipelineModel**: Base class for all custom model implementations

### Important Classes and Functions

1. **SupportedArchitecture** (`max.pipelines.lib.registry`)
   - Registers model with MAX
   - Specifies encodings, capabilities, and weight adapters

2. **PipelineModel** (`max.pipelines.lib`)
   - Base class for model implementation
   - Requires `from_huggingface()` classmethod
   - Must implement `build_graph()` method

3. **WeightData** (`max.graph.weights`)
   - Wrapper for model weights
   - Created with `weight_adapter.adapt(numpy_array)`

4. **Graph** (`max.graph`)
   - Context manager for building computation graphs
   - Define inputs with `TensorType`
   - Build layers and operations within context

## GPT2 Specific Mappings

### Configuration Parameter Mapping
GPT2 uses different naming conventions than standard transformers:
- `n_embd` → `hidden_size`
- `n_head` → `num_attention_heads`
- `n_layer` → `num_hidden_layers`
- `n_positions` → `max_position_embeddings`

### Weight Name Mappings
Key transformations required:
- `h.{i}.attn.c_attn` → Split into `q_proj`, `k_proj`, `v_proj`
- `h.{i}.attn.c_proj` → `o_proj`
- `h.{i}.mlp.c_fc` → `mlp.gate_proj`
- `h.{i}.mlp.c_proj` → `mlp.down_proj`
- `h.{i}.ln_1` → `input_layernorm`
- `h.{i}.ln_2` → `post_attention_layernorm`
- `ln_f` → `model_norm`

### Critical Implementation Details

1. **Weight Splitting for Attention**:
   - GPT2 combines Q, K, V projections in `c_attn` with shape `[768, 2304]`
   - Must split into 3 separate projections of `[768, 768]` each
   - Same applies to biases: `[2304]` → 3×`[768]`

2. **Layer Differences from Llama**:
   - GPT2 uses LayerNorm, not RMSNorm
   - GPT2 has bias in attention and MLP layers
   - GPT2 uses learned position embeddings, not RoPE

3. **Weight Data Creation**:
   - Use `np.ascontiguousarray()` to ensure memory layout
   - Proper shape handling is critical - avoid None dimensions

## Commands

### Serve the Model
```bash
pixi run max serve --model-path distilbert/distilgpt2 --custom-architectures gpt2_lmhead_model
```

### Test the Model (once serving)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "distilbert/distilgpt2",
    "messages": [
      {"role": "user", "content": "Hello! Can you help me with a simple task?"}
    ],
    "max_tokens": 100
  }'
```

## Current Status

- ✅ Architecture registration complete
- ✅ Weight adapter handles GPT2's combined attention weights
- ✅ Configuration properly maps GPT2 parameters
- ✅ Model builds using simplified GPT2 architecture
- ⚠️  Weight loading has shape metadata issue to resolve
- ⏳ Server startup pending weight loading fix

## Debugging Tips

1. **Weight Issues**: Add print statements in `weight_adapters.py` to verify shapes
2. **Architecture Matching**: Model name in `arch.py` must exactly match HuggingFace's model class
3. **Import Errors**: Ensure all MAX modules are properly imported
4. **Shape Errors**: Check that all tensor operations maintain compatible dimensions

## References

- Example architectures: `../modular/max/pipelines/architectures/`
- Llama implementation: Good reference for transformer patterns
- MPNet implementation: Simpler example for understanding basics
