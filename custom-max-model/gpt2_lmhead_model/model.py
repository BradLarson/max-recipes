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

import logging
import time
from collections.abc import Sequence
from typing import Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn import Module, ReturnLogits
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.profiler import traced
from transformers import AutoConfig

from .gpt2_transformer import GPT2Transformer
from .model_config import GPT2Config

logger = logging.getLogger("max.pipelines")


class GPT2Inputs(ModelInputs):
    """A class representing inputs for the GPT2 model."""
    
    tokens: Tensor
    """Tensor containing the input token IDs."""
    
    input_row_offsets: Tensor
    """Tensor containing the offsets for each row in the ragged input sequence."""
    
    return_n_logits: Tensor
    """Number of logits to return."""
    
    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs


class GPT2LMHeadModel(PipelineModel[TextContext]):
    """GPT2 language model with a language modeling head on top."""
    
    model: Model
    """Compiled and initialized model ready for inference."""
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)
    
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return GPT2Config.get_kv_params(
            huggingface_config,
            n_devices,
            kv_cache_config,
            cache_dtype,
        )
    
    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return GPT2Config.get_num_layers(huggingface_config)
    
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return GPT2Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )
    
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the graph."""
        assert isinstance(model_inputs, GPT2Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *curr_kv_cache_inputs,
        )
        
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )
    
    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> GPT2Inputs:
        """Prepare the inputs for the first pass in multistep execution."""
        # Get input_row_offsets: start and end position of each batch
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        
        # Create a ragged token vector
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        
        return GPT2Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )
    
    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> GPT2Inputs:
        """Prepare the inputs for the next token in multistep execution."""
        assert isinstance(prev_model_inputs, GPT2Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        
        return GPT2Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
    
    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=GPT2Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=GPT2Config.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )
    
    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=GPT2Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=GPT2Config.get_num_layers(
                huggingface_config=huggingface_config,
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )
    
    def graph_inputs(self) -> tuple[TensorType]:
        """Define the input types for the computation graph."""
        device_ref = DeviceRef.from_device(self.devices[0])
        
        # Construct input types
        return_n_logits_type = TensorType(
            DType.int64,
            shape=["return_n_logits"],
            device=DeviceRef.CPU(),
        )
        
        kv_inputs = self.kv_manager.input_symbols()
        
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        
        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs[0],
        )
    
    def _get_state_dict(
        self,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
    ) -> dict[str, WeightData]:
        """Get state dict with optional weight adaptation."""
        huggingface_config = self.huggingface_config
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        
        return state_dict
    
    @traced
    def _build_graph(
        self, weights: Weights, adapter: Optional[WeightsAdapter] = None
    ) -> Graph:
        """Build the computation graph for GPT2."""
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        
        # Get state dict
        state_dict = self._get_state_dict(weights, adapter)
        
        # Create model config
        model_config = GPT2Config.from_huggingface_config(
            hf_config=self.huggingface_config,
            dtype=self.dtype,
            devices=[device_ref],
            kv_params=GPT2Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            return_logits=self.return_logits,
        )
        
        # Get graph inputs
        graph_inputs = self.graph_inputs()
        
        # Build graph
        nn_model = GPT2Transformer(model_config)
        
        # Load weights  
        print(f"DEBUG: About to load {len(state_dict)} weights into model")
        try:
            nn_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,  # Enable strict mode for better error messages
            )
            print("DEBUG: Successfully loaded all weights")
        except Exception as e:
            print(f"DEBUG: Weight loading failed with strict=True: {e}")
            print("DEBUG: Retrying with strict=False...")
            nn_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,  # Allow some missing weights
            )
            print("DEBUG: Weight loading completed with strict=False")
        
        self.state_dict = nn_model.state_dict()
        
        with Graph(
            "gpt2",
            input_types=graph_inputs,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            outputs = nn_model(
                tokens.tensor,
                [inp.tensor for inp in kv_cache_inputs],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph
    
    @traced
    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        """Load and compile the model."""
        # Pre-allocate buffer for input_row_offsets
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])
        
        logger.info("Building and compiling GPT2 model...")
        before = time.perf_counter()
        graph = self._build_graph(self.weights, self.adapter)
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        
        return model