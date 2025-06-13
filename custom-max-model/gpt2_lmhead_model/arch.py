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

from max.graph.weights import WeightsFormat
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import PipelineTask
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import GPT2LMHeadModel

gpt2_lmhead_arch = SupportedArchitecture(
    name="GPT2LMHeadModel",  # Must match the Hugging Face model class name
    example_repo_ids=[
        "distilbert/distilgpt2",
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.PAGED],
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.PAGED],
    },
    pipeline_model=GPT2LMHeadModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,  # Start with single GPU support
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
)