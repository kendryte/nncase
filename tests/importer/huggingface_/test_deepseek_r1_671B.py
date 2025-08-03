# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
from huggingface_test_runner import HuggingfaceTestRunner, download_from_huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_deepseek_r1_0528(request):
    cfg = """
    [compile_opt]
    shape_bucket_enable = true
    shape_bucket_range_info = { "sequence_length"=[1,256] }
    shape_bucket_segments_count = 2
    shape_bucket_fix_var_map = { }
    dump_ir = true
    
    [huggingface_options]
    output_logits = true
    output_hidden_states = false

    [generator]
    [generator.inputs]
    method = 'text'
    number = 1
    batch = 1

    [generator.inputs.text]
    args = 'tests/importer/huggingface_/prompt_qwen.txt'

    [generator.calibs]
    method = 'text'
    number = 1
    batch = 1

    [generator.calibs.text]
    args = 'tests/importer/huggingface_/prompt_qwen.txt'

    #TODO: Need remove!
    [target]
    [target.cpu]
    infer = false
    eval = true
    
    [target.xpu]
    infer = false
    eval = false
    
    [target.xpu.target_options]
    CustomOpScheme = "/compiler/yanghaoqi/workspace/nncase/wheel/paged_attn_scheme.json"
    
    [paged_attention_config]
    block_size = 256
    num_blocks = 32
    max_sessions = 1
    kv_type = "float16"
    cache_layout = ["NumBlocks","NumLayers","NumKVHeads","KV","HeadDim","BlockSize"]
    packed_axes = ["HeadDim"]
    lanes = [64]
    sharding_axes = ["NumKVHeads","NumBlocks"]
    axis_policies = [[1],[2,3]]
    hierarchy = [1, 2, 8, 4, 4]
    """
    runner = HuggingfaceTestRunner(request.node.name, overwrite_configs=cfg)

    model_name = "LLM-Research/DeepSeek-R1-0528"

    if os.path.exists(os.path.join(os.path.dirname(__file__), model_name)):
        model_file = os.path.join(os.path.dirname(__file__), model_name)
    else:
        model_file = download_from_huggingface(
            AutoModelForCausalLM, AutoTokenizer, model_name, need_save=True)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vvs', __file__])
