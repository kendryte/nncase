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


def test_qwen3_30B_A3B_static(request):
    cfg = """
    [compile_opt]
    shape_bucket_enable = true
    shape_bucket_range_info = { "sequence_length"=[1, 256] }
    shape_bucket_segments_count = 2
    shape_bucket_fix_var_map = {  }
    dump_ir = true

    [huggingface_options]
    output_logits = true
    output_hidden_states = false
    num_layers = 1

    [generator]
    [generator.inputs]
    method = 'text'
    number = 1
    batch = 1

    [generator.inputs.text]
    args = 'tests/importer/huggingface_/prompt.txt'

    [generator.calibs]
    method = 'text'
    number = 1
    batch = 1

    [generator.calibs.text]
    args = 'tests/importer/huggingface_/prompt.txt'
    """
    runner = HuggingfaceTestRunner(request.node.name, overwrite_configs=cfg)


<< << << < HEAD
   # model_name = "/home/yanghaoqi/workspace/Qwen3-30B-A3B-FP8-dynamic"
   model_name = "/home/yanghaoqi/workspace/Qwen3-30B-A3B_fp8_static"
    # model_name = "/compiler/share/huggingface_cache/hub/LLM-Research/Qwen3-30B-A3B_fp8_static"
== == == =
   model_name = "/compiler/share/huggingface_cache/hub/LLM-Research/Qwen3-30B-A3B_fp8_static"
>>>>>> > 93892ee4(add qwen3moe static test)

   if os.path.exists(os.path.join(os.path.dirname(__file__), model_name)):
        model_file = os.path.join(os.path.dirname(__file__), model_name)
    else:
        model_file = download_from_huggingface(
            AutoModelForCausalLM, AutoTokenizer, model_name, need_save=True)

    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', __file__])
