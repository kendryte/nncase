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
print("getpid", os.getpid())


def test_qwen3_fp8_static(request):
    cfg = """
    [compile_opt]
    dump_ir = true
    shape_bucket_enable = true
    shape_bucket_range_info = { "sequence_length"=[1,512] }
    shape_bucket_segments_count = 2
    shape_bucket_fix_var_map = { "batch_size"=1 }
    
    [huggingface_options]
    output_logits = true
    output_hidden_states = false
    num_layers = -1
    tensor_type = "float16"

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
    [target.xpu]
    infer = true

    [target.cpu.mode.noptq]
    enabled = true
    threshold = 0.98
    """
    runner = HuggingfaceTestRunner(request.node.name, overwrite_configs=cfg)

    model_name = "AngelSlim/Qwen3-0_6B_fp8_static"

    if os.path.exists(os.path.join(os.path.dirname(__file__), model_name)):
        model_file = os.path.join(os.path.dirname(__file__), model_name)
    else:
        model_file = download_from_huggingface(
            AutoModelForCausalLM, AutoTokenizer, model_name, need_save=False)

    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vvs', __file__])
