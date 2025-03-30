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


def test_llama32_1B(request):
    cfg = """
    [huggingface_options]
    output_attentions = true
    output_hidden_states = true
    use_cache = true

    [generator]
    [generator.inputs]
    method = 'text'

    [generator.inputs.text]
    args = 'tests/llm/prompt.txt'

    [generator.calibs]
    method = 'text'

    [generator.calibs.text]
    args = 'tests/llm/prompt.txt'
    
    #TODO: Need remove!
    [target]
    [target.cpu]
    infer = false
    """
    runner = HuggingfaceTestRunner(request.node.name, overwrite_configs=cfg)

    model_name = "LLM-Research/Llama-3.2-1B-Instruct"

    if os.path.exists(os.path.join(os.path.dirname(__file__), model_name)):
        model_file = os.path.join(os.path.dirname(__file__), model_name)
    else:
        model_file = download_from_huggingface(AutoModelForCausalLM, AutoTokenizer, model_name)

    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_llama32_1B.py'])
