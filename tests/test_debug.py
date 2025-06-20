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
"""System test: test demo"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
from tflite_test_runner import TfliteTestRunner
from onnx_test_runner import OnnxTestRunner
import os

def test_debug(request):
    model_path = request.config.getoption("--model_path")
    assert model_path is not None, "Please specify the model path using --model_path"

    runner = OnnxTestRunner(request.node.name)
    runner.set_shape_var({"seq_len": 14, "history_len": 0})
    runner.run(model_path)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_debug.py'])
