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

import pytest
from onnx_test_runner import OnnxTestRunner


def test_resnet152_v1_7(request):
    overwrite_cfg = """
    judge:
      specifics:
        - matchs:
            target: [cpu, k210]
            ptq: true
          threshold: 0.95
        - matchs:
            target: [k510]
            ptq: false
          threshold: 0.98
    """
    runner = OnnxTestRunner(request.node.name, overwrite_configs=overwrite_cfg)
    model_file = 'onnx-models/vision/classification/resnet/model/resnet152-v1-7.onnx'
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_resnet152-v1-7.py'])
