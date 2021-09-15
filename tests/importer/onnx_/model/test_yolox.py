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


def test_yolox(request):
    overwrite_cfg = open('tests/importer/onnx_/model/test_yolox.yml', 'r', encoding="utf8").read()
    runner = OnnxTestRunner(request.node.name, ['cpu', 'k210'],
                            overwrite_configs=overwrite_cfg)
    model_file = 'examples/yolox/model/yolox_nano_224.onnx'
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_yolox.py'])
