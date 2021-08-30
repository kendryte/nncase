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
"""System test: test 20 classes yolo"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
from tflite_test_runner import TfliteTestRunner


def test_20classes_yolo(request):
    runner = TfliteTestRunner(request.node.name)
    # model_file = '/home/curio/project/k510-gnne-compiler-tests/golden-model/mobilenet_v1_1.0.224/tflite/model_f32.tflite'
    # generate model
    # model_file = '/home/curio/project/k510-gnne-compiler-tests/local_test/test/model_0/model_f32.tflite'
    model_file = '/home/curio/github/nncase/tests/importer/tflite/model/model_f32.tflite'
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_20classes_yolo.py'])
