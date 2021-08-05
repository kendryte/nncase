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
"""System test: test mobilenetv2"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import os
import tensorflow as tf
import numpy as np
import sys
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, alpha):
    return tf.keras.applications.MobileNetV2(in_shape, alpha, include_top=False)


in_shapes = [
    (224, 224, 3)
]

alphas = [
    0.5
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('alpha', alphas)
def test_mobilenetv2(in_shape, alpha, request):
    module = _make_module(in_shape, alpha)
    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_mobilenetv2.py'])
