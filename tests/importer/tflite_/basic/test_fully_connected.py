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
"""System test: test fully_connected"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(input_shape, unit, activation, use_bias):
    class FullyConnectedModule(tf.Module):
        def __init__(self):
            super(FullyConnectedModule).__init__()
            self.out = tf.keras.layers.Dense(unit, activation, use_bias)

        @tf.function(input_signature=[tf.TensorSpec(input_shape, dtype=tf.float32)])
        def __call__(self, x):
            out = []
            x = self.out(x)
            y = tf.reshape(x, [1, 1, 560, 80])
            out.append(x)
            out.append(y)
            return out

    return FullyConnectedModule()


input_shapes = [
    [1, 560, 128],
    # [3, 7]
]

units = [
    80,
    # 13
]

activations = [
    None,
    # 'relu',
]

use_biases = [
    True,
    # False
]


@pytest.mark.parametrize('input_shape', input_shapes)
@pytest.mark.parametrize('unit', units)
@pytest.mark.parametrize('activation', activations)
@pytest.mark.parametrize('use_bias', use_biases)
def test_fully_connected(input_shape, unit: int, activation: None, use_bias: bool, request):
    module = _make_module(input_shape, unit, activation, use_bias)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_fully_connected.py'])
