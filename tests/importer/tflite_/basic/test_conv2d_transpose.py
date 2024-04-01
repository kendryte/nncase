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
"""System test: test conv2d transpsoe"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(n, i_channels, i_size, k_size, o_channels, strides, padding, dilations, bias):
    class Conv2DTransposeModule(tf.Module):
        def __init__(self):
            super(Conv2DTransposeModule).__init__()
            self.out = tf.keras.layers.Conv2DTranspose(
                o_channels, k_size, strides, padding, dilation_rate=dilations, use_bias=bias)

        @tf.function(input_signature=[tf.TensorSpec([n, *i_size, i_channels], tf.float32)])
        def __call__(self, x):
            return self.out(x)

    return Conv2DTransposeModule()


n = [
    1,
    3
]

i_channels = [
    1,
    16
]

i_sizes = [
    [1, 1],
    [33, 65]
]

k_sizes = [
    [1, 1],
    [3, 3],
    [5, 5]
]

o_channels = [
    1,
    16
]

strides = [
    [1, 1],
    [1, 3],
    [5, 5]
]

paddings = [
    'SAME',
    'VALID'
]

dilations = [
    [1, 1],
]

biases = [
    True,
    False
]


@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channels', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('k_size', k_sizes)
@pytest.mark.parametrize('o_channels', o_channels)
@pytest.mark.parametrize('strides', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('dilations', dilations)
@pytest.mark.parametrize('bias', biases)
def test_conv2d_transpose(n, i_channels, i_size, k_size, o_channels, strides, padding, dilations, bias, request):
    if k_size[0] >= strides[0] and k_size[1] >= strides[1]:
        module = _make_module(n, i_channels, i_size, k_size, o_channels,
                          strides, padding, dilations, bias)

        #runner = TfliteTestRunner(request.node.name, ['k510'])
        runner = TfliteTestRunner(request.node.name)
        model_file = runner.from_tensorflow(module)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv2d_transpose.py'])
