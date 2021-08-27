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
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, output_channel, axis, keepdim):
    class Module(tf.Module):
        def __init__(self):
            super(Module).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            weight = tf.constant(np.random.rand(
                1, 1, in_shape[3], output_channel).astype(np.float32) - 1)
            conv2d = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')
            output = tf.math.reduce_mean(conv2d, axis=axis, keepdims=keepdim)
            return output
    return Module()


in_shapes = [
    [1, 6, 8, 3]
]

output_channels = [
    4
]

axes = [
    # None,
    # [0],
    [1],
    [2],
    [3],
    [1, 2],
    [1, 3],
    [2, 3]
    # [2, 3, 1],
    # [3, 2, 1, 0]
]

keepdims = [
    True,
    False
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('output_channel', output_channels)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('keepdim', keepdims)
def test_transpose_reduce_motion_transform(in_shape, output_channel, axis, keepdim, request):
    module = _make_module(in_shape, output_channel, axis, keepdim)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_transpose_reduce_motion_transform.py'])
