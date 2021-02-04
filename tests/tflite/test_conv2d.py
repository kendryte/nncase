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
"""System test: test conv2d"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
import test_util


def _make_module(in_shape, k_size, o_channels, strides, padding, dilations):
    class Conv2DModule(tf.Module):
        def __init__(self):
            super(Conv2DModule).__init__()
            self.w = tf.constant(np.random.rand(
                *k_size, in_shape[3], o_channels).astype(np.float32) - 1)

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            out = tf.nn.conv2d(x, self.w, strides, padding,
                               dilations=dilations)
            return out
    return Conv2DModule()


in_shapes = [
    #[1, 1, 1, 1],
    [1, 2, 1, 1],
]

k_sizes = [
    #[1, 1],
    [3, 3]
]

o_channels = [
    #1,
    3,
    #13,
    #64,
    #128
]

strides = [
    [1, 1],
    #[1, 2],
    #[2, 1],
    #[2, 2]
]

paddings = [
    'SAME',
    #'VALID'
]

dilations = [
    [1, 1]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('k_size', k_sizes)
@pytest.mark.parametrize('o_channels', o_channels)
@pytest.mark.parametrize('strides', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('dilations', dilations)
def test_conv2d(in_shape, k_size, o_channels, strides, padding, dilations, request):
    if padding != 'VALID' or (k_size[0] <= in_shape[1] and k_size[1] <= in_shape[2]):
        module = _make_module(in_shape, k_size, o_channels,
                              strides, padding, dilations)
        test_util.test_tf_module(request.node.name, module, ['cpu'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv2d.py'])
