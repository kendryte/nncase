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
"""System test: test conv2d act"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
import test_util


def _make_module(n, i_channels, i_size, k_size, o_channels, strides, padding, dilations, act):
    class Conv2DActModule(tf.Module):
        def __init__(self):
            super(Conv2DActModule).__init__()
            self.w = tf.constant(np.random.rand(
                *k_size, i_channels, o_channels).astype(np.float32) - 1)

        @tf.function(input_signature=[tf.TensorSpec([n, *i_size, i_channels], tf.float32)])
        def __call__(self, x):
            out = tf.nn.conv2d(x, self.w, strides, padding,
                               dilations=dilations)
            if act == 'relu':
                out = tf.nn.relu(out)
            elif act == 'relu6':
                out = tf.nn.relu6(out)
            elif act == 'leaky':
                out = tf.nn.leaky_relu(out)
            return out
    return Conv2DActModule()


n = [
    1
]

i_channels = [
    16
]

i_sizes = [
    [33, 65]
]

k_sizes = [
    [3, 3]
]

o_channels = [
    8
]

strides = [
    [1, 3],
]

paddings = [
    'SAME',
    'VALID'
]

dilations = [
    [1, 1]
]

acts = [
    'relu',
    'relu6',
    'leaky'
]


@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channels', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('k_size', k_sizes)
@pytest.mark.parametrize('o_channels', o_channels)
@pytest.mark.parametrize('strides', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('dilations', dilations)
@pytest.mark.parametrize('act', acts)
def test_conv2d_act(n, i_channels, i_size, k_size, o_channels, strides, padding, dilations, act, request):
    if padding != 'VALID' or (k_size[0] <= i_size[0] and k_size[1] <= i_size[1]):
        module = _make_module(n, i_channels, i_size, k_size, o_channels,
                              strides, padding, dilations, act)
        test_util.test_tf_module(request.node.name, module, ['cpu', 'k210', 'k510'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv2d_act.py'])
