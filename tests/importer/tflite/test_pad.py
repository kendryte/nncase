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
"""System test: test pad"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
import test_util


def _make_module(in_shape, paddings, mode, const):
    class PadModule(tf.Module):
        def __init__(self):
            super(PadModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            return tf.pad(x, paddings, mode, const)
    return PadModule()


in_shapes = [
    [3],
    [64, 3],
    [3, 64, 3],
    [8, 3, 64, 3]
]

paddings = [
    [[1, 0]],
    [[0, 1]],
    [[3, 2], [1, 1]],
    [[0, 1], [3, 2], [1, 1]],
    [[0, 1], [3, 2], [0, 0], [1, 1]],
]

modes = [
    'CONSTANT',

    # tflite bug, padding < in_shape ?  :mixed result
    # [REFLECT] channel dim  if padding.before>1, tflite will pad zero instead of real value
    # 'REFLECT',
    'SYMMETRIC',
    # 'EDGE'
]

constants = [
    0,
    1.5
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('paddings', paddings)
@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('constant', constants)
def test_pad(in_shape, paddings, mode, constant, request):
    if len(in_shape) == len(paddings):
        module = _make_module(in_shape, paddings, mode, constant)
        test_util.test_tf_module(
            request.node.name, module, ['cpu', 'k510'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_pad.py'])
