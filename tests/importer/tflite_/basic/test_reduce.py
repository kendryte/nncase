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
"""System test: test reduce"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_type, in_shape, axis, keep_dims):
    class ReduceModule(tf.Module):
        def __init__(self):
            super(ReduceModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, in_type)])
        def __call__(self, x):
            outs = []
            outs.append(tf.reduce_min(x, axis=axis, keepdims=keep_dims))
            outs.append(tf.reduce_max(x, axis=axis, keepdims=keep_dims))
            outs.append(tf.reduce_mean(x, axis=axis, keepdims=keep_dims))
            outs.append(tf.reduce_sum(x, axis=axis, keepdims=keep_dims))
            return outs
    return ReduceModule()


in_types = [
    tf.float32,
    tf.int32
]

in_shape_axis = [
    ([3], [0]),
    ([64, 3], [0]),
    ([64, 3], [1]),
    ([64, 3], [0, 1]),
    ([3, 64, 3], [0]),
    ([3, 64, 3], [1]),
    ([3, 64, 3], [2]),
    ([3, 64, 3], [0, 1]),
    ([3, 64, 3], [0, 2]),
    ([3, 64, 3], [1, 0]),
    ([3, 64, 3], [1, 2]),
    ([8, 3, 64, 3], [0]),
    ([8, 3, 64, 3], [0, 1]),
    ([8, 3, 64, 3], [0, 2]),
    ([8, 3, 64, 3], [0, 3]),
    ([8, 3, 64, 3], [1, 3]),
    ([8, 3, 64, 3], [1, 2, 3]),
    ([8, 3, 64, 3], [0, 2, 3])
]

keep_dims = [
    True,
    False
]

@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape,axis', in_shape_axis)
@pytest.mark.parametrize('keep_dims', keep_dims)
def test_reduce(in_shape, in_type, axis, keep_dims, request):
    module = _make_module(in_type, in_shape, axis, keep_dims)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_reduce.py'])
