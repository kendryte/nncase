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


def _make_module(in_type, in_shape, axis, keep_dims):
    class ReduceProdModule(tf.Module):
        def __init__(self):
            super(ReduceProdModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, in_type)])
        def __call__(self, x):
            outs = []
            outs.append(tf.math.reduce_prod(x, axis=axis, keepdims=keep_dims))
            return outs
    return ReduceProdModule()


in_types = [
    tf.float32,
    tf.int32
]

in_shapes = [
    [1, 3, 2, 2]
]

axes_list = [
    [1],
    [2],
    [3],
    [-1],
    [-2],
    [-3],
    [2, 3],
    [-2, -1],
    [1, 2, 3],
    [-1, -2, -3],
    [0, 1, 2, 3],
    [-1, -2, -3, -4]
]

keep_dims = [
    True,
    False
]

@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axes', axes_list)
@pytest.mark.parametrize('keep_dims', keep_dims)
def test_reduce_prod(in_type, in_shape, axes, keep_dims, request):
    if len(axes) <= len(in_shape):
        module = _make_module(in_type, in_shape, axes, keep_dims)

        runner = TfliteTestRunner(request.node.name)
        model_file = runner.from_tensorflow(module)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_reduce_prod.py'])
