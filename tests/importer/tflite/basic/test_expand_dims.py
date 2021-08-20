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
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, axes):
    class ExpandDimsModule(tf.Module):
        def __init__(self):
            super(ExpandDimsModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            unary = tf.math.abs(x)
            y = tf.expand_dims(unary, axes)
            return y

    return ExpandDimsModule()


in_shapes = [
    [28],
    [28, 28],
    [3, 28, 28]
]

axes_list = [
    [0],
    [1],
    [2],
    [3],
    [-1],
    [-2],
    [-3],
    [-4]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axes', axes_list)
def test_expand_dims(in_shape, axes, request):
    if len(in_shape) + len(axes) == 4:
        model_def = _make_module(in_shape, axes)

        runner = TfliteTestRunner(request.node.name)
        model_file = runner.from_tensorflow(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_expand_dims.py'])
