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


def _make_module(in_shape, axis, output_type):
    class ReduceModule(tf.Module):
        def __init__(self):
            super(ReduceModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            outs = []
            outs.append(tf.math.argmax(x, axis=axis, output_type=output_type))
            return outs
    return ReduceModule()


in_shapes = [
    [1, 16, 16, 3]
]

axes = [
    None,
    0,
    1,
    2,
    3,
    -1,
    -2,
    -3,
    -4
]

output_types = [
    tf.dtypes.int32,
    tf.dtypes.int64
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('output_type', output_types)
def test_argmax(in_shape, axis, output_type, request):
    module = _make_module(in_shape, axis, output_type)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_argmax.py'])
