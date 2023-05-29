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
"""System test: test binary"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, v_shape):
    class BinaryModule(tf.Module):
        def __init__(self):
            super(BinaryModule).__init__()
            self.v = tf.constant(np.random.rand(*v_shape).astype(np.float32))

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            outs = []
            outs.append(x + self.v)
            # outs.append(x - self.v)
            # outs.append(x * self.v)
            # outs.append(self.v / (2.0 + x))
            # outs.append(tf.minimum(x, self.v))
            # outs.append(tf.maximum(x, self.v))
            return outs
    return BinaryModule()


lhs_shapes = [
    # [3],
    # [64, 3],
    # [3, 64, 3],
    # [8, 3, 64, 3]
    [1,24,24,3]
]

rhs_shapes = [
    # [1],
    # [3],
    # [1, 3],
    # [64, 1],
    # [64, 3],
    # [3, 64, 1],
    # [3, 64, 3],
    # [8, 3, 64, 1],
    # [8, 3, 64, 3],
    # [8, 3, 1, 3],
    # [8, 1, 64, 3],
    # [1, 3, 64, 1]
    [1,24,24,3]
]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
@pytest.mark.parametrize('rhs_shape', rhs_shapes)
def test_binary(lhs_shape, rhs_shape, request):
    module = _make_module(lhs_shape, rhs_shape)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_binary.py'])
