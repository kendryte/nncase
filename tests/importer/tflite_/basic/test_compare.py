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


def _make_module(compare_op, in_type_0, in_shape_0, in_type_1, in_shape_1):
    class CompareModule(tf.Module):
        def __init__(self):
            super(CompareModule).__init__()
            self.v = tf.constant((np.ones(in_shape_1)/2.0).astype(in_type_1))

        @tf.function(input_signature=[tf.TensorSpec(in_shape_0, in_type_0)])
        def __call__(self, x):
            outs = []
            if compare_op == 'equal':
                outs.append(tf.math.equal(x, self.v))

            if compare_op == 'not_equal':
                outs.append(tf.math.not_equal(x, self.v))

            if compare_op == 'greater':
                outs.append(tf.math.greater(x, self.v))

            if compare_op == 'greater_equal':
                outs.append(tf.math.greater_equal(x, self.v))

            if compare_op == 'less':
                outs.append(tf.math.less(x, self.v))

            if compare_op == 'less_equal':
                outs.append(tf.math.less_equal(x, self.v))

            return outs

    return CompareModule()

compare_ops = [
    'equal',
    'not_equal',
    'greater',
    'greater_equal',
    'less',
    'less_equal'
]

in_types = [
    # [tf.uint8, np.uint8],
    [tf.float32, np.float32],
    [tf.int32, np.int32],
    [tf.int64, np.int64],
]

in_shapes = [
    [[1, 3, 16, 16], [1]],
    [[1, 3, 16, 16], [16]],
    [[1, 3, 16, 16], [1, 16]],
    [[1, 3, 16, 16], [1, 16, 16]],
    [[1, 1, 16, 16], [3, 3, 1, 16]],
]

@pytest.mark.parametrize('compare_op', compare_ops)
@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
def test_compare(compare_op, in_type, in_shape, request):
    module = _make_module(compare_op, in_type[0], in_shape[0], in_type[1], in_shape[1])

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_compare.py'])
