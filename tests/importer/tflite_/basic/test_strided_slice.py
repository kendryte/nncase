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
"""System test: test strided slice"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, begin, end, strides):
    class StridedSliceModule(tf.Module):
        def __init__(self):
            super(StridedSliceModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            return tf.strided_slice(x, begin, end, strides)
    return StridedSliceModule()


cases = [
    ([3], [0], [2], [1]),
    ([5], [1], [5], [2]),
    ([6, 3], [1, 0], [5, 3], [2, 1]),
    ([8, 8, 5], [1, 0, 0], [8, 8, 4], [2, 2, 1]),
    ([11, 12, 5], [2, 3, 0], [11, 12, 5], [3, 3, 1]),
    ([3, 512, 512], [1, 2, 3], [3, 512, 512], [2, 3, 1])
]


@pytest.mark.parametrize('in_shape,begin,end,strides', cases)
def test_strided_slice(in_shape, begin, end, strides, request):
    module = _make_module(in_shape, begin, end, strides)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_strided_slice.py'])
