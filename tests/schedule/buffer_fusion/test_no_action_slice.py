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


def _make_module(in_shape, begin, size):
    class SliceModule(tf.Module):
        def __init__(self):
            super(SliceModule).__init__()
            self.w = tf.constant(np.random.rand(
                3, 3, size[3], 21).astype(np.float32) - 1)

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            out = tf.slice(x, begin, size)
            out = tf.nn.conv2d(out, self.w, [1, 1], 'VALID')
            return out
    return SliceModule()


cases = [
    ([1, 20, 20, 4], [0, 0, 0, 0], [1, 18, 15, 3]),
    ([1, 40, 40, 10], [0, 3, 3, 2], [1, 35, 32, 8])
]


@pytest.mark.parametrize('in_shape,begin,size', cases)
def test_no_action_slice(in_shape, begin, size, request):
    module = _make_module(in_shape, begin, size)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_no_action_slice.py'])
