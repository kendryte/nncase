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
"""System test: test hardswish"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape):
    class HardSwishModule(tf.Module):
        def __init__(self):
            super(HardSwishModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            y = x + 3
            y = tf.nn.relu6(y)
            y = x * y
            y = y / 6
            return y

    return HardSwishModule()


in_shapes = [
    [3],
    [64, 3],
    [3, 64, 3],
    [8, 6, 16, 3]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_hardswish(in_shape, request):
    module = _make_module(in_shape)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vv', 'test_hardswish.py'])
