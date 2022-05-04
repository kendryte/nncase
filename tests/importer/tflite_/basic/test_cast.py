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
"""System test: test concat"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shapes, data_type):
    class ConcatModule(tf.Module):
        def __init__(self):
            super(ConcatModule).__init__()
            self.v = tf.constant(np.random.rand(*in_shapes).astype(np.float32) * 100)

        @tf.function(input_signature=[tf.TensorSpec(in_shapes, tf.float32)])
        def __call__(self, x):
            x = tf.cast(x * self.v, data_type)
            return x

    return ConcatModule()


cases = [
    ([1, 32, 32, 3], tf.uint8),
    # Tf not support use 'Cast' to convert fp32 to int8, instead of FlexCast. But tflite can't run it .
    # ([1, 32, 32, 3], tf.int8),

]


@pytest.mark.parametrize('in_shapes, data_type', cases)
def test_cast(in_shapes, data_type, request):
    module = _make_module(in_shapes, data_type)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_cast.py', 's'])
