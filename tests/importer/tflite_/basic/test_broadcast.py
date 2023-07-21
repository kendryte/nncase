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
"""System test: test broadcast"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, out_shape):
    class BroadcastModule(tf.Module):
        def __init__(self):
            super(BroadcastModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            return tf.broadcast_to(x, out_shape)
    return BroadcastModule()


shapes = [
    ([1], [16]),
    ([16, 1], [16, 32]),
    ([5, 1, 1], [3, 5, 2, 32])
]


@pytest.mark.parametrize('in_shape,out_shape', shapes)
def test_broadcast(in_shape, out_shape, request):
    module = _make_module(in_shape, out_shape)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_broadcast.py'])
