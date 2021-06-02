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
from test_runner import TfliteTestRunner

def _make_module(in_shapes, axis):
    class ConcatModule(tf.Module):
        def __init__(self):
            super(ConcatModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32) for in_shape in in_shapes])
        def __call__(self, x0, x1, x2):
            return tf.concat([x0, x1, x2], axis)
    return ConcatModule()


cases = [
    ([[1, 2], [1, 2], [3, 2]], 0),
    ([[1, 2, 3], [1, 2, 1], [1, 2, 2]], 2)
]


@pytest.mark.parametrize('in_shapes,axis', cases)
def test_concat(in_shapes, axis, request):
    module = _make_module(in_shapes, axis)

    runner = TfliteTestRunner(['cpu', 'k210', 'k510'])
    model_file = runner.from_tensorflow(request.node.name, module)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_concat.py'])
