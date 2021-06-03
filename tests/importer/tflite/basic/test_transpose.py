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
"""System test: test transpose"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from test_runner import TfliteTestRunner

def _make_module(in_shape, perm):
    class TransposeModule(tf.Module):
        def __init__(self):
            super(TransposeModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            return tf.transpose(x, perm=perm)
    return TransposeModule()

in_shapes = [
    [3, 2],
    [64, 3],
    [5, 9, 32],
    [8, 3, 64, 3]
]

perms = [
    [1, 0],
    [0, 2, 1],
    [1, 0, 2],
    [2, 0, 1],
    [2, 1, 0],
    [0, 2, 1, 3],
    [0, 1, 3, 2],
    [3, 1, 0, 2],
    [3, 2, 0, 1],
    [3, 2, 1, 0]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('perm', perms)
def test_transpose(in_shape, perm, request):
    if len(perm) == len(in_shape):
        module = _make_module(in_shape, perm)

        runner = TfliteTestRunner(['cpu', 'k210', 'k510'])
        model_file = runner.from_tensorflow(request.node.name, module)
        runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_transpose.py'])