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
"""System test: test onehot"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner

def _make_module(indices, depth, off_value, on_value, axis):
    class OneHotModule(tf.Module):
        def __init__(self):
            super(OneHotModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(np.array(indices).shape, tf.int32)])
        def __call__(self, x):
            return tf.one_hot(x, depth, off_value=off_value, on_value=on_value, axis=axis)
    return OneHotModule()

indices_depth_off_value_on_value_axis = [
    ([[0, 2, 1, 1], [1, -1, 0, 0]], 3, 0, 9, 0),
    ([[0, 2, 1, 1], [1, -1, 0, 0]], 3, 0, 9, 1),
    ([[0, 2, 1, 1], [1, -1, 0, 0]], 3, 0, 9, 2),
]

@pytest.mark.parametrize('indices,depth, off_value,on_value,axis', indices_depth_off_value_on_value_axis)
def test_onehot(indices, depth, off_value, on_value, axis, request):
    module = _make_module(indices, depth, off_value, on_value, axis)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_onehot.py'])
