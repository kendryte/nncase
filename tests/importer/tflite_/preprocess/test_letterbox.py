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
            outs = x + self.v

            return outs
    return BinaryModule()


lhs_shapes = [
    [1, 112, 128, 3],
    [1, 224, 224, 3],
    [1, 304, 320, 3]
]

rhs_shapes = [
    [1],

]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
@pytest.mark.parametrize('rhs_shape', rhs_shapes)
def test_letterbox(lhs_shape, rhs_shape, request):
    module = _make_module(lhs_shape, rhs_shape)
    overwrite_cfg = """
case: 
  preprocess_opt:
    - name: preprocess
      values:
        - true
    - name: swapRB
      values:
        - false
    - name: input_shape
      values:
        - [1,224,224,3]
    - name: mean
      values:
        - [0,0,0]
    - name: std
      values:
        - [1,1,1]
    - name: input_range
      values:
        - [0,255]
    - name: input_type
      values:
        - uint8
    - name: input_layout
      values:
        - NHWC
    - name: output_layout
      values:
        - NHWC
    - name: letter_value
      values:
        - 114.
"""

    runner = TfliteTestRunner(request.node.name, overwrite_configs=overwrite_cfg)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_letterbox.py'])
