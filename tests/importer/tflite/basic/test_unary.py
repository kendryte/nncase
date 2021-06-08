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
"""System test: test unary"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from test_runner import TfliteTestRunner


def _make_module(in_shape):
    class UnaryModule(tf.Module):
        def __init__(self):
            super(UnaryModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            outs = []
            outs.append(tf.math.abs(-x))
            outs.append(tf.math.ceil(x))
            outs.append(tf.math.cos(x))
            outs.append(tf.math.exp(x))
            # outs.append(tf.math.floor(x)) # large errors in ptq
            outs.append(tf.math.log(x + 2))
            outs.append(tf.math.negative(x))
            # outs.append(tf.math.round(x))
            outs.append(tf.math.rsqrt(x + 2))
            outs.append(tf.math.sin(x))
            outs.append(tf.math.sqrt(x + 2))
            outs.append(tf.math.square(x))
            outs.append(tf.math.tanh(x))
            outs.append(tf.math.sigmoid(x))
            return outs
    return UnaryModule()


in_shapes = [
    [3],
    [64, 3],
    [3, 64, 3],
    [8, 6, 16, 3]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_unary(in_shape, request):
    module = _make_module(in_shape)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vv', '/data/huochenghai/GNNE/merge/nncase/nncase/tests/importer/tflite/basic/test_unary.py'])
