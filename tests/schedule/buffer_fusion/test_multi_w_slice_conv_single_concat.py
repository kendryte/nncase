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


def _make_module():
    class Module(tf.Module):
        def __init__(self):
            super(Module).__init__()
            self.w1 = tf.constant(np.random.rand(1, 1, 3, 3).astype(np.float32) - 1)
            self.w2 = tf.constant(np.random.rand(1, 1, 3, 3).astype(np.float32) - 1)

        @tf.function(input_signature=[tf.TensorSpec([1, 4, 10, 3], tf.float32)])
        def __call__(self, x):
            out1 = tf.math.abs(x)[:, :, 5:10, :]
            out2 = tf.math.exp(x)[:, :, 0:5, :]
            s1 = tf.nn.conv2d(out1, self.w1, (1, 1), 'SAME', dilations=(1, 1))
            s2 = tf.nn.conv2d(out2, self.w2, (1, 1), 'SAME', dilations=(1, 1))
            c2 = tf.concat([s1, s2], axis=2)
            return c2
    return Module()


def test_multi_w_slice_conv_single_concat(request):
    module = _make_module()

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_multi_w_slice_conv_single_concat.py'])
