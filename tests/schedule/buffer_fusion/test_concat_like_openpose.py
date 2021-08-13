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

        @tf.function(input_signature=[tf.TensorSpec([1, 4, 4, 3], tf.float32)])
        def __call__(self, x):
            c1 = x
            s0 = tf.nn.pool(c1, (1, 1), 'MAX')
            s1 = tf.nn.conv2d(c1, self.w1, (1, 1), 'SAME', dilations=(1, 1))
            s2 = tf.nn.conv2d(s1, self.w2, (1, 1), 'SAME', dilations=(1, 1))
            c2 = tf.concat([s0, s1, s2], axis=3)
            s3 = tf.math.sin(c2)
            s4 = tf.math.cos(c2)
            c3 = tf.concat([s3, s4, c2], axis=3)
            s5 = tf.math.sin(c3)
            s6 = tf.math.cos(c3)
            c4 = tf.concat([s5, s6], axis=3)

            return c4
    return Module()


def test_concat_like_openpose(request):
    module = _make_module()

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vv', 'test_concat_like_openpose.py'])
