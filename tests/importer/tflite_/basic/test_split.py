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


def _make_module(input_shape, num_split, axis):
    class SplitModule(tf.Module):
        def __init__(self):
            super(SplitModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
        def __call__(self, x):
            return tf.split(x, num_or_size_splits=num_split, axis=axis)
    return SplitModule()


cases = [
    ([4], 2, 0),
    ([9], 3, -1),
    ([16, 8], 4, 0),
    ([16, 16], 4, 1),
    ([16, 32], 4, -1),
    ([10, 32], 5, -2),
    ([16, 8, 9], 8, 0),
    ([1, 8, 9], 2, 1),
    ([1, 8, 9], 3, 2),
    ([1, 8, 18], 3, -1),
    ([1, 3, 18], 3, -2),
    ([6, 8, 18], 3, -3),
    ([32, 8, 18, 7], 4, 0),
    ([32, 8, 18, 7], 2, 1),
    ([32, 8, 18, 7], 1, 2),
    ([32, 8, 18, 7], 7, 3),
    ([32, 8, 18, 2], 2, -1),
    ([32, 8, 18, 2], 3, -2),
    ([32, 8, 18, 2], 4, -3),
    ([36, 8, 18, 2], 6, -4),


]


@pytest.mark.parametrize('in_shapes,num_split,axis', cases)
def test_split(in_shapes, num_split, axis, request):
    module = _make_module(in_shapes, num_split, axis)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_split.py'])
