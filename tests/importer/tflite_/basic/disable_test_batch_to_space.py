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
"""System test: test batch_to_space"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
from tflite_test_runner import TfliteTestRunner


def _make_module(batch_coff, in_shape, block_shape, crops):
    class BatchToSpaceModule(tf.Module):
        def __init__(self):
            super(BatchToSpaceModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec([batch_coff * np.prod(block_shape), *in_shape], tf.float32)])
        def __call__(self, x):
            return tf.batch_to_space(x, block_shape, crops)
    return BatchToSpaceModule()


batch_coffs = [
    1,
    2
]

in_shapes = [
    #[3, 4],
    [8, 4, 5]
]

block_shapes = [
    [2, 2],
    [3, 3]
]

crops = [
    [[0, 0], [0, 0]],
    [[0, 1], [0, 1]],
    [[1, 1], [1, 1]]
]


@pytest.mark.parametrize('batch_coff', batch_coffs)
@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('block_shape', block_shapes)
@pytest.mark.parametrize('crops', crops)
def test_batch_to_space(batch_coff, in_shape, block_shape, crops, request):
    module = _make_module(batch_coff, in_shape, block_shape, crops)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_batch_to_space.py'])
