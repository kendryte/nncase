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
"""System test: test gather"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, indice, axis, batch_dims):
    class GatherModule(tf.Module):
        def __init__(self):
            super(GatherModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            return tf.gather(x, indice, axis=axis, batch_dims=batch_dims)
    return GatherModule()


in_shape_indice_axis_batch_dims = [
    ([11], [1, 3, 10, 0, 2], 0, 0),
    ([11], [[2, 4], [1, 3]], 0, 0),
    ([7, 5], [1, 3], 0, 0),
    ([7, 5], [[1, 4, 3]], 1, 0),
    ([2, 3, 5], [1, 0, 1], 0, 0),
    ([2, 3, 5], [[2, 1], [1, 1], [1, 2]], 1, 0),
    ([2, 3, 5], [2, 4, 1], 2, 0),
    ([4, 5, 8, 3], [1, 0, 2], 1, 0),
    ([2, 3, 4, 7], [[1, 1], [1, 2]], 2, 0),
    ([4, 6, 5, 7], [[[1], [2]], [[3], [1]]], 3, 0),
    ([2, 3, 4, 7], [[1, 1], [1, 2]], -1, 0),
]


@pytest.mark.parametrize('in_shape,indice,axis,batch_dims', in_shape_indice_axis_batch_dims)
def test_gather(in_shape, indice, axis, batch_dims, request):
    module = _make_module(in_shape, indice, axis, batch_dims)
    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_gather.py'])
