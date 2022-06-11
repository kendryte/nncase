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
"""System test: test strided slice"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import tensorflow as tf
import numpy as np
from tflite_test_runner import TfliteTestRunner


def _make_module(in_shape, begin, end, strides, begin_masks=0, ellipsis_masks=0, end_masks=0, new_axis_masks=0, shrink_axis_masks=0):
    class StridedSliceModule(tf.Module):
        def __init__(self):
            super(StridedSliceModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            return tf.strided_slice(x, begin, end, strides,
                                    begin_mask=begin_masks, ellipsis_mask=ellipsis_masks, end_mask=end_masks, new_axis_mask=new_axis_masks, shrink_axis_mask=shrink_axis_masks)
    return StridedSliceModule()


cases = [
    ([3], [0], [2], [1], 0, 0, 0, 0, 0),
    ([5], [1], [5], [2], 0, 0, 0, 0, 0),
    ([6, 3], [1, 0], [5, 3], [2, 1], 0, 0, 0, 0, 0),
    ([4, 4, 3], [1, 0, 0], [4, 4, 3], [2, 2, 1], 0, 0, 0, 0, 0),
    ([11, 12, 5], [2, 3, 0], [11, 12, 5], [3, 3, 1], 0, 0, 0, 0, 0),
    ([3, 512, 512], [1, 2, 3], [3, 512, 512], [2, 3, 1], 0, 0, 0, 0, 0),
    ([11, 12, 6], [2, 3, 1], [11, 12, 6], [3, 3, 2], 0, 0, 0, 0, 0),
    ([3, 512, 512], [1, 2, 2], [3, 512, 510], [2, 3, 4], 0, 0, 0, 0, 0),
    ([3, 20, 20], [0, -2, 0], [0, 1, 0], [1, 1, 1], 5, 0, 5, 0, 2)
]


@pytest.mark.parametrize('in_shape,begin,end,strides, begin_mask, ellipsis_mask,end_mask,  new_axis_mask, shrink_axis_mask', cases)
def test_strided_slice(in_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask, request):
    module = _make_module(in_shape, begin, end, strides, begin_mask,
                          ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_strided_slice.py'])
