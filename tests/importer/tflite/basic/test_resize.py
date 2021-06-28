# Copyright 2019-2021 Canaan Inc.
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
"""System test: test resize"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
from test_runner import TfliteTestRunner
import tensorflow as tf

def _make_module(in_shape, size, mode, align_corners, half_pixel_centers):
    class ResizeModule(tf.Module):
        def __init__(self):
            super(ResizeModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            if mode == tf.image.ResizeMethod.BILINEAR:
                return tf.compat.v1.image.resize_bilinear(x, size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
            else:
                return tf.compat.v1.image.resize_nearest_neighbor(x, size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
    return ResizeModule()

in_shape = [
    [2, 32, 32, 3]
]

sizes = [
    [16, 16],
    [64, 64],
    [11, 11],
    [37, 37],
    [41, 41]
]

modes = [
    tf.image.ResizeMethod.BILINEAR,
    tf.image.ResizeMethod.NEAREST_NEIGHBOR
]

align_corners = [
    True,
    False
]

half_pixel_centers = [
    True,
    False
]

@pytest.mark.parametrize('in_shape', in_shape)
@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('align_corners', align_corners)
@pytest.mark.parametrize('half_pixel_centers', half_pixel_centers)
def test_resize(in_shape, size, mode, align_corners, half_pixel_centers, request):
    if mode == tf.image.ResizeMethod.BILINEAR and align_corners and half_pixel_centers:
        return
    module = _make_module(in_shape, size, mode, align_corners, half_pixel_centers)
    runner = TfliteTestRunner(request.node.name, ['cpu', 'k210'])
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_resize.py'])
