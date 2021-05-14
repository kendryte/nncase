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
import os
import torch
import numpy as np
import sys
import test_util

def _make_module(in_shape, out_channel, kernel_size, dim):

    class SqueezeModule(torch.nn.Module):
        def __init__(self):
            super(SqueezeModule, self).__init__()
            self.conv2d = torch.nn.Conv2d(in_shape[1], out_channel, kernel_size)

        def forward(self, x):
            x = self.conv2d(x)
            x = torch.squeeze(x)

            # There is something wrong when converting pytorch into onnx. Use tf.squeeze(x, dim) instead.
            # x = torch.squeeze(x, dim)
            return x

    return SqueezeModule()

in_shapes = [
    [1, 4, 60, 72],
    [1, 3, 224, 224]
]

out_channels = [
    1,
    3,
]

kernel_sizes = [
    1,
    3,
]

axes = [
    0,
    1,
    2,
    3
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('out_channel', out_channels)
@pytest.mark.parametrize('kernel_size', kernel_sizes)
@pytest.mark.parametrize('axis', axes)
def test_squeeze(in_shape, out_channel, kernel_size, axis, request):
    out_shape = [in_shape[0], out_channel, in_shape[2] - kernel_size + 1, in_shape[3] - kernel_size + 1]
    dim = axis if out_shape[axis] == 1 else None
    module = _make_module(in_shape, out_channel, kernel_size, dim)

    # test_util.test_onnx_module(request.node.name, module, in_shape, ['cpu', 'k210', 'k510'])
    test_util.test_onnx_module(request.node.name, module, in_shape, ['k510'])

if __name__ == "__main__":
    pytest.main(['-vv', 'test_squeeze.py'])