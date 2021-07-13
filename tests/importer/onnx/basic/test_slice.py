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
import torch
from onnx_test_runner import OnnxTestRunner

def _make_module(in_shape, out_channel, kernel_size, dim, start, length):

    class SqueezeModule(torch.nn.Module):
        def __init__(self):
            super(SqueezeModule, self).__init__()
            self.conv2d = torch.nn.Conv2d(2, out_channel, kernel_size)

        def forward(self, x):
            x = torch.narrow(x, dim=dim, start=start, length=length)
            x = self.conv2d(x)
            return x

    return SqueezeModule()

in_shapes = [
    [1, 4, 60, 72],
    [1, 3, 224, 224]
]

out_channels = [
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
def test_slice(in_shape, out_channel, kernel_size, axis, request):
    module = _make_module(in_shape, out_channel, kernel_size, 1, 0, 2)

    # depend on pad to slice patch
    # runner = OnnxTestRunner(request.node.name)
    # model_file = runner.from_torch(module, in_shape)
    # runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_slice.py'])