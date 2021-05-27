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
import test_util

def _make_module(i_channel, k_size, o_channel, stride, padding, padding_mode, dilation):

    class Conv2dModule(torch.nn.Module):
        def __init__(self):
            super(Conv2dModule, self).__init__()
            self.conv = torch.nn.Conv2d(i_channel, o_channel, k_size, stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation)

        def forward(self, x):
            x = self.conv(x)
            return x

    return Conv2dModule()

n = [
    1,
    3
]

i_channels = [
    3,
    16
]

i_sizes = [
    [224, 224],
    [112, 65]
]

k_sizes = [
    [1, 1],
    [3, 3],
]

o_channels = [
    1,
    16
]

strides = [
    (1, 1),
    (5, 5)
]

paddings = [
    1,
    (2, 3)
]

padding_modes = [
    'zeros'
]

dilations = [
    [1, 1],
    [2, 2]
]

@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channel', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('k_size', k_sizes)
@pytest.mark.parametrize('o_channel', o_channels)
@pytest.mark.parametrize('stride', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('padding_mode', padding_modes)
@pytest.mark.parametrize('dilation', dilations)
def test_conv2d(n, i_channel, i_size, k_size, o_channel, stride, padding, padding_mode, dilation, request):

    module = _make_module(i_channel, k_size, o_channel, stride, padding, padding_mode, dilation)
    in_shape = [n, i_channel]
    in_shape.extend(i_size)

    # test_util.test_onnx_module(request.node.name, module, in_shape, ['cpu', 'k210', 'k510'])
    test_util.test_onnx_module(request.node.name, module, in_shape, ['k510'])

if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv2d.py'])