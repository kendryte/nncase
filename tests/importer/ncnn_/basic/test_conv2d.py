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
import numpy as np
from ncnn_test_runner import Net, NcnnTestRunner


def _make_module(i_channels, i_size, k_size, o_channels, strides, padding, dilations):
    w = np.random.rand(*k_size, i_channels, o_channels).astype(np.float32) - 0.5

    net = Net()
    input = net.Input("input", i_size[1], i_size[0], i_channels)
    net.Convolution("conv1", input, o_channels, k_size[1], k_size[0], dilations[1], dilations[0],
                    strides[1], strides[0], padding[0], padding[1], padding[2], padding[3], 0.0, w)

    return net


i_channels = [
    3,
    16
]

i_sizes = [
    [12, 24],
    [38, 65]
]

k_sizes = [
    [1, 1],
    [3, 3],
]

o_channels = [
    1,
    8
]

strides = [
    (1, 1),
    (5, 5)
]

paddings = [
    (1, 1, 1, 1)
]

padding_modes = [
    'zeros'
]

dilations = [
    [1, 1],
    [2, 2]
]


@pytest.mark.parametrize('i_channels', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('k_size', k_sizes)
@pytest.mark.parametrize('o_channels', o_channels)
@pytest.mark.parametrize('strides', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('dilations', dilations)
def test_conv2d(i_channels, i_size, k_size, o_channels, strides, padding, dilations, request):
    module = _make_module(i_channels, i_size, k_size, o_channels,
                          strides, padding, dilations)
    runner = NcnnTestRunner(request.node.name)
    model_param, model_bin = runner.from_ncnn(module)
    runner.run(model_param, model_bin)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv2d.py'])
