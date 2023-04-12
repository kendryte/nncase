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

import math
import pytest
import onnx
import torch
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np


def _make_module(in_channel, out_channel, kernel_size, stride, dilation, pad, group, bias):
    class ConvTransposeModule(torch.nn.Module):
        def __init__(self):
            super(ConvTransposeModule, self).__init__()
            self.conv_transpose = torch.nn.ConvTranspose2d(
                in_channel, out_channel, kernel_size, stride, pad, [0, 0], group, bias, dilation)

        def forward(self, x):
            return self.conv_transpose(x)

    return ConvTransposeModule()


in_sizes = [
    [16, 16],
    [33, 65],
]

in_channels = [
    1,
    3,
    16
]

out_channels = [
    1,
    16
]

kernel_sizes = [
    [1, 1],
    [3, 3],
]

strides = [
    1,
    [2, 2]
]

dilations = [
    1
]

pads = [
    0,
    [1, 1],
]

groups = [
    1
]

biases = [
    True,
    False
]


@pytest.mark.parametrize('in_size', in_sizes)
@pytest.mark.parametrize('in_channel', in_channels)
@pytest.mark.parametrize('out_channel', out_channels)
@pytest.mark.parametrize('kernel_size', kernel_sizes)
@pytest.mark.parametrize('stride', strides)
@pytest.mark.parametrize('dilation', dilations)
@pytest.mark.parametrize('pad', pads)
@pytest.mark.parametrize('group', groups)
@pytest.mark.parametrize('bias', biases)
def test_conv_transpose(in_size, in_channel, out_channel, kernel_size, stride, dilation, pad, group, bias, request):
    model_file = _make_module(in_channel, out_channel, kernel_size,
                              stride, dilation, pad, group, bias)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(model_file, [1, in_channel, *in_size])
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', __file__])
