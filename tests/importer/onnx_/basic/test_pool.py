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


def _make_module(kernel_size, stride, padding, count_include_pad):

    class PoolModule(torch.nn.Module):
        def __init__(self):
            super(PoolModule, self).__init__()
            # self.avgpool2d = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding) # dsp pad
            self.avgpool2d = torch.nn.AvgPool2d(
                kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
            self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.maxpool2d = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
            self.global_maxpool = torch.nn.AdaptiveMaxPool2d(1)

        def forward(self, x):
            if(count_include_pad):
                return self.avgpool2d(x)

            outs = []
            outs.append(self.avgpool2d(x))
            outs.append(self.global_avgpool(x))
            outs.append(self.maxpool2d(x))
            outs.append(self.global_maxpool(x))  # maxpool2d in fact

            return outs

    return PoolModule()


in_shapes = [
    [1, 3, 60, 72],
]

kernel_sizes = [
    (3, 3),
]

strides = [
    1,
    2,
    [2, 1]
]

paddings = [
    (0, 0),
    (1, 1),
    (2, 3)
]

count_include_pads = [
    False,
    True
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('kernel_size', kernel_sizes)
@pytest.mark.parametrize('stride', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('count_include_pad', count_include_pads)
def test_pool(in_shape, kernel_size, stride, padding, count_include_pad, request):
    if kernel_size[0] / 2 > padding[0] and kernel_size[1] / 2 > padding[1]:
        module = _make_module(kernel_size, stride, padding, count_include_pad)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_torch(module, in_shape)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_pool.py'])
