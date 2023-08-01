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
# import test_util
from onnx_test_runner import OnnxTestRunner


def _make_module():
    class UnaryModule(torch.nn.Module):
        def __init__(self):
            super(UnaryModule, self).__init__()

        def forward(self, x):
            outs = []
            outs.append(torch.abs(-x))
            outs.append(torch.acos(x))
            outs.append(torch.asin(x))
            outs.append(torch.ceil(x))
            outs.append(torch.cos(x))
            outs.append(torch.exp(x))
            outs.append(torch.floor(x * 10))
            outs.append(torch.log(x + 2))
            outs.append(torch.neg(x))
            outs.append(torch.round(x))
            outs.append(torch.sin(x))
            outs.append(torch.sqrt(x + 2))
            outs.append(torch.tanh(x))
            return outs

    return UnaryModule()


in_shapes = [
    [16],
    [1, 3, 16, 16]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_unary(in_shape, request):
    module = _make_module()

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, in_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_unary.py'])
