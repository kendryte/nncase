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
from test_runner import OnnxTestRunner

def _make_module(dim, keepdim):

    class ReduceModule(torch.nn.Module):
        def __init__(self):
            super(ReduceModule, self).__init__()

        def forward(self, x):
            outs = []
            outs.append(torch.max(x, dim, keepdim)[0])
            outs.append(torch.min(x, dim, keepdim)[0])

            # depend on Gather op
            # outs.append(torch.mean(x, dim, keepdim)[0])
            # outs.append(torch.sum(x, dim, keepdim)[0])

            return outs

    return ReduceModule()

in_shapes = [
    [1],
    [3, 4],
    [3, 4, 5],
    [1, 3, 224, 224]
]

dims = [
    0,
    1,
    2,
    3
]

keepdims = [
    False,
    True
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('dim', dims)
@pytest.mark.parametrize('keepdim', keepdims)
def test_reduce(in_shape, dim, keepdim, request):
    if len(in_shape) > dim :
        module = _make_module(dim, keepdim)

        # test_util.test_onnx_module(request.node.name, module, in_shape, ['cpu', 'k210', 'k510'])
        runner = OnnxTestRunner(['cpu', 'k210', 'k510'])
        model_file = runner.from_torch(request.node.name, module, in_shape)
        runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_reduce.py'])