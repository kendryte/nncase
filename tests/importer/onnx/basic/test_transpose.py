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

def _make_module(dim0, dim1):

    class TransposeModule(torch.nn.Module):
        def __init__(self):
            super(TransposeModule, self).__init__()

        def forward(self, x):
            x = torch.transpose(x, dim0=dim0, dim1=dim1)
            return x

    return TransposeModule()

in_shapes = [
    [3, 4],
    [3, 4, 5],
    [2, 3, 224, 224]
]

axes = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3]
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
def test_transpose(in_shape, axis, request):
    if len(in_shape) > axis[1]:
        module = _make_module(axis[0], axis[1])

        # test_util.test_onnx_module(request.node.name, module, in_shape, ['cpu', 'k210', 'k510'])
        runner = OnnxTestRunner(['cpu', 'k210', 'k510'])
        model_file = runner.from_torch(request.node.name, module, in_shape)
        runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_transpose.py'])