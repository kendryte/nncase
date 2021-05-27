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

def _make_module(in_shape, dim):

    class ConcatModule(torch.nn.Module):
        def __init__(self):
            super(ConcatModule, self).__init__()

        def forward(self, x):
            x = torch.cat((x, x, x), dim)
            return x

    return ConcatModule()

in_shapes = [
    [1],
    [3, 4],
    [3, 4, 5],
    [1, 3, 224, 224]
]

axes = [
    0,
    1,
    2,
    3
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
def test_concat(in_shape, axis, request):
    if len(in_shape) > axis:
        module = _make_module(in_shape, axis)

        test_util.test_onnx_module(request.node.name, module, in_shape, ['cpu', 'k210', 'k510'])

if __name__ == "__main__":
    pytest.main(['-vv', 'test_concat.py'])