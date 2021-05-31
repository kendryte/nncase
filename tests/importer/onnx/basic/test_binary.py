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

def _make_module():
    class BinaryModule(torch.nn.Module):
        def __init__(self):
            super(BinaryModule, self).__init__()

        def forward(self, x):
            add = torch.add(x, 6)
            mul = torch.mul(add, x)
            sub = torch.sub(mul, x)
            max = torch.max(sub, x)
            div = torch.div(max, 2)
            min = torch.min(div, x)
            return min

    return BinaryModule()

in_shapes = [
    [3],
    [64, 3],
    [3, 64, 3],
    [8, 6, 16, 3]
]

@pytest.mark.parametrize('in_shape', in_shapes)
def test_binary(in_shape, request):
    module = _make_module()

    runner = OnnxTestRunner(['cpu', 'k210', 'k510'])
    model_file = runner.from_torch(request.node.name, module, in_shape)
    runner.run(model_file)
    
if __name__ == "__main__":
    pytest.main(['-vv', 'test_binary.py'])