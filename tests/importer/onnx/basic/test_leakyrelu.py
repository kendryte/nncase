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

def _make_module(negative_slope):

    class LeakyReluModule(torch.nn.Module):
        def __init__(self):
            super(LeakyReluModule, self).__init__()
            self.leakyrelu = torch.nn.LeakyReLU(negative_slope)

        def forward(self, x):
            x = self.leakyrelu(x)
            return x

    return LeakyReluModule()

in_shapes = [
    [1],
    [1, 3, 224, 224]
]

negative_slopes = [
    0,
    0.01,
    0.4,
    0.8
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('negative_slope', negative_slopes)
def test_leakyrelu(in_shape, negative_slope, request):
    module = _make_module(negative_slope)

    runner = OnnxTestRunner(['cpu', 'k210', 'k510'])
    model_file = runner.from_torch(request.node.name, module, in_shape)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_leakyrelu.py'])