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


def _make_module(num, init):

    class PReluModule(torch.nn.Module):
        def __init__(self):
            super(PReluModule, self).__init__()
            self.prelu = torch.nn.PReLU(num_parameters=num, init=init)

        def forward(self, x):
            x = self.prelu(x)
            return x

    return PReluModule()


in_shapes = [
    [1],
    [1, 3, 224, 224]
]

inits = [
    0.1,
    0.25,
    1.4,
    -0.5, 
    -1.2
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('init', inits)
def test_prelu(in_shape, init, request):
    num = 1 if len(in_shape) < 2 else in_shape[1]
    module = _make_module(num, init)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, in_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_prelu.py'])
