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

    class SigmoidModule(torch.nn.Module):
        def __init__(self):
            super(SigmoidModule, self).__init__()

        def forward(self, x):
            x = torch.sigmoid(x)
            return x

    return SigmoidModule()

in_shapes = [
    [1],
    [1, 3, 224, 224]
]

@pytest.mark.parametrize('in_shape', in_shapes)
def test_sigmoid(in_shape, request):
    module = _make_module()

    runner = OnnxTestRunner(['cpu', 'k210', 'k510'])
    model_file = runner.from_torch(request.node.name, module, in_shape)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_sigmoid.py'])