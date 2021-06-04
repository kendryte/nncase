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

def _make_module(in_shape, axis):

    class FlattenModule(torch.nn.Module):
        def __init__(self):
            super(FlattenModule, self).__init__()
            self.conv2d = torch.nn.Conv2d(in_shape[1], 3, 3)

        def forward(self, x):
            x = self.conv2d(x)
            x = torch.flatten(x, start_dim=axis)

            return x

    return FlattenModule()

in_shapes = [
    [1, 3, 224, 224]
]

axes = [
    1,
    2,
    3
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
def test_flatten(in_shape, axis, request):
    if len(in_shape) > axis:
        module = _make_module(in_shape, axis)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_torch(module, in_shape)
        runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_flatten.py'])