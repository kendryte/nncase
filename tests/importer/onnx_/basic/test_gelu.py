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
import torch.nn as nn
from onnx_test_runner import OnnxTestRunner


def _make_module(v_shape):
    class GELUModule(torch.nn.Module):
        def __init__(self,):
            super(GELUModule, self).__init__()
            self.gelu = nn.GELU();
        def forward(self, x):
            x = self.gelu(x)
            return x


    return GELUModule()


lhs_shapes = [
    [1, 3, 16, 32],
    [1, 3, 16]
]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
def test_gelu(lhs_shape, request):
    module = _make_module(lhs_shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, lhs_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_gelu.py'])
