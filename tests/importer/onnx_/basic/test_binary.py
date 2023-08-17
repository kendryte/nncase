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
import numpy as np
from onnx_test_runner import OnnxTestRunner


def _make_module(v_shape):
    class BinaryModule(torch.nn.Module):
        def __init__(self):
            super(BinaryModule, self).__init__()
            self.v = torch.from_numpy(np.random.rand(*v_shape).astype(np.float32))

        def forward(self, x):
            outs = []
            outs.append(torch.add(x, self.v))
            outs.append(torch.mul(x, self.v))
            outs.append(torch.sub(x, self.v))
            outs.append(torch.max(x, self.v))
            outs.append(torch.div(x, self.v))
            outs.append(torch.min(x, self.v))
            outs.append(torch.fmod(x, self.v))
            return outs

    return BinaryModule()


lhs_shapes = [
    [3],
    [64, 3],
    [3, 64, 3],
    [8, 3, 64, 3],
    [1, 3, 24, 24]
]

rhs_shapes = [
    [1],
    [3],
    [1, 3],
    [64, 1],
    [64, 3],
    [3, 64, 1],
    [3, 64, 3],
    [8, 3, 64, 1],
    [8, 3, 64, 3],
    [8, 3, 1, 3],
    [8, 1, 64, 3],
    [1, 3, 64, 1],
    [1, 3, 24, 24]
]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
@pytest.mark.parametrize('rhs_shape', rhs_shapes)
def test_binary(lhs_shape, rhs_shape, request):
    module = _make_module(rhs_shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, lhs_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_binary.py'])
