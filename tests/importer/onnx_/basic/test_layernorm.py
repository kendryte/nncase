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
import numpy as np
from onnx_test_runner import OnnxTestRunner


def _make_module(v_shape, axis):
    class LayerNormModule(torch.nn.Module):
        def __init__(self) -> None:
            super(LayerNormModule, self).__init__()
            reduce_dim = [v_shape[i] for i in range(len(v_shape)) if i > axis]
            self.scale = torch.from_numpy(np.random.rand(*reduce_dim).astype(np.float32))
            self.bias = torch.from_numpy(np.random.rand(*reduce_dim).astype(np.float32))
            self.axis = [i for i in range(len(v_shape)) if i > axis]

        def forward(self, x):
            reduce_mean = torch.mean(x, self.axis, keepdim=True)
            x_sub = x - reduce_mean
            x_pow = torch.pow(x_sub, 2)
            x_pow_mean = torch.mean(x_pow, self.axis, keepdim=True)
            x_add = x_pow_mean + 1e-06
            x_sqrt = torch.sqrt(x_add)
            x_div = x_sub / x_sqrt
            x_mul = x_div * self.scale
            x_add_bias = x_mul + self.bias
            return x_add_bias

    return LayerNormModule()


lhs_shapes = [
    [1, 3, 16, 32],
    [1, 3, 16]
]

axises = [
    1
]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
@pytest.mark.parametrize('axis', axises)
def test_LayerNorm(lhs_shape, axis, request):
    module = _make_module(lhs_shape, axis)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, lhs_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_LayerNorm.py'])
