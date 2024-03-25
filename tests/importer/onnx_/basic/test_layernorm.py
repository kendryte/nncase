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
        def __init__(self, channel_size=3):
            super(LayerNormModule, self).__init__()
            reduce_dim = [v_shape[i] for i in range(len(v_shape)) if i > axis]
            self.scale = torch.from_numpy(np.random.rand(*reduce_dim).astype(np.float32))
            self.bias = torch.from_numpy(np.random.rand(*reduce_dim).astype(np.float32))
            # torch.layernorm init scale and bias with [1,0] because they are learnable. When the model exports to onnx, scale and bias were eliminated.
            # So, random data are used to pretend parameters.
            self.layernorm = nn.LayerNorm(normalized_shape=reduce_dim,
                                          elementwise_affine=True, eps=1e-03)

        def forward(self, x):
            x = self.layernorm(x)
            x = x * self.scale + self.bias
            return x

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
