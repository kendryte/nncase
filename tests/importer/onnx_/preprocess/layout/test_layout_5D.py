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
            self.v = torch.from_numpy(np.ones_like(*v_shape).astype(np.uint8))

        def forward(self, x):
            x = torch.add(x, self.v)
            return x

    return BinaryModule()


lhs_shapes = [
    [1, 3, 6, 9, 12]
]

rhs_shapes = [
    [1]
]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
@pytest.mark.parametrize('rhs_shape', rhs_shapes)
def test_layout_5D(lhs_shape, rhs_shape, request):
    module = _make_module(rhs_shape)
    overwrite_cfg = """
case:
  preprocess_opt:
    - name: preprocess
      values:
        - true
    - name: swapRB
      values:
        - false
    - name: mean
      values:
        - [0,0,0]
    - name: std
      values:
        - [1,1,1]
    - name: input_range
      values:
        - [0,255]
    - name: input_type
      values:
        - uint8
    - name: input_shape
      values:
        - [1, 3, 9, 12, 6]
    - name: input_layout
      values:
        - "0,1,4,2,3"
    - name: output_layout
      values:
        - "0,1,4,2,3"
    - name: model_layout
      values:
        - NCHW
    - name: letterbox_value
      values:
        - 0.
"""

    runner = OnnxTestRunner(request.node.name, overwrite_configs=overwrite_cfg)
    model_file = runner.from_torch(module, lhs_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_layout_5D.py'])
