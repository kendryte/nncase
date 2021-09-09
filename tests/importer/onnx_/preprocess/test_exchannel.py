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
    [1, 3, 224, 224]
]

rhs_shapes = [
    [1]
]


@pytest.mark.parametrize('lhs_shape', lhs_shapes)
@pytest.mark.parametrize('rhs_shape', rhs_shapes)
def test_exchannel(lhs_shape, rhs_shape, request):
    module = _make_module(rhs_shape)
    overwrite_cfg = """
case: 
  preprocess_opt:
    - name: preprocess
      values:
        - true
    - name: exchange_channel
      values:
        - true
    - name: input_shape
      values:
        - [1,3,224,224]
    - name: mean
      values:
        - [0,0,0]
    - name: scale
      values:
        - [1,1,1]
    - name: input_range
      values:
        - [0,255]
    - name: input_type
      values:
        - uint8
    - name: input_layout
      values:
        - NCHW
    - name: output_layout
      values:
        - NCHW
"""

    runner = OnnxTestRunner(request.node.name, overwirte_configs=overwrite_cfg)
    # runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, lhs_shape)
    runner.run(model_file)


'''  exchange_channel: true
    input_shape: [1,3,224,224]
    mean: [0,0,0]
    scale: [1,1,1]
    input_range: [0,1]
    input_type: float32
    input_layout: NCHW
    output_layout: NCHW'''
if __name__ == "__main__":
    pytest.main(['-vv', 'test_exchannel.py'])
""" - name: exchange_channel
      values:
        - true
        - flase
    - name: input_shape
      values:
        - [1,3,320,320]
    - name: mean
      values:
        - [104,117,123]
    - name: scale
      values:
        - [1,1,1]
    - name: input_range
      values:
        - [0,255]
    - name: input_type
      values:
        - uint8
    - name: input_layout
      values:
        - NHWC
    - name: output_layout
      values:
        - NHWC
        - NCHW  """
