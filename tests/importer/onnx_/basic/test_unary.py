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

import sys
import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np

def _make_module(op, in_type, in_shape):
    inputs = []
    outputs = []
    # initializers = []
    attributes_dict = {}
    nodes = []

    # input1
    input1 = helper.make_tensor_value_info('input1', in_type, in_shape)
    inputs.append('input1')

    output_shape = in_shape
    output = helper.make_tensor_value_info('output', in_type, output_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        op,
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    
    nodes.append(node)
    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input1],
        [output],
        initializer=None)
        
    model_def = helper.make_model(graph_def, producer_name='onnx')
    return model_def


in_shapes = [
    # [16],
    [1, 3, 16, 16]
]

ops = [
    # 'Rsqrt', 'Square'  这 2 个目前不支持
    # 'Ceil',  # 已更新 1
    'Floor', # 已更新 2
    # 'Round', # 已更新 3
    # 'Sqrt',  # 已更新 4
    # 'Tanh',  # 已更新 5
    # 'Erf',   # 已更新 6
    # 'Abs',   # 已更新 7
    # 'Acos',  # 已更新 8
    # 'Asin',  # 已更新 9
    # 'Exp',   # 已更新 10
    # 'Log',   # 已更新 11
    # 'Neg',   # 已更新 12
    # 'Not',   # 已更新 13
    # 'Sign',  # 已更新 14
    # 'Sin',   # 已更新 15
    # 'Cos',   # 已更新 16
]

in_types = [
    TensorProto.FLOAT,
    # TensorProto.INT32,
    # TensorProto.INT8,
    # TensorProto.BOOL,
    # TensorProto.INT64,
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('op', ops)
def test_unary(op, in_type, in_shape, request):
    model_def = _make_module(op, in_type, in_shape)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)
    
'''
import pytest
import torch
import test_util
from onnx_test_runner import OnnxTestRunner
def _make_module():
    class UnaryModule(torch.nn.Module):
        def __init__(self):
            super(UnaryModule, self).__init__()

        def forward(self, x):
            outs = []
            outs.append(torch.abs(-x))
            outs.append(torch.acos(x))
            outs.append(torch.asin(x))
            outs.append(torch.ceil(x))
            outs.append(torch.cos(x))
            outs.append(torch.exp(x))
            outs.append(torch.floor(x * 10))
            outs.append(torch.log(x + 2))
            outs.append(torch.neg(x))
            outs.append(torch.round(x))
            outs.append(torch.sin(x))
            outs.append(torch.sqrt(x + 2))
            outs.append(torch.tanh(x))
            outs.append(torch.rsqrt(x + 2))
            return outs

    return UnaryModule()
@pytest.mark.parametrize('in_shape', in_shapes)
def test_unary(in_shape, request):
    module = _make_module()

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, in_shape)
    runner.run(model_file)
'''

if __name__ == "__main__":
    pytest.main(['-vv', 'test_unary.py'])
