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
    [16],
    [1, 3, 16, 16]
]

# calc operators
ops = [
    # 'Rsqrt', 'Square'  # 这 2 个算子目前不支持
    'Ceil',  
    'Floor',
    'Round',
    'Sqrt',
    'Tanh',
    'Erf',
    'Abs',
    'Acos',
    'Asin',
    'Exp',
    'Log',
    'Neg',
    'Sign',
    'Sin',
    'Cos',
]

# calc operators data type
in_types = [
    TensorProto.FLOAT,
    # TensorProto.INT32,  // Not supported at present
    # TensorProto.INT8,   // Not supported at present
    # TensorProto.INT64,  // Not supported at present
]

# logical operators
logical_ops = [
    'Not'
]

# logical operators data type
logical_types = [
    TensorProto.BOOL
]

# operators and types group
op_type_pairs = [
    [logical_ops, logical_types], 
    [ops, in_types]
]

def get_case_data(in_datas):
    case_data = []
    for op_types in in_datas:
        _ops = op_types[0]
        _types = op_types[1]
        for _op in _ops:
            for _type in _types:
                tmp_pair = []
                tmp_pair.append(_op)
                tmp_pair.append(_type)
                case_data.append(tmp_pair)
    return case_data        
    pass

class TestUnaryModule(object):
 
    def setup_class(self):
        pass
 
    def teardown_class(self):
        pass
 
    # get the test case
    case_data=get_case_data(op_type_pairs)
    print(case_data)
    
    @pytest.mark.parametrize('in_shape', in_shapes)
    @pytest.mark.parametrize('op, in_type', case_data)
    def test_unary(self, op, in_type, in_shape, request):
        model_def = _make_module(op, in_type, in_shape)
        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)
        pass
    
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
    pytest.main(['-v', 'test_unary.py'])
