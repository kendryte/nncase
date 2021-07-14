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
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np

def _make_module(in_shape, padding, constant_value, mode, op_version):

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)

    initializers = None

    out_shape = in_shape.copy()
    out_shape[2] += padding[2] + padding[6]
    out_shape[3] += padding[3] + padding[7]

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)

    if op_version == 1:
        if constant_value is None:
            node = onnx.helper.make_node(
                'Pad',
                inputs=['input'],
                outputs=['output'],
                mode=mode,
                paddings=padding)
        else:
            node = onnx.helper.make_node(
                'Pad',
                inputs=['input'],
                outputs=['output'],
                mode=mode,
                paddings=padding,
                value=constant_value)
    elif op_version == 2:
        if constant_value is None:
            node = onnx.helper.make_node(
                'Pad',
                inputs=['input'],
                outputs=['output'],
                mode=mode,
                pads=padding)
        else:
            node = onnx.helper.make_node(
                'Pad',
                inputs=['input'],
                outputs=['output'],
                mode=mode,
                pads=padding,
                value=constant_value)
    else:
        # opset 11/13
        initializers = []
        dims_list = []
        dims_list.append(len(padding))
        pads = helper.make_tensor(
            "pads",
            TensorProto.INT64,
            dims=dims_list,
            vals=padding)

        initializers.append(pads)

        inputs = ['input', 'pads']
        if constant_value is not None:
            cv_list = []
            cv_list.append(constant_value)
            cv = helper.make_tensor(
                "constant_value",
                TensorProto.FLOAT,
                dims=[1],
                vals=cv_list)

            initializers.append(cv)
            inputs.append('constant_value')

        node = onnx.helper.make_node(
            'Pad',
            inputs=inputs,
            outputs=['output'],
            mode=mode)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [input],
        [output],
        initializer=initializers)

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def

in_shapes = [
    [1, 3, 56, 56],
]

paddings = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 1],
]

constant_values = [
    None,
    0.0,
    1.2
]

modes = [
    'constant',
    'reflect',
    'edge'
]

op_versions = [
    # opset 1 is not supported by onnx runntime
    2,
    11,
    13
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('constant_value', constant_values)
@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('op_version', op_versions)
def test_pad(in_shape, padding, constant_value, mode, op_version, request):
    model_def = _make_module(in_shape, padding, constant_value, mode, op_version)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_pad.py'])
