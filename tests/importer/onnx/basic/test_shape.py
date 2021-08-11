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


def _make_module(in_shape, shape, start, stop):
    inputs = []
    outputs = []
    initializers = []
    nodes = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)

    # abs
    abs = onnx.helper.make_node(
        'Abs',
        inputs=['input'],
        outputs=['abs_output'],
    )
    nodes.append(abs)
    inputs.append('abs_output')

    # reshape1
    new_shape = helper.make_tensor(
        'new_shape',
        TensorProto.INT64,
        dims=[len(shape)],
        vals=shape
    )
    initializers.append(new_shape)

    reshape1 = onnx.helper.make_node(
        'Reshape',
        inputs=['input', 'new_shape'],
        outputs=['reshape1_output'],
    )
    nodes.append(reshape1)

    # shape
    begin = 0
    end = len(shape)
    if start is not None:
        begin = start
        attributes_dict['start'] = start

    if stop is not None:
        end = stop
        attributes_dict['end'] = stop

    slice_shape = shape[begin:end]

    shape = onnx.helper.make_node(
        'Shape',
        inputs=['reshape1_output'],
        outputs=['shape_output']
    )
    nodes.append(shape)
    inputs.append('shape_output')

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, slice_shape)
    outputs.append('output')

    # reshape2
    reshape2 = onnx.helper.make_node(
        'Reshape',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(reshape2)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 2, 3, 4]
]

shapes = [
    [1, 2, 12]
]

# disable it as opset 15 is not supported by onnxruntime now
start_and_ends = [
    [None, None],
    # [None, -1],
    # [0, None],
    # [0, 3],
    # [1, 3]
]

# Note: To test the Shape test case, disable onnxsim.simplify


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('shape', shapes)
@pytest.mark.parametrize('start_and_end', start_and_ends)
def test_shape(in_shape, shape, start_and_end, request):
    start, end = start_and_end
    model_def = _make_module(in_shape, shape, start, end)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_shape.py'])
