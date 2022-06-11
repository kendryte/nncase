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

import math
import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np


def _make_module(in_type, in_shape, axes, keepdims):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', in_type, in_shape)
    inputs.append('input')

    # output
    kd = 1 if keepdims is None else keepdims
    data = np.ones(in_shape)
    out_shape = np.prod(data, axis=tuple(axes), keepdims=kd).shape
    output = helper.make_tensor_value_info('output', in_type, out_shape)
    outputs.append('output')

    # axes
    attributes_dict['axes'] = axes

    # keepdims
    if keepdims is not None:
        attributes_dict['keepdims'] = keepdims

    node = onnx.helper.make_node(
        'ReduceProd',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    # graph
    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')
    return model_def


in_types = [
    TensorProto.FLOAT,
    TensorProto.INT32
]

# The input data for ReduceProd should be located in a reasonable range, such as [1, 2]
in_shapes = [
    [1, 3, 2, 2]
]

axes_list = [
    [1],
    [2],
    [3],
    [-1],
    [-2],
    [-3],
    [2, 3],
    [-2, -1],
    [1, 2, 3],
    [-1, -2, -3],
    [0, 1, 2, 3],
    [-1, -2, -3, -4]
]

keepdims_lists = [
    None,
    0
]

@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axes', axes_list)
@pytest.mark.parametrize('keepdims', keepdims_lists)
def test_reduce_prod(in_type, in_shape, axes, keepdims, request):
    if len(axes) <= len(in_shape):
        model_def = _make_module(in_type, in_shape, axes, keepdims)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_reduce_prod.py'])
