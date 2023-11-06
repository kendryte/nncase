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
import random


def _make_module(in_shape_0, condition_shape, axis=None):
    inputs = []
    outputs = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape_0)
    inputs.append('input')

    # output
    x = np.random.rand(*in_shape_0).astype(np.float32)
    condition = np.array(np.random.rand(*condition_shape) > .5).astype(np.bool_)
    if(condition.sum() == 0):
        print(condition.sum())
        condition[-1] = True

    output_shape = np.compress(condition, x, axis=axis).shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    outputs.append('output')

    condi_data = helper.make_tensor(
        'condi_Constant',
        TensorProto.BOOL,
        dims=condition_shape,
        vals=condition.astype(np.bool).flatten()
    )
    weights_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["condi"],
        value=condi_data,
        name="condition")

    nodes.append(weights_constant)
    if axis != None:
        attributes_dict['axis'] = axis
    node = helper.make_node(
        'Compress',
        inputs=['input', 'condi'],
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes_0 = [
    [1],
    [16],
    [1, 16],
    [16, 16],
    [1, 15, 16],
    [1, 3, 3, 3]
]

condition = [
    [1],
    [3],
    [6],

]

axes = [
    None,
    -1,
    0,
    1,
    2,
    3
]


@pytest.mark.parametrize('in_shape_0', in_shapes_0)
@pytest.mark.parametrize('condition', condition)
@pytest.mark.parametrize('axes', axes)
def test_compress(in_shape_0, condition, axes, request):
    size = 1
    for x in in_shape_0:
        size *= x
    if((axes != None and axes < len(in_shape_0) and condition[0] <= in_shape_0[axes])
            or (axes == None and condition[0] <= size)):
        model_def = _make_module(in_shape_0, condition, axes)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_compress.py'])
