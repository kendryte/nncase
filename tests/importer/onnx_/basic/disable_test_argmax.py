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
import copy
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, axis, keepdim, select_last_index):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # output
    output_shape = copy.deepcopy(in_shape)
    real_axis = 0 if axis is None else axis
    kd = True if keepdim is None else keepdim
    output_shape[real_axis] = 1
    if not kd:
        output_shape.pop(real_axis)
    output = helper.make_tensor_value_info('output', TensorProto.INT64, output_shape)
    outputs.append('output')

    if axis is not None:
        attributes_dict['axis'] = axis

    if keepdim is not None:
        attributes_dict['keepdims'] = keepdim

    if select_last_index is not None:
        attributes_dict['select_last_index'] = select_last_index

    node = onnx.helper.make_node(
        'ArgMax',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )

    nodes = []
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]

axes = [
    None,
    0,
    1,
    2,
    3,
    -1,
    -2,
    -3,
    -4
]

keepdims = [
    None,
    True,
    False
]

select_last_indices = [
    None,
    True,
    False
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('keepdim', keepdims)
@pytest.mark.parametrize('select_last_index', select_last_indices)
def test_argmax(in_shape, axis, keepdim, select_last_index, request):
    model_def = _make_module(in_shape, axis, keepdim, select_last_index)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_argmax.py'])
