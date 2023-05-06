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
from tflite_test_runner import TfliteTestRunner
import numpy as np


def _make_module(bc_type, in_shape_0, in_shape_1, in_shape_2):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input0 = helper.make_tensor_value_info('input0', TensorProto.BOOL, in_shape_0)
    inputs.append('input0')

    input1 = helper.make_tensor_value_info('input1', bc_type, in_shape_1)
    inputs.append('input1')

    input2 = helper.make_tensor_value_info('input2', bc_type, in_shape_2)
    inputs.append('input2')

    # output
    x = (np.random.randn(*in_shape_0) > 0).astype(bool)
    y = np.zeros(in_shape_1)
    z = np.zeros(in_shape_2)
    output_shape = np.where(x, y, z).shape
    output = helper.make_tensor_value_info('output', bc_type, output_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        'Where',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input0, input1, input2],
        [output],
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


in_shapes_0 = [
    [1],
    [16],
    [1, 16],
    [16, 16],
    [1, 16, 16],
    [3, 16, 16],
    [1, 3, 16, 16]
]

in_shapes_1 = [
    [1],
    [16],
    [1, 16],
    [16, 16],
    [1, 16, 16],
    [3, 16, 16],
    [1, 3, 16, 16]
]

in_shapes_2 = [
    [1],
    [16],
    [1, 16],
    [16, 16],
    [1, 16, 16],
    [3, 16, 16],
    [1, 3, 16, 16]
]

bc_types = [
    TensorProto.FLOAT
]


@pytest.mark.parametrize('in_shape_0', in_shapes_0)
@pytest.mark.parametrize('in_shape_1', in_shapes_1)
@pytest.mark.parametrize('in_shape_2', in_shapes_2)
@pytest.mark.parametrize('bc_type', bc_types)
def test_where(in_shape_0, in_shape_1, in_shape_2, bc_type, request):
    model_def = _make_module(bc_type, in_shape_0, in_shape_1, in_shape_2)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_where.py'])
