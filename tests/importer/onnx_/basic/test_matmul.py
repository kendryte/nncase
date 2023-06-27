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
import copy


def _make_module(in_a_shape, in_b_shape, b_format):
    inputs = []
    inputs_value_info = []
    outputs = []
    outputs_value_info = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input A
    input_a = helper.make_tensor_value_info('A', TensorProto.FLOAT, in_a_shape)
    inputs.append('A')
    inputs_value_info.append(input_a)

    # input B
    if b_format == 'constant':
        B = helper.make_tensor(
            'B',
            TensorProto.FLOAT,
            dims=in_b_shape,
            vals=np.random.randn(*in_b_shape).astype(np.float32).flatten().tolist()
        )
        initializers.append(B)
        inputs.append('B')
    else:
        input_b = helper.make_tensor_value_info('B', TensorProto.FLOAT, in_b_shape)
        inputs.append('B')
        inputs_value_info.append(input_b)

    # output
    data_a = np.ones(in_a_shape)
    data_b = np.ones(in_b_shape)
    out_shape = np.matmul(data_a, data_b).shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    outputs.append('output')
    outputs_value_info.append(output)

    node = onnx.helper.make_node(
        'MatMul',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        inputs_value_info,
        outputs_value_info,
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


in_shapes = [
    [[6], [6]],
    [[6], [6, 3]],
    [[6], [1, 6, 3]],
    [[6], [3, 6, 3]],
    [[6], [2, 3, 6, 3]],

    [[4, 6], [6]],
    [[4, 1], [1, 3]],
    [[4, 6], [1, 6, 3]],
    [[4, 6], [3, 6, 3]],
    [[4, 6], [2, 3, 6, 3]],

    [[1, 4, 6], [6]],
    [[1, 4, 6], [6, 3]],
    [[1, 4, 6], [1, 6, 3]],
    [[1, 4, 6], [3, 6, 3]],
    [[1, 4, 6], [2, 3, 6, 3]],

    [[3, 4, 6], [6]],
    [[3, 4, 6], [6, 3]],
    [[3, 4, 6], [1, 6, 3]],
    [[3, 4, 6], [3, 6, 3]],
    [[3, 4, 6], [2, 3, 6, 3]],

    [[2, 3, 4, 6], [6]],
    [[2, 3, 4, 6], [6, 3]],
    [[2, 3, 4, 6], [1, 6, 3]],
    [[2, 3, 4, 6], [3, 6, 3]],
    [[2, 3, 4, 6], [2, 3, 6, 3]],

    [[2, 3, 4, 6], [1, 3, 6, 3]],
    [[1, 3, 4, 6], [2, 1, 6, 3]],
    [[3, 2, 3, 4, 6], [1, 1, 3, 6, 3]],
]

b_formats = [
    'constant',
    'non_constant',
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('b_format', b_formats)
def test_matmul(in_shape, b_format, request):
    in_a_shape, in_b_shape = in_shape
    model_def = _make_module(in_a_shape, in_b_shape, b_format)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_matmul.py'])
