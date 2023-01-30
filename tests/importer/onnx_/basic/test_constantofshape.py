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
from onnx import AttributeProto, TensorProto, GraphProto, numpy_helper
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, value):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}

    # input
    input_const = helper.make_tensor(
        'input',
        TensorProto.INT64,
        dims=[len(in_shape)],
        vals=in_shape
    )
    inputs.append('input')
    initializers.append(input_const)

    type = TensorProto.FLOAT
    if value is not None:
        type = value[0]
        tensor = onnx.helper.make_tensor("value", type, [1], [value[1]])
        attributes_dict['value'] = tensor

    # output
    output = helper.make_tensor_value_info('output', type, in_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )

    nodes = []
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]

values = [
    None,
    [TensorProto.FLOAT, 0],
    [TensorProto.FLOAT, 1],
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('value', values)
def test_constantofshape(in_shape, value, request):
    model_def = _make_module(in_shape, value)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_constantofshape.py'])
