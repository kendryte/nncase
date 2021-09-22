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


def _make_module(in_shape):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, in_shape[0])
    inputs.append('input1')

    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, in_shape[1])

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape[0])
    outputs.append('output')

    # shape
    shape_tensor = helper.make_tensor(
        'shape',
        TensorProto.INT64,
        dims=[len(in_shape[0])],
        vals=in_shape[0]
    )
    initializers.append(shape_tensor)

    # Expand
    expand = onnx.helper.make_node(
        'Expand',
        inputs=['input2', 'shape'],
        outputs=['expand_output']
    )
    inputs.append('expand_output')
    nodes.append(expand)

    # Mul
    mul = onnx.helper.make_node(
        'Mul',
        inputs=inputs,
        outputs=outputs,
    )
    nodes.append(mul)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input1, input2],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [[1, 3, 16, 16], [16, 1]]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_meshnet_fusion(in_shape, request):
    model_def = _make_module(in_shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_meshnet_fusion.py'])
