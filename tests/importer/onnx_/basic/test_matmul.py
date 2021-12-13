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

def _make_module(in_a_shape, in_b_shape):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input A
    input = helper.make_tensor_value_info('A', TensorProto.FLOAT, in_a_shape)
    inputs.append('A')

    # input B
    B = helper.make_tensor(
        'B',
        TensorProto.FLOAT,
        dims=in_b_shape,
        vals=np.random.randn(*in_b_shape).astype(np.float32).flatten().tolist()
    )
    initializers.append(B)
    inputs.append('B')

    # output
    data_a = np.ones(in_a_shape)
    data_b = np.ones(in_b_shape)
    out_shape = np.matmul(data_a, data_b).shape

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    outputs.append('output')

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
        [input],
        [output],
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def

in_a_shapes = [
    [16],
    [16, 16],
    [1, 3, 16, 16]
]

in_b_shapes = [
    [16],
    [16, 16]
]

@pytest.mark.parametrize('in_a_shape', in_a_shapes)
@pytest.mark.parametrize('in_b_shape', in_b_shapes)
def test_matmul(in_a_shape, in_b_shape, request):
    model_def = _make_module(in_a_shape, in_b_shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_matmul.py'])
