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

def _make_module(in_type, in_shape):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', in_type, in_shape)
    inputs.append('input')

    # output
    x = np.random.randn(*in_shape)
    for cell in np.nditer(x, op_flags=['readwrite']):
        if (cell > 0.5):
            cell[...] = float("inf")
    output_shape = in_shape
    output = helper.make_tensor_value_info('output', TensorProto.BOOL, output_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        'IsInf',
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
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def

in_types = [
    TensorProto.FLOAT,
]

in_shapes = [
    [1],
    [16],
    [1, 16],
    [1, 16, 16],
    [3, 3, 1, 16],
    [1, 1, 16, 16],
    [1, 3, 16, 16]
]

# pytest with multiprocess will fail on uppercase names
@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
def test_isinf(in_type, in_shape, request):
    model_def = _make_module(in_type, in_shape)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_isinf.py'])
