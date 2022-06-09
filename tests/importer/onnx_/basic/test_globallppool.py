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

def _make_module(in_type, in_shape, p):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', in_type, in_shape)
    inputs.append('input')

    # output
    data = np.ones(in_shape)

    out_shape = in_shape
    for i in range(2, len(out_shape)):
        out_shape[i] = 1

    output = helper.make_tensor_value_info('output', in_type, out_shape)
    outputs.append('output')

    attributes_dict['p'] = p

    node = onnx.helper.make_node(
        'GlobalLpPool',
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
    TensorProto.FLOAT
]

in_shapes = [
    [1, 3, 16, 17]
]

ps = [
    1,
    2,
    3
]

@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('p', ps)
def test_globallppool(in_type, in_shape, p, request):
    model_def = _make_module(in_type, in_shape, p)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_globallppool.py'])
