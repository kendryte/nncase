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

def _make_module(in_shape, repeat):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # repeats
    rep = helper.make_tensor(
        'repeats',
        TensorProto.INT64,
        dims=[len(repeat)],
        vals=repeat
    )
    initializers.append(rep)
    inputs.append('repeats')

    # output
    out_shape = np.tile(np.ones(in_shape), tuple(repeat)).shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        'Tile',
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

in_shapes = [
    [1, 3, 16, 16]
]

repeats = [
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 3, 2],
    [1, 2, 3, 2],
    [3, 2, 3, 2],
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('repeat', repeats)
def test_tile(in_shape, repeat, request):
    model_def = _make_module(in_shape, repeat)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_tile.py'])
