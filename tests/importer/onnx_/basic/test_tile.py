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


def _make_module(in_shape, axis, repeat, op_version):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # other inputs according to op_version
    if (op_version == 1):
        # tile
        tiles_tensor = helper.make_tensor(
            'tiles',
            TensorProto.FLOAT,
            dims=[1],
            vals=repeat
        )
        inputs.append('tiles')
        tiles_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['tiles'],
            value=tiles_tensor
        )
        nodes.append(tiles_node)

        # axis
        axis_tensor = helper.make_tensor(
            'axis',
            TensorProto.FLOAT,
            dims=[1],
            vals=axis
        )
        inputs.append('axis')
        axis_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['axis'],
            value=axis_tensor
        )
        nodes.append(axis_node)

    else:
        # op_version 6/11
        # repeats
        repeats_tensor = helper.make_tensor(
            'repeats',
            TensorProto.INT64,
            dims=[len(repeat)],
            vals=repeat
        )
        initializers.append(repeats_tensor)
        inputs.append('repeats')

    # output
    if op_version == 1:
        out_shape = copy.deepcopy(in_shape)
        out_shape[axis[0]] *= repeat[0]
    else:
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

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='onnx', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]

axes = [
    [0],
    [1],
    [2],
    [3]
]

repeats = [
    [1],
    [2],
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 3, 2],
    [1, 2, 3, 2],
    [3, 2, 3, 2],
]

op_versions = [
    # NOT_IMPLEMENTED : Could not find an implementation for the node Tile(1)
    # 1,
    6,
    11
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('repeat', repeats)
@pytest.mark.parametrize('op_version', op_versions)
def test_tile(in_shape, axis, repeat, op_version, request):
    if ((op_version == 1 and len(repeat) == 1) or (op_version in [6, 11] and len(in_shape) == len(repeat))):
        model_def = _make_module(in_shape, axis, repeat, op_version)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_tile.py'])
