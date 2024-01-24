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

def _make_module(in_shape, axis, op_version):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []


    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    data = np.ones(in_shape)
    out_shape = np.transpose(data, axis).shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    outputs.append('output')

    if axis is not None:
        attributes_dict['perm'] = axis


    node = onnx.helper.make_node(
        'Transpose',
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

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='onnx', opset_imports=[op])
    return model_def


in_shapes = [
    [16, 16],
    [3, 16, 16],
    [1, 3, 16, 16]
]

axes = [
    None,
    [0, 1],
    [1, 0],
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [3, 2, 1, 0]
]

op_versions = [
    1,
    13
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('op_version', op_versions)
def test_transpose(in_shape, axis, op_version, request):
    if axis is None or (len(axis) == len(in_shape)):
        model_def = _make_module(in_shape, axis, op_version)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_transpose.py'])
