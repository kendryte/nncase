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
from test_runner import OnnxTestRunner
import numpy as np

def _make_module(in_shape, axes, op_version):

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, in_shape)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, in_shape)
    add = onnx.helper.make_node(
        'Add',
        inputs=['x', 'y'],
        outputs=['sum'],
    )

    initializers = []

    # infer output shape
    out = np.ones(in_shape)
    out_shape = np.expand_dims(out, axis=axes).shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)

    # unsqueeze-1 and unsqueeze-11
    if op_version == 11:
        unsqueeze = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['sum'],
            outputs=['output'],
            axes=axes
        )
    else:
        # unsqueeze-13
        axes_len = []
        axes_len.append(len(axes))
        axes = onnx.helper.make_tensor('axes', onnx.TensorProto.INT64, axes_len, np.array(axes).astype(np.int64))
        initializers.append(axes)
        unsqueeze = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['sum', 'axes'],
            outputs=['output'],
        )

    graph_def = helper.make_graph(
        [add, unsqueeze],
        'test-model',
        [x, y],
        [output],
        initializer=initializers
    )

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def

in_shapes = [
    [224],
    [224, 224],
    [3, 224, 224]
]

axes_list = [
    [0],
    [1],
    [2],
    [3],
    [-1],
    [-2],
    [-3],
    [-4],
    [0, 1],
    [0, 2],
    [1, -1],
    [-2, -1],
    [2, 1],
]

op_versions = [
    11,
    13
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axes', axes_list)
@pytest.mark.parametrize('op_version', op_versions)
def test_unsqueeze(in_shape, axes, op_version, request):
    if len(in_shape) + len(axes) == 4:
        model_def = _make_module(in_shape, axes, op_version)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_unsqueeze.py'])