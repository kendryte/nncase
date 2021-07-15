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
"""System test: test onehot"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np

def onnx_make_tensor(name, type, val):
    return helper.make_tensor(name, type, np.array(val).shape, np.array(val).flatten().tolist())

def get_onehot_shape(indices, axis, depth):
    indices_shape = list(np.array(indices).shape)
    # return indices[:axis] + [depth] + indices[axis:]
    return indices_shape[:axis] + [depth] + indices_shape[axis:]

def _make_module(indices, depth, axis):
    out_shape = get_onehot_shape(indices, axis, depth)
    indices = onnx_make_tensor('indices', TensorProto.INT64, indices)
    # indices = helper.make_tensor_value_info('indices', TensorProto.INT64, indices)
    depth = onnx_make_tensor('depth', TensorProto.FLOAT, depth)
    # values = onnx_make_tensor('values', TensorProto.INT64, values)
    values = helper.make_tensor_value_info('values', TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    initializers = [indices, depth]

    node = onnx.helper.make_node(
        'OneHot',
        inputs=['indices', 'depth', 'values'],
        outputs=['output'],
        axis=axis
    )
    graph_def = helper.make_graph(
        [node],
        'test-model',
        [values],
        [output],
        initializer=initializers
    )

    return helper.make_model(graph_def, producer_name='kendryte')

indices_depth_axis = [
    ([3, 2, 4, 0], 5, 0),
    ([3, 2, 4, 0], 5, 1),
    ([[0, 2, 1, 1], [1, 1, 0, 0]], 3, 0),
    ([[0, 2, 1, 1], [1, 1, 0, 0]], 3, 1),
    ([[0, 2, 1, 1], [1, 1, 0, 0]], 3, 2),
    ([[[0, 3], [2, 4], [1, 0]], [[3, 0], [4, 2], [0, 1]]], 5, 0),
    ([[[0, 3], [2, 4], [1, 0]], [[3, 0], [4, 2], [0, 1]]], 5, 1),
    ([[[0, 3], [2, 4], [1, 0]], [[3, 0], [4, 2], [0, 1]]], 5, 2),
    ([[[0, 3], [2, 4], [1, 0]], [[3, 0], [4, 2], [0, 1]]], 5, 3),
    ([[[0, -3], [2, -4], [-1, 0]], [[-3, 0], [4, -2], [0, -1]]], 5, 0),
    ([[[0, -3], [2, -4], [-1, 0]], [[-3, 0], [4, -2], [0, -1]]], 5, 1),
    ([[[0, -3], [2, -4], [-1, 0]], [[-3, 0], [4, -2], [0, -1]]], 5, 2),
    ([[[0, -3], [2, -4], [-1, 0]], [[-3, 0], [4, -2], [0, -1]]], 5, 3),
]

@pytest.mark.parametrize('indices,depth,axis', indices_depth_axis)
def test_onehot(indices, depth, axis, request):
    model_def = _make_module(indices, depth, axis)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_onehot.py'])
