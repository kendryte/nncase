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
"""System test: test gather nd"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np


def result_shape(p_shape, i_shape, batch_dims=0):
    if batch_dims < 0:
        batch_dims = len(p_shape) + batch_dims
    return i_shape[:-1] + p_shape[i_shape[-1] + batch_dims:]


def _make_module(in_shape, indices, batch_dims):
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    i_shape = list(np.array(indices).shape)
    indices = helper.make_tensor('indices', TensorProto.INT64, np.array(
        indices).shape, np.array(indices).flatten().tolist())
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, result_shape(in_shape, i_shape, batch_dims))
    initializers = []
    initializers.append(indices)

    node = onnx.helper.make_node(
        'GatherND',
        inputs=['input', 'indices'],
        outputs=['output'],
        batch_dims=batch_dims
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [input],
        [output],
        initializer=initializers
    )

    # todo: support other opset
    op = onnx.OperatorSetIdProto()
    op.version = 12
    return helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])


in_shapes_indices_dim = [
    ([11], [[0], [7], [5]], 0),
    ([3, 5], [[[0, 2], [0, 4]]], 0),
    ([3, 5], [[1], [4], [3]], 1),
    ([2, 3, 1], [[0], [1]], 0),
    ([2, 3, 1], [[[0], [0], [0]], [[0], [0], [0]]], 0),
    ([5, 7, 5], [1, 4, 3], 0),
    ([2, 3, 5], [[0, 1], [1, 0]], 0),
    ([2, 3, 5], [[[0, 4]], [[2, 0]]], 1),
    ([2, 3, 5], [[[4], [3], [0]], [[2], [1], [0]]], 2),
    ([5, 4, 3, 2], [[1, 0, 2], [1, 2, 2]], 0),
    ([5, 5, 7, 7], [[1, 2, 3], [1, 2, 3]], 0),
    ([5, 5, 7, 7], [[1, 2, 3, 1], [1, 2, 3, 1]], 0),
    ([5, 4, 3, 2], [[1, 0, 2], [1, 2, 2]], 0),
    ([2, 4, 3, 5], [[1, 0, 2], [1, 2, 2]], 1),
    ([2, 3, 3, 5], [[[2, 1], [0, 1], [1, 0]], [[0, 1], [2, 2], [1, 1]]], 2),
    ([2, 3, 3, 5], [[[[4], [1], [3]], [[2], [0], [1]], [[4], [2], [3]]],
                    [[[3], [1], [4]], [[1], [0], [2]], [[3], [2], [4]]]], 3)
]


@pytest.mark.parametrize('in_shape,indices,dim', in_shapes_indices_dim)
def test_gather_nd(in_shape, indices, dim, request):
    model_def = _make_module(in_shape, indices, dim)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_gather_nd.py'])
