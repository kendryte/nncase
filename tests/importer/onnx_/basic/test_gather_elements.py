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
"""System test: test gather"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np


def _make_module(in_shape, index, axis):
    initializers = []
    attributes_dict = {}

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)

    # i_shape = list(np.array(indices).shape)
    indices = helper.make_tensor(
        'indices',
        TensorProto.INT32,
        np.array(index).shape,
        np.array(index).astype(np.int32).flatten())
    initializers.append(indices)

    # axis
    if axis is not None:
        default_axis = axis
    else:
        default_axis = 0
    attributes_dict['axis'] = default_axis

    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, np.array(index).shape)

    node = onnx.helper.make_node(
        'GatherElements',
        inputs=['input', 'indices'],
        outputs=['output'],
        **attributes_dict
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [input],
        [output],
        initializer=initializers
    )

    return helper.make_model(graph_def, producer_name='onnx')


# input_shape, indices_data, axis
# input_shape[i] >= indices_data.shape[i]
in_shapes_indices_dim = [
    ([2, 2], [[0, 0], [1, 0]], 0),
    ([2, 3], [[0, 2], [1, 0]], 1),
    ([2, 3], [[0, -2], [-1, 0]], 1),
    ([1, 3, 3], [[[1, 2, 0], [2, 0, 0]]], 2),
    ([1, 3, 3], [[[1, 2, 0], [2, 0, 0]]], 1),
    ([4, 2, 3], [[[1, 2, 0], [2, 0, 0]], [[3, 2, 1], [2, 3, 1]]], 0),
    ([1, 5, 3], [[[1, 2, 0], [2, 0, 0]]], 1),
    ([2, 5, 6], [[[1, 2, 0], [2, 0, 0]], [[3, 2, 1], [2, 3, 1]]], 2),

]


@pytest.mark.parametrize('in_shape, indices, axis', in_shapes_indices_dim)
def test_gather_elements(in_shape, indices, axis, request):
    model_def = _make_module(in_shape, indices, axis)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_gather_elements.py'])
