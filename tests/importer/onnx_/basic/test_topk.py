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


def _make_module(in_shape, k, axis, largest, sorted):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # k input
    k_tensor = helper.make_tensor(
        'K',
        TensorProto.INT64,
        dims=[1],
        vals=[k]
    )
    initializers.append(k_tensor)
    inputs.append('K')

    # output
    idx = -1 if axis is None else axis
    out_shape = copy.deepcopy(in_shape)
    out_shape[idx] = k

    output_values = helper.make_tensor_value_info('output_values', TensorProto.FLOAT, out_shape)
    outputs.append('output_values')

    output_indices = helper.make_tensor_value_info('output_indices', TensorProto.INT64, out_shape)
    outputs.append('output_indices')

    # axis
    if axis is not None:
        attributes_dict['axis'] = axis

    # largest
    if largest is not None:
        attributes_dict['largest'] = largest

    # sorted
    if sorted is not None:
        attributes_dict['sorted'] = sorted

    # TopK node
    node = onnx.helper.make_node(
        'TopK',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output_values, output_indices],
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


in_shapes = [
    [2, 3, 16, 16]
]

ks = [
    1,
    2,
    4,
    16
]

axes = [
    None,
    0,
    1,
    2,
    3,
    -1,
    -2,
    -3,
    -4
]

largest_list = [
    None,
    1,
    0
]

sorted_list = [
    None,
    1,
    # we cannot determin the correct sequence without sorting
    # 0
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('k', ks)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('largest', largest_list)
@pytest.mark.parametrize('sorted', sorted_list)
def test_topk(in_shape, k, axis, largest, sorted, request):
    idx = -1 if axis is None else axis
    if (k <= in_shape[idx]):
        model_def = _make_module(in_shape, k, axis, largest, sorted)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_topk.py'])
