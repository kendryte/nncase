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
import random


def _make_module(in_shape_0, condition, axis=None):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape_0)
    inputs.append('input')

    conditions = helper.make_tensor_value_info('conditions', TensorProto.BOOL, condition)
    inputs.append('conditions')

    x = np.random.rand(*in_shape_0).astype(np.float32)
    condition = np.random.rand(*condition) > .5
    output_shape = np.compress(condition, x, axis=axis).shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    outputs.append('output')
    # output
    attributes_dict['axis'] = axis
    # Cosh
    node = onnx.helper.make_node(
        'Compress',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input, conditions],
        [output],
        initializer=initializers
    )
    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes_0 = [
    # [1],
    # [16],
    # [1, 16],
    # [16, 16],
    [1, 16, 16],


    # [3, 16, 16],
    # [1, 3, 16, 16]
]

condition = [
    # [1],
    [16],
    # [6],

]

axes = [
    # -1,
    1,
    # 1,
    # 2,
    # 3
]


@ pytest.mark.parametrize('in_shape_0', in_shapes_0)
@ pytest.mark.parametrize('condition', condition)
@ pytest.mark.parametrize('axes', axes)
def test_compress(in_shape_0, condition, axes, request):
    model_def = _make_module(in_shape_0, condition, axes)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_compress.py'])
