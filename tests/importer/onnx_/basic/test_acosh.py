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


def _make_module(in_shape):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    outputs.append('output')

    # Abs
    abs = onnx.helper.make_node(
        'Abs',
        inputs=inputs,
        outputs=['abs_output'],
    )
    inputs.append('abs_output')
    nodes.append(abs)

    # Add
    two = helper.make_tensor("two",
                             TensorProto.FLOAT,
                             dims=[1],
                             vals=np.full([1], 2, dtype=np.float32).flatten().tolist())
    initializers.append(two)
    add = onnx.helper.make_node(
        'Add',
        inputs=['abs_output', "two"],
        outputs=['add_output'],
    )
    inputs.append('add_output')
    nodes.append(add)

    # Acosh
    acosh = onnx.helper.make_node(
        'Acosh',
        inputs=['add_output'],
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(acosh)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_acosh(in_shape, request):
    model_def = _make_module(in_shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_acosh.py'])
