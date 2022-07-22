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


def _make_module(in_shape, expand_shape, value_format):

    inputs = []
    outputs = []
    initializers = []
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # output
    out = np.ones(in_shape) * np.ones(expand_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out.shape)
    outputs.append('output')

    # shape
    shape_tensor = helper.make_tensor(
        'shape',
        TensorProto.INT64,
        dims=[len(expand_shape)],
        vals=expand_shape
    )

    if value_format == 'initializer':
        initializers.append(shape_tensor)
    else:
        shape_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['shape'],
            value=shape_tensor
        )
        nodes.append(shape_node)
    inputs.append('shape')

    # Expand
    expand = onnx.helper.make_node(
        'Expand',
        inputs=inputs,
        outputs=outputs,
    )
    nodes.append(expand)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [3, 1]
]

expand_shapes = [
    [1],
    [1, 1],
    [3, 4],
    [2, 1, 6]
]

value_formats = [
    ['initializer'],
    ['constant']
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('expand_shape', expand_shapes)
@pytest.mark.parametrize('value_format', value_formats)
def test_expand(in_shape, expand_shape, value_format, request):
    model_def = _make_module(in_shape, expand_shape, value_format)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_expand.py'])
