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


def _make_module(in_shape, scale):
    inputs = []
    outputs = []
    initializers = []
    nodes = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # reshape1
    scales = helper.make_tensor(
        'scales',
        TensorProto.FLOAT,
        dims=[len(scale)],
        vals=scale
    )
    initializers.append(scales)
    inputs.append('scales')

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, list(
        [math.floor(in_shape[i] * scale[i]) for i in range(len(in_shape))]))
    outputs.append('output')

    # upsample
    node = onnx.helper.make_node(
        'Upsample',
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
    op.version = 9
    model_def = helper.make_model(graph_def, producer_name='onnx', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 2, 3, 4],
]

scales = [
    [1, 1, 2, 2],
    [1, 1, 3, 4]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('scale', scales)
def test_upsample(in_shape, scale, request):
    model_def = _make_module(in_shape, scale)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_upsample.py'])
