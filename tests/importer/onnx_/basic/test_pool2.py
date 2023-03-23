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
import math


def _make_module(in_shape, kernel_size, stride, padding, count_include_pad, ceil_mode):
    nodes = []
    initializers = []
    inputs = []
    outputs = []

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')
    
    out_shape = in_shape.copy()
    out_shape[2] = (in_shape[2] + padding[0] + padding[2] - kernel_size[0]) // stride[0] + 1 if ceil_mode == 0 else math.ceil((in_shape[2] + padding[0] + padding[2] - kernel_size[0]) / stride[0]) + 1
    out_shape[3] = (in_shape[3] + padding[1] + padding[3] - kernel_size[1]) // stride[1] + 1 if ceil_mode == 0 else math.ceil((in_shape[3] + padding[1] + padding[3] - kernel_size[1]) / stride[1]) + 1
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
		'MaxPool',
		inputs=inputs,
		outputs=outputs,
		kernel_shape=kernel_size,
        strides=stride,
        ceil_mode=ceil_mode,
        pads=padding)

    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    op = onnx.OperatorSetIdProto()
    op.version = 11
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 3, 60, 72],
]

kernel_sizes = [
    (3, 3),
]

strides = [
    (1, 1),
    (2, 2),
    [2, 1]
]

paddings = [
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (1, 1, 1, 2)
]

count_include_pads = [
    False,
    True
]

ceil_modes = [
    False,
    True
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('kernel_size', kernel_sizes)
@pytest.mark.parametrize('stride', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('count_include_pad', count_include_pads)
@pytest.mark.parametrize('ceil_mode', ceil_modes)
def test_pool2(in_shape, kernel_size, stride, padding, count_include_pad, ceil_mode, request):
    if kernel_size[0] / 2 > padding[0] and kernel_size[1] / 2 > padding[1]:
        module = _make_module(in_shape, kernel_size, stride, padding, count_include_pad, ceil_mode)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(module)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_pool2.py'])
