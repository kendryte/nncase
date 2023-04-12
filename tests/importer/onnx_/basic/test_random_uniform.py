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


def _make_module(dtype, low, high, seed, shape):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', dtype, shape)
    inputs.append('input')

    # output
    output = helper.make_tensor_value_info('output', dtype, shape)
    outputs.append('output')

    # dtype
    if dtype is not None:
        attributes_dict['dtype'] = dtype

    # low
    if low is not None:
        attributes_dict['low'] = low

    # high
    if high is not None:
        attributes_dict['high'] = high

    # seed
    if seed is not None:
        attributes_dict['seed'] = seed

    # shape
    if shape is not None:
        attributes_dict['shape'] = shape

    # RandomUniform node
    ru_output = helper.make_tensor_value_info('ru_output', dtype, shape)
    ru = onnx.helper.make_node(
        'RandomUniform',
        inputs=[],
        outputs=['ru_output'],
        **attributes_dict
    )
    nodes.append(ru)

    # add node
    add = onnx.helper.make_node(
        'Add',
        inputs=['input', 'ru_output'],
        outputs=['output'],
    )
    nodes.append(add)

    # graph
    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


dtypes = [
    TensorProto.FLOAT,
]

lows = [
    None,
    1.0,
]

highs = [
    None,
    2.0,
]

seeds = [
    # None will lead to generate different random number and cannot be compared with onnx runtime
    # None,
    1.0,
]

shapes = [
    [1, 3, 16, 16]
]


@pytest.mark.parametrize('dtype', dtypes)
@pytest.mark.parametrize('low', lows)
@pytest.mark.parametrize('high', highs)
@pytest.mark.parametrize('seed', seeds)
@pytest.mark.parametrize('shape', shapes)
def test_random_uniform(dtype, low, high, seed, shape, request):
    model_def = _make_module(dtype, low, high, seed, shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_random_uniform.py'])
