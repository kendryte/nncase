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


def _make_module(in_shape, input_dtype, output_dtype, mean, scale, seed):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', input_dtype, in_shape)
    inputs.append('input')

    # output
    dtype = input_dtype if output_dtype is None else output_dtype
    output = helper.make_tensor_value_info('output', dtype, in_shape)
    outputs.append('output')

    # dtype
    if output_dtype is not None:
        attributes_dict['dtype'] = output_dtype

    # mean
    if mean is not None:
        attributes_dict['mean'] = mean

    # scale
    if scale is not None:
        attributes_dict['scale'] = scale

    # seed
    if seed is not None:
        attributes_dict['seed'] = seed

    # RandomNormalLike node
    rnl_output = helper.make_tensor_value_info('rnl_output', dtype, in_shape)
    rnl = onnx.helper.make_node(
        'RandomNormalLike',
        inputs=['input'],
        outputs=['rnl_output'],
        **attributes_dict
    )
    nodes.append(rnl)

    # add node
    add = onnx.helper.make_node(
        'Add',
        inputs=['input', 'rnl_output'],
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


in_shapes = [
    [1, 3, 16, 16]
]

input_dtypes = [
    TensorProto.FLOAT,
]

output_dtypes = [
    None,
    TensorProto.FLOAT,
]

means = [
    None,
    1.0,
]

scales = [
    None,
    2.0,
]

seeds = [
    # None will lead to generate different random number and cannot be compared with onnx runtime
    # None,
    1.0,
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('input_dtype', input_dtypes)
@pytest.mark.parametrize('output_dtype', output_dtypes)
@pytest.mark.parametrize('mean', means)
@pytest.mark.parametrize('scale', scales)
@pytest.mark.parametrize('seed', seeds)
def test_random_normal_like(in_shape, input_dtype, output_dtype, mean, scale, seed, request):
    model_def = _make_module(in_shape, input_dtype, output_dtype, mean, scale, seed)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_random_normal_like.py'])
