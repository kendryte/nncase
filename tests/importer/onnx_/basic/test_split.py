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


def _make_module(in_shape, axis, split, output_size, op_version, value_format):
    input_names = []
    output_names = []
    inputs = []
    outputs = []
    initializers = []
    nodes = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    input_names.append('input')
    inputs.append(input)

    # output
    out_shape = copy.deepcopy(in_shape)
    dim = axis if axis is not None else 0
    for i in range(output_size):
        output_name = 'output_{0}'.format(i)
        output_names.append(copy.deepcopy(output_name))
        out_shape[dim] = split[i] if split is not None else in_shape[dim] // output_size
        output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, out_shape)
        outputs.append(copy.deepcopy(output))

    if axis is not None:
        attributes_dict['axis'] = axis

    # opset 1/2/11
    if op_version == 1 or op_version == 2 or op_version == 11:
        if split is not None:
            attributes_dict['split'] = split
    else:
        # opset 13
        if split is not None:
            split_tensor = helper.make_tensor(
                'split',
                TensorProto.INT64,
                dims=[len(split)],
                vals=split
            )

            if value_format == 'initializer':
                initializers.append(split_tensor)
            else:
                const_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['split'],
                    value=split_tensor
                )
                nodes.append(const_node)

            input_names.append('split')

    split_node = onnx.helper.make_node(
        'Split',
        inputs=input_names,
        outputs=output_names,
        **attributes_dict
    )
    nodes.append(split_node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        inputs,
        outputs,
        initializer=initializers
    )

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 4, 8, 8]
]

axes = [
    None,
    -2,
    1
]

splits = [
    None,
    [2, 2],
    [1, 1, 2],
    [1, 2, 1],
    [3, 2, 2, 1],
    [2, 1, 1, 1,3],
]

output_sizes = [
    2,
    3,
    4,
    5
]

op_versions_and_value_formats = [
    # opset 1 is not supported by onnx runtime yet.
    # [1, 'attribute'],
    [2, 'attribute'],
    [11, 'attribute'],
    [13, 'initializer'],
    [13, 'constant']
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('split', splits)
@pytest.mark.parametrize('output_size', output_sizes)
@pytest.mark.parametrize('op_versions_and_value_format', op_versions_and_value_formats)
def test_split(in_shape, axis, split, output_size, op_versions_and_value_format, request):
    op_version, value_format = op_versions_and_value_format
    dim = axis if axis is not None else 0
    if (split is None and in_shape[dim] % output_size == 0 ) or (split is not None and len(split) == output_size and sum(split) == in_shape[dim]):
        model_def = _make_module(in_shape, axis, split, output_size, op_version, value_format)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_split.py'])
