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


def _make_module(in_shape, minimum, maximum, op_version, value_format):

    inputs = []
    outputs = []
    initializers = []
    nodes = []
    attributes_dict = {}

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    outputs.append('output')

    # opset 1/6
    if op_version == 1 or op_version == 6:
        if minimum is not None:
            attributes_dict['min'] = minimum

        if maximum is not None:
            attributes_dict['max'] = maximum
    else:
        # opset 11/12/13
        if minimum is not None:
            mins = []
            mins.append(minimum)
            min_tensor = helper.make_tensor(
                'min',
                TensorProto.FLOAT,
                dims=[1],
                vals=mins
            )
            if value_format == 'initializer':
                initializers.append(min_tensor)
            else:
                min_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['min'],
                    value=min_tensor)
                nodes.append(min_node)
            inputs.append('min')

        if maximum is not None:
            maxes = []
            maxes.append(maximum)
            max_tensor = helper.make_tensor(
                'max',
                TensorProto.FLOAT,
                dims=[1],
                vals=maxes
            )
            if value_format == 'initializer':
                initializers.append(max_tensor)
            else:
                max_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['max'],
                    value=max_tensor)
                nodes.append(max_node)
            inputs.append('max')

    clip = onnx.helper.make_node(
        'Clip',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(clip)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 3, 8, 8]
]

minimums = [
    None,
    -1.0,
]

maximums = [
    None,
    6.0
]

op_versions_and_value_formats = [
    [1, 'attribute'],
    [6, 'attribute'],
    [11, 'initializer'],
    [11, 'constant'],
    [12, 'initializer'],
    [12, 'constant'],
    [13, 'initializer'],
    [13, 'constant']
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('minimum', minimums)
@pytest.mark.parametrize('maximum', maximums)
@pytest.mark.parametrize('op_versions_and_value_format', op_versions_and_value_formats)
def test_clip(in_shape, minimum, maximum, op_versions_and_value_format, request):
    op_version, value_format = op_versions_and_value_format

    if minimum is None or maximum is None or minimum <= maximum:
        model_def = _make_module(in_shape, minimum, maximum, op_version, value_format)

        runner = OnnxTestRunner(request.node.name, ['k510'])
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_clip.py'])
