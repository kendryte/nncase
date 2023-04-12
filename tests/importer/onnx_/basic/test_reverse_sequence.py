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


def _make_module(in_shape, sequence_len, batch_axis, time_axis):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # sequence_lens
    sequence_lens = helper.make_tensor(
        'sequence_lens',
        TensorProto.INT64,
        dims=[len(sequence_len)],
        vals=sequence_len
    )
    initializers.append(sequence_lens)
    inputs.append('sequence_lens')

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    outputs.append('output')

    # batch_axis
    if batch_axis is not None:
        attributes_dict['batch_axis'] = batch_axis

    # time_axis
    if time_axis is not None:
        attributes_dict['time_axis'] = time_axis

    node = onnx.helper.make_node(
        'ReverseSequence',
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

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


in_shapes = [
    [2, 3, 2, 2]
]

sequence_lens = [
    [1, 1],
    [1, 2],
    [2, 2],
    [3, 3],
    [1, 1, 1],
    [1, 2, 2],
    [2, 2, 2],
]

batch_axes = [
    None,
    0,
    1
]

time_axes = [
    None,
    0,
    1
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('sequence_len', sequence_lens)
@pytest.mark.parametrize('batch_axis', batch_axes)
@pytest.mark.parametrize('time_axis', time_axes)
def test_reverse_sequence(in_shape, sequence_len, batch_axis, time_axis, request):
    if (batch_axis in [None, 1] and time_axis in [None, 0] and in_shape[1] == len(sequence_len)) or (batch_axis == 0 and time_axis == 1 and in_shape[0] == len(sequence_len)):
        model_def = _make_module(in_shape, sequence_len, batch_axis, time_axis)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_reverse_sequence.py'])
