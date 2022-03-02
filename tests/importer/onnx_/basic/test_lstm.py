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

def _make_module(direction, hidden_size, seq_length, batch_size, input_size, bias, sequence_lens, initial_h, initial_c, Y, Y_h, Y_c):
    nodes_inputs = []
    nodes_outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []
    graph_inputs = []
    graph_outputs = []

    num_directions = 2 if direction == 'bidirectional' else 1
    if direction is not None:
        attributes_dict['direction'] = direction
    attributes_dict['hidden_size'] = hidden_size

    # input
    input_shape = [seq_length, batch_size, input_size]
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    nodes_inputs.append('input')
    graph_inputs.append(input)

    w_shape = [num_directions, 4 * hidden_size, input_size]
    w_tensor = helper.make_tensor(
        'W',
        TensorProto.FLOAT,
        dims=w_shape,
        vals=np.random.rand(*w_shape).astype(np.float32).flatten().tolist()
    )
    nodes_inputs.append('W')
    initializers.append(w_tensor)

    r_shape = [num_directions, 4 * hidden_size, hidden_size]
    r_tensor = helper.make_tensor(
        'R',
        TensorProto.FLOAT,
        dims=r_shape,
        vals=np.random.rand(*r_shape).astype(np.float32).flatten().tolist()
    )
    nodes_inputs.append('R')
    initializers.append(r_tensor)

    # bias
    if bias is None:
        nodes_inputs.append('')
    else:
        bias_shape = [num_directions, 8 * hidden_size]
        bias_tensor = helper.make_tensor(
            'B',
            TensorProto.FLOAT,
            dims=bias_shape,
            vals=np.random.rand(*bias_shape).astype(np.float32).flatten().tolist()
        )
        nodes_inputs.append('B')
        initializers.append(bias_tensor)

    if sequence_lens is None:
        nodes_inputs.append('')
    else:
        sequence_lens_shape = [batch_size]
        sequence_lens_tensor = helper.make_tensor(
            'sequence_lens',
            TensorProto.INT32,
            dims=sequence_lens_shape,
            vals=np.full(sequence_lens_shape, seq_length).flatten().tolist()
        )
        nodes_inputs.append('sequence_lens')
        initializers.append(sequence_lens_tensor)

    if initial_h is None:
        nodes_inputs.append('')
    else:
        initial_h_shape = [num_directions, batch_size, hidden_size]
        initial_h_tensor = helper.make_tensor(
            'initial_h',
            TensorProto.FLOAT,
            dims=initial_h_shape,
            vals=np.random.rand(*initial_h_shape).astype(np.float32).flatten().tolist()
        )
        nodes_inputs.append('initial_h')
        initializers.append(initial_h_tensor)

    if initial_c is None:
        nodes_inputs.append('')
    else:
        initial_c_shape = [num_directions, batch_size, hidden_size]
        initial_c_tensor = helper.make_tensor(
            'initial_c',
            TensorProto.FLOAT,
            dims=initial_c_shape,
            vals=np.random.rand(*initial_c_shape).astype(np.float32).flatten().tolist()
        )
        nodes_inputs.append('initial_c')
        initializers.append(initial_c_tensor)

    # output
    if Y is None:
        nodes_outputs.append('')
    else:
        output_shape = [seq_length, num_directions, batch_size, hidden_size]
        output = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)
        nodes_outputs.append('Y')
        graph_outputs.append(output)

    if Y_h is None:
        nodes_outputs.append('')
    else:
        h_shape = [num_directions, batch_size, hidden_size]
        y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, h_shape)
        nodes_outputs.append('Y_h')
        graph_outputs.append(y_h)

    if Y_c is None:
        nodes_outputs.append('')
    else:
        c_shape = [num_directions, batch_size, hidden_size]
        y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT, c_shape)
        nodes_outputs.append('Y_c')
        graph_outputs.append(y_c)

    # lstm node
    node = onnx.helper.make_node(
        'LSTM',
        inputs=nodes_inputs,
        outputs=nodes_outputs,
        **attributes_dict
    )
    nodes.append(node)

    # graph
    graph_def = helper.make_graph(
        nodes,
        'test-model',
        graph_inputs,
        graph_outputs,
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def

directions = [
    None,
    'forward',
    'reverse',
    'bidirectional'
]

hidden_sizes = [
    16
]

seq_lengths = [
    8
]

batch_sizes = [
    1,
]

input_sizes = [
    16,
]

biases = [
    None,
    1
]

sequence_lenses = [
    None,
]

initial_hs = [
    None,
    1
]

initial_cs = [
    None,
    1
]

Ys = [
    # None,
    1
]


Y_hs = [
    None,
    1
]

Y_cs = [
    None,
    1
]

@pytest.mark.parametrize('direction', directions)
@pytest.mark.parametrize('hidden_size', hidden_sizes)
@pytest.mark.parametrize('seq_length', seq_lengths)
@pytest.mark.parametrize('batch_size', batch_sizes)
@pytest.mark.parametrize('input_size', input_sizes)
@pytest.mark.parametrize('bias', biases)
@pytest.mark.parametrize('sequence_lens', sequence_lenses)
@pytest.mark.parametrize('initial_h', initial_hs)
@pytest.mark.parametrize('initial_c', initial_cs)
@pytest.mark.parametrize('Y', Ys)
@pytest.mark.parametrize('Y_h', Y_hs)
@pytest.mark.parametrize('Y_c', Y_cs)
def test_lstm(direction, hidden_size, seq_length, batch_size, input_size, bias, sequence_lens, initial_h, initial_c, Y, Y_h, Y_c, request):
    model_def = _make_module(direction, hidden_size, seq_length, batch_size, input_size, bias, sequence_lens, initial_h, initial_c, Y, Y_h, Y_c)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_lstm.py'])
