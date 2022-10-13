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


def _make_module(in_shape, kernel_output_channel, bias_shape, auto_pad_mode, dilation, group, kernel_shape, pad, stride):
    inputs = []
    initializers = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # weight
    w_shape = []
    w_shape.append(kernel_output_channel)
    w_shape.append(in_shape[1])
    w_shape.extend(kernel_shape)
    weight = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        dims=w_shape,
        vals=np.random.rand(*w_shape).astype(np.float32).flatten().tolist()
    )
    inputs.append('weight')
    initializers.append(weight)

    # bias
    if bias_shape is not None:
        bias = helper.make_tensor(
            'bias',
            TensorProto.FLOAT,
            dims=bias_shape,
            vals=np.random.rand(*bias_shape).astype(np.float32).flatten().tolist()
        )
        inputs.append('bias')
        initializers.append(bias)

    # dilation
    d = [1, 1] if dilation is None else dilation

    # stride
    s = [1, 1] if stride is None else stride

    # pad
    padding = [0, 0, 0, 0]
    if (auto_pad_mode is None or auto_pad_mode == 'NOTSET') and pad is not None:
        padding = pad
    elif auto_pad_mode == 'SAME_UPPER':
        output_h = math.floor((in_shape[2] + s[0] - 1) / s[0])
        pad_nums = max(0, (output_h - 1) * s[0] + (w_shape[2] - 1) * d[0] + 1 - in_shape[2])
        padding[0] = math.floor(pad_nums / 2)
        padding[2] = pad_nums - padding[0]

        output_w = math.floor((in_shape[3] + s[1] - 1) / s[1])
        pad_nums = max(0, (output_w - 1) * s[1] + (w_shape[3] - 1) * d[1] + 1 - in_shape[3])
        padding[1] = math.floor(pad_nums / 2)
        padding[3] = pad_nums - padding[1]
    elif auto_pad_mode == 'SAME_LOWER':
        output_h = math.floor((in_shape[2] + s[0] - 1) / s[0])
        pad_nums = max(0, (output_h - 1) * s[0] + (w_shape[2] - 1) * d[0] + 1 - in_shape[2])
        padding[0] = math.ceil(pad_nums / 2)
        padding[2] = pad_nums - padding[0]

        output_w = math.floor((in_shape[3] + s[1] - 1) / s[1])
        pad_nums = max(0, (output_w - 1) * s[1] + (w_shape[3] - 1) * d[1] + 1 - in_shape[3])
        padding[1] = math.ceil(pad_nums / 2)
        padding[3] = pad_nums - padding[1]

    # output
    out_shape = []
    out_shape.append(in_shape[0])
    out_shape.append(w_shape[0])
    out_shape.append(math.floor(
        (in_shape[2] + padding[0] + padding[2] - ((w_shape[2] - 1) * d[0] + 1) + s[0]) / s[0]))
    out_shape.append(math.floor(
        (in_shape[3] + padding[1] + padding[3] - ((w_shape[3] - 1) * d[1] + 1) + s[1]) / s[1]))
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)

    attributes_dict = {}

    if auto_pad_mode is not None:
        attributes_dict['auto_pad'] = auto_pad_mode

    if dilation is not None:
        attributes_dict['dilations'] = dilation

    if group is not None:
        attributes_dict['group'] = group

    if kernel_shape is not None:
        attributes_dict['kernel_shape'] = kernel_shape

    if pad is not None:
        attributes_dict['pads'] = padding

    if stride is not None:
        attributes_dict['strides'] = stride

    node = onnx.helper.make_node(
        'Conv',
        inputs=inputs,
        outputs=['output'],
        **attributes_dict
    )

    nodes = []
    nodes.append(node)

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

kernel_output_channels = [
    2
]

bias_shapes = [
    None,
]
bias_shapes.extend(list([[x] for x in kernel_output_channels]))

auto_pad_modes = [
    None,
    'NOTSET',
    'SAME_UPPER',
    'SAME_LOWER',
    'VALID'
]

dilations = [
    None,
    [2, 2]
]

groups = [
    None,
    1
]

kernel_shapes = [
    [1, 1],
    [3, 3],
]

pads = [
    None,
    [0, 0, 1, 0],
]

strides = [
    None,
    [2, 2]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('kernel_output_channel', kernel_output_channels)
@pytest.mark.parametrize('bias_shape', bias_shapes)
@pytest.mark.parametrize('auto_pad_mode', auto_pad_modes)
@pytest.mark.parametrize('dilation', dilations)
@pytest.mark.parametrize('group', groups)
@pytest.mark.parametrize('kernel_shape', kernel_shapes)
@pytest.mark.parametrize('pad', pads)
@pytest.mark.parametrize('stride', strides)
def test_conv(in_shape, kernel_output_channel, bias_shape, auto_pad_mode, dilation, group, kernel_shape, pad, stride, request):
    if (bias_shape is None or (bias_shape is not None and bias_shape[0] == kernel_output_channel)) and ((auto_pad_mode is not None and pad is None) or (auto_pad_mode is None and pad is not None)) and (dilation is None or auto_pad_mode is None or auto_pad_mode == 'NOTSET'):
        model_def = _make_module(in_shape, kernel_output_channel, bias_shape,
                                 auto_pad_mode, dilation, group, kernel_shape, pad, stride)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv.py'])
