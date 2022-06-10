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

def _make_module(in_type, in_shape, auto_pad, kernel_shape, p, pads, strides):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', in_type, in_shape)
    inputs.append('input')

    # output
    data = np.ones(in_shape)

    out_shape = in_shape
    if (auto_pad == 'NOTSET'):
        out_shape[2] = (int)((in_shape[2] + pads[0] + pads[2] - kernel_shape[0]) / strides[0] + 1)
        out_shape[3] = (int)((in_shape[3] + pads[1] + pads[3] - kernel_shape[1]) / strides[1] + 1)
    elif (auto_pad == 'VALID'):
        out_shape[2] = math.ceil(float((in_shape[2] - kernel_shape[0] + 1)) / strides[0])
        out_shape[3] = math.ceil(float((in_shape[3] - kernel_shape[1] + 1)) / strides[1])
    elif (auto_pad == 'SAME_UPPER' or auto_pad == 'SAME_LOWER'):
        out_shape[2] = math.ceil(float(in_shape[2]) / strides[0])
        out_shape[3] = math.ceil(float(in_shape[3]) / strides[1])

    output = helper.make_tensor_value_info('output', in_type, out_shape)
    outputs.append('output')

    attributes_dict['auto_pad'] = auto_pad
    attributes_dict['kernel_shape'] = kernel_shape
    attributes_dict['p'] = p
    if (auto_pad == 'NOTSET'):
        attributes_dict['pads'] = pads
    attributes_dict['strides'] = strides

    node = onnx.helper.make_node(
        'LpPool',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    # graph
    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')
    return model_def

in_types = [
    TensorProto.FLOAT
]

in_shapes = [
    [1, 3, 16, 17]
]

auto_pads = [
    'NOTSET',
    'SAME_UPPER',
    'SAME_LOWER',
    #'VALID'
]

kernel_shapes = [
    [3, 3]
]

ps = [
    1,
    2
]

pads = [
    [0, 0, 2 ,2],
    [1, 1, 1, 1],
]

strides = [
    [1, 1],
    [2, 2],
    [1, 2]
]

@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('auto_pad', auto_pads)
@pytest.mark.parametrize('kernel_shape', kernel_shapes)
@pytest.mark.parametrize('p', ps)
@pytest.mark.parametrize('pad', pads)
@pytest.mark.parametrize('stride', strides)
def test_lppool(in_type, in_shape, auto_pad, kernel_shape, p, pad, stride, request):
    model_def = _make_module(in_type, in_shape, auto_pad, kernel_shape, p, pad, stride)
    overwrite_cfg = """
     judge:
       specifics:
         - matchs:
             target: [cpu, k210]
             ptq: true
           threshold: 0.91
     """
    runner = OnnxTestRunner(request.node.name, overwrite_configs=overwrite_cfg)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_lppool.py'])
