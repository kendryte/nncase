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
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, alpha):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    outputs.append('output')

    # alpha
    if alpha is not None:
        attributes_dict['alpha'] = alpha

    tensor = helper.make_tensor(
        'input2',
        TensorProto.FLOAT,
        dims=in_shape,
        vals=(np.random.rand(*in_shape) + 2).astype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[TensorProto.FLOAT]).flatten().tolist()
    )
    # inputs.append('input2')
    initializers.append(tensor)

    # enable default alphas: None -> 1
    node = onnx.helper.make_node(
        'Mul',
        inputs=[inputs[0], 'input2'],
        outputs=['0'],
    )
    nodes.append(node)

    # Celu node
    node = onnx.helper.make_node(
        'ThresholdedRelu',
        inputs=['0'],
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

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

alphas = [
    None,
    0.5,
    1.5
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('alpha', alphas)
def test_threadholdrelu(in_shape, alpha, request):
    model_def = _make_module(in_shape, alpha)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_threadholdrelu.py'])
