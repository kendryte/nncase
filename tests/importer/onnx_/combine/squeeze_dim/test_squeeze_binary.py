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

def _make_module(op, in_type, in_shape_0, in_shape_1):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input1
    input1 = helper.make_tensor_value_info('input1', in_type, in_shape_0)
    inputs.append('input1')

    # set input2 to avoid SIGFPE for div op.
    tensor = helper.make_tensor(
        'input2',
        in_type,
        dims=in_shape_1,
        vals=(np.random.rand(*in_shape_1) + 2).astype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[in_type]).flatten().tolist()
    )
    inputs.append('input2')
    initializers.append(tensor)

    # output
    x = np.random.randn(*in_shape_0)
    y = np.random.randn(*in_shape_1)
    output_shape = np.add(x, y).shape
    output = helper.make_tensor_value_info('output', in_type, output_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        op,
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input1],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')
    return model_def

ops = [
    'Add',
]

in_types = [
    TensorProto.FLOAT,
]

in_shapes = [
    [[1, 3, 4, 5, 2], [1]],   
    [[4, 3, 4, 5, 2], [2]],   
    [[1, 3, 4, 5, 2], [1, 3, 4, 1, 1]],   
    [[1, 3, 16, 16, 2], [1, 1, 1, 16, 1]],
    [[1, 3, 16, 16, 2], [1, 3, 1, 16, 1]],
    [[2, 3, 16, 16, 2], [2, 1, 16, 1, 2]],
    [[1, 3, 16, 16, 2, 3], [1, 3, 1, 16, 1, 1]],
    [[1, 3, 16, 16, 2, 3], [1, 3, 1, 16, 2, 1]],
]

@pytest.mark.parametrize('op', ops)
@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
def test_squeeze_binary(op, in_type, in_shape, request):
    model_def = _make_module(op, in_type, in_shape[0], in_shape[1])

    runner = OnnxTestRunner(request.node.name, ['cpu', 'k210', 'k510'])
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_squeeze_binary.py'])
