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

def _make_module(compare_op, in_type, in_shape_0, in_shape_1):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input1 = helper.make_tensor_value_info('input1', in_type, in_shape_0)
    inputs.append('input1')

    input2 = helper.make_tensor_value_info('input2', in_type, in_shape_1)
    inputs.append('input2')

    # output
    x = np.random.randn(*in_shape_0)
    y = np.random.randn(*in_shape_1)
    output_shape = np.equal(x, y).shape
    output = helper.make_tensor_value_info('output', TensorProto.BOOL, output_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        compare_op,
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input1, input2],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def

compare_ops = {
    'Equal',
    'Greater',
    'GreaterOrEqual',
    'Less',
    'LessOrEqual'
}

in_types = [
    TensorProto.BOOL,
    TensorProto.FLOAT,
    TensorProto.INT32,
    TensorProto.INT64
]

in_shapes = [
    [[1, 3, 16, 16], [1]],
    [[1, 3, 16, 16], [16]],
    [[1, 3, 16, 16], [1, 16]],
    [[1, 3, 16, 16], [1, 16, 16]],
    [[1, 1, 16, 16], [3, 3, 1, 16]],
]

# pytest with multiprocess will fail on uppercase names
@pytest.mark.parametrize('compare_op', compare_ops, ids=['equal','greater','greater_or_equal','less','less_or_equal'])
@pytest.mark.parametrize('in_type', in_types)
@pytest.mark.parametrize('in_shape', in_shapes)
def test_compare(compare_op, in_type, in_shape, request):
    # TensorProto.BOOL is supported by Equal only.
    if compare_op == "Equal" or in_type != TensorProto.BOOL:
        model_def = _make_module(compare_op, in_type, in_shape[0], in_shape[1])
        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_compare.py'])
