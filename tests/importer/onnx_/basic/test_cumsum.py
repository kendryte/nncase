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
from onnx import AttributeProto, TensorProto, GraphProto, numpy_helper
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, axis, exclusive, reverse):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    outputs.append('output')

    # axis
    axis = helper.make_tensor(
        'axis',
        TensorProto.INT32,
        dims=[1],
        vals=[axis]
    )
    inputs.append('axis')
    initializers.append(axis)

    # exclusive
    if exclusive is not None:
        attributes_dict['exclusive'] = exclusive

    # reverse
    if reverse is not None:
        attributes_dict['reverse'] = reverse

    node = onnx.helper.make_node(
        'CumSum',
        inputs=inputs,
        outputs=outputs,
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

axes = [
    # axis 0 ONNXRuntimeError
    # 0,
    1,
    2,
    3,
    -1,
    -2,
    -3,
    # -4
]

exclusives = [
    None,
    0,
    1
]

reverses = [
    None,
    0,
    1
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('exclusive', exclusives)
@pytest.mark.parametrize('reverse', reverses)
def test_cumsum(in_shape, axis, exclusive, reverse, request):
    model_def = _make_module(in_shape, axis, exclusive, reverse)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_cumsum.py'])
