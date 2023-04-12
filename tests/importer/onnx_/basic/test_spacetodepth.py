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


def _make_module(in_shape, blocksize):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # output
    output_shape = [in_shape[0], in_shape[1] *
                    (blocksize ** 2), in_shape[2] // blocksize, in_shape[3] // blocksize]
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    outputs.append('output')

    # blocksize attribute
    attributes_dict['blocksize'] = blocksize

    node = onnx.helper.make_node(
        'SpaceToDepth',
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
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]

blocksizes = [
    2,
    4
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('blocksize', blocksizes)
def test_spacetodepth(in_shape, blocksize, request):
    model_def = _make_module(in_shape, blocksize)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_spacetodepth.py'])
