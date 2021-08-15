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
from onnx import helper, OperatorSetIdProto
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np
from collections import defaultdict
from typing import List


class BinaryGraph():
    def __init__(self, inverses: List[bool]) -> None:
        self.initializers = []
        self.constant_count = 0
        self.nodes = []
        self.op_nodes = defaultdict(list)
        self.inverses = inverses

    def make_node(self, name: str, input: TensorProto, dims=[1], vals=[1], final=False) -> TensorProto:
        op_type = name.split('_')[0].capitalize()
        inputs = [input.name, f'Constant_{self.constant_count}']
        if self.inverses[self.constant_count] and op_type != 'Div':
            inputs = inputs[::-1]
        node = helper.make_node(op_type, inputs, ['output' if final else name], name=name)
        self.initializers.append(helper.make_tensor(
            f'Constant_{self.constant_count}', TensorProto.FLOAT, dims=dims, vals=vals))
        self.constant_count += 1
        self.nodes.append(node)
        return node


def _make_module(in_shape, op_version: int, inverse: int):
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)

    # NOTE the  a/b != b/a; a-0 != 0-a
    #  (the random data will be zero, can't not be divide)
    g = BinaryGraph(inverse)
    x = g.make_node('mul_0', input, vals=[1])
    x = g.make_node('div_0', x, vals=[1])
    x = g.make_node('mul_1', x, vals=[3])
    x = g.make_node('sub_0', x, vals=[0])
    x = g.make_node('add_0', x, vals=[-2])
    x = g.make_node('div_1', x, vals=[1])
    x = g.make_node('sub_1', x, vals=[0])
    x = g.make_node('mul_2', x, vals=[1])
    x = g.make_node('div_2', x, vals=[1])
    x = g.make_node('add_1', x, vals=[-2], final=True)

    graph_def = helper.make_graph(g.nodes,
                                  'test-model', [input], [output], initializer=g.initializers)
    op = OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])
    print(model_def)

    return model_def


in_shapes = [
    [1, 12, 14, 14],
    [1, 64, 56, 56],
]

op_versions = [
    11,
]

# this code for generate bin inverse
# for i in range(5):
#     print(np.random.randint(0, 2, 12))
inverses = [
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('op_version', op_versions)
@pytest.mark.parametrize('inverse', inverses)
def test_unary_with_clamp(in_shape, op_version, inverse, request):
    model_def = _make_module(in_shape, op_version, inverse)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_nonsence_binary.py'])
