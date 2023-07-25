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


def _make_module(in_shape, kvalue, upper):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    k = helper.make_tensor('k', TensorProto.INT64, np.array(
        kvalue).shape, np.array(kvalue).flatten().tolist())
    initializers.append(k)

# output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    outputs.append('output')

    attributes_dict['upper'] = upper

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['input', 'k'],
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

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]

ks = [
    1,
    2,
    -1,
    -2
]

uppers = [
    0,
    1
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('k', ks)
@pytest.mark.parametrize('upper', uppers)
def test_erf(in_shape, upper, k, request):
    model_def = _make_module(in_shape, k, upper)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_trilu.py'])
