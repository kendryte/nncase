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


def _make_module(in_shape, alpha, beta, bias, size):
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)

    node = onnx.helper.make_node(
        'LRN',
        inputs=['input'],
        outputs=['output'],
        alpha=alpha,
        beta=beta,
        bias=bias,
        size=size
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [input],
        [output]
    )

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 3, 60, 72],
    [1, 3, 224, 224]
]

alphas = [
    0.00009999999747378752,
    0.22
]

betas = [
    0.20,
    0.75
]

biases = [
    0.75,
    1.0
]

sizes = [
    3,
    5
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('alpha', alphas)
@pytest.mark.parametrize('beta', betas)
@pytest.mark.parametrize('bias', biases)
@pytest.mark.parametrize('size', sizes)
def test_lrn(in_shape, alpha, beta, bias, size, request):
    model_def = _make_module(in_shape, alpha, beta, bias, size)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_lrn.py'])
