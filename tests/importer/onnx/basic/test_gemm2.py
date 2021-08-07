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
import numpy as np
from onnx_test_runner import OnnxTestRunner


def _make_module():
    input_A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [112, 224])
    input_B = helper.make_tensor("B", TensorProto.FLOAT,
                                 dims=(56, 224),
                                 vals=np.random.randn(56, 224).astype(np.float32).flatten().tolist())
    input_C = helper.make_tensor("C", TensorProto.FLOAT,
                                 dims=(1, 56),
                                 vals=np.random.randn(1,56).astype(np.float32).flatten().tolist())
    initializers = []
    initializers.append(input_B)
    initializers.append(input_C)

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [112, 56])

    node_def = helper.make_node(
        'Gemm',
        ['A', 'B', 'C'],
        ['output'],
        alpha=2.0,
        beta=3.0,
        transA=0,
        transB=1
    )

    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input_A],
        [output],
        initializer=initializers
    )

    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


def test_gemm2(request):
    model_def = _make_module()

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_gemm2.py'])
