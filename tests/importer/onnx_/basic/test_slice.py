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


def _make_module(in_shape, start, end, axes, step, outshape, op_version, value_format):
    input_names = []
    output_names = []
    inputs = []
    outputs = []
    initializers = []
    nodes = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    input_names.append('input')
    inputs.append(input)

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, outshape)
    output_names.append('output')
    outputs.append(output)

    # opset 1
    if op_version == 1:
        if axes is not None:
            attributes_dict['axes'] = axes

        attributes_dict['starts'] = start
        attributes_dict['ends'] = end
    else:
        # opset 10/11/13
        # starts
        start_tensor = helper.make_tensor(
            'starts',
            TensorProto.INT64,
            dims=[len(start)],
            vals=start
        )

        if value_format == 'initializer':
            initializers.append(start_tensor)
        else:
            start_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['starts'],
                value=start_tensor
            )
            nodes.append(start_node)

        input_names.append('starts')

        # ends
        end_tensor = helper.make_tensor(
            'ends',
            TensorProto.INT64,
            dims=[len(end)],
            vals=end
        )

        if value_format == 'initializer':
            initializers.append(end_tensor)
        else:
            end_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['ends'],
                value=end_tensor
            )
            nodes.append(end_node)

        input_names.append('ends')

        # axes
        if axes is not None:
            axes_tensor = helper.make_tensor(
                'axes',
                TensorProto.INT64,
                dims=[len(end)],
                vals=axes
            )

            if value_format == 'initializer':
                initializers.append(axes_tensor)
            else:
                axes_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['axes'],
                    value=axes_tensor
                )
                nodes.append(axes_node)

            input_names.append('axes')

        # steps
        if step is not None:
            step_tensor = helper.make_tensor(
                'steps',
                TensorProto.INT64,
                dims=[len(step)],
                vals=step
            )

            if value_format == 'initializer':
                initializers.append(step_tensor)
            else:
                step_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['steps'],
                    value=step_tensor
                )
                nodes.append(step_node)

            input_names.append('steps')

    slice_node = onnx.helper.make_node(
        'Slice',
        inputs=input_names,
        outputs=output_names,
        **attributes_dict
    )
    nodes.append(slice_node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        inputs,
        outputs,
        initializer=initializers
    )

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def


in_shapes_starts_ends_axes_steps_outshapes = [
    [[1, 20, 10, 5], [0, 0], [3, 10], [1, 2], [1, 1], [1, 3, 10, 5]],
    [[1, 20, 10, 5], [0, 0, 0, 3], [1, 20, 10, 4], None, None, [1, 20, 10, 1]],
    [[1, 20, 10, 5], [0, 0, 0, 3], [1, 20, 10, 4], [0, 1, 2, 3], None, [1, 20, 10, 1]],
    [[1, 20, 10, 5], [1], [1000], [2], [1], [1, 20, 9, 5]],
    [[1, 20, 10, 5], [0], [-1], [1], [1], [1, 19, 10, 5]],
    [[1, 20, 10, 5], [0, 0, 3], [20, 10, 4], [-3, -2, -1], None, [1, 20, 10, 1]],
    [[1, 20, 10, 5], [ 20, 10, 4], [ 0, 0, 1], [1, 2, 3], [-1, -3, -2], [1, 20, 4, 2]],
    [[1, 20, 10, 5], [-1], [-9223372036854775807], [1], [-1], [1, 20, 10, 5]]
]


op_versions_and_value_formats = [
    [1, 'attribute'],
    [10, 'initializer'],
    [10, 'constant'],
    [11, 'initializer'],
    [11, 'constant'],
    [13, 'initializer'],
    [13, 'constant']
]


@pytest.mark.parametrize('in_shape_start_end_axis_step_outshape', in_shapes_starts_ends_axes_steps_outshapes)
@pytest.mark.parametrize('op_versions_and_value_format', op_versions_and_value_formats)
def test_slice(in_shape_start_end_axis_step_outshape, op_versions_and_value_format, request):
    in_shape, start, end, axes, step, outshape = in_shape_start_end_axis_step_outshape
    op_version, value_format = op_versions_and_value_format
    if op_version != 1 or (op_version == 1 and step is not None and all([x == 1 for x in step])):
        model_def = _make_module(in_shape, start, end, axes, step,
                                 outshape, op_version, value_format)

        runner = OnnxTestRunner(request.node.name)
        model_file = runner.from_onnx_helper(model_def)
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_slice.py'])
