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
import copy

def _make_module(in_shape, rois, batch_indices, mode, output_height, output_width, sampling_ratio, spatial_scale, op_version):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}
    nodes = []

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # roi input
    rois_array = np.array(rois).reshape(-1, 4)
    rois_tensor = helper.make_tensor(
        'rois',
        TensorProto.FLOAT,
        dims=rois_array.shape,
        vals=rois_array.flatten().tolist()
    )
    initializers.append(rois_tensor)
    inputs.append('rois')

    # batch_indices input
    batch_indices_tensor = helper.make_tensor(
        'batch_indices',
        TensorProto.INT64,
        dims=[len(batch_indices)],
        vals=batch_indices
    )
    initializers.append(batch_indices_tensor)
    inputs.append('batch_indices')

    # output
    out_shape = [rois_array.shape[0], in_shape[1], output_height if output_height is not None else 1, output_width if output_width is not None else 1]
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
    outputs.append('output')

    # mode
    if mode is not None:
        attributes_dict['mode'] = mode

    # output_height
    if output_height is not None:
        attributes_dict['output_height'] = output_height

    # output_width
    if output_width is not None:
        attributes_dict['output_width'] = output_width

    # sampling_ratio
    if sampling_ratio is not None:
        attributes_dict['sampling_ratio'] = sampling_ratio

    # spatial_scale
    if spatial_scale is not None:
        attributes_dict['spatial_scale'] = spatial_scale

    # RoiAlign node
    node = onnx.helper.make_node(
        'RoiAlign',
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
        initializer=initializers
    )

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='onnx helper', opset_imports=[op])

    return model_def

in_shapes = [
    [1, 3, 16, 16]
]

rois = [
    [[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]]
]

batch_indices = [
    [0, 0, 0],
]

modes = [
    None,
    'avg',
    'max'
]

output_heights = [
    None,
    5
]

output_widths = [
    None,
    5
]

sampling_ratios = [
    None,
    2
]

spatial_scales = [
    None,
    1.0
]

op_versions = [
    10
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('roi', rois)
@pytest.mark.parametrize('batch_index', batch_indices)
@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('output_height', output_heights)
@pytest.mark.parametrize('output_width', output_widths)
@pytest.mark.parametrize('sampling_ratio', sampling_ratios)
@pytest.mark.parametrize('spatial_scale', spatial_scales)
@pytest.mark.parametrize('op_version', op_versions)
def test_roi_align(in_shape, roi, batch_index, mode, output_height, output_width, sampling_ratio, spatial_scale, op_version, request):
    model_def = _make_module(in_shape, roi, batch_index, mode, output_height, output_width, sampling_ratio, spatial_scale, op_version)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_roi_align.py'])
