/* Copyright 2020 Alexey Chernov <4ernov@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../onnx_importer.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/roi_align.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_RoiAlign(const NodeProto &node)
{
    assert(node.input().size() == 3);
    assert(node.output().size() == 1);
    const auto &op_name { generate_name(node) };

    // input
    const auto &input = node.input()[0];
    const datatype_t input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    assert(input_shape.size() == 4);

    // rois input
    const auto rois = node.input()[1];
    const datatype_t rois_type = get_datatype(rois).value();
    auto rois_shape = get_shape(rois);
    assert(rois_shape.size() == 2 && rois_shape[1] == 4);
    auto rois_vec = get_constant_value<float>(rois);
    auto rois_const = graph_.emplace<constant>(rois_type, rois_shape, rois_vec);

    // batch_indices input
    const auto batch_indices = node.input()[2];
    auto batch_indices_shape = get_shape(batch_indices);
    assert(batch_indices_shape.size() == 1 && batch_indices_shape[0] == rois_shape[0]);
    auto batch_indices_vec = get_constant_value<int64_t>(batch_indices);
    auto batch_indices_const = graph_.emplace<constant>(dt_int64, batch_indices_shape, batch_indices_vec);

    // output
    const auto &output = node.output()[0];

    // mode
    auto mode_str = get_attribute<std::string>(node, "mode").value_or("avg");
    roi_align_mode_t mode = mode_str == "avg" ? roi_align_avg : roi_align_max;

    // spatial_scale
    auto spatial_scale = get_attribute<float>(node, "spatial_scale").value_or(1.0);

    // output_height
    auto output_height = get_attribute<int64_t>(node, "output_height").value_or(1);

    // output_width
    auto output_width = get_attribute<int64_t>(node, "output_width").value_or(1);

    // sampling_ratio
    auto sampling_ratio = get_attribute<int64_t>(node, "sampling_ratio").value_or(0);

    auto op = graph_.emplace<roi_align>(input_type, input_shape, rois_shape, batch_indices_shape, mode, spatial_scale,
        output_height, output_width, sampling_ratio);
    op->name(op_name + "/roi_aligh");
    op->rois().connect(rois_const->output());
    op->batch_indices().connect(batch_indices_const->output());

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
