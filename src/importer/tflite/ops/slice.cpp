/* Copyright 2019-2021 Canaan Inc.
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
#include "../tflite_importer.h"
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/slice.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SLICE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto begin = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto size = load_axis<int32_t>(get_tensor(op.inputs(), 2));
    axis_t end(begin.size());
    for (size_t i = 0; i < begin.size(); i++)
        end[i] = begin[i] + size[i];

    [[maybe_unused]] auto &options = *op.builtin_options_as_SliceOptions();
    auto node = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}

DEFINE_TFLITE_LOWER(STRIDED_SLICE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    auto begin = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto end = load_axis<int32_t>(get_tensor(op.inputs(), 2));
    auto strides = load_axis<int32_t>(get_tensor(op.inputs(), 3));
    auto &options = *op.builtin_options_as_StridedSliceOptions();
    auto node = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end, strides, options.begin_mask(),
        options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
    node->name(get_tensor(op.outputs(), 0).name()->string_view());
    auto rshape = graph_.emplace<bitcast>(node->output().type(), node->output().shape(), get_shape(output.shape()));
    rshape->name(node->name() + "/reshape");
    rshape->input().connect(node->output());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &rshape->output());
}
