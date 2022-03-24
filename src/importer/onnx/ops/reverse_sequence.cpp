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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/slice.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_ReverseSequence(const NodeProto &node)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);
    const auto &op_name { generate_name(node) };

    // input
    const auto &input = node.input()[0];
    auto input_shape = get_shape(input);
    auto input_shape_size = input_shape.size();
    assert(input_shape_size >= 2);
    const datatype_t input_type = get_datatype(input).value();

    // sequence_lens
    auto sequence_lens = get_constant_value<int, int64_t>(node.input()[1]);

    // batch_axis
    int32_t batch_axis = get_attribute<int>(node, "batch_axis").value_or(1);
    assert(batch_axis == 0 || batch_axis == 1);
    assert(sequence_lens.size() == input_shape[batch_axis]);

    // time_axis
    int32_t time_axis = get_attribute<int>(node, "time_axis").value_or(0);
    assert(time_axis == 0 || time_axis == 1);

    axis_t forward_stride(input_shape_size, 1);
    std::vector<output_connector *> batch_output_connectors;
    batch_output_connectors.reserve(input_shape[batch_axis]);
    shape_t batch_sl_shape;
    for (auto i = 0; i < input_shape[batch_axis]; i++)
    {
        // 1. slice at batch_axis first
        axis_t batch_begin(input_shape_size, 0);
        batch_begin[batch_axis] = i;

        axis_t batch_end(input_shape.begin(), input_shape.end());
        batch_end[batch_axis] = i + 1;

        auto batch_sl = graph_.emplace<slice>(input_type, input_shape, batch_begin, batch_end, forward_stride, 0, 0, 0, 0, 0);
        batch_sl->name(op_name + ".slice_" + std::to_string(i) + "(batch_axis)");
        input_tensors_.emplace(&batch_sl->input(), input);

        // 2. reverse along time_axis
        std::vector<slice *> slices;
        slices.reserve(2);
        batch_sl_shape = batch_sl->output().shape();
        assert(sequence_lens[i] <= input_shape[time_axis]);
        if (sequence_lens[i] == 1)
        {
            batch_output_connectors.push_back(&batch_sl->output());
            continue;
        }

        axis_t time_begin(batch_sl_shape.size(), 0);
        time_begin[batch_axis] = 0;
        time_begin[time_axis] = sequence_lens[i] - 1;

        axis_t time_end(batch_sl_shape.begin(), batch_sl_shape.end());
        time_end[batch_axis] = 1;
        time_end[time_axis] = -1;
        int32_t time_end_mask = (1 << time_axis);

        axis_t time_stride(batch_sl_shape.size(), 1);
        time_stride[time_axis] = -1;

        auto backward_sl = graph_.emplace<slice>(input_type, batch_sl_shape, time_begin, time_end, time_stride, 0, time_end_mask, 0, 0, 0);
        backward_sl->name(op_name + ".slice(backward)");
        backward_sl->input().connect(batch_sl->output());
        slices.push_back(backward_sl);

        // 3. copy the unreversed slices
        if (sequence_lens[i] < input_shape[time_axis])
        {
            axis_t time_begin(input_shape.size(), 0);
            time_begin[batch_axis] = 0;
            time_begin[time_axis] = sequence_lens[i];

            axis_t time_end(batch_sl_shape.begin(), batch_sl_shape.end());
            time_end[batch_axis] = 1;

            auto forward_sl = graph_.emplace<slice>(input_type, batch_sl_shape, time_begin, time_end, forward_stride, 0, 0, 0, 0, 0);
            forward_sl->name(op_name + ".slice(forward)");
            forward_sl->input().connect(batch_sl->output());
            slices.push_back(forward_sl);
        }

        // 4. try to concat the time slices
        assert(slices.size() >= 1);
        if (slices.size() == 1)
        {
            // only one slice
            batch_output_connectors.push_back(&slices[0]->output());
        }
        else
        {
            std::vector<shape_t> time_concat_shapes;
            time_concat_shapes.reserve(slices.size());
            for (auto &s : slices)
            {
                time_concat_shapes.push_back(s->output().shape());
            }
            auto time_concat = graph_.emplace<concat>(input_type, time_concat_shapes, time_axis);
            time_concat->name(op_name + ".concat_" + std::to_string(i) + "(time_axis)");

            size_t idx = 0;
            for (auto &s : slices)
            {
                time_concat->input_at(idx++).connect(s->output());
            }
            batch_output_connectors.push_back(&time_concat->output());
        }
    }

    // concat the batch slices
    std::vector<shape_t> batch_concat_shapes(input_shape[batch_axis], batch_sl_shape);
    auto batch_concat = graph_.emplace<concat>(input_type, batch_concat_shapes, batch_axis);
    batch_concat->name(op_name + "/concat(batch_axis)");

    size_t idx = 0;
    for (auto &out : batch_output_connectors)
    {
        batch_concat->input_at(idx++).connect(*out);
    }

    output_tensors_.emplace(node.output()[0], &batch_concat->output());
}
