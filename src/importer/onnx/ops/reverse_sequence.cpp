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
    assert(input_shape.size() >= 2);
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

    shape_t batch_shape = input_shape;
    batch_shape[batch_axis] = 1;
    std::vector<shape_t> batch_concat_shapes(input_shape[batch_axis], batch_shape);
    auto batch_concat = graph_.emplace<concat>(input_type, batch_concat_shapes, batch_axis);
    batch_concat->name(op_name + "/concat_batch_axis");
    size_t batch_concat_idx = 0;

    axis_t stride(input_shape.size(), 1);
    for (auto i = 0; i < input_shape[batch_axis]; i++)
    {
        // slice at batch_axis first
        std::cout << std::endl
                  << "batch: i = " << i << std::endl;
        axis_t batch_begin(input_shape.size(), 0);
        batch_begin[batch_axis] = i;
        // std::cout << "batch_begin: ";
        // for (auto axis : batch_begin)
        //     std::cout << axis << " ";
        // std::cout << std::endl;

        axis_t batch_end(input_shape.begin(), input_shape.end());
        batch_end[batch_axis] = i + 1;
        // std::cout << "batch_end: ";
        // for (auto axis : batch_end)
        //     std::cout << axis << " ";
        // std::cout << std::endl;

        auto batch_sl = graph_.emplace<slice>(input_type, input_shape, batch_begin, batch_end, stride, 0, 0, 0, 0);
        batch_sl->name(op_name + ".slice_" + std::to_string(i) + "(batch_axis)");
        input_tensors_.emplace(&batch_sl->input(), input);

        // reverse along time_axis
        shape_t time_shape = batch_sl->output().shape();
        assert(sequence_lens[i] <= input_shape[time_axis]);
        if (sequence_lens[i] <= 1)
        {
            // do not need to reverse
            batch_concat->input_at(batch_concat_idx++).connect(batch_sl->output());
        }
        else
        {
            std::vector<slice *> slices;
            slices.reserve(input_shape[time_axis]);

            // reverse the first sequence_lens[i] elements
            for (auto j = sequence_lens[i] - 1; j >= 0; j--)
            {
                axis_t time_begin(input_shape.size(), 0);
                time_begin[batch_axis] = i;
                time_begin[time_axis] = j;
                std::cout << "time: j = " << j << std::endl;
                std::cout << "time_begin: ";
                for (auto axis : time_begin)
                    std::cout << axis << " ";
                std::cout << std::endl;

                axis_t time_end(time_shape.begin(), time_shape.end());
                time_end[batch_axis] = i + 1;
                time_end[time_axis] = j + 1;
                std::cout << "time_end: ";
                for (auto axis : time_end)
                    std::cout << axis << " ";
                std::cout << std::endl;

                auto time_sl = graph_.emplace<slice>(input_type, time_shape, time_begin, time_end, stride, 0, 0, 0, 0);
                time_sl->name(op_name + ".slice_" + std::to_string(j) + "(time_axis)");
                time_sl->input().connect(batch_sl->output());

                slices.push_back(time_sl);
            }

            // the left elements will be one slice
            if (sequence_lens[i] < input_shape[time_axis])
            {
                axis_t time_begin(input_shape.size(), 0);
                time_begin[batch_axis] = i;
                time_begin[time_axis] = sequence_lens[i];
                std::cout << "time_begin: ";
                for (auto axis : time_begin)
                    std::cout << axis << " ";
                std::cout << std::endl;

                axis_t time_end(time_shape.begin(), time_shape.end());
                time_end[batch_axis] = i + 1;
                std::cout << "time_end: ";
                for (auto axis : time_end)
                    std::cout << axis << " ";
                std::cout << std::endl;

                auto time_sl = graph_.emplace<slice>(input_type, time_shape, time_begin, time_end, stride, 0, 0, 0, 0);
                time_sl->name(op_name + ".slice(time_axis)");
                time_sl->input().connect(batch_sl->output());

                slices.push_back(time_sl);
            }

            // concat the slices
            std::vector<shape_t> concat_shapes;
            std::cout << "concat shapes: " << std::endl;
            for (auto &s : slices)
            {
                std::cout << "name: (" << s->name() << "): ";
                concat_shapes.push_back(s->output().shape());
                for (auto dim : s->output().shape())
                {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
            }

            auto time_concat = graph_.emplace<concat>(input_type, concat_shapes, time_axis);
            time_concat->name(op_name + ".concat_" + std::to_string(i) + "(time_axis)");

            size_t time_con_idx = 0;
            for (auto &s : slices)
            {
                time_concat->input_at(time_con_idx++).connect(s->output());
            }

            // concat the time concats
            batch_concat->input_at(batch_concat_idx++).connect(time_concat->output());
        }
    }

    output_tensors_.emplace(node.output()[0], &batch_concat->output());
}
