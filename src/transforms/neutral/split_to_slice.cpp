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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/split.h>
#include <nncase/ir/visitor.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/neutral/split_to_slice.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool split_to_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (auto old_split = node_cast<split>(node))
    {
        context.inputs.emplace_back(&old_split->input());
        for (size_t i = 0; i < old_split->outputs().size(); i++)
            context.outputs.emplace_back(&old_split->output_at(i));

        context.matched_nodes.emplace_back(old_split);
        return true;
    }

    return false;
}

void split_to_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    std::vector<std::span<input_connector *const>> inputs_vec = {};
    for (size_t i = 0; i < context.outputs.size(); i++)
        inputs_vec.push_back(context.outputs[i]->connections());

    auto &old_split = static_cast<split &>(*context.matched_nodes[0]);
    std::vector<size_t> indices_or_sections = old_split.indices_or_sections();
    int32_t axis = old_split.axis();
    if (axis == -1)
    {
        axis = output.shape().size() + axis;
    }
    bool is_indices = old_split.is_indices();

    std::vector<slice *> new_slices = {};
    axis_t begin = {}, end = {}, strides = {};

    for (size_t i = 0; i < output.shape().size(); i++)
    {
        if (axis == (int32_t)i)
        {
            begin.push_back(0);
            end.push_back(0);
            strides.push_back(1);
        }
        else
        {
            begin.push_back(0);
            end.push_back(output.shape()[i]);
            strides.push_back(1);
        }
    }

    if (is_indices)
    {
        size_t indices_count = 0;

        for (size_t i = 0; i < indices_or_sections.size(); i++)
            indices_count += indices_or_sections[i];
        // might should be '==' below, but the model might be wrong, so use '<=' currently.
        assert(indices_count <= output.shape()[axis]);

        for (size_t i = 0; i < indices_or_sections.size(); i++)
        {
            end[axis] += indices_or_sections[i];
            new_slices.push_back(context.graph.emplace<slice>(output.type(), output.shape(), begin, end, strides, 0, 0, 0, 0));
            begin[axis] = end[axis];
        }
        if (indices_count < output.shape()[axis])
        {
            end[axis] = output.shape()[axis];
            new_slices.push_back(context.graph.emplace<slice>(output.type(), output.shape(), begin, end, strides, 0, 0, 0, 0));
        }
    }
    else
    {
        assert(indices_or_sections.size() == 1);
        assert(output.shape()[axis] % indices_or_sections[0] == 0);
        for (size_t i = 0; i < indices_or_sections[0]; i++)
        {
            end[axis] += output.shape()[axis] / indices_or_sections[0];
            new_slices.push_back(context.graph.emplace<slice>(output.type(), output.shape(), begin, end, strides, 0, 0, 0, 0));
            begin[axis] = end[axis];
        }
    }

    for (size_t i = 0; i < new_slices.size(); i++)
        new_slices[i]->input().connect(output);

    for (size_t i = 0; i < inputs_vec.size(); i++)
    {
        for (auto &in : dup(inputs_vec[i]))
        {
            in->connect(new_slices[i]->output());
        }
    }
}
