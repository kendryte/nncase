/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/take.h>
#include <nncase/ir/visitor.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/neutral/take_to_slice.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool take_to_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (auto t = node_cast<take>(node))
    {
        if (try_get_direct_parent<constant>(*t, 1))
        {
            context.inputs.emplace_back(&t->input());
            context.inputs.emplace_back(&t->indices());
            context.outputs.emplace_back(&t->output());

            context.matched_nodes.emplace_back(t);
            return true;
        }
    }

    return false;
}

void take_to_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &indices = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_take = static_cast<take &>(*context.matched_nodes[0]);
    auto indices_const = node_cast<constant>(indices.owner());
    const int32_t *indices_values = reinterpret_cast<const int32_t *>(indices_const->data().data());
    size_t indices_size = xt::compute_size(indices_const->output().shape());
    std::vector<int32_t> vec(indices_values, indices_values + indices_size);
    sort(vec.begin(), vec.end());

    int32_t t_axis = old_take.axis();
    std::string t_mode = old_take.mode();

    assert(vec[0] >= 0);
    const int32_t start = vec[0], step = vec.size() == 1 ? 1 : vec[1] - vec[0];
    const int32_t stop = vec[indices_size - 1] + step;
    for (size_t i = 0; i < vec.size() - 1; i++)
    {
        assert(step - (vec[i + 1] - vec[i]) == 0);
    }

    axis_t begin = {}, end = {}, strides = {};

    // TODO: double check if the interval is open or close.
    for (size_t i = 0; i < output.shape().size(); i++)
    {
        // TODO: double check the axis.
        if ((int32_t)i != t_axis)
        {
            begin.push_back(0);
            strides.push_back(1);
            end.push_back(output.shape()[i]);
        }
        else
        {
            begin.push_back(start);
            strides.push_back(step);
            if (t_mode == "clip")
            {
                if (stop <= (int32_t)output.shape()[i])
                {
                    end.push_back(stop);
                }
                else
                {
                    end.push_back(output.shape()[i]);
                    assert((stop - start) % step == 0);
                }
            }
            else if (t_mode == "wrap")
            {
                if (stop <= (int32_t)output.shape()[i])
                {
                    end.push_back(stop);
                }
                else
                {
                    end.push_back(start + ((output.shape()[i] - start) / step) * step);
                }
            }
            else
            {
                end.push_back(stop);
            }
        }
    }

    auto new_slice = context.graph.emplace<slice>(output.type(), output.shape(), begin, end, strides, 0, 0, 0, 0, 0);

    new_slice->input().connect(output);
    old_take.indices().clear_connection();

    for (auto &in : dup(inputs))
        in->connect(new_slice->output());
}
